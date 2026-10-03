"""官方 SDK 0.7.2 的 HTTP 契约、错误与重试；推理使用可控 Engine。"""

import asyncio
from collections import Counter
from contextlib import contextmanager
import socket
import threading
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient
import httpx
import pytest
from typesafe_sdk import (
    AsyncTypeSafeClient, Choice, ChoiceAnswer, Noul, NoulAnswer, RetryPolicy,
    Score, ScoreAnswer, SystemOneResponse, TypeSafeClient,
    TypeSafeInternalServerError, TypeSafeUnprocessableEntityError,
)
import uvicorn

from laya.api.app import create_app
from laya.engine.core import EngineError
from laya.api.jev import choice_confidence, score_confidence


PAYLOAD = {
    "model": "jev-latest", "state": {"message": "测试"},
    "questions": {
        "owner": {"type": "choice", "instructions": {"task": "选择"},
                  "criteria": {"甲": {"role": "测试"}, "乙": None}},
        "rating": {"type": "score", "criteria": ["低", {"level": "中"}, ["高"]]},
        "holds": {"type": "noul", "criteria": {"true": ["是"], "false": None}},
    },
}


class FakeEngine:
    def __init__(self):
        self.counters = Counter()
        self.info = {"model": "local-model"}
        self.ready = False
        self.payloads = []
        self.cancelled = 0
        self.retry_calls = 0

    async def start(self):
        self.ready = True

    async def close(self):
        self.ready = False

    async def submit(self, data):
        self.payloads.append(data)
        state = data["state"]
        if state == "retry":
            self.retry_calls += 1
            if self.retry_calls == 1:
                raise EngineError("queue_full", 429)
        if state in ("queue_full", "not_ready", "deadline"):
            raise EngineError(state, {"queue_full": 429, "not_ready": 503, "deadline": 504}[state])
        if state == "too_long":
            raise EngineError("state_token_budget_exceeded", 422)
        if state == "crash":
            raise RuntimeError("private backend details")
        if state == "hang":
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
        answers = {}
        for key, q in data["questions"].items():
            if q["type"] == "noul":
                answers[key] = {"noul": 0.75, "confidence": 0.999, "act_probability": 0.5}
            else:
                n = len(q["criteria"])
                p = [0.8, 0.2] if n == 2 else [0, 0.5, 0.5] if n == 3 else [1 / n] * n
                keys = list(q["criteria"]) if q["type"] == "choice" else [str(i) for i in range(n)]
                answers[key] = {"probabilities": dict(zip(keys, p)), "confidence": 0.999,
                                "act_probability": 0.5}
        return {"answers": answers, "model": self.info, "usage": {"input_tokens": 12},
                "timings": {"total_ms": 1}}

    def metrics(self):
        return dict(self.counters)


def make_app(**overrides):
    config = SimpleNamespace(model_dir="/tmp/local-model", request_timeout=1,
                             max_body_bytes=16384, max_inflight=2)
    for key, value in overrides.items():
        setattr(config, key, value)
    engine = FakeEngine()
    return create_app(config, engine), engine


@contextmanager
def live_server(app):
    """真实 TCP；SDK 使用 httpx2，不用 httpx mock transport 替代网络。"""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 10
            while not server.started:
                if not thread.is_alive() or time.monotonic() > deadline:
                    pytest.fail("测试 HTTP 服务未就绪")
                time.sleep(0.01)
            yield f"http://127.0.0.1:{port}"
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            assert not thread.is_alive(), "测试 HTTP 服务未退出"


@pytest.fixture
def server():
    app, engine = make_app()
    with live_server(app) as url:
        yield url, engine
    assert not engine.ready


def sdk_options(url):
    return dict(base_url=url, api_key="local", timeout=5, retry=RetryPolicy(max_retries=0))


@pytest.mark.parametrize("p,expected", [([0.5, 0.5], 0), ([1, 0], 1), ([0.8, 0.2], 0.6),
                                       ([0.6, 0.3, 0.1], 0.4)])
def test_choice_confidence_formula(p, expected):
    assert choice_confidence(p) == pytest.approx(expected)


@pytest.mark.parametrize("p,expected", [([0.5, 0.5], 0), ([1, 0], 1), ([0, 0.5, 0.5], 0.25),
                                       ([0.5, 0, 0.5], 0), ([0, 0.57, 0.43], 0.355)])
def test_score_confidence_formula(p, expected):
    assert score_confidence(p) == pytest.approx(expected)


def test_wire_fields_and_structured_content():
    app, engine = make_app()
    with TestClient(app) as client:
        response = client.post("/v1/systemone", json=PAYLOAD)
        assert response.status_code == 200, response.text
        assert response.headers["x-typesafe-request-id"]
        data = response.json()
        assert set(data) == {"model", "answers", "usage"}
        assert data["model"] == "local-model"
        assert data["usage"] == {"input_tokens": 12, "output_tokens": 0}
        assert data["answers"]["holds"] == {"type": "noul", "noul": 0.75}
        assert set(data["answers"]["owner"]) == {"type", "choice", "probabilities", "confidence"}
        assert data["answers"]["owner"]["confidence"] == pytest.approx(0.6)
        rating = data["answers"]["rating"]
        assert set(rating) == {"type", "score", "legend", "probabilities", "confidence"}
        assert rating["score"] == 1.5 and rating["confidence"] == 0.25
        assert rating["legend"] == {"0": "低", "1": {"level": "中"}, "2": ["高"]}
        q = engine.payloads[0]["questions"]
        assert q["owner"]["instructions"] == '{"task": "选择"}'
        assert q["owner"]["criteria"] == {"甲": '{"role": "测试"}', "乙": None}
        assert q["rating"]["instructions"] == ""
        assert q["holds"]["criteria"] == {"true": '["是"]'}
        # 无鉴权；即使带无效 Authorization 也不会切换认证行为。
        assert client.post("/v1/systemone", json=PAYLOAD, headers={"Authorization": "garbage"}).status_code == 200
        schema = client.get("/openapi.json").json()
        assert "/v1/systemone" in schema["paths"] and "/v1/models" in schema["paths"]
        assert "security" not in schema["paths"]["/v1/systemone"]["post"]

        def check_refs(node):
            if isinstance(node, dict):
                if "$ref" in node:
                    target = schema
                    for part in node["$ref"].removeprefix("#/").split("/"):
                        target = target[part.replace("~1", "/").replace("~0", "~")]
                for value in node.values():
                    check_refs(value)
            elif isinstance(node, list):
                for value in node:
                    check_refs(value)

        check_refs(schema["paths"]["/v1/systemone"])
        check_refs(schema["components"])



@pytest.mark.parametrize("body,field", [
    ({"state": "x", "questions": {"q": {"type": "noul"}}}, "model"),
    ({**PAYLOAD, "model": "typo"}, "model"),
    ({**PAYLOAD, "state": None}, "state"),
    ({**PAYLOAD, "questions": {}}, "questions"),
    ({**PAYLOAD, "questions": {"q": {"instructions": "x"}}}, "questions"),
    ({**PAYLOAD, "questions": {"q": {"type": "unknown"}}}, "questions"),
    ({**PAYLOAD, "questions": {"q": {"type": "choice", "criteria": ["a", "b"]}}}, "questions"),
    ({**PAYLOAD, "questions": {"q": {"type": "choice", "criteria": {"a": None}}}}, "questions"),
    ({**PAYLOAD, "questions": {"q": {"type": "score", "criteria": []}}}, "questions"),
    ({**PAYLOAD, "questions": {"q": {"type": "score", "criteria": ["a"]}}}, "questions"),
    ({**PAYLOAD, "questions": {"q": {"type": "score", "criteria": ["a"] * 11}}}, "questions"),
    ({**PAYLOAD, "questions": {"q": {"type": "choice", "criteria": {str(i): None for i in range(17)}}}}, "questions"),
    ({**PAYLOAD, "questions": {str(i): {"type": "noul"} for i in range(17)}}, "questions"),
    ({**PAYLOAD, "state": "too_long"}, "state"),
])
def test_validation_errors(body, field):
    app, _ = make_app()
    with TestClient(app) as client:
        response = client.post("/v1/systemone", json=body)
        assert response.status_code == 422, response.text
        assert response.headers["x-typesafe-request-id"]
        details = response.json()["detail"]
        assert all("input" not in item and "ctx" not in item for item in details)
        assert any(field in item["loc"] for item in details)


def test_local_limits_and_wire_extra_field_policy():
    app, _ = make_app()
    with TestClient(app) as client:
        for q in [{"type": "choice", "criteria": {str(i): None for i in range(16)}},
                  {"type": "score", "criteria": ["level"] * 10}]:
            assert client.post("/v1/systemone", json={**PAYLOAD, "questions": {"q": q}}).status_code == 200
        body = {**PAYLOAD, "future": True, "questions": {
            str(i): {"type": "noul", "instructions": None, "future": True} for i in range(16)}}
        assert client.post("/v1/systemone", json=body).status_code == 200
        assert client.post("/v1/systemone", content="{").status_code == 422
        assert client.post("/v1/systemone", content=b'\xff').status_code == 422
        assert client.post("/v1/systemone", content="x" * 16385).status_code == 413
        # 旧接口依然要求 instructions，也仍拒绝 model。
        assert client.post("/v1/decisions", json=PAYLOAD).status_code == 422
        assert client.post("/v1/decisions", json={"state": "x", "questions": {"q": {"type": "noul"}}}).status_code == 422


def test_official_sync_sdk(server):
    url, _ = server
    with TypeSafeClient(**sdk_options(url)) as client:
        models = client.models.list()
        assert {model.name for model in models.models} == {"local-model", "jev-latest", "laya", "laya-latest"}
        assert models.request_id
        for model in [entry.name for entry in models.models]:
            result = client.system_one(state=PAYLOAD["state"], questions=PAYLOAD["questions"], model=model)
            assert isinstance(result.choices["owner"], ChoiceAnswer)
            assert isinstance(result.scores["rating"], ScoreAnswer)
            assert isinstance(result.nouls["holds"], NoulAnswer)
            assert result.scores["rating"].legend == {0: "低", 1: {"level": "中"}, 2: ["高"]}
            assert set(result.scores["rating"].probabilities) == {0, 1, 2}
            assert result.request_id and result.raw_http_response.status_code == 200

        class CustomResponse(SystemOneResponse):
            owner: ChoiceAnswer

        custom = client.system_one(state="x", questions={
            "owner": Choice(criteria={"甲": None, "乙": None}),
        }, response_model=CustomResponse)
        assert custom.owner.choice == "甲"
        with pytest.raises(TypeSafeUnprocessableEntityError) as error:
            client.system_one(state="x", questions={"q": Noul()}, model="unknown")
        assert error.value.status == 422 and error.value.request_id
        assert "model" in str(error.value)


async def test_official_async_sdk_concurrency(server):
    url, _ = server
    async with AsyncTypeSafeClient(**sdk_options(url)) as client:
        assert (await client.models.list()).models
        results = await asyncio.gather(*(client.system_one(state=["x", i], questions={
            f"q{i}": Choice(criteria={"甲": None, "乙": None}),
            "s": Score(criteria=["低", "高"]), "n": Noul(instructions=["是否成立"]),
        }) for i in range(2)))
        assert len({r.request_id for r in results}) == 2
        for i, result in enumerate(results):
            assert result.choices[f"q{i}"].choice == "甲"
            assert result.nouls["n"].noul == 0.75


@pytest.mark.parametrize("state", ["queue_full", "not_ready", "deadline"])
def test_sdk_overload_and_retry_headers(server, state):
    url, _ = server
    with TypeSafeClient(**sdk_options(url)) as client:
        with pytest.raises(TypeSafeInternalServerError) as error:
            client.system_one(state=state, questions={"q": Noul()})
        assert error.value.status == 529
        assert error.value.headers["Retry-After"] == "1"
        assert error.value.request_id


def test_sdk_retries_transient_overload(server):
    url, engine = server
    options = {**sdk_options(url), "retry": RetryPolicy(max_retries=1)}
    with TypeSafeClient(**options) as client:
        result = client.system_one(state="retry", questions={"q": Noul()})
        assert result.nouls["q"].noul == 0.75
        assert engine.retry_calls == 2


def test_internal_errors_are_sanitized(server):
    url, _ = server
    with TypeSafeClient(**sdk_options(url)) as client:
        with pytest.raises(TypeSafeInternalServerError) as error:
            client.system_one(state="crash", questions={"q": Noul()})
        assert error.value.status == 500
        assert "private backend details" not in str(error.value.body)
        assert error.value.request_id
        assert client.system_one(state="next", questions={"q": Noul()}).nouls["q"].noul == 0.75


async def test_ingress_deadline_and_cancellation():
    app, engine = make_app(request_timeout=0.15, max_inflight=1)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        first = asyncio.create_task(client.post("/v1/systemone", json={**PAYLOAD, "state": "hang"}))
        while not engine.payloads:
            await asyncio.sleep(0)
        rejected = await client.post("/v1/systemone", json=PAYLOAD)
        assert rejected.status_code == 529 and rejected.headers["Retry-After"] == "1"
        assert rejected.headers["x-typesafe-request-id"]
        assert (await client.post("/v1/decisions", json={})).status_code == 429
        assert (await first).status_code == 529
        assert engine.cancelled == 1
        assert (await client.post("/v1/systemone", json=PAYLOAD)).status_code == 200
        cancelled = asyncio.create_task(client.post("/v1/systemone", json={**PAYLOAD, "state": "hang"}))
        while len(engine.payloads) < 3:
            await asyncio.sleep(0)
        cancelled.cancel()
        await asyncio.gather(cancelled, return_exceptions=True)
        assert engine.cancelled == 2
        assert (await client.post("/v1/systemone", json=PAYLOAD)).status_code == 200
