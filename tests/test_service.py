import asyncio
import sys
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from laya.api.app import create_app
from laya.api.contracts import Question
from laya.engine.core import Engine, EngineError


WORKER = """import sys,json,time
print(json.dumps({"ready":{"fingerprint":"fake"}}),flush=True)
for line in sys.stdin:
 p=json.loads(line)
 if p.get("crash"): sys.exit(7)
 time.sleep(p.get("delay",0))
 print(json.dumps({"result":{"answers":p,"timings":{}}}),flush=True)
"""


def make_engine(**overrides):
    config = SimpleNamespace(
        queue_size=1,
        request_timeout=0.2,
        startup_timeout=5,
        shutdown_timeout=0.1,
        max_body_bytes=1024,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return Engine(config, [sys.executable, "-u", "-c", WORKER])


@pytest.mark.parametrize(
    "question",
    [
        {"type": "choice", "instructions": "x", "criteria": ["a"]},
        {"type": "choice", "instructions": "x", "criteria": ["a", "a"]},
        {"type": "score", "instructions": "x", "criteria": {"a": "b"}},
        {"type": "noul", "instructions": "x", "criteria": {"other": "a"}},
    ],
)
def test_reject_invalid_question(question):
    with pytest.raises(ValidationError):
        Question.model_validate(question)


async def test_backpressure_and_association():
    engine = make_engine(request_timeout=2)
    await engine.start()
    try:
        first = asyncio.create_task(engine.submit({"delay": 0.1, "id": 1}))
        while engine.active is None:
            await asyncio.sleep(0)
        second = asyncio.create_task(engine.submit({"id": 2}))
        await asyncio.sleep(0)
        with pytest.raises(EngineError, match="queue_full"):
            await engine.submit({"id": 3})
        assert (await first)["answers"]["id"] == 1
        assert (await second)["answers"]["id"] == 2
    finally:
        await engine.close()
    assert engine.process.returncode is not None


async def test_cancel_does_not_poison_next_response():
    engine = make_engine(request_timeout=2)
    await engine.start()
    try:
        first = asyncio.create_task(engine.submit({"delay": 0.1, "id": 1}))
        while engine.active is None:
            await asyncio.sleep(0)
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)
        assert (await engine.submit({"id": 2}))["answers"]["id"] == 2
        assert engine.counters["discarded"] == 1
    finally:
        await engine.close()


async def test_deadline_and_worker_crash_are_bounded():
    engine = make_engine(request_timeout=0.03, shutdown_timeout=0.2)
    await engine.start()
    try:
        with pytest.raises(EngineError, match="deadline"):
            await engine.submit({"delay": 0.08})
        await asyncio.sleep(0.08)
        with pytest.raises(EngineError, match="worker"):
            await engine.submit({"crash": True})
        assert not engine.ready
    finally:
        await engine.close()


async def test_api_body_bound_readiness_and_health_during_compute():
    engine = make_engine(request_timeout=2)
    await engine.start()
    app = create_app(engine.config, engine)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            assert (await client.get("/health/ready")).status_code == 200
            assert (
                await client.post("/v1/decisions", content="x" * 1025)
            ).status_code == 413
            assert (
                await client.post("/v1/decisions", json={"state": "x", "questions": {}})
            ).status_code == 422
            pending = asyncio.create_task(engine.submit({"delay": 0.1}))
            assert (await client.get("/health/live")).json() == {"alive": True}
            await pending
    finally:
        await engine.close()


async def test_shutdown_fails_active_and_queued():
    engine = make_engine(request_timeout=2)
    await engine.start()
    active = asyncio.create_task(engine.submit({"delay": 1}))
    while engine.active is None:
        await asyncio.sleep(0)
    queued = asyncio.create_task(engine.submit({}))
    await asyncio.sleep(0)
    await engine.close()
    results = await asyncio.gather(active, queued, return_exceptions=True)
    assert all(isinstance(r, EngineError) for r in results)
    await engine.close()
