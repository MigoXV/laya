"""Jev HTTP 协议适配；SDK 0.7.2 的私有 wire schema 仅在此处引用。"""

import asyncio
import json
import math
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from typesafe_sdk import (
    ChoiceAnswer, ListModelsResponse, ModelMetadata, NoulAnswer, ScoreAnswer,
    SystemOneResponse, Usage,
)
from typesafe_sdk._schemas.models import HTTPValidationError, SystemOneRequest

from laya.inferencers.contracts import InferenceRequest
from laya.engine.core import EngineError


PATHS = {"/v1/systemone", "/v1/models"}
# 此日期描述本地兼容接口的发布，不冒充上游权重训练日期。
RELEASE_DATE = "2026-10-03"


def invalid(message, location=(), *, kind="value_error", status=422):
    body = HTTPValidationError(detail=[{
        "loc": ["body", *location], "msg": message, "type": kind,
    }])
    return JSONResponse(body.model_dump(exclude_none=True), status_code=status)


def validation_error(exc):
    # 不返回 input/context，防止把完整用户状态或不可序列化的异常写入响应。
    body = HTTPValidationError(detail=[{
        "loc": ["body", *error["loc"]], "msg": error["msg"], "type": error["type"],
    } for error in exc.errors(include_url=False)])
    return JSONResponse(body.model_dump(exclude_none=True), status_code=422)


def unavailable(message="Service temporarily overloaded"):
    return JSONResponse({"error": {"message": message}}, status_code=529,
                        headers={"Retry-After": "1"})


def internal_error():
    return JSONResponse({"error": {"message": "Internal server error"}}, status_code=500)


def text_content(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def to_inference(data):
    json.dumps(data.state, allow_nan=False)
    questions = {}
    for name, wrapper in data.questions.items():
        q = wrapper.root
        if q.type == "noul":
            criteria = q.criteria.model_dump(exclude_none=True) if q.criteria else None
        else:
            criteria = q.criteria
        if isinstance(criteria, dict):
            criteria = {key: None if value is None else text_content(value)
                        for key, value in criteria.items()}
        elif isinstance(criteria, list):
            criteria = [text_content(value) for value in criteria]
        questions[name] = {
            "type": q.type, "instructions": text_content(q.instructions), "criteria": criteria,
        }
    return InferenceRequest(state=data.state, questions=questions)


def choice_confidence(p):
    n = len(p)
    return min(1.0, max(0.0, (max(p) - 1 / n) / (1 - 1 / n)))


def score_confidence(p):
    n = len(p)
    peak = max(range(n), key=p.__getitem__)
    spread = sum(probability * abs(i - peak) for i, probability in enumerate(p))
    uniform = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return max(0.0, 1 - spread / uniform)


def to_response(data, result, model):
    answers = {}
    for name, wrapper in data.questions.items():
        q, raw = wrapper.root, result["answers"][name]
        if q.type == "noul":
            probability = raw["noul"]
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("Invalid backend probability")
            answers[name] = NoulAnswer(noul=float(probability))
            continue
        keys = list(q.criteria) if q.type == "choice" else [str(i) for i in range(len(q.criteria))]
        p = [float(raw["probabilities"][key]) for key in keys]
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in p) or not math.isclose(sum(p), 1, abs_tol=1e-5):
            raise ValueError("Invalid backend probabilities")
        if q.type == "choice":
            answers[name] = ChoiceAnswer(
                choice=keys[max(range(len(p)), key=p.__getitem__)],
                probabilities=dict(zip(keys, p)), confidence=choice_confidence(p),
            )
        else:
            answers[name] = ScoreAnswer(
                score=sum(i * v for i, v in enumerate(p)), confidence=score_confidence(p),
                legend=dict(enumerate(q.criteria)), probabilities=dict(enumerate(p)),
            )
    return SystemOneResponse(model=model, answers=answers,
                             usage=Usage(input_tokens=result["usage"]["input_tokens"], output_tokens=0))


def install_openapi(app):
    """将 SDK 的请求定义注册为组件，使 $ref 在整个 OpenAPI 文档中可解析。"""
    schema = SystemOneRequest.model_json_schema(ref_template="#/components/schemas/Jev{model}")
    components = {f"Jev{name}": value for name, value in schema.pop("$defs", {}).items()}
    components["JevSystemOneRequest"] = schema
    generate = app.openapi

    def openapi():
        document = generate()
        document.setdefault("components", {}).setdefault("schemas", {}).update(components)
        return document

    app.openapi = openapi


def router_for(config, engine):
    router = APIRouter()
    model = Path(getattr(config, "model_dir", "laya-multilingual")).name
    aliases = list(dict.fromkeys([model, "jev-latest", "laya", "laya-latest"]))

    @router.get("/v1/models", response_model=ListModelsResponse)
    async def models():
        return ListModelsResponse(models=tuple(ModelMetadata(
            name=name, description=f"Local Laya model: {model}; Jev protocol alias, not Jev weights.",
            release_date=RELEASE_DATE,
        ) for name in aliases))

    @router.post("/v1/systemone", response_model=SystemOneResponse, responses={
        422: {"model": HTTPValidationError},
        529: {"description": "Service temporarily overloaded"},
    }, openapi_extra={"requestBody": {"required": True, "content": {
        "application/json": {"schema": {"$ref": "#/components/schemas/JevSystemOneRequest"}},
    }}})
    async def systemone(request: Request):
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > config.max_body_bytes:
                return invalid("Request body exceeds local byte budget", status=413)
        try:
            data = SystemOneRequest.model_validate_json(body)
        except ValidationError as exc:
            return validation_error(exc)
        if data.model not in aliases:
            return invalid("Unknown model; use GET /v1/models", ("model",))
        for name, wrapper in data.questions.items():
            if wrapper.root.type == "score" and not 2 <= len(wrapper.root.criteria) <= 10:
                return invalid("Score requires 2–10 levels", ("questions", name, "criteria"))
        try:
            payload = to_inference(data)
        except ValidationError as exc:
            return validation_error(exc)
        except ValueError:
            return invalid("Content must contain finite JSON values")
        deadline = asyncio.get_running_loop().time() + config.request_timeout
        task = asyncio.create_task(engine.submit(payload.model_dump()))
        try:
            while not task.done():
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    return unavailable()
                done, _ = await asyncio.wait({task}, timeout=min(0.1, remaining))
                if not done and await request.is_disconnected():
                    return JSONResponse({"error": {"message": "Client disconnected"}}, status_code=499)
            result = task.result()
        except EngineError as exc:
            if exc.status == 422:
                location = ("state",) if str(exc) == "state_token_budget_exceeded" else ("questions",)
                return invalid(str(exc), location)
            if exc.status in (429, 503, 504):
                return unavailable()
            raise
        finally:
            # 同时覆盖中间件 deadline 与客户端取消，不能留下孤立的 submit task。
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        return to_response(data, result, model)

    return router
