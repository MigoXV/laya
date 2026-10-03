"""Laya decisions 协议适配。"""
import asyncio
import json
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from laya.engine.core import EngineError
from .contracts import DecisionRequest


def router_for(config, engine):
    router = APIRouter()

    @router.post(
        "/v1/decisions",
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {"schema": DecisionRequest.model_json_schema()}
                },
            }
        },
    )
    async def decisions(request: Request):
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > config.max_body_bytes:
                return JSONResponse({"error": "body_too_large"}, status_code=413)
        try:
            data = DecisionRequest.model_validate(json.loads(body))
        except (ValidationError, ValueError, UnicodeDecodeError):
            return JSONResponse({"error": "invalid_request"}, status_code=422)
        task = asyncio.create_task(engine.submit(data.model_dump()))
        try:
            while not task.done():
                done, _ = await asyncio.wait({task}, timeout=0.1)
                if not done and await request.is_disconnected():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    return JSONResponse(
                        {"error": "client_disconnected"}, status_code=499
                    )
            return task.result()
        except EngineError as exc:
            return JSONResponse({"error": str(exc)}, status_code=exc.status)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    return router
