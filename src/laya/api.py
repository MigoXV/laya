"""HTTP 边界；正文不记录，状态只存在内存。"""

import asyncio
from contextlib import asynccontextmanager
import json

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .contracts import DecisionRequest
from .engine import Engine, EngineError


def create_app(config, engine=None):
    engine = engine or Engine(config)

    @asynccontextmanager
    async def lifespan(app):
        await engine.start()
        try:
            yield
        finally:
            await engine.close()

    app = FastAPI(title="Laya 本地决策服务", lifespan=lifespan)
    app.state.engine = engine
    ingress = asyncio.Semaphore(getattr(config, "max_inflight", 64))

    @app.middleware("http")
    async def bounded_ingress(request, call_next):
        if request.url.path != "/v1/decisions":
            return await call_next(request)
        if ingress.locked():
            engine.counters["ingress_rejected"] += 1
            return JSONResponse({"error": "inflight_full"}, status_code=429)
        async with ingress:
            try:
                return await asyncio.wait_for(
                    call_next(request), config.request_timeout + 1
                )
            except asyncio.TimeoutError:
                engine.counters["ingress_timeout"] += 1
                return JSONResponse({"error": "request_deadline"}, status_code=504)

    @app.get("/health/live")
    async def live():
        return {"alive": True}

    @app.get("/health/ready")
    async def ready():
        return JSONResponse(
            {"ready": engine.ready}, status_code=200 if engine.ready else 503
        )

    @app.get("/v1/info")
    async def info():
        return {**engine.info, "ready": engine.ready, "protocol": "laya-decisions-v1"}

    @app.get("/metrics")
    async def metrics():
        return engine.metrics()

    @app.post(
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

    return app
