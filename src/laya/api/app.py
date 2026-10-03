"""HTTP 边界；正文不记录，状态只存在内存。"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from laya.engine.core import Engine
from . import jev, laya
from .middleware import install_ingress


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
    install_ingress(app, config, engine)

    app.include_router(jev.router_for(config, engine))
    jev.install_openapi(app)

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

    app.include_router(laya.router_for(config, engine))

    return app
