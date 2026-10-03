"""Demo 只代理 HTTP；不导入推理运行时、不加载权重。"""

from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DemoConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LAYA_DEMO_", env_file=".env", extra="ignore"
    )
    service_url: AnyHttpUrl = "http://127.0.0.1:10002"
    timeout: float = Field(default=60, gt=0, le=300)
    host: str = "0.0.0.0"
    port: int = Field(default=10013, ge=1, le=65535)


def create_app(config=None, dist_dir=None, transport=None):
    config = config or DemoConfig()
    dist_dir = Path(dist_dir or Path(__file__).parent / "web" / "dist")
    if not (dist_dir / "index.html").is_file():
        raise RuntimeError("Demo 尚未构建，请执行 pnpm --dir examples/demo01/web build")

    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(
            base_url=str(config.service_url).rstrip("/"),
            timeout=config.timeout,
            transport=transport,
        ) as client:
            app.state.client = client
            yield

    app = FastAPI(title="Laya 推理测试工作台", lifespan=lifespan)

    async def forward(method, path, content=None):
        try:
            reply = await app.state.client.request(
                method, path, content=content,
                headers={"content-type": "application/json"},
            )
        except httpx.TimeoutException:
            return JSONResponse({"error": "service_timeout"}, status_code=504)
        except httpx.RequestError:
            return JSONResponse({"error": "service_unavailable"}, status_code=502)
        return Response(
            reply.content, status_code=reply.status_code,
            media_type=reply.headers.get("content-type", "application/json"),
        )

    @app.get("/api/connection")
    async def connection():
        return {"service_url": str(config.service_url).rstrip("/")}

    @app.get("/api/health/ready")
    async def ready():
        return await forward("GET", "/health/ready")

    @app.get("/api/v1/info")
    async def info():
        return await forward("GET", "/v1/info")

    @app.post("/api/v1/decisions")
    async def decisions(request: Request):
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 262144:
                return JSONResponse({"error": "body_too_large"}, status_code=413)
        return await forward("POST", "/v1/decisions", bytes(body))

    @app.api_route("/api", methods=["GET", "HEAD"], include_in_schema=False)
    @app.api_route("/api/{path:path}", methods=["GET", "HEAD"], include_in_schema=False)
    async def api_not_found(path=""):
        raise HTTPException(status_code=404, detail="未知 Demo API")

    # 页面无客户端路由；真实静态资源、未知 API 和未知路径均有明确边界。
    app.mount("/", StaticFiles(directory=dist_dir, html=True), name="demo")
    return app
