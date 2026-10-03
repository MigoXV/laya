"""HTTP 入口容量、deadline 和 Jev 请求标识。"""
import asyncio
import logging
from uuid import uuid4
from fastapi.responses import JSONResponse
from . import jev


def install_ingress(app, config, engine):
    ingress = asyncio.Semaphore(getattr(config, "max_inflight", 64))

    @app.middleware("http")
    async def bounded_ingress(request, call_next):
        is_jev = request.url.path in jev.PATHS
        request_id = uuid4().hex if is_jev else None

        async def handle():
            if request.url.path not in ("/v1/decisions", "/v1/systemone"):
                return await call_next(request)
            if ingress.locked():
                engine.counters["ingress_rejected"] += 1
                return jev.unavailable() if is_jev else JSONResponse({"error": "inflight_full"}, status_code=429)
            async with ingress:
                try:
                    return await asyncio.wait_for(call_next(request), config.request_timeout + 1)
                except asyncio.TimeoutError:
                    engine.counters["ingress_timeout"] += 1
                    return jev.unavailable() if is_jev else JSONResponse({"error": "request_deadline"}, status_code=504)

        try:
            response = await handle()
        except Exception:
            if not is_jev:
                raise
            logging.getLogger(__name__).exception("Jev request failed: %s", request_id)
            response = jev.internal_error()
        if is_jev:
            response.headers["x-typesafe-request-id"] = request_id
        return response

