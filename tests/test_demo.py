"""代理边界与静态文件验证；推理正确性由真实 E2E 检查。"""

import httpx
import pytest

from examples.demo01.app import DemoConfig, create_app


@pytest.fixture
def dist(tmp_path):
    (tmp_path / "index.html").write_text("<html>demo</html>")
    (tmp_path / "asset.js").write_text("console.log('demo')")
    return tmp_path


def test_missing_build_is_explicit(tmp_path):
    with pytest.raises(RuntimeError, match="Demo 尚未构建"):
        create_app(dist_dir=tmp_path)


async def test_proxy_routes_preserve_status_and_payload(dist):
    calls = []

    def upstream(request):
        calls.append(request)
        if request.url.path == "/v1/decisions":
            return httpx.Response(422, json={"error": "invalid_request"})
        return httpx.Response(200, json={"ready": True})

    app = create_app(
        DemoConfig(service_url="http://service:10002"), dist,
        httpx.MockTransport(upstream),
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://demo"
        ) as client:
            assert (await client.get("/")).text == "<html>demo</html>"
            assert (await client.get("/asset.js")).status_code == 200
            for accept in ("text/html", "application/json"):
                assert (await client.get("/api/unknown", headers={"accept": accept})).status_code == 404
            assert (await client.get("/unknown")).status_code == 404
            assert (await client.get("/api/connection")).json()["service_url"] == "http://service:10002"
            assert (await client.get("/api/health/ready")).json() == {"ready": True}
            assert (await client.get("/api/v1/info")).json() == {"ready": True}
            reply = await client.post("/api/v1/decisions", content=b'{"bad":true}')
            assert reply.status_code == 422
            assert reply.json() == {"error": "invalid_request"}
            assert calls[-1].content == b'{"bad":true}'
            assert (await client.post("/api/v1/decisions", content="x" * 262145)).status_code == 413
            assert len(calls) == 3


@pytest.mark.parametrize(
    "failure,status,code",
    [
        (httpx.ConnectError, 502, "service_unavailable"),
        (httpx.ReadTimeout, 504, "service_timeout"),
    ],
)
async def test_proxy_connection_errors_are_recoverable(dist, failure, status, code):
    def upstream(request):
        raise failure("unavailable", request=request)

    app = create_app(dist_dir=dist, transport=httpx.MockTransport(upstream))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://demo"
        ) as client:
            reply = await client.get("/api/health/ready")
            assert reply.status_code == status
            assert reply.json() == {"error": code}
