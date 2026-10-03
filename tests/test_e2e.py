"""显式启用的真实 HTTP/Worker/模型端到端测试。"""

import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import httpx
import pytest

from laya.config import Config


pytestmark = pytest.mark.e2e
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=[("cpu", "fp32"), ("cuda:0", "fp32"), ("cuda:0", "fp16")], scope="module")
def service(request, tmp_path_factory):
    if os.getenv("LAYA_RUN_E2E") != "1":
        pytest.skip("设置 LAYA_RUN_E2E=1 启用真实模型测试")
    import torch

    assert torch.__version__.split("+")[0] == "2.8.0"
    device, dtype = request.param
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("当前环境无 CUDA")
    model_dir = os.getenv("LAYA_E2E_MODEL_DIR") or str(Config().model_dir)
    model_args = ["--model-dir", model_dir]
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    log_path = tmp_path_factory.mktemp("service") / "service.log"
    worker_pid = None
    with log_path.open("w+") as logs:
        process = subprocess.Popen(
            [sys.executable, "-m", "laya.commands.app", "serve", *model_args,
             "--device", device, "--dtype", dtype, "--host", "127.0.0.1", "--port", str(port)],
            cwd=ROOT, stdout=logs, stderr=logs,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        )
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=45) as client:
                deadline = time.monotonic() + 180
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        pytest.fail(log_path.read_text()[-6000:])
                    try:
                        if client.get("/health/ready", timeout=1).status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(0.1)
                else:
                    pytest.fail("服务未在 180 秒内就绪\n" + log_path.read_text()[-6000:])
                metadata = client.get("/v1/info").json()
                assert metadata["dtype"] == dtype
                assert metadata["autocast"] == (dtype == "fp16")
                assert metadata["parameter_count"] == 321908995
                assert metadata["parameter_bytes"] == 321908995 * (2 if dtype == "fp16" else 4)
                worker_pid = metadata["worker_pid"]
                yield client, metadata
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            if worker_pid:
                with pytest.raises(ProcessLookupError):
                    os.kill(worker_pid, 0)


def test_real_decisions_and_validation(service):
    client, metadata = service
    assert metadata["ready"] and metadata["torch_version"].startswith("2.8.0")
    assert client.get("/health/live").json() == {"alive": True}
    payload = {
        "state": "小李负责测试，小王只负责发布。",
        "questions": {
            "owner": {"type": "choice", "instructions": "谁负责测试？", "criteria": ["小李", "小王"]},
            "importance": {"type": "score", "instructions": "测试的重要程度？", "criteria": ["低", "中", "高"]},
            "holds": {"type": "noul", "instructions": "小李负责测试吗？"},
        },
    }
    reply = client.post("/v1/decisions", json=payload)
    reply.raise_for_status()
    result = reply.json()
    assert result["answers"]["owner"]["choice"] == "小李"
    for answer in result["answers"].values():
        probabilities = list(answer["probabilities"].values())
        assert all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
        assert abs(sum(probabilities) - 1) < 1e-6
    assert 0 <= result["answers"]["importance"]["score"] <= 2
    assert 0 <= result["answers"]["holds"]["noul"] <= 1
    again = client.post("/v1/decisions", json=payload).json()
    assert again["answers"] == result["answers"]
    assert result["model"]["fingerprint"] == metadata["fingerprint"]
    assert client.post("/v1/decisions", json={"state": "x", "questions": {}}).status_code == 422
    oversized = client.post("/v1/decisions", json={**payload, "state": "背景材料。" * 2000})
    assert oversized.status_code == 422
    assert oversized.json()["error"] == "state_token_budget_exceeded"
    assert client.post("/v1/decisions", content="x" * 262145).status_code == 413
    assert client.get("/health/ready").status_code == 200
    print(json.dumps({"device": metadata["device"], "dtype": metadata["dtype"], "torch": metadata["torch_version"],
                      "fingerprint": metadata["fingerprint"], "answers": result["answers"],
                      "timings": result["timings"]}, ensure_ascii=False))


def test_reference_alignment(service):
    from laya.checks import check_reference

    _, metadata = service
    model_dir = Path(os.getenv("LAYA_E2E_MODEL_DIR") or Config().model_dir)
    snapshot_dir = Path(os.getenv("LAYA_E2E_SNAPSHOT_DIR", ROOT / "model-bin/convaiinnovations/laya"))
    result = check_reference(model_dir, snapshot_dir, metadata["device"], metadata["dtype"])
    assert result["fingerprint"] == metadata["fingerprint"]
    print(json.dumps({"alignment": result}, ensure_ascii=False))
