"""真实 HTTP 并发、静态 buffer 复用、错误隔离、Worker 故障与重启。"""

import asyncio
import gc
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import httpx
import pytest

from laya.config import Config
from laya.contracts import DecisionRequest
from scripts.benchmark_support import compare, stop


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.e2e


@pytest.mark.parametrize("runner", ["cuda-graph", "cuda-graph-compile"])
def test_real_optimized_concurrency_failure_and_restart(runner, tmp_path):
    if os.getenv("LAYA_RUN_OPTIMIZED_E2E") != "1":
        pytest.skip("设置 LAYA_RUN_OPTIMIZED_E2E=1 启用优化服务测试")
    import torch
    from laya.runtime import Runtime

    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA")
    payloads = [{"state": state, "questions": {
        "owner": {"type": "choice", "instructions": "谁负责测试？", "criteria": ["小李", "小王"]},
        "importance": {"type": "score", "instructions": "测试的重要程度？", "criteria": ["低", "中", "高"]},
        "holds": {"type": "noul", "instructions": "小李负责测试吗？"},
    }} for state in ["小李负责测试，小王负责发布。", "小王负责测试，小李负责发布。"]]
    baseline = Runtime(Config(runner="eager", dtype="fp16", max_batch_size=1))
    expected = [baseline.infer(DecisionRequest.model_validate(payload)) for payload in payloads]
    shapes = sorted({(len(item["ids"]), len(item["markers"])) for payload in payloads
                     for _, _, item in baseline.prepare(DecisionRequest.model_validate(payload))})
    baseline.close()
    del baseline
    gc.collect()
    torch.cuda.empty_cache()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    args = [sys.executable, "-m", "laya.commands.app", "serve", "--runner", runner,
            "--dtype", "fp16", "--max-batch-size", "16", "--graph-streams", "8",
            "--host", "127.0.0.1", "--port", str(port)]
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "LAYA_STARTUP_TIMEOUT": "900", "LAYA_SHUTDOWN_TIMEOUT": "15",
           "LAYA_REQUEST_TIMEOUT": "120",
           "LAYA_GRAPH_PREWARM_PROFILES": json.dumps([(1, *shape) for shape in shapes])}
    worker = None
    for attempt in range(2):
        with (tmp_path / f"{runner}-{attempt}.log").open("w") as logs:
            process = subprocess.Popen(args, cwd=ROOT, env=env, stdout=logs, stderr=logs)
            try:
                with httpx.Client(base_url=url, timeout=180, trust_env=False) as client:
                    deadline = time.monotonic() + 900
                    while time.monotonic() < deadline:
                        if process.poll() is not None:
                            pytest.fail((tmp_path / f"{runner}-{attempt}.log").read_text()[-6000:])
                        try:
                            if client.get("/health/ready", timeout=1).status_code == 200:
                                break
                        except httpx.HTTPError:
                            pass
                        time.sleep(0.1)
                    else:
                        pytest.fail("优化服务未就绪")
                    info = client.get("/v1/info").json()
                    worker = info["worker_pid"]
                    assert info["runner"] == runner and info["max_batch_size"] == 16
                    assert info["graph_prewarm_profiles"] == [[1, *shape] for shape in shapes]

                    async def concurrent():
                        async with httpx.AsyncClient(base_url=url, timeout=180, trust_env=False) as requests:
                            invalid = {**payloads[0], "state": "背景材料。" * 2000}
                            return await asyncio.gather(*(requests.post("/v1/decisions", json=invalid if i == 7 else payloads[i % 2]) for i in range(32)))

                    responses = asyncio.run(concurrent())
                    actual = []
                    for i, response in enumerate(responses):
                        if i == 7:
                            assert response.status_code == 422
                            assert response.json()["error"] == "state_token_budget_exceeded"
                            continue
                        response.raise_for_status()
                        actual.append(response.json())
                    differences = compare([expected[i % 2] for i in range(32) if i != 7], actual)
                    assert differences["choice_mismatches"] == 0
                    assert differences["max_abs"]["probabilities"] <= 0.003
                    assert differences["max_abs"]["confidence"] <= 0.005
                    assert differences["max_abs"]["score"] <= 0.005
                    assert differences["max_abs"]["noul"] <= 0.003
                    assert differences["max_abs"]["act_probability"] <= 1e-6
                    assert any(int(size) > 1 for size in client.get("/metrics").json()["batching"]["request_batch_histogram"])
                    invalid = {**payloads[0], "state": "背景材料。" * 2000}
                    assert client.post("/v1/decisions", json=invalid).status_code == 422
                    assert client.post("/v1/decisions", content="x" * 262145).status_code == 413
                    assert client.post("/v1/decisions", json=payloads[0]).status_code == 200
                    if attempt == 0:
                        os.kill(worker, signal.SIGKILL)
                        deadline = time.monotonic() + 3
                        while client.get("/health/ready").status_code != 503:
                            assert time.monotonic() < deadline
                            time.sleep(0.01)
                        assert client.post("/v1/decisions", json=payloads[0]).status_code == 503
            finally:
                stop(process)
                assert process.returncode is not None
                if worker:
                    with pytest.raises(ProcessLookupError):
                        os.kill(worker, 0)
        # 同一端口已释放，可立即启动全新的设备所有者。
    print(json.dumps({"runner": runner, "concurrency": 32, "restart": True}, ensure_ascii=False))
