"""Arietta 导出精度接入的独立回归测试。"""

import pytest


def test_bf16_config(tmp_path):
    from laya.configs.settings import Config

    for name in ("model.safetensors", "tokenizer.json", "config.json"):
        (tmp_path / name).touch()
    assert (
        Config(model_dir=tmp_path, device="cpu", dtype="bf16", _env_file=None).dtype
        == "bf16"
    )


def test_bf16_eager_autocast():
    torch = pytest.importorskip("torch")
    from laya.runners.eager import EagerRunner

    class Model(torch.nn.Module):
        def forward(self, *args):
            assert torch.is_autocast_enabled("cpu")
            assert torch.get_autocast_dtype("cpu") == torch.bfloat16
            return torch.zeros(1, 2), torch.zeros(1, 2)

    batch = {
        k: torch.zeros(1, 1)
        for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
    }
    logits, actions = EagerRunner(Model(), torch.device("cpu"), torch.bfloat16).execute(
        batch
    )
    assert logits.shape == (1, 2)
    assert actions.sum() == 1


def test_exported_bf16_http_lifecycle():
    import os
    import socket
    import subprocess
    import sys
    import time
    from pathlib import Path
    import httpx

    if not os.getenv("ARIETTA_BF16_E2E_MODEL"):
        pytest.skip("显式设置 ARIETTA_BF16_E2E_MODEL 启用 GPU0 导出模型服务验收")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "laya.commands.app",
            "serve",
            "--model-dir",
            os.environ["ARIETTA_BF16_E2E_MODEL"],
            "--dtype",
            "bf16",
            "--device",
            "cuda:0",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=Path(__file__).resolve().parents[1],
    )
    worker = None
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30) as client:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                assert process.poll() is None
                try:
                    if client.get("/health/ready", timeout=1).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.1)
            else:
                pytest.fail("BF16 worker startup timeout")
            info = client.get("/v1/info").json()
            worker = info["worker_pid"]
            assert info["dtype"] == "bf16" and info["autocast"]
            response = client.post(
                "/v1/decisions",
                json={
                    "state": "支持这个方案。",
                    "questions": {
                        "q": {"type": "noul", "instructions": "说话人是否支持方案？"}
                    },
                },
            )
            response.raise_for_status()
            assert 0 <= response.json()["answers"]["q"]["noul"] <= 1
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        if worker:
            with pytest.raises(ProcessLookupError):
                os.kill(worker, 0)


def test_modern_rope_asset_uses_same_theta_on_legacy_transformers():
    from laya.models.loading import encoder_config_for_runtime
    cfg=encoder_config_for_runtime({"model_type":"modernbert", "rope_parameters": {
        "full_attention":{"rope_type":"default","rope_theta":160000},
        "sliding_attention":{"rope_type":"default","rope_theta":160000}}})
    assert cfg.global_rope_theta==160000
    assert cfg.local_rope_theta==160000
