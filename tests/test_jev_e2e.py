"""真实权重、HTTP、Worker 和官方 TypeSafe SDK 的端到端验收。"""

import asyncio
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import httpx
import pytest
from typesafe_sdk import (
    AsyncTypeSafeClient, Choice, Noul, RetryPolicy, Score, TypeSafeClient,
    TypeSafeUnprocessableEntityError,
)


pytestmark = pytest.mark.e2e
ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = {
    "owner": Choice(instructions="谁负责测试？", criteria={"小李": None, "小王": None}),
    "importance": Score(instructions="测试的重要程度？", criteria=["低", "中", "高"]),
    "holds": Noul(instructions="小李负责测试吗？"),
}
STATE = "小李负责测试，小王负责发布。"


@pytest.fixture(scope="module")
def real_service(tmp_path_factory):
    if os.getenv("LAYA_RUN_JEV_E2E") != "1":
        pytest.skip("设置 LAYA_RUN_JEV_E2E=1 和 LAYA_E2E_MODEL_DIR 启用真实模型 SDK 测试")
    model = Path(os.environ["LAYA_E2E_MODEL_DIR"]).resolve()
    assert (model / "model.safetensors").is_file()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    log = tmp_path_factory.mktemp("jev-e2e") / "service.log"
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "LAYA_STARTUP_TIMEOUT": "180", "LAYA_REQUEST_TIMEOUT": "60",
           "LAYA_GRAPH_PREWARM_PROFILES": "[]", "LAYA_THREADS": "4"}
    device = os.getenv("LAYA_JEV_E2E_DEVICE", "cpu")
    dtype = os.getenv("LAYA_JEV_E2E_DTYPE", "fp32")
    runner = os.getenv("LAYA_JEV_E2E_RUNNER", "eager")
    worker_pid = None
    with log.open("w") as output:
        process = subprocess.Popen([
            sys.executable, "-m", "laya.commands.app", "serve", "--model-dir", str(model),
            "--device", device, "--dtype", dtype, "--runner", runner, "--max-batch-size", "2",
            "--host", "127.0.0.1", "--port", str(port),
        ], cwd=ROOT, env=env, stdout=output, stderr=output)
        try:
            with httpx.Client(base_url=url, timeout=1, trust_env=False) as client:
                deadline = time.monotonic() + 180
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        pytest.fail(log.read_text()[-6000:])
                    try:
                        if client.get("/health/ready").status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(0.1)
                else:
                    pytest.fail("服务启动超时\n" + log.read_text()[-6000:])
                info = client.get("/v1/info").json()
                worker_pid = info["worker_pid"]
                assert info["device"] == device and info["dtype"] == dtype
                assert info["runner"] == runner
            yield url, model.name
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            if worker_pid:
                with pytest.raises(ProcessLookupError):
                    os.kill(worker_pid, 0)
            with socket.socket() as connection:
                assert connection.connect_ex(("127.0.0.1", port)) != 0


def options(url):
    return dict(base_url=url, api_key="local", timeout=60, retry=RetryPolicy(max_retries=0))


def test_real_sync_sdk_and_legacy_probabilities(real_service):
    url, model = real_service
    with TypeSafeClient(**options(url)) as client, httpx.Client(base_url=url, timeout=60, trust_env=False) as raw:
        result = client.system_one(state=STATE, questions=QUESTIONS)
        assert result.model == model and result.request_id
        assert result.choices["owner"].choice == "小李"
        assert result.usage.input_tokens > 0 and result.usage.output_tokens == 0
        assert model in {entry.name for entry in client.models.list().models}
        legacy = raw.post("/v1/decisions", json={"state": STATE,
                          "questions": {key: q.model_dump() for key, q in QUESTIONS.items()}})
        legacy.raise_for_status()
        old = legacy.json()["answers"]
        assert result.choices["owner"].probabilities == pytest.approx(old["owner"]["probabilities"], abs=1e-6)
        assert result.choices["owner"].choice == old["owner"]["choice"]
        assert result.scores["importance"].probabilities == pytest.approx(
            {int(k): v for k, v in old["importance"]["probabilities"].items()}, abs=1e-6)
        assert result.scores["importance"].score == pytest.approx(old["importance"]["score"], abs=1e-6)
        assert result.nouls["holds"].noul == pytest.approx(old["holds"]["noul"], abs=1e-6)
        wire = result.raw_http_response.json()
        assert wire["answers"]["holds"].keys() == {"type", "noul"}
        assert wire["answers"]["importance"]["legend"] == {"0": "低", "1": "中", "2": "高"}
        assert wire.keys() == {"model", "answers", "usage"}


async def test_real_async_sdk_batching_and_isolation(real_service):
    url, _ = real_service
    async with AsyncTypeSafeClient(**options(url)) as client:
        assert (await client.models.list()).request_id
        states = [STATE, "小王负责测试，小李负责发布。"]
        baseline = [await client.system_one(state=state, questions=QUESTIONS) for state in states]
        replies = await asyncio.gather(
            *(client.system_one(state=states[i % 2], questions=QUESTIONS) for i in range(4)),
            client.system_one(state="背景材料。" * 2000, questions=QUESTIONS),
            return_exceptions=True,
        )
        assert isinstance(replies[-1], TypeSafeUnprocessableEntityError)
        results = replies[:-1]
        assert len({r.request_id for r in results}) == 4
        for i, result in enumerate(results):
            expected = baseline[i % 2]
            assert result.choices["owner"].choice == expected.choices["owner"].choice
            assert result.choices["owner"].probabilities == pytest.approx(expected.choices["owner"].probabilities, abs=1e-5)
            assert result.nouls["holds"].noul == pytest.approx(expected.nouls["holds"].noul, abs=1e-5)


def test_real_structured_inputs_and_errors(real_service):
    url, _ = real_service
    with TypeSafeClient(**options(url)) as client:
        result = client.system_one(state={"text": STATE}, questions={
            "owner": Choice(instructions={"task": "谁负责测试？"}, criteria={"小李": ["测试"], "小王": None}),
            "level": Score(instructions=["测试的重要程度？"], criteria=[{"level": "低"}, {"level": "高"}]),
            "holds": Noul(instructions="小李负责测试吗？", criteria={"true": {"meaning": "是"}}),
        })
        assert result.scores["level"].legend == {0: {"level": "低"}, 1: {"level": "高"}}
        # 缺省指令也必须经过真实 Worker，不能只在 API 模拟测试通过。
        assert client.system_one(state=STATE, questions={"q": Choice(criteria={"测试": None, "财务": None})}).choices["q"]
        with pytest.raises(TypeSafeUnprocessableEntityError) as error:
            client.system_one(state="背景材料。" * 2000, questions=QUESTIONS)
        assert error.value.status == 422 and error.value.request_id
        assert "state" in str(error.value)
        with pytest.raises(TypeSafeUnprocessableEntityError) as error:
            client.system_one(state=STATE, questions={"q": Choice(instructions="判断。" * 400, criteria={"是": None, "否": None})})
        assert "questions" in str(error.value)
        assert client.system_one(state=STATE, questions=QUESTIONS).choices["owner"].choice == "小李"
