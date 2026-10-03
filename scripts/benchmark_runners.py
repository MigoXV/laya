"""Poetry 环境中运行隔离实例，验证完整决策契约后测量后端矩阵。"""

import importlib.metadata as metadata
import csv
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from statistics import pstdev, mean
from datetime import datetime, timezone

import httpx
import typer

ROOT = Path(__file__).resolve().parents[1]
app = typer.Typer()


def command(args):
    return subprocess.check_output(args, text=True).strip()


def gpu_processes():
    return command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv"])


def wait_for_gpu(uuid):
    # 进程退出后，驱动可能稍晚释放/更新 CUDA context。
    deadline = time.monotonic() + 10
    while uuid in gpu_processes():
        if time.monotonic() >= deadline:
            raise RuntimeError("GPU 0 存在推理进程，不能做隔离性能测试")
        time.sleep(0.25)


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def alignment_cases():
    questions = [
        {"type": "choice", "instructions": "谁负责测试？", "criteria": ["小李", "小王"]},
        {"type": "score", "instructions": "测试的重要程度？", "criteria": ["低", "中", "高"]},
        {"type": "noul", "instructions": "小李负责测试吗？"},
    ]
    cases = [
        {"state": state, "questions": {"q": q}}
        for state in ("小李负责测试。", "小李负责测试，小王只负责发布。",
                      "讨论背景。" * 180 + "小李负责测试，小王只负责发布。")
        for q in questions
    ]
    cases.extend([
        {"state": {"测试": "小李", "发布": "小王"}, "questions": {
            str(i): q for i, q in enumerate(questions)}},
        {"state": "Alice owns testing; Bob handles releases.", "questions": {"q": {
            "type": "choice", "instructions": "Who owns testing?", "criteria": ["Alice", "Bob"]}}},
        {"state": "版本编号是 7。", "questions": {"q": {
            "type": "choice", "instructions": "版本编号？", "criteria": [str(i) for i in range(16)]}}},
    ])
    # 由公共 tokenizer 构造真实 1024 token 边界输入。
    from laya.config import Config
    from laya.contracts import DecisionRequest
    from laya.runtime import checked_sequence
    from transformers import PreTrainedTokenizerFast

    config = json.loads((Config().model_dir / "config.json").read_text())
    tok = PreTrainedTokenizerFast(tokenizer_file=str(Config().model_dir / "tokenizer.json"), **config["tokenizer"])
    q = DecisionRequest.model_validate(cases[0]).questions["q"]
    head, _ = checked_sequence(tok, "", q, config["input_limits"])
    state = ("a " * (1024 - len(head))).strip()
    ids, _ = checked_sequence(tok, state, q, config["input_limits"])
    assert len(ids) == 1024
    cases.append({"state": state, "questions": {"q": questions[0]}})
    return cases


def compare(reference, actual):
    maxima = dict(probabilities=0.0, confidence=0.0, score=0.0, noul=0.0, act_probability=0.0)
    choices = 0
    for expected, result in zip(reference, actual):
        assert expected["usage"] == result["usage"]
        assert expected["answers"].keys() == result["answers"].keys()
        for key, a in expected["answers"].items():
            b = result["answers"][key]
            assert a.keys() == b.keys()
            choices += int(a.get("choice") != b.get("choice"))
            for option, value in a["probabilities"].items():
                maxima["probabilities"] = max(maxima["probabilities"], abs(value - b["probabilities"][option]))
            for field in set(maxima) - {"probabilities"}:
                if field in a:
                    maxima[field] = max(maxima[field], abs(a[field] - b[field]))
    return {"max_abs": maxima, "choice_mismatches": choices}


def summarize(output):
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["completed"] and len(manifest["sessions"]) == len(manifest["order"])
    results, resources = [], []
    groups = {}
    for session in manifest["sessions"]:
        groups.setdefault((session["runner"], session["dtype"]), []).append(session)
    for (runner, dtype), sessions in groups.items():
        for case in ("short", "long"):
            files = [f"{session['label']}-{case}.jsonl" for session in sessions]
            rows = [json.loads(line) for name in files for line in (output / name).read_text().splitlines()]
            assert len(rows) == 3 * len(sessions)
            raw = [sample for row in rows for sample in row["raw"]]
            assert len(raw) == 384 * len(sessions) and all(sample["status"] == 200 and sample["aligned"] for sample in raw)
            times = sorted(sample["ms"] for sample in raw)
            qps = [row["successful_qps"] for row in rows]
            results.append({
                "runner": runner, "dtype": dtype, "case": case,
                "input_tokens": rows[0]["reference"]["usage"]["input_tokens"],
                "successful": sum(row["successful"] for row in rows),
                "errors": sum(row["errors"] for row in rows),
                "answer_mismatches": sum(row["answer_mismatches"] for row in rows),
                "successful_qps": sum(row["successful"] for row in rows) / sum(row["elapsed_s"] for row in rows),
                "latency_ms": {f"p{p}": times[int((len(times) - 1) * p / 100)] for p in (50, 95, 99)},
                "round_qps": qps, "qps_cv_percent": pstdev(qps) / mean(qps) * 100,
                "inference_median_ms": sorted(sample["timings"]["inference_ms"] for sample in raw)[len(raw) // 2],
                "queue_median_ms": sorted(sample["timings"]["queue_ms"] for sample in raw)[len(raw) // 2],
                "raw_files": files,
            })
        samples = []
        for session in sessions:
            with (output / f"{session['label']}-gpu.csv").open() as file:
                samples.extend({key.strip(): value.strip() for key, value in row.items()} for row in csv.DictReader(file))
        resources.append({"runner": runner, "dtype": dtype,
                          "peak_gpu_mib": max(int(row["memory.used [MiB]"].split()[0]) for row in samples),
                          "peak_gpu_utilization_percent": max(int(row["utilization.gpu [%]"].split()[0]) for row in samples)})
    summary = {"results": results, "resources": resources,
               "successful": sum(row["successful"] for row in results),
               "errors": sum(row["errors"] for row in results),
               "answer_mismatches": sum(row["answer_mismatches"] for row in results)}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


@app.command()
def main(
    output: Path = typer.Option(ROOT / "benchmarks/vllm_16"),
    repeats: int = typer.Option(2, min=1, max=2),
    resume: bool = typer.Option(False),
):
    """要求物理 GPU 0 空闲；不终止其他进程，不自动修改生产配置。"""
    output.mkdir(parents=True, exist_ok=True)
    logs_dir = ROOT / "outputs/vllm-benchmark"
    logs_dir.mkdir(parents=True, exist_ok=True)
    uuid = command(["nvidia-smi", "--id=0", "--query-gpu=uuid", "--format=csv,noheader"])
    wait_for_gpu(uuid)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "LAYA_THREADS": "4", "LAYA_STARTUP_TIMEOUT": "900", "LAYA_SHUTDOWN_TIMEOUT": "15"}
    cases = alignment_cases()
    (output / "alignment-inputs.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2))
    order = [("eager", "fp16"), ("eager", "fp32"), ("vllm-eager", "fp16"),
             ("vllm-eager", "fp32"), ("vllm", "fp16"), ("vllm", "fp32")]
    if repeats == 2:
        order += list(reversed(order))
    manifest = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "revision": command(["git", "rev-parse", "HEAD"]),
        "workspace_runtime_patch": command(["git", "diff", "--", "src/laya/config.py", "src/laya/runtime.py"]),
        "versions": {name: metadata.version(name) for name in ("torch", "vllm", "transformers", "httpx")},
        "gpu": command(["nvidia-smi", "--id=0", "--query-gpu=uuid,name,driver_version", "--format=csv"]),
        "processes_before": gpu_processes(), "order": order, "sessions": [],
        "concurrency": 16, "rounds": 3, "requests_per_round": 128,
        "batch_size": 1, "canonical_reference": "eager/fp16",
        "backend_tolerance": {"probabilities": 0.003, "confidence": 0.005, "score": 0.005,
                              "noul": 0.003, "act_probability": 1e-6, "choice_mismatches": 0},
    }
    references = {}
    if resume:
        saved = json.loads((output / "manifest.json").read_text())
        assert saved["revision"] == manifest["revision"]
        assert saved["versions"] == manifest["versions"]
        assert saved["workspace_runtime_patch"] == manifest["workspace_runtime_patch"]
        assert saved["order"] == [list(item) for item in order[:len(saved["order"])]]
        manifest["sessions"] = saved["sessions"]
        manifest["utc"] = saved["utc"]
        for session in manifest["sessions"]:
            if session["runner"] == "eager":
                references[session["dtype"]] = session["alignment_outputs"]
    try:
        for session_id, (runner, dtype) in enumerate(order):
            if session_id < len(manifest["sessions"]):
                continue
            label = f"{session_id}-{runner}-{dtype}"
            print(f"{label}: 启动", flush=True)
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            args = [sys.executable, "-m", "laya.commands.app", "serve", "--runner", runner,
                    "--dtype", dtype, "--host", "127.0.0.1", "--port", str(port)]
            log_path = logs_dir / f"{label}.log"
            with log_path.open("w") as logs:
                started = time.monotonic()
                process = subprocess.Popen(args, cwd=ROOT, env=env, stdout=logs, stderr=logs)
                worker = None
                try:
                    with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=60, trust_env=False) as client:
                        deadline = time.monotonic() + 900
                        while time.monotonic() < deadline:
                            if process.poll() is not None:
                                raise RuntimeError(log_path.read_text()[-9000:])
                            try:
                                if client.get("/health/ready", timeout=1).status_code == 200:
                                    break
                            except httpx.HTTPError:
                                pass
                            time.sleep(0.1)
                        else:
                            raise TimeoutError("服务未就绪")
                        info = client.get("/v1/info").json()
                        worker = info["worker_pid"]
                        assert info["runner"] == runner and info["dtype"] == dtype
                        assert info["parameter_count"] == 321908995
                        assert info["threads"] == 4
                        if runner != "eager":
                            assert info["worker_threads"] == 4
                            assert info["loaded_parameter_count"] == info["parameter_count"]
                            assert info["loaded_parameter_bytes"] == info["parameter_bytes"]
                        session = {"label": label, "runner": runner, "dtype": dtype, "command": args,
                                   "startup_s": time.monotonic() - started, "model": info}
                        actual = []
                        for payload in cases:
                            response = client.post("/v1/decisions", json=payload)
                            response.raise_for_status()
                            actual.append(response.json())
                        session["alignment_outputs"] = actual
                        if runner == "eager":
                            references[dtype] = actual
                        session["vs_eager_fp16"] = compare(references["fp16"], actual)
                        session["vs_same_precision_eager"] = compare(references[dtype], actual)
                        checks = session["vs_same_precision_eager"]
                        assert checks["choice_mismatches"] == 0, checks
                        assert all(value <= manifest["backend_tolerance"][field]
                                   for field, value in checks["max_abs"].items()), checks
                        session["backend_aligned"] = True
                        # HTTP 校验与故障恢复也经过真实 Worker 和模型。
                        for invalid in ({"state": "x", "questions": {}},
                                        {**cases[0], "state": "背景材料。" * 2000}):
                            assert client.post("/v1/decisions", json=invalid).status_code == 422
                        assert client.post("/v1/decisions", content="x" * 262145).status_code == 413
                        assert client.get("/health/ready").status_code == 200
                        monitor_file = (output / f"{label}-gpu.csv").open("w")
                        monitor = subprocess.Popen([
                            "nvidia-smi", "--id=0", "--query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,power.draw",
                            "--format=csv", "-l", "1"], stdout=monitor_file)
                        try:
                            for case in (("short", "long") if session_id % 2 == 0 else ("long", "short")):
                                print(f"{label}/{case}: 16 并发，3 × 128 请求", flush=True)
                                bench = [sys.executable, "-c", "import logging; logging.getLogger('httpx').setLevel(logging.WARNING); from laya.commands.app import app; app()",
                                         "benchmark", "--url", f"http://127.0.0.1:{port}", "--concurrency", "16",
                                         "--samples", "128", "--rounds", "3", "--warmup", "16",
                                         "--input-path", str(output / f"{case}.json")]
                                raw_path = output / f"{label}-{case}.jsonl"
                                with raw_path.open("w") as raw:
                                    subprocess.run(bench, cwd=ROOT, env=env, stdout=raw, stderr=logs, check=True, timeout=300)
                                rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
                                assert len(rows) == 3 and all(row["errors"] == row["answer_mismatches"] == 0 for row in rows)
                                print(json.dumps({"label": label, "case": case, "qps": [round(r["successful_qps"], 2) for r in rows]}), flush=True)
                            session["metrics_after"] = client.get("/metrics").json()
                        finally:
                            stop(monitor)
                            monitor_file.close()
                        manifest["sessions"].append(session)
                finally:
                    stop(process)
                    if worker and Path(f"/proc/{worker}").exists():
                        raise RuntimeError(f"Worker 未退出: {worker}")
                    wait_for_gpu(uuid)
            (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        manifest["completed"] = True
    finally:
        manifest["processes_after"] = gpu_processes()
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    summarize(output)


if __name__ == "__main__":
    app()
