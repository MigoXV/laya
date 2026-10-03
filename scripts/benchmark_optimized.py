"""隔离 GPU 0，交错重复测量整模型 Graph／多流／编译及 vLLM。"""

import asyncio
import csv
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import socket
from statistics import mean, pstdev
import subprocess
import sys
import time
from datetime import datetime, timezone

import httpx
import typer

from scripts.benchmark_runners import command, compare, gpu_processes, stop, wait_for_gpu


ROOT = Path(__file__).resolve().parents[1]
app = typer.Typer()
VARIANTS = [
    {"label": "eager", "runner": "eager", "batch": 1, "streams": 1},
    {"label": "graph-single", "runner": "cuda-graph", "batch": 1, "streams": 1},
    {"label": "graph-serial", "runner": "cuda-graph", "batch": 16, "streams": 1},
    {"label": "graph-streams4", "runner": "cuda-graph", "batch": 16, "streams": 4},
    {"label": "graph-streams8", "runner": "cuda-graph", "batch": 16, "streams": 8},
    {"label": "graph-compile", "runner": "cuda-graph-compile", "batch": 16, "streams": 8},
    {"label": "vllm-eager", "runner": "vllm-eager", "batch": 16, "streams": 1},
    {"label": "vllm", "runner": "vllm", "batch": 16, "streams": 1},
]


async def prewarm(url, payload):
    async with httpx.AsyncClient(base_url=url, timeout=120, trust_env=False) as client:
        for level in [1, 2, 4, 8, 16]:
            for _ in range(2):
                responses = await asyncio.gather(*(client.post("/v1/decisions", json=payload) for _ in range(level)))
                for response in responses:
                    response.raise_for_status()


def summarize(output, manifest):
    groups = {}
    for session in manifest["sessions"]:
        groups.setdefault(session["variant"]["label"], []).append(session)
    summaries = []
    for label, sessions in groups.items():
        for case in ["short", "long"]:
            raw_files = [f"{session['label']}-{case}.jsonl" for session in sessions]
            rows = [json.loads(line) for name in raw_files for line in (output / name).read_text().splitlines()]
            for concurrency in [1, 16]:
                selected = [row for row in rows if row["concurrency"] == concurrency]
                samples = [sample for row in selected for sample in row["raw"]]
                latencies = sorted(sample["ms"] for sample in samples)
                rates = [row["successful_qps"] for row in selected]
                summaries.append({
                    "variant": label, "case": case, "concurrency": concurrency,
                    "input_tokens": selected[0]["reference"]["usage"]["input_tokens"],
                    "successful": sum(row["successful"] for row in selected),
                    "errors": sum(row["errors"] for row in selected),
                    "mismatches": sum(row["answer_mismatches"] for row in selected),
                    "successful_qps": sum(row["successful"] for row in selected) / sum(row["elapsed_s"] for row in selected),
                    "latency_ms": {f"p{p}": latencies[int((len(latencies) - 1) * p / 100)] for p in (50, 95, 99)},
                    "round_qps": rates, "qps_cv_percent": pstdev(rates) / mean(rates) * 100,
                    "inference_median_ms": sorted(sample["timings"]["inference_ms"] for sample in samples)[len(samples) // 2],
                    "queue_median_ms": sorted(sample["timings"]["queue_ms"] for sample in samples)[len(samples) // 2],
                    "raw_files": raw_files,
                })
    baseline = {(row["case"], row["concurrency"]): row for row in summaries if row["variant"] == "eager"}
    for row in summaries:
        original = baseline[row["case"], row["concurrency"]]
        row["throughput_speedup"] = row["successful_qps"] / original["successful_qps"]
        row["p95_ratio"] = row["latency_ms"]["p95"] / original["latency_ms"]["p95"]
    resources = []
    for label, sessions in groups.items():
        samples = []
        for session in sessions:
            with (output / f"{session['label']}-gpu.csv").open() as file:
                samples.extend({k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(file))
        resources.append({"variant": label, "peak_gpu_mib": max(int(row["memory.used [MiB]"].split()[0]) for row in samples),
                          "max_gpu_utilization": max(int(row["utilization.gpu [%]"].split()[0]) for row in samples)})
    result = {"results": summaries, "resources": resources,
              "successful": sum(row["successful"] for row in summaries),
              "errors": sum(row["errors"] for row in summaries),
              "mismatches": sum(row["mismatches"] for row in summaries)}
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


@app.command()
def main(
    output: Path = typer.Option(ROOT / "benchmarks/optimized_16"),
    include_compile: bool = typer.Option(True),
    repeats: int = typer.Option(2, min=1, max=2),
):
    alignment = json.loads((output / "alignment.json").read_text())
    assert alignment["completed"], "必须先通过完整 logits 对齐"
    output.mkdir(parents=True, exist_ok=True)
    logs_dir = ROOT / "outputs/optimization/benchmark"
    logs_dir.mkdir(parents=True, exist_ok=True)
    uuid = command(["nvidia-smi", "--id=0", "--query-gpu=uuid", "--format=csv,noheader"])
    wait_for_gpu(uuid)
    variants = [variant for variant in VARIANTS if include_compile or variant["label"] != "graph-compile"]
    order = variants + (list(reversed(variants)) if repeats == 2 else [])
    manifest = {
        "utc": datetime.now(timezone.utc).isoformat(), "revision": command(["git", "rev-parse", "HEAD"]),
        "workspace_diff": command(["git", "diff", "--", "src/laya"]),
        "versions": {name: metadata.version(name) for name in ["torch", "vllm", "transformers", "triton"]},
        "gpu": command(["nvidia-smi", "--id=0", "--query-gpu=uuid,name,driver_version", "--format=csv"]),
        "processes_before": gpu_processes(), "order": order, "sessions": [], "completed": False,
        "concurrency": [1, 16], "rounds": 3, "requests_per_round": 128,
        "acceptance": {"throughput_speedup_min": 1.1, "p95_ratio_max": 1.0,
                       "errors": 0, "mismatches": 0, "canonical_reference": "eager/fp16/batch1"},
    }
    for case in ["short", "long"]:
        source = ROOT / "benchmarks/vllm_16" / f"{case}.json"
        (output / source.name).write_bytes(source.read_bytes())
    expected = alignment["baseline"]
    references = alignment["cases"]
    baseline_answers = None
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "LAYA_THREADS": "4",
           "LAYA_STARTUP_TIMEOUT": "900", "LAYA_SHUTDOWN_TIMEOUT": "15", "LAYA_GRAPH_CACHE_SIZE": "16",
           "LAYA_MAX_BATCH_TOKENS": "8192", "LAYA_BATCH_WAIT_MS": "2"}
    env["TOKENIZERS_PARALLELISM"] = "false"
    try:
        for index, variant in enumerate(order):
            label = f"{index}-{variant['label']}"
            print(f"{label}: 启动", flush=True)
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            args = [sys.executable, "-m", "laya.commands.app", "serve", "--runner", variant["runner"],
                    "--dtype", "fp16", "--host", "127.0.0.1", "--port", str(port)]
            session_env = {**env, "LAYA_MAX_BATCH_SIZE": str(variant["batch"]), "LAYA_GRAPH_STREAMS": str(variant["streams"])}
            profiles = ([[batch, length, 2] for length in [27, 512] for batch in [1, 2, 4, 8, 16]
                         if batch <= variant["batch"]] if variant["runner"].startswith("cuda-graph") else [])
            session_env["LAYA_GRAPH_PREWARM_PROFILES"] = json.dumps(profiles)
            with (logs_dir / f"{label}.log").open("w") as logs:
                started = time.monotonic()
                process = subprocess.Popen(args, cwd=ROOT, env=session_env, stdout=logs, stderr=logs)
                worker, monitor = None, None
                try:
                    url = f"http://127.0.0.1:{port}"
                    with httpx.Client(base_url=url, timeout=120, trust_env=False) as client:
                        deadline = time.monotonic() + 900
                        while time.monotonic() < deadline:
                            if process.poll() is not None:
                                raise RuntimeError((logs_dir / f"{label}.log").read_text()[-6000:])
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
                        assert info["runner"] == variant["runner"] and info["dtype"] == "fp16"
                        assert info["max_batch_size"] == variant["batch"]
                        assert info["parameter_count"] == expected["parameter_count"]
                        assert info["parameter_bytes"] == expected["parameter_bytes"]
                        source = expected if variant["runner"] == "eager" else next(
                            item["model"] for item in alignment["results"] if item["runner"] == variant["runner"]
                        )
                        assert info["runtime_code_sha256"] == source["runtime_code_sha256"]
                        session = {"label": label, "variant": variant, "command": args,
                                   "environment": {key: value for key, value in session_env.items() if key.startswith("LAYA_")},
                                   "startup_s": time.monotonic() - started, "model": info,
                                   "cases": {}, "gpu_processes_started": gpu_processes()}
                        with (output / f"{label}-gpu.csv").open("w") as samples:
                            monitor = subprocess.Popen(["nvidia-smi", "--id=0", "--query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,power.draw",
                                                        "--format=csv", "-l", "1"], stdout=samples)
                            for case in (["short", "long"] if index % 2 == 0 else ["long", "short"]):
                                payload = json.loads((output / f"{case}.json").read_text())
                                asyncio.run(prewarm(url, payload))
                                before = client.get("/metrics").json()
                                bench = [sys.executable, "-c", "import logging; logging.getLogger('httpx').setLevel(logging.WARNING); from laya.commands.app import app; app()",
                                         "benchmark", "--url", url, "--concurrency", "1", "--concurrency", "16",
                                         "--samples", "128", "--rounds", "3", "--input-path", str(output / f"{case}.json")]
                                with (output / f"{label}-{case}.jsonl").open("w") as raw:
                                    subprocess.run(bench, cwd=ROOT, env=session_env, stdout=raw, stderr=logs, check=True, timeout=900)
                                after = client.get("/metrics").json()
                                if variant["runner"].startswith("cuda-graph"):
                                    assert after["runner"]["counters"]["captures"] == before["runner"]["counters"]["captures"], "稳态测量出现冷捕获"
                                session["cases"][case] = {"metrics_before": before, "metrics_after": after}
                                rows = [json.loads(line) for line in (output / f"{label}-{case}.jsonl").read_text().splitlines()]
                                assert all(row["errors"] == row["answer_mismatches"] == 0 for row in rows)
                                print(json.dumps({"label": label, "case": case, "qps": [round(row["successful_qps"], 2) for row in rows]}), flush=True)
                            stop(monitor)
                            monitor = None
                        # 完整数值门槛与 HTTP 并发／故障测试已在本矩阵前通过；
                        # 重复 HTTP 契约核验放在稳态测量之后，防止其 shape 淘汰预热缓存。
                        actual = []
                        for payload in references:
                            response = client.post("/v1/decisions", json=payload)
                            response.raise_for_status()
                            actual.append(response.json())
                        if variant["label"] == "eager":
                            baseline_answers = actual
                        differences = compare(baseline_answers, actual)
                        assert differences["choice_mismatches"] == 0
                        assert all(value <= alignment["tolerance"][field] for field, value in differences["max_abs"].items())
                        session["public_alignment"] = differences
                        session["gpu_processes_after_measurements"] = gpu_processes()
                        initial_owners = {row for row in session["gpu_processes_started"].splitlines() if uuid in row}
                        final_owners = {row for row in session["gpu_processes_after_measurements"].splitlines() if uuid in row}
                        # 显存会随缓存变化；对比 PID，避免把外部实例混入测量。
                        assert {row.split(",")[1].strip() for row in initial_owners} == {row.split(",")[1].strip() for row in final_owners}
                        manifest["sessions"].append(session)
                finally:
                    if monitor:
                        stop(monitor)
                    stop(process)
                    if worker and Path(f"/proc/{worker}").exists():
                        raise RuntimeError(f"Worker 未清理: {worker}")
                    wait_for_gpu(uuid)
            (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        manifest["completed"] = True
    finally:
        manifest["processes_after"] = gpu_processes()
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    summarize(output, manifest)


if __name__ == "__main__":
    app()
