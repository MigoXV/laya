"""保存后的 W8A8 权重、冻结精度输出与三个真实 HTTP runner 的针对性验证。"""

import asyncio
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from time import monotonic, sleep

import httpx
import numpy as np
import torch
import typer
from safetensors.torch import load_file

from laya.config import Config
from laya.contracts import DecisionRequest
from laya.cuda_runner import INPUT_KEYS
from laya.quantization import explicit_heads, eligible, file_record
from laya.reference import collate_items
from laya.runtime import Runtime
from .accuracy.evaluate import load_cases, request_from_case
from .benchmark import capture
from .kernels import quantize_weight


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
app = typer.Typer()


def release(runtime):
    runtime.close()
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def check_weights(source, saved):
    baseline = Runtime(Config(model_dir=source, dtype="fp16", runner="eager"))
    explicit_heads(baseline.runner.model)
    weights = load_file(str(saved / "model.safetensors"))
    modules = eligible(baseline.runner.model)
    quantized = set()
    for name, module in modules:
        qweight, scales = quantize_weight(module.weight)
        assert torch.equal(qweight.cpu(), weights[f"{name}.qweight"]), name
        assert torch.equal(scales.cpu(), weights[f"{name}.weight_scales"]), name
        quantized.add(f"{name}.weight")
    remaining = 0
    for name, value in baseline.runner.model.state_dict().items():
        if name not in quantized:
            assert torch.equal(value.cpu(), weights[name]), name
            remaining += 1
    result = {"quantized_modules_bitwise_equal": len(modules),
              "quantized_weight_count": sum(m.weight.numel() for _, m in modules),
              "unquantized_tensors_bitwise_equal": remaining,
              "source_weights": file_record(source / "model.safetensors"),
              "saved_weights": file_record(saved / "model.safetensors")}
    release(baseline)
    return result


@torch.inference_mode()
def check_frozen(saved):
    runtime = Runtime(Config(model_dir=saved, runner="eager", dtype="fp16", max_batch_size=16))
    cases, _ = load_cases(HERE / "accuracy/data")
    frozen_path = HERE / "accuracy/results.json"
    frozen = json.loads(frozen_path.read_text())
    expected = {(row["id"], row["qid"]): row for row in frozen["raw"]}
    work = []
    for case in cases:
        for qid, question, item in runtime.prepare(request_from_case(case["request"])):
            work.append((case["id"], qid, question, item))
    batch_size, length, markers = 16, 256, 6

    def inputs(rows):
        items = [row[3] for row in rows]
        items += [items[0]] * (batch_size - len(items))
        batch = collate_items([items], runtime.tok.pad_token_id)
        tensors = {"input_ids": torch.full((batch_size, length), runtime.tok.pad_token_id, dtype=torch.long),
                   "attention_mask": torch.zeros(batch_size, length, dtype=torch.long),
                   "marker_pos": torch.zeros(batch_size, markers, dtype=torch.long),
                   "marker_mask": torch.zeros(batch_size, markers, dtype=torch.bool), "qtype": batch["qtype"]}
        for key in INPUT_KEYS[:-1]:
            tensors[key][:, :batch[key].shape[1]] = batch[key]
        return [tensors[key] for key in INPUT_KEYS]

    static = [t.cuda() for t in inputs(work[:16])]
    prepared = runtime.runner.function
    prepared(*static)  # 先收集真实 INT8 指令，不在 Dynamo trace 中读取 asm。
    compiled = torch.compile(prepared, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
    graph, outputs = capture(lambda function=compiled, tensors=static: function(*tensors))
    maxima = dict(logits=0.0, probabilities=0.0, score=0.0)
    label_mismatches, correct, classification_n = 0, 0, 0
    for offset in range(0, len(work), batch_size):
        rows = work[offset:offset + batch_size]
        for target, value in zip(static, inputs(rows)):
            target.copy_(value)
        graph.replay()
        logits, acts = [tensor.float().cpu().numpy() for tensor in outputs]
        assert np.isfinite(logits).all() and np.isfinite(acts).all()
        for i, (ident, qid, question, item) in enumerate(rows):
            row = expected[(ident, qid)]
            original = row["predictions"]["w8a8-native-compile"]
            k = len(item["markers"])
            maxima["logits"] = max(maxima["logits"], float(np.abs(logits[i, :k] - original["logits"]).max()))
            answer = runtime.answer(question, item, logits[i, :k], torch.tensor(acts[i]).softmax(-1).numpy())
            probabilities = np.array(list(answer["probabilities"].values()))
            maxima["probabilities"] = max(maxima["probabilities"], float(np.abs(probabilities - original["probabilities"]).max()))
            if question.type == "score":
                maxima["score"] = max(maxima["score"], abs(answer["score"] - original["answer"]["score"]))
            else:
                label = answer["choice"] if question.type == "choice" else answer["noul"] >= 0.5
                label_mismatches += label != original["label"]
                correct += label == row["gold"]
                classification_n += 1
    from laya.int8_kernels import EVIDENCE
    result = {"profile": [batch_size, length, markers], "questions": len(work),
              "label_mismatches_vs_frozen": label_mismatches, "max_abs_vs_frozen": maxima,
              "classification_correct": correct, "classification_n": classification_n,
              "accuracy": correct / classification_n, "model": runtime.info,
              "int8_evidence": dict(EVIDENCE), "frozen_results_sha256": file_record(frozen_path)["sha256"]}
    assert label_mismatches == 0, result
    assert maxima["probabilities"] <= 1e-5 and maxima["score"] <= 1e-5, result
    assert result["int8_evidence"]["int8_tensor_core_instructions"]
    graph.reset()
    del graph, outputs, compiled, prepared, static
    release(runtime)
    return result


def check_service(saved, runner, log_dir):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    args = [sys.executable, "-m", "laya.commands.app", "serve", "--model-dir", str(saved),
            "--runner", runner, "--max-batch-size", "16", "--host", "127.0.0.1", "--port", str(port)]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "TORCHINDUCTOR_COMPILE_THREADS": "4", "LAYA_STARTUP_TIMEOUT": "900",
           "LAYA_REQUEST_TIMEOUT": "300", "LAYA_SHUTDOWN_TIMEOUT": "15", "LAYA_GRAPH_CACHE_SIZE": "4"}
    payloads = [json.loads((ROOT / "scripts/inputs" / f"{name}.json").read_text()) for name in ("short", "long")]
    payloads.extend([
        {"state": "Alice handles testing; Bob handles releases.", "questions": {"owner": {
            "type": "choice", "instructions": "Who handles testing?", "criteria": ["Alice", "Bob"]}}},
        {"state": "小李负责测试，小王负责发布。", "questions": {
            "owner": {"type": "choice", "instructions": "谁负责测试？", "criteria": ["小李", "小王"]},
            "score": {"type": "score", "instructions": "测试重要吗？", "criteria": ["低", "中", "高"]},
            "holds": {"type": "noul", "instructions": "小李负责测试吗？"}}},
    ])
    # 每个后端以自身离线执行作为 oracle，避免把跨后端浮点/量化舍入当作存储错误。
    offline = Runtime(Config(model_dir=saved, runner=runner, dtype="fp16", max_batch_size=16,
                             graph_cache_size=4, compile_cache_size=16))
    expected = [offline.infer(DecisionRequest.model_validate(p)) for p in payloads]
    release(offline)
    del offline
    gc.collect()
    torch.cuda.empty_cache()
    worker = None
    log_path = log_dir / f"{runner}.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(args, cwd=ROOT, env=env, stdout=log, stderr=log)
        try:
            with httpx.Client(base_url=url, trust_env=False, timeout=300) as client:
                deadline = monotonic() + 900
                while monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(log_path.read_text()[-6000:])
                    try:
                        if client.get("/health/ready", timeout=1).status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    sleep(0.1)
                else:
                    raise TimeoutError("W8A8 service not ready")
                info = client.get("/v1/info").json()
                worker = info["worker_pid"]
                assert info["quantization"]["method"] == "laya_w8a8" and not info["autocast"]
                assert info["parameter_count"] == 321908995
                assert info["parameter_bytes"] == 518775302
                answers = []
                for p, reference in zip(payloads, expected):
                    response = client.post("/v1/decisions", json=p)
                    response.raise_for_status()
                    actual = response.json()
                    assert actual["answers"] == reference["answers"], (runner, actual, reference)
                    again = client.post("/v1/decisions", json=p)
                    again.raise_for_status()
                    assert again.json()["answers"] == actual["answers"]
                    answers.append(actual["answers"])
                invalid = {**payloads[0], "state": "背景材料。" * 2000}
                assert client.post("/v1/decisions", json=invalid).status_code == 422

                async def concurrent():
                    async with httpx.AsyncClient(base_url=url, trust_env=False, timeout=300) as requests:
                        return await asyncio.gather(*(requests.post("/v1/decisions", json=invalid if i == 7 else payloads[i % 2]) for i in range(16)))

                responses = asyncio.run(concurrent())
                max_prob_diff = 0.0
                choice_mismatches = 0
                for i, response in enumerate(responses):
                    if i == 7:
                        assert response.status_code == 422
                        continue
                    response.raise_for_status()
                    actual = response.json()["answers"]
                    reference = expected[i % 2]["answers"]
                    for key, value in actual.items():
                        max_prob_diff = max(max_prob_diff, max(abs(v - reference[key]["probabilities"][k]) for k, v in value["probabilities"].items()))
                        choice_mismatches += value.get("choice") != reference[key].get("choice")
                # B1/B16 可以改变浮点舍入与动态量化边界；记录实际漂移，不混淆保存对齐。
                assert client.get("/health/ready").status_code == 200
                metrics = client.get("/metrics").json()
                result = {"runner": runner, "model": info, "command": args,
                          "offline_service_answers_equal": True, "payloads": payloads, "answers": answers,
                          "concurrency": 16, "invalid_request_isolated": True,
                          "concurrent_max_probability_drift_vs_b1": max_prob_diff,
                          "concurrent_choice_mismatches_vs_b1": choice_mismatches, "metrics": metrics}
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            if worker:
                try:
                    os.kill(worker, 0)
                except ProcessLookupError:
                    pass
                else:
                    raise AssertionError(f"worker leaked: {worker}")
    return {**result, "shutdown_clean": True}


@app.command()
def main(model_dir: Path = typer.Option(..., envvar="LAYA_MODEL_DIR"),
         source_dir: Path = typer.Option(...), output: Path = typer.Option(HERE / "saved-validation.json"),
         services: bool = typer.Option(True)):
    if not (HERE / "accuracy/results.json").is_file():
        raise typer.BadParameter(
            "请先用原 FP16 仓库运行 poetry run python -m tests.w8a8.accuracy.evaluate，生成对齐参考"
        )
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")
    report = {"created_at": datetime.now(timezone.utc).isoformat(),
              "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "workspace_status": subprocess.check_output(["git", "status", "--short"], text=True),
              "model_dir": str(model_dir), "source_dir": str(source_dir), "completed": False}
    log_dir = ROOT / "outputs/w8a8-saved-validation"
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        report["weights"] = check_weights(source_dir, model_dir)
        print("saved weights: bitwise equal", flush=True)
        report["frozen"] = check_frozen(model_dir)
        print("684 frozen questions: aligned", flush=True)
        report["services"] = []
        if services:
            for runner in ("eager", "cuda-graph", "cuda-graph-compile"):
                report["services"].append(check_service(model_dir, runner, log_dir))
                print(f"{runner}: HTTP/concurrency/shutdown passed", flush=True)
        report["completed"] = True
    finally:
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    app()
