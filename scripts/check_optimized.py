"""按 token 输入关联原始双头 logits，审计串行、混合请求、并发批与缓存复用。"""

import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import typer

from laya.configs.settings import Config
from laya.api.contracts import DecisionRequest
from laya.inferencers.decision import DecisionInferencer
from scripts.benchmark_support import alignment_cases, compare


app = typer.Typer()
TOLERANCE = {"decision_rtol": 0.005, "decision_atol": 0.02,
             "act_rtol": 0.005, "act_atol": 0.01,
             "probabilities": 0.003, "confidence": 0.005, "score": 0.005,
             "noul": 0.003, "act_probability": 1e-6}


def key_for(batch, row):
    length = int(batch["attention_mask"][row].sum())
    count = int(batch["marker_mask"][row].sum())
    fields = [batch["input_ids"][row, :length].tolist(),
              batch["marker_pos"][row, :count].tolist(), int(batch["qtype"][row])]
    return hashlib.sha256(json.dumps(fields).encode()).hexdigest(), count


@app.command()
def main(output: Path = typer.Option(Path("benchmarks/torch_28_optimized")),
         model_dir: Path | None = typer.Option(None, envvar="LAYA_MODEL_DIR"),
         include_compile: bool = typer.Option(True),
         runner: list[str] | None = typer.Option(None),
         resume: bool = typer.Option(False)):
    output.mkdir(parents=True, exist_ok=True)
    selected = Config(**({"model_dir": model_dir} if model_dir is not None else {})).model_dir
    cases = alignment_cases(selected)
    payloads = [DecisionRequest.model_validate(case) for case in cases]
    oracle, checks = {}, []
    baseline = DecisionInferencer(Config(model_dir=selected, runner="eager", dtype="fp16", max_batch_size=1))

    def record(batch, logits, acts):
        logits, acts = logits.float().cpu().numpy(), acts.float().cpu().numpy()
        for row in range(len(logits)):
            key, count = key_for(batch, row)
            oracle[key] = {"decision": logits[row, :count].tolist(), "act": acts[row].tolist()}

    baseline.runtime.runner.raw_observer = record
    expected = [baseline.infer(payload) for payload in payloads]
    baseline_info = baseline.runtime.info
    baseline.close()
    del baseline
    gc.collect()
    torch.cuda.empty_cache()
    report = {"canonical_reference": "eager/fp16/batch1", "tolerance": TOLERANCE,
              "baseline": baseline_info, "oracle": oracle, "cases": cases,
              "results": checks, "completed": False}
    path = output / "alignment.json"
    if resume:
        previous = json.loads(path.read_text())
        assert previous["completed"]
        assert previous["baseline"] == baseline_info
        assert previous["oracle"] == oracle and previous["tolerance"] == TOLERANCE
        checks.extend(previous["results"])
    variants = [("cuda-graph", 1), ("cuda-graph", 8)]
    if include_compile:
        variants.append(("cuda-graph-compile", 8))
    if runner:
        variants = [variant for variant in variants if variant[0] in runner]
        assert variants and set(runner) == {variant[0] for variant in variants}
    for runner, streams in variants:
        if any(item["runner"] == runner and item["streams"] == streams and item.get("aligned") for item in checks):
            continue
        inferencer = DecisionInferencer(Config(model_dir=selected, runner=runner, dtype="fp16", max_batch_size=16,
                                 graph_streams=streams, graph_cache_size=4))
        stats = {"runner": runner, "streams": streams, "model": inferencer.runtime.info, "raw_samples": 0,
                 "max_decision_abs": 0.0, "max_act_abs": 0.0, "max_act_rel": 0.0,
                 "checks": []}
        checks.append(stats)

        def audit(batch, logits, acts):
            logits, acts = logits.float().cpu().numpy(), acts.float().cpu().numpy()
            for row in range(len(logits)):
                key, count = key_for(batch, row)
                a, b = np.array(oracle[key]["decision"]), np.array(oracle[key]["act"])
                stats["raw_samples"] += 1
                stats["max_decision_abs"] = max(stats["max_decision_abs"], float(abs(a - logits[row, :count]).max()))
                stats["max_act_abs"] = max(stats["max_act_abs"], float(abs(b - acts[row]).max()))
                stats["max_act_rel"] = max(stats["max_act_rel"], float((abs(b - acts[row]) / np.maximum(abs(b), 1)).max()))
                np.testing.assert_allclose(logits[row, :count], a, rtol=TOLERANCE["decision_rtol"], atol=TOLERANCE["decision_atol"])
                np.testing.assert_allclose(acts[row], b, rtol=TOLERANCE["act_rtol"], atol=TOLERANCE["act_atol"])

        inferencer.runtime.runner.raw_observer = audit
        try:
            for count in [1, 2, 4, 8, 16]:
                actual = []
                for payload in payloads:
                    group = inferencer.infer_many([payload] * count)
                    actual.append(group[0])
                    assert all(result["answers"] == group[0]["answers"] for result in group)
                differences = compare(expected, actual)
                assert differences["choice_mismatches"] == 0
                assert all(value <= TOLERANCE[field] for field, value in differences["max_abs"].items())
                stats["checks"].append({"count": count, **differences})
                print(f"{runner}, streams={streams}, count={count}: 对齐通过", flush=True)
            actual = inferencer.infer_many(payloads)
            differences = compare(expected, actual)
            assert differences["choice_mismatches"] == 0
            assert all(value <= TOLERANCE[field] for field, value in differences["max_abs"].items())
            stats["mixed_requests"] = differences
            stats["runner_metrics"] = getattr(inferencer.runtime.runner, "metrics", lambda: {})()
            stats["aligned"] = True
        finally:
            inferencer.close()
            del inferencer
            gc.collect()
            torch.cuda.empty_cache()
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    report["completed"] = True
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    app()
