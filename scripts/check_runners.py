"""审计完整 logits，避免饱和的动作概率掩盖动作头误差。"""

import gc
import json
from pathlib import Path

import numpy as np
import torch
import typer

from laya.config import Config
from laya.contracts import DecisionRequest
from laya.runtime import Runtime
from scripts.benchmark_runners import compare

app = typer.Typer()


def collect(runtime, cases):
    vectors = []
    if runtime.info["runner"] == "eager":
        def capture(model, args, outputs):
            vectors.append(np.concatenate([value[0].detach().float().cpu().numpy() for value in outputs]))

        handle = runtime.runner.model.register_forward_hook(capture)
    else:
        encode = runtime.runner.llm.encode

        def capture(*args, **kwargs):
            outputs = encode(*args, **kwargs)
            vectors.extend(output.outputs.data.float().cpu().numpy() for output in outputs)
            return outputs

        runtime.runner.llm.encode = capture
    try:
        answers = [runtime.infer(DecisionRequest.model_validate(case)) for case in cases]
    finally:
        if runtime.info["runner"] == "eager":
            handle.remove()
        else:
            runtime.runner.llm.encode = encode
    return {"model": runtime.info, "outputs": answers, "logits": [vector.tolist() for vector in vectors]}


@app.command()
def main(output: Path = typer.Option(Path("benchmarks/vllm_16"))):
    cases = json.loads((output / "alignment-inputs.json").read_text())
    oracle = {}
    report = {"canonical_reference": "eager/fp16", "results": [], "tolerance": {
        "fp16": {"logits_rtol": 0.005, "logits_atol": 0.02, "act_rtol": 0.005, "act_atol": 0.01},
        "fp32": {"logits_rtol": 1e-5, "logits_atol": 1e-4, "act_rtol": 1e-5, "act_atol": 1e-4}}}
    try:
        for runner, dtype in [("eager", "fp16"), ("eager", "fp32"), ("vllm-eager", "fp16"),
                              ("vllm-eager", "fp32"), ("vllm", "fp16"), ("vllm", "fp32")]:
            print(f"完整 logits 审计：{runner}/{dtype}", flush=True)
            runtime = Runtime(Config(runner=runner, dtype=dtype))
            try:
                result = collect(runtime, cases)
            finally:
                runtime.close()
                del runtime
                gc.collect()
                torch.cuda.empty_cache()
            if runner == "eager":
                oracle[dtype] = result
            result["vs_same_precision_eager"] = compare(oracle[dtype]["outputs"], result["outputs"])
            result["vs_eager_fp16"] = compare(oracle["fp16"]["outputs"], result["outputs"])
            max_decision, max_act, max_act_relative = 0.0, 0.0, 0.0
            tolerance = report["tolerance"][dtype]
            assert len(result["logits"]) == len(oracle[dtype]["logits"])
            for expected, actual in zip(oracle[dtype]["logits"], result["logits"]):
                expected, actual = np.array(expected), np.array(actual)
                assert expected.shape == actual.shape
                max_decision = max(max_decision, float(abs(expected[:-2] - actual[:-2]).max()))
                max_act = max(max_act, float(abs(expected[-2:] - actual[-2:]).max()))
                max_act_relative = max(max_act_relative, float((abs(expected[-2:] - actual[-2:]) / np.maximum(abs(expected[-2:]), 1)).max()))
                np.testing.assert_allclose(actual[:-2], expected[:-2], rtol=tolerance["logits_rtol"], atol=tolerance["logits_atol"])
                np.testing.assert_allclose(actual[-2:], expected[-2:], rtol=tolerance["act_rtol"], atol=tolerance["act_atol"])
            result.update(max_abs_decision_logits=max_decision, max_abs_act_logits=max_act,
                          max_rel_act_logits=max_act_relative, aligned=True)
            report["results"].append(result)
        report["completed"] = True
    finally:
        (output / "logits-alignment.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    app()
