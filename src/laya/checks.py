"""显式真实模型验证；不属于默认无设备测试。"""

import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from .config import Config
from .contracts import DecisionRequest
from .runtime import Runtime, InputTooLong, move_model


def check_reference(model_dir: Path, snapshot_dir: Path, device="cuda:0", dtype="fp16"):
    # 重构对齐使用相同计算精度；跨精度漂移另行报告，不替代行为基线。
    tolerance = {
        "logits": 1e-5,
        "act_rtol": 1e-6,
        "act_atol": 1e-5,
        "probabilities": 0.000051,
        "score": 0.000051,
        "confidence": 0.000051,
    }
    # 仅此审计入口执行用户给定快照代码；线上路径从不导入权重目录源码。
    for name in ("rl_common", "rl_agent_api"):
        spec = importlib.util.spec_from_file_location(
            name, snapshot_dir / (name + ".py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    runtime = Runtime(Config(model_dir=model_dir, device=device, dtype=dtype))
    # Oracle 独立读取原始多语言快照，避免使用转换后的配置自我对照。
    # 原始快照也是 Transformers 5 配置。独立读取并临时转换供 v4 审计，
    # 不修改快照、不复用线上配置转换，以免二者同时采用错误的默认频率。
    api = sys.modules["rl_agent_api"]
    reference_builder = api.build_model
    with TemporaryDirectory(prefix="laya-oracle-config-") as directory:
        source = json.loads(
            (snapshot_dir / "multilingual/encoder/config.json").read_text()
        )
        rope = source.get("rope_parameters", {})
        if "full_attention" in rope:
            source["global_rope_theta"] = rope["full_attention"]["rope_theta"]
        if "sliding_attention" in rope:
            source["local_rope_theta"] = rope["sliding_attention"]["rope_theta"]
        (Path(directory) / "config.json").write_text(json.dumps(source))
        api.build_model = lambda cfg, encoder_dir=None: reference_builder(
            cfg, directory if encoder_dir else None
        )
        try:
            original = sys.modules["rl_agent_api"].RLAgent(
                str(snapshot_dir / "multilingual"), device="cpu"
            )
        finally:
            api.build_model = reference_builder
    reference_dtype = {"fp16": torch.float16, "fp32": torch.float32}[dtype]
    move_model(original.model, torch.device(device), reference_dtype)
    original.device = torch.device(device)
    # Oracle 与被测服务同精度，覆盖上游 GPU 默认的 BF16 autocast。
    original.dtype = reference_dtype
    reference_buffers = dict(original.model.named_buffers())
    for name, buffer in runtime.runner.model.named_buffers():
        if "inv_freq" in name:
            assert buffer.dtype == reference_buffers[name].dtype
            np.testing.assert_array_equal(
                buffer.cpu().numpy(), reference_buffers[name].cpu().numpy()
            )
    assert runtime.tok.special_tokens_map == original.tok.special_tokens_map
    assert json.loads(runtime.tok.backend_tokenizer.to_str()) == json.loads(
        original.tok.backend_tokenizer.to_str()
    ), "分词器后端配置发生变化"
    maxima, act_maxima, act_relative_maxima, probability_maxima = [], [], [], []
    act_probability_maxima = []
    score_maxima, noul_maxima, confidence_maxima = [], [], []
    reference_logits = []
    new_logits = []
    reference_inputs, new_inputs = [], []

    def capture(outputs, inputs):
        def hook(model, args, result):
            inputs.append([x.detach().cpu().numpy() for x in args])
            outputs.append([x.detach().float().cpu().numpy() for x in result])

        return hook

    h1 = original.model.register_forward_hook(
        capture(reference_logits, reference_inputs)
    )
    h2 = runtime.runner.model.register_forward_hook(
        capture(new_logits, new_inputs)
    )
    cases = []
    for state in (
        "小李负责测试。",
        "小李负责测试，小王只负责发布。",
        "讨论背景。" * 180 + "小李负责测试，小王只负责发布。",
    ):
        for question in (
            {
                "type": "choice",
                "instructions": "谁负责测试？",
                "criteria": ["小李", "小王"],
            },
            {
                "type": "score",
                "instructions": "测试的重要程度？",
                "criteria": ["低", "中", "高"],
            },
            {"type": "noul", "instructions": "小李负责测试吗？"},
        ):
            cases.append({"state": state, "questions": {"q": question}})
    started = perf_counter()
    try:
        for data in cases:
            actual = runtime.infer(DecisionRequest.model_validate(data))
            expected = original.system_one(data["state"], data["questions"])
            assert len(reference_inputs[-1]) == len(new_inputs[-1])
            for reference_input, new_input in zip(reference_inputs[-1], new_inputs[-1]):
                np.testing.assert_array_equal(reference_input, new_input)
            delta = float(np.max(np.abs(reference_logits[-1][0] - new_logits[-1][0])))
            maxima.append(delta)
            assert delta <= tolerance["logits"], delta
            act_delta = float(np.max(np.abs(reference_logits[-1][1] - new_logits[-1][1])))
            act_maxima.append(act_delta)
            act_relative_maxima.append(float(np.max(
                np.abs(reference_logits[-1][1] - new_logits[-1][1])
                / np.maximum(np.abs(reference_logits[-1][1]), 1)
            )))
            # 动作 logits 量级约 1e3，采用与推理精度对应的组合容差。
            np.testing.assert_allclose(
                new_logits[-1][1], reference_logits[-1][1],
                rtol=tolerance["act_rtol"], atol=tolerance["act_atol"],
            )
            a, b = actual["answers"]["q"], expected["answers"]["q"]
            act_probability_maxima.append(
                abs(a["act_probability"] - b["rl_agent"]["act_probability"])
            )
            assert act_probability_maxima[-1] <= 1e-6
            if "probabilities" in b:
                probability_delta = max(
                    abs(a["probabilities"][key] - value)
                    for key, value in b["probabilities"].items()
                )
                probability_maxima.append(probability_delta)
                assert probability_delta <= tolerance["probabilities"], probability_delta
                confidence_maxima.append(abs(a["confidence"] - b["confidence"]))
                assert confidence_maxima[-1] <= tolerance["confidence"]
            if a["type"] == "choice":
                assert a["choice"] == b["choice"]
            elif a["type"] == "score":
                score_maxima.append(abs(a["score"] - b["score"]))
                assert score_maxima[-1] <= tolerance["score"]
            else:
                noul_maxima.append(abs(a["noul"] - b["noul"]))
                assert noul_maxima[-1] <= tolerance["probabilities"]
        # 参考实现会静默截断；封装必须在推理前拒绝。
        for data in (
            {"state": "背景材料。" * 2000, "questions": cases[0]["questions"]},
            {
                "state": "短文本",
                "questions": {
                    "q": {
                        "type": "choice",
                        "instructions": "判断。" * 300,
                        "criteria": ["甲", "乙"],
                    }
                },
            },
            {
                "state": "短文本",
                "questions": {
                    "q": {
                        "type": "choice",
                        "instructions": "判断",
                        "criteria": ["甲" * 200, "乙"],
                    }
                },
            },
        ):
            try:
                runtime.infer(DecisionRequest.model_validate(data))
            except InputTooLong:
                pass
            else:
                raise AssertionError("必须拒绝截断")
    finally:
        h1.remove()
        h2.remove()
    return {
        "cases": len(cases),
        "truncation_rejections": 3,
        "max_abs_logits": max(maxima),
        "max_abs_act_logits": max(act_maxima),
        "max_rel_act_logits": max(act_relative_maxima),
        "max_abs_act_probability": max(act_probability_maxima),
        "max_abs_rounded_probabilities": max(probability_maxima),
        "max_abs_score": max(score_maxima),
        "max_abs_noul": max(noul_maxima),
        "max_abs_confidence": max(confidence_maxima),
        "tokenizer_identical": True,
        "position_buffers_identical": True,
        "model_inputs_identical": True,
        "device": device,
        "dtype": dtype,
        "reference_dtype": dtype,
        "tolerance": tolerance,
        "elapsed_seconds": perf_counter() - started,
        "fingerprint": runtime.info["fingerprint"],
    }
