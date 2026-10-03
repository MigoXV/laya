"""显式真实模型验证；不属于默认无设备测试。"""

import importlib.util
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

from .config import Config
from .contracts import DecisionRequest
from .runtime import Runtime, InputTooLong


def check_reference(model_dir: Path, snapshot_dir: Path, device="cpu"):
    # 仅此审计入口执行用户给定快照代码；线上路径从不导入权重目录源码。
    for name in ("rl_common", "rl_agent_api"):
        spec = importlib.util.spec_from_file_location(
            name, snapshot_dir / (name + ".py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    runtime = Runtime(Config(model_dir=model_dir, device=device))
    original = sys.modules["rl_agent_api"].RLAgent(str(model_dir), device="cpu")
    original.model.to(torch.device(device))
    original.device = torch.device(device)
    # 参考 API 的 GPU 默认 autocast 为 BF16；本轮明确比较 FP32，不混入精度差异。
    original.dtype = torch.float32
    maxima = []
    reference_logits = []
    new_logits = []
    h1 = original.model.register_forward_hook(
        lambda m, args, result: reference_logits.append(
            result[0].detach().float().cpu().numpy()
        )
    )
    h2 = runtime.runner.model.register_forward_hook(
        lambda m, args, result: new_logits.append(
            result[0].detach().float().cpu().numpy()
        )
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
            delta = float(np.max(np.abs(reference_logits[-1] - new_logits[-1])))
            maxima.append(delta)
            assert delta <= 1e-5, delta
            a, b = actual["answers"]["q"], expected["answers"]["q"]
            if a["type"] == "choice":
                assert a["choice"] == b["choice"]
            elif a["type"] == "score":
                assert abs(a["score"] - b["score"]) <= 0.000051
            else:
                assert abs(a["noul"] - b["noul"]) <= 0.000051
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
        "device": device,
        "elapsed_seconds": perf_counter() - started,
        "fingerprint": runtime.info["fingerprint"],
    }
