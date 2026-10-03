"""vLLM 执行策略；与 eager 共用输入、校准和领域输出。"""

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch


@contextmanager
def model_output():
    """CLI 的 stdout 也只输出领域 JSON；子进程日志沿 stderr 继承。"""
    sys.stdout.flush()
    saved = os.dup(sys.stdout.fileno())
    try:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, sys.stdout.fileno())
        os.close(saved)


def native_config(cfg, tok):
    native = dict(cfg["encoder"])
    # vLLM 0.14.1 + Transformers 4 接受平铺 RoPE 字段。
    rope = native.pop("rope_parameters", {})
    native.pop("layer_types", None)
    for kind, field in (("full_attention", "global_rope_theta"), ("sliding_attention", "local_rope_theta")):
        if kind in rope:
            if rope[kind].get("rope_type", "default") != "default":
                raise ValueError("unsupported native RoPE scaling")
            native[field] = rope[kind]["rope_theta"]
    # HF 的窗口包含距离 == radius，vLLM backend 使用 distance < window。
    # 原生 ModernBERT 直接传 radius 会少一圈；翻译成 radius + 1。
    native["laya_local_attention"] = native["local_attention"]
    native["local_attention"] += 2
    native.update(
        architectures=["LayaForDecisions"],
        laya_head_layers=cfg["decision_head"]["layers"],
        laya_num_actions=cfg["decision_head"]["num_actions"],
        laya_mask_id=tok.mask_token_id,
        laya_type_prefixes=[
            tok(kind + " question:", add_special_tokens=False)["input_ids"]
            for kind in ("choice", "score", "noul")
        ],
    )
    return native


class VllmRunner:
    @model_output()
    def __init__(self, config, cfg, tok):
        from vllm import LLM, __version__
        from .vllm_plugin import register

        register()
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        self.directory = TemporaryDirectory(prefix="laya-vllm-config-")
        native = native_config(cfg, tok)
        native["laya_threads"] = config.threads
        (Path(self.directory.name) / "config.json").write_text(json.dumps(native))
        enforce_eager = config.runner == "vllm-eager"
        self.llm = LLM(
            model=str(config.model_dir), hf_config_path=self.directory.name,
            runner="pooling", dtype={"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}[config.dtype],
            skip_tokenizer_init=True, trust_remote_code=False,
            max_model_len=cfg["input_limits"]["max_len"], max_num_seqs=1,
            max_num_batched_tokens=cfg["input_limits"]["max_len"],
            enable_chunked_prefill=False, enable_prefix_caching=False,
            gpu_memory_utilization=0.1, enforce_eager=enforce_eager,
            disable_log_stats=True,
            worker_extension_cls="laya.vllm_plugin.LayaWorkerExtension",
            # 编译器默认可消除 FP16 中间舍入；行为基线要求保留 eager 舍入。
            compilation_config={
                "inductor_compile_config": {"emulate_precision_casts": True},
                "custom_ops": ["none", "+rotary_embedding"],
            },
        )
        compilation = self.llm.llm_engine.vllm_config.compilation_config
        self.info = {
            "vllm_version": __version__, "enforce_eager": enforce_eager,
            "compilation_mode": compilation.mode.name,
            "cudagraph_mode": compilation.cudagraph_mode.name,
            "cudagraph_capture_sizes": compilation.cudagraph_capture_sizes,
            "head_compilation": False,
            "emulate_eager_rounding": True,
            "max_gpu_sequences": 1,
            "batch_execution": "vllm_serial_scheduler",
            **self.llm.collective_rpc("laya_model_info", timeout=30)[0],
        }
        self.raw_observer = None

    @model_output()
    def execute(self, batch):
        lengths = batch["attention_mask"].sum(-1).tolist()
        prompts = [
            {"prompt_token_ids": ids[:length].tolist()}
            for ids, length in zip(batch["input_ids"], lengths)
        ]
        outputs = self.llm.encode(prompts, pooling_task="embed", use_tqdm=False)
        kmax = batch["marker_mask"].shape[1]
        logits = np.full((len(outputs), kmax), -1e4, dtype=np.float32)
        acts = []
        raw_acts = []
        for i, output in enumerate(outputs):
            k = int(batch["marker_mask"][i].sum())
            data = output.outputs.data.float().cpu()
            if data.numel() != k + 2:
                raise RuntimeError("laya_vllm_output_shape_mismatch")
            logits[i, :k] = data[:k].numpy()
            acts.append(torch.softmax(data[k:], -1).numpy())
            if self.raw_observer is not None:
                raw_acts.append(data[k:])
        if self.raw_observer is not None:
            self.raw_observer(batch, torch.from_numpy(logits), torch.stack(raw_acts))
        return logits, np.stack(acts)

    def close(self):
        try:
            self.llm.llm_engine.engine_core.shutdown()
        finally:
            self.directory.cleanup()
