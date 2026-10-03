"""设备、模型、tokenizer 和 runner 的资源生命周期。"""
import hashlib
import json
import os
import numpy as np
import torch
from safetensors.torch import load_file
from safetensors import safe_open
from transformers import AutoModel, PreTrainedTokenizerFast
from laya.configs.settings import Config
from laya.models.decision import DecisionModel
from laya.models.loading import move_model, encoder_config_for_runtime
from laya.quantization.config import read_quantization
from laya.quantization.transforms import explicit_heads, install_linears, validate_weights, fuse_norms
from laya.runners.eager import EagerRunner
from .fingerprints import fingerprint, runtime_source_fingerprint


class Runtime:
    def __init__(self, config: Config):
        torch.set_num_threads(config.threads)
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        tokenizer_parallelism = os.environ["TOKENIZERS_PARALLELISM"].lower() not in (
            "", "0", "false", "off", "no", "n", "f",
        )
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[config.dtype]
        root = config.model_dir
        self.cfg = json.loads((root / "config.json").read_text())
        if self.cfg.get("format_version") != 1:
            raise ValueError("模型 config.json 的 format_version 必须为 1")
        quantization = read_quantization(self.cfg)
        if quantization and (self.device.type != "cuda" or self.dtype != torch.float16):
            raise ValueError("w8a8_requires_cuda_and_fp16")
        if self.device.type == "cuda":
            if (
                not torch.cuda.is_available()
                or self.device.index >= torch.cuda.device_count()
            ):
                raise ValueError("requested_cuda_unavailable")
            if quantization and torch.cuda.get_device_capability(self.device) < (8, 0):
                raise ValueError("w8a8_requires_sm80_or_newer")
            torch.cuda.set_device(self.device)
        self.tok = PreTrainedTokenizerFast(
            tokenizer_file=str(root / "tokenizer.json"), **self.cfg["tokenizer"]
        )
        ecfg = encoder_config_for_runtime(self.cfg["encoder"])
        ecfg.reference_compile = False
        encoder = AutoModel.from_config(
            ecfg, dtype=self.dtype, attn_implementation="sdpa", trust_remote_code=False
        )
        head = self.cfg["decision_head"]
        model = DecisionModel(encoder, head["layers"], head["num_actions"])
        if quantization:
            explicit_heads(model)
            install_linears(model, quantization)
        state = load_file(str(root / "model.safetensors"))
        if quantization:
            validate_weights(model, state)
        model.load_state_dict(state, strict=True)
        del state
        move_model(model, self.device, self.dtype).eval()
        if quantization:
            fuse_norms(model)
            # 确保 CUDA 依赖在启动时检查，避免首个请求才发现缺少后端。
            from laya.quantization import int8_kernels

            del int8_kernels
        if config.runner.startswith("cuda-graph"):
            from laya.runners.cuda_graph import CudaGraphRunner

            self.runner = CudaGraphRunner(model, self.device, self.dtype, config,
                                          self.cfg["input_limits"]["max_len"], quantized=bool(quantization))
        else:
            self.runner = EagerRunner(model, self.device, self.dtype, bool(quantization),
                                      self.cfg["input_limits"]["max_len"])
        with safe_open(root / "model.safetensors", framework="pt") as weights:
            parameter_count = sum(
                int(np.prod(weights.get_slice(name).get_shape()))
                for name in weights.keys() if name != "temperature" and not name.endswith(".weight_scales")
            )
            parameter_bytes = sum(
                weights.get_tensor(name).numel() * weights.get_tensor(name).element_size()
                for name in weights.keys() if name != "temperature" and not name.endswith(".weight_scales")
            ) if quantization else parameter_count * torch.empty((), dtype=self.dtype).element_size()
            scale_bytes = sum(weights.get_tensor(name).numel() * 4 for name in weights.keys()
                              if name.endswith(".weight_scales"))
        code_hash = runtime_source_fingerprint(config.runner, bool(quantization))
        identity = hashlib.sha256(
            (
                fingerprint(root)
                + code_hash
                + torch.__version__
                + str(self.device)
                + config.dtype
                + config.runner
                + json.dumps({"max_batch_size": config.max_batch_size,
                              "max_batch_tokens": config.max_batch_tokens,
                              "threads": config.threads,
                              "tokenizer_parallelism": tokenizer_parallelism}, sort_keys=True)
                + json.dumps(getattr(self.runner, "info", {}), sort_keys=True)
            ).encode()
        ).hexdigest()
        self.info = {
            "runtime_code_sha256": code_hash,
            "torch_version": torch.__version__,
            "model": root.name,
            "fingerprint": identity,
            "device": str(self.device),
            "dtype": config.dtype,
            "autocast": not quantization and self.dtype in (torch.float16, torch.bfloat16),
            "parameter_count": parameter_count,
            "parameter_bytes": parameter_bytes,
            "quantization": ({"method": quantization.method, "version": quantization.version,
                              "weight_bits": 8, "activation_bits": 8,
                              "activation_scheme": quantization.activation_scheme,
                              "quantized_modules": len(quantization.quantized_modules),
                              "scale_bytes": scale_bytes} if quantization else None),
            "runner": config.runner,
            **getattr(self.runner, "info", {}),
            "max_len": self.cfg["input_limits"]["max_len"],
            "head_max_len": self.cfg["input_limits"]["head_max_len"],
            "calibrated": False,
            "batch_size": 1,
            "max_batch_size": config.max_batch_size,
            "max_batch_tokens": config.max_batch_tokens,
            "batch_wait_ms": config.batch_wait_ms,
            "threads": config.threads,
            "tokenizer_parallelism": tokenizer_parallelism,
        }

    def close(self):
        if hasattr(self.runner, "close"):
            self.runner.close()
