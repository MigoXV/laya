"""唯一设备所有者使用的本地推理与公共领域适配。"""

import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from safetensors.torch import load_file
from safetensors import safe_open
from transformers import AutoConfig, AutoModel, PreTrainedTokenizerFast

from .config import Config
from .contracts import DecisionRequest
from .quantization import read_quantization, explicit_heads, install_linears, validate_weights, fuse_norms
from .reference import (
    DecisionModel,
    QTYPES,
    collate_items,
    confidence_from_probs,
    render_options,
    serialize_state,
    temp_bucket,
)


class InputTooLong(ValueError):
    pass


def move_model(model, device, dtype):
    """转换参数精度，保留 RoPE 频率等浮点缓冲区的原始精度。"""
    buffers = {
        name: buffer for name, buffer in model.named_buffers()
        if buffer.is_floating_point()
    }
    model.to(device=device, dtype=dtype)
    for name, buffer in buffers.items():
        owner, _, attribute = name.rpartition(".")
        setattr(model.get_submodule(owner), attribute, buffer.to(device=device))
    return model


def fingerprint(root: Path):
    digest = hashlib.sha256()
    for name in (
        "config.json",
        "tokenizer.json",
        "model.safetensors",
    ):
        digest.update(name.encode())
        with (root / name).open("rb") as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def checked_sequence(tok, state, question, config):
    """先验证参考构建器不会截断任何指令、选项或原文。"""
    q = {"t": question.type, "ins": question.instructions, "crit": question.criteria}
    mask = tok.mask_token
    state_text = serialize_state(state)
    rendered = render_options(q)
    if (
        mask in state_text
        or mask in question.instructions
        or any(mask in o for o in rendered)
    ):
        raise ValueError("输入包含模型保留的 MASK 标记")
    backend = tok.backend_tokenizer
    backend.no_truncation()
    backend.no_padding()
    texts = [f"{q['t']} question: {q['ins']}", *[" " + o for o in rendered], state_text]
    head, *options, state_ids = [encoding.ids for encoding in backend.encode_batch(
        texts, add_special_tokens=False
    )]
    option_len = sum(1 + len(o) for o in options)
    budget = config["head_max_len"] - option_len
    if any(len(o) > 48 for o in options) or budget < 16 or len(head) > max(8, budget):
        raise InputTooLong("question_head_exceeded")
    if 4 + len(head) + option_len + len(state_ids) > config["max_len"]:
        raise InputTooLong("state_token_budget_exceeded")
    # 上面的预算已证明不需要截断；复用同一批 token，避免参考构建器重复分词。
    ids = [tok.cls_token_id] + head + [tok.sep_token_id]
    markers = []
    for option in options:
        markers.append(len(ids))
        ids.extend([tok.mask_token_id] + option)
    return ids + [tok.sep_token_id] + state_ids + [tok.sep_token_id], markers


class EagerRunner:
    def __init__(self, model, device, dtype, quantized=False, max_len=1024):
        self.model, self.device, self.dtype = model, device, dtype
        self.raw_observer = None
        self.autocast = not quantized and dtype in (torch.float16, torch.bfloat16)
        self.function = model
        if quantized:
            from .cuda_runner import PreparedModel

            self.function = PreparedModel(model, max_len).eval()

    @torch.inference_mode()
    def execute(self, batch):
        tensors = [
            batch[k].to(self.device)
            for k in (
                "input_ids",
                "attention_mask",
                "marker_pos",
                "marker_mask",
                "qtype",
            )
        ]
        with torch.autocast(
            device_type=self.device.type, dtype=self.dtype,
            enabled=self.autocast,
        ):
            logits, acts = self.function(*tensors)
        if self.raw_observer is not None:
            self.raw_observer(batch, logits, acts)
        return logits.float().cpu().numpy(), torch.softmax(
            acts.float(), -1
        ).cpu().numpy()


def encoder_config_for_runtime(raw):
    encoder_config = dict(raw)
    # Transformers 4.x reads legacy theta fields; 5.x writes rope_parameters.
    # Preserve the asset's position encoding instead of silently using defaults.
    rope = encoder_config.get("rope_parameters", {})
    for kind, legacy in (("full_attention", "global_rope_theta"), ("sliding_attention", "local_rope_theta")):
        if kind in rope:
            if rope[kind].get("rope_type", "default") != "default":
                raise ValueError("unsupported native RoPE scaling")
            encoder_config[legacy] = rope[kind]["rope_theta"]
    return AutoConfig.for_model(**encoder_config)


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
            from . import int8_kernels

            del int8_kernels
        if config.runner.startswith("cuda-graph"):
            from .cuda_runner import CudaGraphRunner

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
        code_hash = hashlib.sha256()
        code_files = ["reference.py", "runtime.py", "contracts.py", "config.py"]
        if config.runner.startswith("cuda-graph") or quantization:
            code_files += ["cuda_runner.py"]
        if quantization:
            code_files += ["quantization.py", "int8_kernels.py"]
        for name in code_files:
            code_hash.update(Path(__file__).with_name(name).read_bytes())
        identity = hashlib.sha256(
            (
                fingerprint(root)
                + code_hash.hexdigest()
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
            "runtime_code_sha256": code_hash.hexdigest(),
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

    def prepare(self, request: DecisionRequest):
        prepared = []
        for qid, question in request.questions.items():
            ids, markers = checked_sequence(
                self.tok, request.state, question, self.cfg["input_limits"]
            )
            prepared.append(
                (
                    qid,
                    question,
                    {
                        "ids": ids,
                        "markers": markers,
                        "qtype": QTYPES[question.type],
                        "target": [0.0] * len(markers),
                        "label": -1,
                        "episode": 0,
                        "ep_step": 0,
                        "ep_len": 1,
                        "src": "api",
                    },
                )
            )
        return prepared

    def answer(self, question, item, logits, act):
        k, qt = len(item["markers"]), item["qtype"]
        temperature = self.cfg["calibration"]["temperature_by_options"].get(
            temp_bucket(qt, k), self.cfg["calibration"]["temperature"][qt]
        )
        z = logits[:k] / temperature
        probabilities = np.exp(z - z.max())
        probabilities /= probabilities.sum()
        keys = list(question.criteria) if question.type == "choice" else [str(i) for i in range(k)]
        answer = {
            "type": question.type,
            "probabilities": dict(zip(keys, map(float, probabilities))),
            "confidence": confidence_from_probs(probabilities, k),
            "act_probability": float(act[0]),
        }
        if question.type == "choice":
            answer["choice"] = keys[int(probabilities.argmax())]
        elif question.type == "score":
            answer["score"] = float((np.arange(k) * probabilities).sum())
        else:
            answer["noul"] = float(probabilities[1])
        return answer

    def infer_many(self, requests):
        """跨请求组批；预处理错误只拒绝其所属请求，结果始终按输入顺序返回。"""
        started = perf_counter()
        prepared, results = [], []
        for index, request in enumerate(requests):
            prep_start = perf_counter()
            try:
                entries = self.prepare(request)
            except ValueError as exc:
                results.append(exc)
                continue
            results.append({
                "model": self.info, "answers": {},
                "usage": {"input_tokens": sum(len(item["ids"]) for _, _, item in entries), "output_tokens": 0},
                "timings": {"preprocess_ms": (perf_counter() - prep_start) * 1000},
            })
            prepared.extend((index, qid, question, item) for qid, question, item in entries)
        # 同一长度桶内 oldest-first；所有结果通过 index/qid 恢复关联。
        groups = {}
        for entry in prepared:
            length = len(entry[3]["ids"])
            bucket = ((length, len(entry[3]["markers"])) if self.config.runner.startswith("cuda-graph")
                      else 1 << (length - 1).bit_length())
            groups.setdefault(bucket, []).append(entry)
        execution = perf_counter()
        batches = []
        for entries in groups.values():
            cursor = 0
            while cursor < len(entries):
                group = []
                max_length = 0
                while cursor < len(entries) and len(group) < self.config.max_batch_size:
                    candidate = entries[cursor]
                    length = max(max_length, len(candidate[3]["ids"]))
                    size = len(group) + 1
                    padded_size = 1 << (size - 1).bit_length() if self.config.runner.startswith("cuda-graph") else size
                    if group and length * padded_size > self.config.max_batch_tokens:
                        break
                    group.append(candidate)
                    max_length = length
                    cursor += 1
                batch = collate_items([[entry[3] for entry in group]], self.tok.pad_token_id)
                logits, acts = self.runner.execute(batch)
                padded_size = (1 << (len(group) - 1).bit_length()
                               if self.config.runner.startswith("cuda-graph") else len(group))
                batches.append({"size": len(group), "padded_size": padded_size, "length": max_length,
                                "tokens": batch["n_tokens"]})
                for row, (index, qid, question, item) in enumerate(group):
                    results[index]["answers"][qid] = self.answer(question, item, logits[row], acts[row])
        execution_ms = (perf_counter() - execution) * 1000
        for request, result in zip(requests, results):
            if isinstance(result, Exception):
                continue
            result["answers"] = {key: result["answers"][key] for key in request.questions}
            result["timings"].update(inference_ms=execution_ms, total_ms=(perf_counter() - started) * 1000)
        self.last_batches = batches
        return results

    def infer(self, request: DecisionRequest):
        result = self.infer_many([request])[0]
        if isinstance(result, Exception):
            raise result
        return result
