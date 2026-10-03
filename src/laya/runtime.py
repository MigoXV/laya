"""唯一设备所有者使用的本地 eager FP32 推理。"""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .config import Config
from .contracts import DecisionRequest
from .reference import (
    DecisionModel,
    QTYPES,
    build_sequence,
    collate_items,
    confidence_from_probs,
    render_options,
    serialize_state,
    temp_bucket,
)


class InputTooLong(ValueError):
    pass


def fingerprint(root: Path):
    digest = hashlib.sha256()
    for name in (
        "rl_agent_config.json",
        "encoder/config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
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
    if (
        mask in serialize_state(state)
        or mask in question.instructions
        or any(mask in o for o in render_options(q))
    ):
        raise ValueError("输入包含模型保留的 MASK 标记")
    head = tok(f"{q['t']} question: {q['ins']}", add_special_tokens=False)["input_ids"]
    options = [
        tok(" " + o, add_special_tokens=False)["input_ids"] for o in render_options(q)
    ]
    option_len = sum(1 + len(o) for o in options)
    budget = config["head_max_len"] - option_len
    if any(len(o) > 48 for o in options) or budget < 16 or len(head) > max(8, budget):
        raise InputTooLong("question_head_exceeded")
    state_len = len(tok(serialize_state(state), add_special_tokens=False)["input_ids"])
    if 4 + len(head) + option_len + state_len > config["max_len"]:
        raise InputTooLong("state_token_budget_exceeded")
    return build_sequence(tok, state, q, config["max_len"], config["head_max_len"])


class EagerRunner:
    def __init__(self, model, device):
        self.model, self.device = model, device

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
        logits, acts = self.model(*tensors)
        return logits.float().cpu().numpy(), torch.softmax(
            acts.float(), -1
        ).cpu().numpy()


class Runtime:
    def __init__(self, config: Config):
        torch.set_num_threads(config.threads)
        self.device = torch.device(config.device)
        if self.device.type == "cuda":
            if (
                not torch.cuda.is_available()
                or self.device.index >= torch.cuda.device_count()
            ):
                raise ValueError("requested_cuda_unavailable")
            torch.cuda.set_device(self.device)
        root = config.model_dir
        self.cfg = json.loads((root / "rl_agent_config.json").read_text())
        self.tok = AutoTokenizer.from_pretrained(
            root / "tokenizer", local_files_only=True, trust_remote_code=False
        )
        ecfg = AutoConfig.from_pretrained(
            root / "encoder", local_files_only=True, trust_remote_code=False
        )
        ecfg.reference_compile = False
        encoder = AutoModel.from_config(
            ecfg, attn_implementation="sdpa", trust_remote_code=False
        )
        model = DecisionModel(
            encoder, self.cfg["head_layers"], len(self.cfg["act_costs"]) + 1
        )
        model.load_state_dict(load_file(str(root / "model.safetensors")), strict=True)
        model.to(device=self.device, dtype=torch.float32).eval()
        self.runner = EagerRunner(model, self.device)
        code_hash = hashlib.sha256()
        for name in ("reference.py", "runtime.py", "contracts.py"):
            code_hash.update(Path(__file__).with_name(name).read_bytes())
        identity = hashlib.sha256(
            (
                fingerprint(root)
                + code_hash.hexdigest()
                + torch.__version__
                + str(self.device)
                + "fp32"
            ).encode()
        ).hexdigest()
        self.info = {
            "runtime_code_sha256": code_hash.hexdigest(),
            "torch_version": torch.__version__,
            "model": root.name,
            "fingerprint": identity,
            "device": str(self.device),
            "dtype": "fp32",
            "runner": "eager",
            "max_len": self.cfg["max_len"],
            "head_max_len": self.cfg["head_max_len"],
            "calibrated": False,
            "batch_size": 1,
        }

    def infer(self, request: DecisionRequest):
        started = perf_counter()
        prepared = []
        for qid, question in request.questions.items():
            ids, markers = checked_sequence(self.tok, request.state, question, self.cfg)
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
        prep_ms = (perf_counter() - started) * 1000
        answers, tokens = {}, 0
        execution = perf_counter()
        for qid, question, item in prepared:
            batch = collate_items([[item]], self.tok.pad_token_id)
            logits, act = self.runner.execute(batch)
            k, qt = len(item["markers"]), item["qtype"]
            temperature = self.cfg.get("temperature_by_options", {}).get(
                temp_bucket(qt, k), self.cfg.get("temperature", [1, 1, 1])[qt]
            )
            z = logits[0, :k] / temperature
            probabilities = np.exp(z - z.max())
            probabilities /= probabilities.sum()
            keys = (
                list(question.criteria)
                if question.type == "choice"
                else [str(i) for i in range(k)]
            )
            answer = {
                "type": question.type,
                "probabilities": dict(zip(keys, map(float, probabilities))),
                "confidence": confidence_from_probs(probabilities, k),
                "act_probability": float(act[0, 0]),
            }
            if question.type == "choice":
                answer["choice"] = keys[int(probabilities.argmax())]
            elif question.type == "score":
                answer["score"] = float((np.arange(k) * probabilities).sum())
            else:
                answer["noul"] = float(probabilities[1])
            answers[qid] = answer
            tokens += len(item["ids"])
        return {
            "model": self.info,
            "answers": answers,
            "usage": {"input_tokens": tokens, "output_tokens": 0},
            "timings": {
                "preprocess_ms": prep_ms,
                "inference_ms": (perf_counter() - execution) * 1000,
                "total_ms": (perf_counter() - started) * 1000,
            },
        }
