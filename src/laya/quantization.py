"""Laya W8A8 v1：独立权重格式、导出及纯模型模块；CUDA 内核按需导入。"""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file, save_file


class QuantizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: Literal["laya_w8a8"] = "laya_w8a8"
    version: Literal[1] = 1
    weight_dtype: Literal["int8"] = "int8"
    weight_scheme: Literal["symmetric_per_output_channel"] = "symmetric_per_output_channel"
    weight_layout: Literal["out_in"] = "out_in"
    activation_dtype: Literal["int8"] = "int8"
    activation_scheme: Literal["symmetric_dynamic_per_token"] = "symmetric_dynamic_per_token"
    compute_dtype: Literal["fp16"] = "fp16"
    quantized_modules: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_modules(self):
        if len(self.quantized_modules) != len(set(self.quantized_modules)):
            raise ValueError("duplicate_quantized_modules")
        return self


def read_quantization(config):
    return QuantizationConfig.model_validate(config["quantization"]) if "quantization" in config else None


def quantize_weight(weight):
    scale = weight.float().abs().amax(-1).clamp_min(1.0e-8) / 127.0
    quantized = (weight.float() / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
    return quantized.contiguous(), scale.contiguous()


class ExplicitAttention(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.heads = original.num_heads
        self.qkv = nn.Linear(original.embed_dim, 3 * original.embed_dim,
                             bias=original.in_proj_bias is not None,
                             device=original.in_proj_weight.device, dtype=original.in_proj_weight.dtype)
        with torch.no_grad():
            self.qkv.weight.copy_(original.in_proj_weight)
            if self.qkv.bias is not None:
                self.qkv.bias.copy_(original.in_proj_bias)
        self.proj = original.out_proj

    def forward(self, x, mask):
        b, length, dim = x.shape
        q, k, v = self.qkv(x).view(b, length, 3, self.heads, dim // self.heads).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=~mask[:, None, None, :], dropout_p=0.0)
        return self.proj(out.transpose(1, 2).contiguous().view(b, length, dim))


class ExplicitHead(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.attn = ExplicitAttention(original.self_attn)
        self.norm1, self.norm2 = original.norm1, original.norm2
        self.linear1, self.linear2 = original.linear1, original.linear2
        self.activation = original.activation
        if not original.norm_first:
            raise ValueError("w8a8_requires_pre_norm_head")

    def forward(self, x, src_key_padding_mask):
        x = x + self.attn(self.norm1(x), src_key_padding_mask)
        return x + self.linear2(self.activation(self.linear1(self.norm2(x))))


def explicit_heads(model):
    if model.head is not None:
        for i, layer in enumerate(model.head.layers):
            model.head.layers[i] = ExplicitHead(layer).eval()


def eligible(model):
    return [(name, module) for name, module in model.named_modules()
            if isinstance(module, nn.Linear) and module.in_features % 32 == 0
            and module.out_features >= 256 and not name.startswith("act_head.")]


class QuantizedActivation:
    def __init__(self, values, scales, shape):
        self.values, self.scales, self.shape = values, scales, shape


class Int8Linear(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.register_buffer("qweight", torch.empty_like(source.weight, dtype=torch.int8))
        self.register_buffer("weight_scales", torch.empty(source.out_features, dtype=torch.float32,
                                                         device=source.weight.device))
        self.register_buffer("bias", source.bias.detach().to(dtype=torch.float16).clone()
                             if source.bias is not None else None)

    def forward(self, x):
        from .int8_kernels import quantize, triton_gemm

        if isinstance(x, QuantizedActivation):
            values, scales, shape = x.values, x.scales, x.shape
        else:
            shape = x.shape
            values, scales = quantize(x, None, True)
        out = triton_gemm(values, self.qweight, scales, self.weight_scales, self.bias, False)
        return out.view(*shape[:-1], self.qweight.shape[0])


class QuantizedNorm(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.norm = norm

    def forward(self, x):
        from .int8_kernels import norm_quantize

        values, scales = norm_quantize(x, self.norm, None, True)
        return QuantizedActivation(values, scales, x.shape)


def install_linears(model, metadata):
    candidates = dict(eligible(model))
    if set(candidates) != set(metadata.quantized_modules):
        raise ValueError("w8a8_module_list_mismatch")
    for name in metadata.quantized_modules:
        owner, _, attribute = name.rpartition(".")
        setattr(model.get_submodule(owner), attribute, Int8Linear(candidates[name]))


def validate_weights(model, state):
    expected = model.state_dict()
    if state.keys() != expected.keys():
        raise ValueError(f"w8a8_tensor_keys_mismatch: missing={sorted(expected.keys() - state.keys())}, "
                         f"unexpected={sorted(state.keys() - expected.keys())}")
    for name, tensor in state.items():
        prototype = expected[name]
        dtype = torch.int8 if name.endswith(".qweight") else (
            torch.float32 if name.endswith(".weight_scales") or name == "temperature" else torch.float16)
        if tensor.shape != prototype.shape or tensor.dtype != dtype:
            raise ValueError(f"w8a8_tensor_shape_or_dtype_mismatch: {name}")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"w8a8_non_finite_tensor: {name}")
        if name.endswith(".weight_scales") and not (tensor > 0).all():
            raise ValueError(f"w8a8_invalid_weight_scales: {name}")
        if name.endswith(".qweight") and (tensor == -128).any():
            raise ValueError(f"w8a8_invalid_quantized_range: {name}")


def fuse_norms(model):
    targets = []
    for layer in model.encoder.layers:
        if isinstance(layer.attn_norm, nn.LayerNorm):
            targets.append((layer, "attn_norm"))
        targets.append((layer, "mlp_norm"))
    if model.head is not None:
        for layer in model.head.layers:
            targets.extend([(layer, "norm1"), (layer, "norm2")])
    targets.append((model.scorer, "0"))
    for owner, attribute in targets:
        setattr(owner, attribute, QuantizedNorm(getattr(owner, attribute)))


def file_record(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


@torch.inference_mode()
def export_w8a8(source: Path, destination: Path, device: str):
    from transformers import AutoModel
    from .reference import DecisionModel
    from .runtime import encoder_config_for_runtime, move_model

    source, destination = source.resolve(), destination.absolute()
    if destination.exists():
        raise ValueError(f"output_directory_exists: {destination}")
    raw = json.loads((source / "config.json").read_text())
    if raw.get("format_version") != 1 or "quantization" in raw:
        raise ValueError("export_requires_unquantized_laya_v1")
    target = torch.device(device)
    if target.type != "cuda" or target.index is None or not torch.cuda.is_available():
        raise ValueError("export_requires_explicit_cuda_device")
    torch.cuda.set_device(target)
    torch.set_num_threads(4)
    cfg = encoder_config_for_runtime(raw["encoder"])
    cfg.reference_compile = False
    encoder = AutoModel.from_config(cfg, dtype=torch.float16, attn_implementation="sdpa", trust_remote_code=False)
    head = raw["decision_head"]
    model = DecisionModel(encoder, head["layers"], head["num_actions"])
    state = load_file(str(source / "model.safetensors"))
    if any(t.dtype != (torch.float32 if name == "temperature" else torch.float16)
           for name, t in state.items()):
        raise ValueError("export_requires_fp16_source_weights")
    model.load_state_dict(state, strict=True)
    del state
    move_model(model, torch.device("cpu"), torch.float16).eval()
    explicit_heads(model)
    candidates = eligible(model)
    tensors = dict(model.state_dict())
    for name, module in candidates:
        qweight, scales = quantize_weight(module.weight.to(target))
        tensors.pop(f"{name}.weight")
        tensors[f"{name}.qweight"] = qweight.cpu()
        tensors[f"{name}.weight_scales"] = scales.cpu()
    raw["quantization"] = QuantizationConfig(quantized_modules=[name for name, _ in candidates]).model_dump()
    install_linears(model, read_quantization(raw))
    validate_weights(model, tensors)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        save_file({name: value.contiguous() for name, value in tensors.items()}, str(temporary / "model.safetensors"))
        (temporary / "config.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n")
        for name in ("tokenizer.json", "LICENSE"):
            shutil.copyfile(source / name, temporary / name)
        original = json.loads((source / "manifest.json").read_text())
        manifest = {"format_version": 1, "name": destination.name, "source": original["source"],
                    "license": original["license"], "encoder": original["encoder"],
                    "parent_files": {name: file_record(source / name) for name in
                                     ("config.json", "model.safetensors", "tokenizer.json")},
                    "export_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    "export_code_sha256": file_record(Path(__file__))["sha256"],
                    "quantized_modules": len(candidates),
                    "quantized_weight_count": sum(m.weight.numel() for _, m in candidates),
                    "files": {name: file_record(temporary / name) for name in
                              ("config.json", "model.safetensors", "tokenizer.json")},
                    "transformations": ["动态 W8A8；权重逐输出通道 INT8，激活逐 token 动态 INT8，浮点部分 FP16。",
                                        "决策头显式 QKV；97 个大矩阵量化，embedding 与动作头保留 FP16。"]}
        (temporary / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        (temporary / "README.md").write_text(
            f"# {destination.name}\n\n"
            "当前 Laya 多语言模型的动态 W8A8 仓库，供 Laya 项目加载；模型实现位于项目中。\n\n"
            "## 格式\n\n"
            "量化信息统一写入 config.json 的 quantization。laya_w8a8 v1 保存 INT8 qweight（out,in）"
            "与 FP32 weight_scales；其余参数为 FP16。尺度为每行 absmax.clamp_min(1e-8)/127，"
            "round ties-to-even 后限制至 [-127,127]。激活逐 token 动态量化，不需要固定激活尺度。"
            "决策头采用显式 QKV；归一化融合由项目在加载之后建立。\n\n"
            f"量化 {manifest['quantized_modules']} 个矩阵、{manifest['quantized_weight_count']:,} 个权重。"
            "embedding、SDPA、动作头及小末端层仍使用浮点计算。\n\n"
            "## 加载\n\n```bash\n"
            f"LAYA_REQUEST_TIMEOUT=300 poetry run laya serve --model-dir {destination} --runner cuda-graph-compile --max-batch-size 16\n"
            "```\n\n"
            "量化格式自动识别；仅支持 CUDA SM80+ 与 FP16 浮点计算，现有 A100 已验证。"
            "没有默认模型路径，必须显式选择模型仓库。\n\n"
            "## 精度与来源\n\n"
            "导出源为原 FP16 多语言模型，权重哈希及上游 revision 见 manifest.json。"
            "未经 QAT；冻结小测试集 659 个分类问题的实验 FP16/W8A8 准确率为 79.97%/79.82%，"
            "23 个答案发生变化。这是小测试集结果，不是业务精度保证。"
            "保存后模型的对齐结果见项目 tests/w8a8 的验证报告。\n\n"
            "许可沿用原仓库 Apache-2.0，见 LICENSE。\n")
        # 仅发布完整仓库；不覆盖任何已有目录。
        if destination.exists():
            raise ValueError(f"output_directory_exists: {destination}")
        (temporary / "model.safetensors").chmod(0o644)
        temporary.chmod(0o755)
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest
