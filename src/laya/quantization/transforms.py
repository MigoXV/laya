"""量化权重与模型替换。"""
import torch
from torch import nn
from .modules import ExplicitHead, Int8Linear, QuantizedNorm

def quantize_weight(weight):
    scale = weight.float().abs().amax(-1).clamp_min(1.0e-8) / 127.0
    quantized = (weight.float() / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
    return quantized.contiguous(), scale.contiguous()


def explicit_heads(model):
    if model.head is not None:
        for i, layer in enumerate(model.head.layers):
            model.head.layers[i] = ExplicitHead(layer).eval()


def eligible(model):
    return [(name, module) for name, module in model.named_modules()
            if isinstance(module, nn.Linear) and module.in_features % 32 == 0
            and module.out_features >= 256 and not name.startswith("act_head.")]


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
