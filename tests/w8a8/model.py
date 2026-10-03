"""仅实验使用的 Linear 替换和显式决策头注意力。"""

import torch
from torch import nn
from torch.nn import functional as F

from .kernels import int_mm, norm_quantize, quantize, quantize_weight, triton_gemm


class ExplicitAttention(nn.Module):
    """展开 MHA 的隐式 F.linear，以便 QKV 与输出投影也能量化。"""

    def __init__(self, original):
        super().__init__()
        self.heads = original.num_heads
        self.qkv = nn.Linear(original.embed_dim, 3 * original.embed_dim,
                             bias=original.in_proj_bias is not None,
                             device=original.in_proj_weight.device,
                             dtype=original.in_proj_weight.dtype)
        with torch.no_grad():
            self.qkv.weight.copy_(original.in_proj_weight)
            if self.qkv.bias is not None:
                self.qkv.bias.copy_(original.in_proj_bias)
        self.proj = original.out_proj

    def forward(self, x, mask):
        b, length, dim = x.shape
        q, k, v = self.qkv(x).view(b, length, 3, self.heads, dim // self.heads).permute(2, 0, 3, 1, 4)
        mask = ~mask[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        return self.proj(out.transpose(1, 2).contiguous().view(b, length, dim))


class ExplicitHead(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.attn = ExplicitAttention(original.self_attn)
        self.norm1, self.norm2 = original.norm1, original.norm2
        self.linear1, self.linear2 = original.linear1, original.linear2
        self.activation = original.activation
        assert original.norm_first

    def forward(self, x, src_key_padding_mask):
        x = x + self.attn(self.norm1(x), src_key_padding_mask)
        return x + self.linear2(self.activation(self.linear1(self.norm2(x))))


def explicit_heads(model):
    for i, layer in enumerate(model.head.layers):
        model.head.layers[i] = ExplicitHead(layer).eval()


def eligible(model):
    return [(name, module) for name, module in model.named_modules()
            if isinstance(module, nn.Linear) and module.in_features % 32 == 0
            and module.out_features >= 256 and not name.startswith("act_head.")]


@torch.inference_mode()
def calibrate(model, inputs):
    """只用两个测速输入估计静态量化尺度；不作为精度校准数据集。"""
    scales, hooks = {}, []
    for name, module in eligible(model):
        def hook(module, args, name=name):
            amax = args[0].float().abs().amax()
            scales[name] = torch.maximum(scales.get(name, amax), amax)
        hooks.append(module.register_forward_pre_hook(hook))
    for tensors in inputs:
        model(*tensors)
    for hook in hooks:
        hook.remove()
    return {name: (value.clamp_min(1.0e-8) / 127.0).clone() for name, value in scales.items()}


class Int8Linear(nn.Module):
    def __init__(self, source, scale, mode):
        super().__init__()
        weight, scales = quantize_weight(source.weight)
        self.register_buffer("qweight", weight)
        self.register_buffer("weight_scales", scales)
        self.register_buffer("scale", scale)
        self.register_buffer("bias", source.bias)
        self.mode = mode

    def forward(self, x):
        if isinstance(x, QuantizedActivation):
            out = triton_gemm(x.values, self.qweight, x.scales, self.weight_scales, self.bias, False)
            return out.view(*x.shape[:-1], self.qweight.shape[0])
        shape = (*x.shape[:-1], self.qweight.shape[0])
        a = x.contiguous().view(-1, x.shape[-1])
        if self.mode in ("triton-static-fused", "triton-static-ln"):
            out = triton_gemm(a, self.qweight, self.scale, self.weight_scales, self.bias, True)
        else:
            q, scales = quantize(a, self.scale, dynamic="dynamic" in self.mode)
            fn = int_mm if self.mode.startswith("cublas") else triton_gemm
            if fn is int_mm:
                out = fn(q, self.qweight, scales, self.weight_scales, self.bias)
            else:
                out = fn(q, self.qweight, scales, self.weight_scales, self.bias, False)
        return out.view(shape)


class QuantizedActivation:
    def __init__(self, values, scales, shape):
        self.values, self.scales, self.shape = values, scales, shape


class QuantizedNorm(nn.Module):
    def __init__(self, norm, linear):
        super().__init__()
        self.norm = norm
        self.register_buffer("scale", linear.scale)
        self.dynamic = "dynamic" in linear.mode

    def forward(self, x):
        q, scales = norm_quantize(x, self.norm, self.scale, self.dynamic)
        return QuantizedActivation(q, scales, x.shape)


def replace_norms(prepared):
    decision = prepared.model
    targets = []
    for layer in decision.encoder.layers:
        if isinstance(layer.attn_norm, nn.LayerNorm):
            targets.append((layer, "attn_norm", layer.attn.Wqkv))
        targets.append((layer, "mlp_norm", layer.mlp.Wi))
    for layer in decision.head.layers:
        targets.extend([(layer, "norm1", layer.attn.qkv), (layer, "norm2", layer.linear1)])
    targets.append((decision.scorer, "0", decision.scorer[1]))
    originals = []
    for owner, attribute, linear in targets:
        norm = getattr(owner, attribute)
        originals.append((owner, attribute, norm))
        setattr(owner, attribute, QuantizedNorm(norm, linear))
    return originals


def replace_linears(model, scales, mode):
    originals = []
    for name, module in eligible(model):
        owner_name, _, attribute = name.rpartition(".")
        owner = model.get_submodule(owner_name)
        originals.append((owner, attribute, module))
        setattr(owner, attribute, Int8Linear(module, scales[name], mode))
    if mode.endswith("-ln"):
        originals.extend(replace_norms(model))
    return originals


def restore_linears(originals):
    for owner, attribute, module in originals:
        setattr(owner, attribute, module)
