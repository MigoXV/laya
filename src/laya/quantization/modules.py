"""W8A8 模型模块；CUDA 内核按需导入。"""
import torch
from torch import nn
from torch.nn import functional as F

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
        from laya.quantization.int8_kernels import quantize, triton_gemm

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
        from laya.quantization.int8_kernels import norm_quantize

        values, scales = norm_quantize(x, self.norm, None, True)
        return QuantizedActivation(values, scales, x.shape)
