"""Ampere INT8 Tensor Core GEMM：动态逐行量化与静态融合量化。"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def quantize_kernel(X, Q, S, STATIC, K: tl.constexpr, DYNAMIC: tl.constexpr,
                    BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(X + row * K + cols, cols < K, other=0).to(tl.float32)
    if DYNAMIC:
        scale = tl.maximum(tl.max(tl.abs(x)), 1.0e-8) / 127.0
    else:
        scale = tl.load(STATIC)
    q = libdevice.nearbyint(x / scale)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8)
    tl.store(Q + row * K + cols, q, cols < K)
    tl.store(S + row, scale)


@triton.jit
def norm_quantize_kernel(X, W, BIAS, Q, S, STATIC, K: tl.constexpr, EPS: tl.constexpr,
                         DYNAMIC: tl.constexpr, HAS_BIAS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(X + row * K + cols, cols < K, other=0).to(tl.float32)
    mean = tl.sum(x) / K
    centered = tl.where(cols < K, x - mean, 0.0)
    variance = tl.sum(centered * centered) / K
    value = centered * tl.rsqrt(variance + EPS)
    value *= tl.load(W + cols, cols < K, other=0).to(tl.float32)
    if HAS_BIAS:
        value += tl.load(BIAS + cols, cols < K, other=0).to(tl.float32)
    # 保留原 FP16 LayerNorm 输出的舍入点，再在寄存器里量化。
    value = value.to(tl.float16).to(tl.float32)
    value = tl.where(cols < K, value, 0.0)
    if DYNAMIC:
        scale = tl.maximum(tl.max(tl.abs(value)), 1.0e-8) / 127.0
    else:
        scale = tl.load(STATIC)
    q = libdevice.nearbyint(value / scale)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8)
    tl.store(Q + row * K + cols, q, cols < K)
    tl.store(S + row, scale)


@triton.jit
def dequantize_kernel(C, S, WS, BIAS, Y, M: tl.constexpr, N: tl.constexpr,
                      HAS_BIAS: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = idx < M * N
    row, col = idx // N, idx % N
    val = tl.load(C + idx, valid, other=0).to(tl.float32)
    val *= tl.load(S + row, valid, other=0) * tl.load(WS + col, valid, other=0)
    if HAS_BIAS:
        val += tl.load(BIAS + col, valid, other=0)
    tl.store(Y + idx, val, valid)


@triton.jit
def gemm_kernel(A, W, S, WS, BIAS, Y, M: tl.constexpr, N: tl.constexpr,
                K: tl.constexpr, FUSED_QUANT: tl.constexpr, HAS_BIAS: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.int32)
    if FUSED_QUANT:
        a_scale = tl.load(S)
    for start in range(tl.cdiv(K, BK)):
        kk = start * BK + ks
        a = tl.load(A + rows[:, None] * K + kk[None, :],
                    (rows[:, None] < M) & (kk[None, :] < K), other=0)
        if FUSED_QUANT:
            a = libdevice.nearbyint(a.to(tl.float32) / a_scale)
            a = tl.minimum(tl.maximum(a, -127.0), 127.0).to(tl.int8)
        # W 存储为 [N,K]，以转置视图参与实际 s8*s8 -> s32 MMA。
        w = tl.load(W + cols[None, :] * K + kk[:, None],
                    (cols[None, :] < N) & (kk[:, None] < K), other=0)
        acc = tl.dot(a, w, acc)
    if FUSED_QUANT:
        scales = a_scale
    else:
        scales = tl.load(S + rows, rows < M, other=0)[:, None]
    out = acc.to(tl.float32) * scales * tl.load(WS + cols, cols < N, other=0)[None, :]
    if HAS_BIAS:
        out += tl.load(BIAS + cols, cols < N, other=0)[None, :]
    tl.store(Y + rows[:, None] * N + cols[None, :], out,
             (rows[:, None] < M) & (cols[None, :] < N))


TILES = [(32, 64, 64, 4, 3), (32, 128, 64, 4, 3),
         (64, 128, 64, 4, 3), (64, 128, 128, 4, 3), (128, 128, 64, 8, 3)]
TUNING = {}
EVIDENCE = {}


def quantize(x, static_scale, dynamic):
    x = x.contiguous().view(-1, x.shape[-1])
    q = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
    quantize_kernel[(x.shape[0],)](x, q, scales, static_scale, x.shape[1], dynamic,
                                  triton.next_power_of_2(x.shape[1]))
    return q, scales


def norm_quantize(x, norm, static_scale, dynamic):
    x = x.contiguous().view(-1, x.shape[-1])
    q = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
    norm_quantize_kernel[(x.shape[0],)](
        x, norm.weight, norm.bias, q, scales, static_scale, x.shape[1], norm.eps,
        dynamic, norm.bias is not None, triton.next_power_of_2(x.shape[1]))
    return q, scales


def triton_gemm(a, w, scales, weight_scales, bias, fused, tile=None):
    m, k = a.shape
    n = w.shape[0]
    key = (m, n, k, fused)
    tile = tile or TUNING.get(key, {}).get("tile", TILES[0])
    bm, bn, bk, warps, stages = tile
    out = torch.empty((m, n), device=a.device, dtype=torch.float16)
    kernel = gemm_kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
        a, w, scales, weight_scales, bias, out, m, n, k, fused, bias is not None,
        bm, bn, bk, num_warps=warps, num_stages=stages,
    )
    if not EVIDENCE and not torch.compiler.is_compiling():
        instructions = [line.strip() for line in kernel.asm["ptx"].splitlines()
                        if "mma.sync" in line and "s8.s8" in line]
        EVIDENCE.update({"int8_tensor_core_instructions": sorted(set(instructions)),
                         "kernel_name": kernel.name})
    return out


def int_mm(a, w, scales, weight_scales, bias):
    rows = a.shape[0]
    # Torch 2.8 的 cuBLAS 路径要求 M>16；选项打分层常只有 2 行。
    if rows <= 16:
        a = torch.nn.functional.pad(a, (0, 0, 0, 32 - rows))
    acc = torch._int_mm(a, w.t())[:rows]
    out = torch.empty_like(acc, dtype=torch.float16)
    dequantize_kernel[(triton.cdiv(out.numel(), 256),)](
        acc, scales, weight_scales, bias, out, *out.shape, bias is not None, 256)
    return out


def quantize_weight(weight):
    scale = weight.float().abs().amax(-1).clamp_min(1.0e-8) / 127.0
    quantized = (weight.float() / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
    return quantized.contiguous(), scale.contiguous()
