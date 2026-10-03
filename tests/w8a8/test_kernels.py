"""显式开启的 GPU 检查：验证真实 INT8 GEMM 及尾部 mask。"""

import os

import pytest


pytestmark = [pytest.mark.e2e, pytest.mark.skipif(
    os.environ.get("RUN_W8A8") != "1", reason="设置 RUN_W8A8=1 才运行 INT8 CUDA 实验")]


@pytest.mark.parametrize("shape", [(2, 768, 768), (27, 768, 768),
                                  (32, 768, 2304), (17, 1152, 768)])
def test_int8_dot(shape):
    import torch
    from .kernels import EVIDENCE, int_mm, quantize, quantize_weight, triton_gemm

    torch.manual_seed(42)
    m, k, n = shape
    x = torch.randn(m, k, device="cuda", dtype=torch.float16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.03
    bias = torch.randn(n, device="cuda", dtype=torch.float16)
    w, ws = quantize_weight(weight)
    static = x.float().abs().max() / 127
    for dynamic in (True, False):
        q, scales = quantize(x, static, dynamic)
        expected_q = (x.float() / scales[:, None]).round().clamp(-127, 127).to(torch.int8)
        torch.testing.assert_close(q, expected_q, rtol=0, atol=0)
        # 整数值范围很小，FP32 matmul 可精确表示此 INT32 累加结果。
        reference = ((q.float() @ w.float().t()) * scales[:, None] * ws[None, :] + bias).half()
        for result in (int_mm(q, w, scales, ws, bias),
                       triton_gemm(q, w, scales, ws, bias, False)):
            torch.testing.assert_close(result, reference, rtol=0.002, atol=0.004)
        if not dynamic:
            fused = triton_gemm(x, w, static, ws, bias, True)
            torch.testing.assert_close(fused, reference, rtol=0.002, atol=0.004)
    assert EVIDENCE["int8_tensor_core_instructions"]


@pytest.mark.parametrize("bias", [False, True])
def test_fused_norm_quantize(bias):
    import torch
    from .kernels import norm_quantize, quantize

    torch.manual_seed(7)
    x = torch.randn(27, 768, device="cuda", dtype=torch.float16)
    norm = torch.nn.LayerNorm(768, bias=bias, device="cuda", dtype=torch.float16).eval()
    with torch.inference_mode():
        y = norm(x)
    static = y.float().abs().max() / 127
    for dynamic in (False, True):
        reference, ref_scales = quantize(y, static, dynamic)
        fused, scales = norm_quantize(x, norm, static, dynamic)
        torch.testing.assert_close(scales, ref_scales, rtol=0.001, atol=1.0e-7)
        torch.testing.assert_close(fused, reference, rtol=0, atol=1)
