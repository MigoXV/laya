> 以下保留 Torch 2.8.0／Transformers 5.18.0 的历史验收数据，不作为当前环境的测试结论。Torch 2.9.1、Transformers 4.57.6、vLLM 0.14.1 的重新对齐、16 路并发性能及兼容性记录见 [当前性能测试](../../benchmarks/cuda_precision_16/README.md)。

# Demo 01 验证记录

验证日期：2026-10-03。线上与 Demo 均不执行模型目录中的 Python 文件；对齐检查独立读取原始快照作为 FP32 Oracle。

## 当前配置

| 项目 | 实际值 |
| --- | --- |
| Python / Torch / Transformers | 3.10.18 / 2.8.0+cu128 / 5.18.0 |
| GPU | NVIDIA A100-PCIE-40GB，cuda:0 |
| 默认执行 | CUDA FP16 参数 + FP16 autocast，eager，batch 1 |
| 精度敏感部分 | 归一化、RoPE 位置频率、softmax 与概率保留必要的 FP32 精度 |
| 支持组合 | CPU FP32、CUDA FP32、CUDA FP16；CPU FP16 明确拒绝 |
| 模型目录 | /workspace/model-bin/MigoXV/laya-multilingual |
| 模型参数 | 321,908,995；FP16 参数占 643,817,990 字节 |
| Token 预算 | max_len=1024，head_max_len=256 |
| Node.js / pnpm / 浏览器 | 24.18.0 / 11.24.0 / Playwright Chromium |

FP16 参数占用是 FP32 的一半；该数字只计算参数，不是进程总内存或显存。权重与分词器文件未改动。运行时指纹包含代码、模型文件、设备与精度，因此精度切换及代码修改会改变指纹。

| 执行组合 | 模型及运行时指纹 |
| --- | --- |
| CPU FP32 | `976abd4df507fa0b38fb091d1e91b51c5420d505fef614bf8c3021d731aba33d` |
| CUDA FP32 | `7a2c1e5b2e408fa3d162fde11f851115d1de090c20746b61d87c49cd6896fb14` |
| CUDA FP16 | `e793d2c6b799149d0a1033b967d6d788092bb9f9ee4e3ff8d069a07ee2831505` |

## 数值对齐

每种执行组合分别运行 9 个用例（三种问题 × 三种文本长度）和 3 个超长输入拒绝场景。Oracle 独立加载原始快照的 `multilingual/`，始终使用同设备 FP32，关闭上游默认 BF16 autocast。分词器特殊 Token、完整后端结构、位置频率缓冲区与全部模型输入张量一致。

容差在最终执行前确定，FP32 延续原有严格门槛；FP16 决策 logits 最大绝对差为 0.01，动作 logits 使用 atol=0.01、rtol=0.005，概率／真假概率为 0.003，评分及置信度为 0.005，动作概率为 1e-6。

| 最大误差 | CPU FP32 | CUDA FP32 | CUDA FP16 |
| --- | --- | --- | --- |
| 决策 logits 绝对差 | 0 | 8.344650268554688e-7 | 0.008616209030151367 |
| 动作 logits 绝对差 | 0 | 0.000244140625 | 1.01953125 |
| 动作 logits 相对差 | 0 | 1.5346725490417157e-7 | 0.0007630084292031825 |
| 动作概率绝对差 | 0 | 0 | 0 |
| 与上游四位小数概率的差 | 4.065806865691246e-5 | 4.074747562407555e-5 | 0.0008043989181518718 |
| 评分绝对差 | 4.025502204896281e-5 | 4.082126617432902e-5 | 0.00021931931972507535 |
| 真假概率绝对差 | 2.6915836334184817e-5 | 2.5723743438677005e-5 | 0.0005104875564575506 |
| 置信度绝对差 | 3.5477352142321283e-5 | 3.4523677825915033e-5 | 0.0013014204025268428 |

所有选择结果一致，所有组合满足各自容差。FP16 与 FP32 不逐位相同；上游概率等数值仅保留四位小数，表中对应误差包含舍入差异。FP32 动作 logits 量级约为 1,000–1,800，采用 atol=1e-5、rtol=1e-6，输出概率／评分／置信度仍使用原有 0.000051 容差。

直接转换全部张量为 FP16 的初次检查未通过长文本阈值。定位后保留位置频率缓冲区的原始精度，并启用 FP16 autocast 处理敏感算子；矩阵与参数仍为 FP16，最终使用同一组 FP16 门槛通过。CPU FP16 未完成对应数值验收，因此不作为支持组合，不静默改用 FP32。

可复现对照命令：

```bash
LAYA_RUN_E2E=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 poetry run pytest tests/test_e2e.py -q -s
```

测试通过 `/v1/info` 核对服务设备、dtype、autocast、参数数量与实际参数字节数，并确认对齐 Runtime 与真实服务指纹一致。显式审计入口也可调用：

```python
from pathlib import Path
from laya.checks import check_reference
from laya.config import DEFAULT_MODEL_DIR

snapshot = Path("model-bin/convaiinnovations/laya")
for device, dtype in (("cpu", "fp32"), ("cuda:0", "fp32"), ("cuda:0", "fp16")):
    print(check_reference(DEFAULT_MODEL_DIR, snapshot, device, dtype))
```

## 已通过的检查

| 验证 | 结果 |
| --- | --- |
| `poetry check --lock` | 通过；保留既有 Poetry scripts 格式的弃用提示 |
| `poetry run ruff check src tests examples` | 通过 |
| `poetry run pytest -q` | 16 通过，6 个真实模型测试默认跳过 |
| 真实 HTTP 服务与参考对齐测试 | 6 通过；3 种组合各覆盖服务与数值对齐 |
| `pnpm --dir examples/demo01/web lint` | ESLint 与 TypeScript 通过 |
| `pnpm --dir examples/demo01/web test:e2e` | 默认 CUDA FP16，4 个浏览器测试通过 |

小型精度回归检查覆盖 FP16／FP32 动作头输入、有限输出、位置频率值及 dtype 不变、缓冲区非持久属性不变，以及 CPU FP16 明确拒绝。设备与精度的配置优先级（参数、进程环境变量、`.env`、内置默认）逐项验证通过。

真实 HTTP 检查覆盖三种问题、重复请求一致性、概率归一化、参数占用、模型指纹、422 非法／超长输入、413 超大正文，以及错误后的就绪状态；关闭后确认 Worker PID 消失。

浏览器主流程使用真实 Demo 代理、服务、Worker 与 CUDA FP16 模型。界面设备／精度文字、三种答案、概率条、百分比和 JSON 与实际响应对照；断连恢复场景仅模拟网络错误。检查覆盖 320/768/1280px、CSS 200% 缩放、长中文、键盘 Tab／Enter、减少动效，以及 axe-core WCAG A/AA，无违规。

## 模型整理与验证边界

推理只需根目录的 `config.json`、`model.safetensors` 和 `tokenizer.json`，另附中文 README、许可证与 SHA-256 清单。权重／分词器原始字节保留，编码器、决策与分词器配置合并并删除未使用的训练字段。此前配置合并对照旧项目 Runtime 的 CUDA 9 个用例，答案、决策及动作 logits 完全一致；本次精度切换按上述独立 FP32 Oracle 验证。

这份记录是正确性、交互与生命周期验证，不是模型准确率评测或吞吐基准；有限用例不能保证任意输入结果不变。未验证 Firefox、Safari 或高并发长时间运行。浏览器报告／截图位于 `web/playwright-report`、`web/test-results`，HTTP 日志在 pytest 临时目录，运行产物不进入 Git。
