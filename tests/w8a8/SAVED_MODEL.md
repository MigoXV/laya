# W8A8 仓库导出与主程序加载验证

2026-10-03，Torch 2.8.0+cu128、A100-PCIE-40GB，全部推理与导出使用 GPU 0。原 FP16 服务和 Demo 保持运行；验证使用临时端口，结束后测试 Worker 全部退出。本轮没有做性能基准。

## 交付

新模型目录为 `/workspace/model-bin/MigoXV/laya-multilingual-w8a8`，保留六个独立文件，无 Python 实现代码。权重文件 **519,401,730 字节（约 519 MB / 495 MiB）**；加上约 34 MB 的分词器等文件，完整目录约 **554 MB / 528 MiB**。原 FP16 权重为 643,835,514 字节；embedding 仍为 FP16，因此没有减半。

`config.json` 中的 `quantization` 自动决定加载格式，`model.safetensors` 真正保存 INT8 权重和 FP32 通道尺度。加载时不需要原始 FP16 仓库，也不重新量化。权重哈希为 `99d7b1ff310adf077164f018f068e996b3ce6e02c11158c1aec78720e5b97206`，来源和文件哈希见模型 `manifest.json`。

项目取消所有内置默认模型路径。显式选择目录即可：

```bash
LAYA_REQUEST_TIMEOUT=300 poetry run laya serve \
  --model-dir /workspace/model-bin/MigoXV/laya-multilingual-w8a8 \
  --runner cuda-graph-compile --max-batch-size 16
```

首次编译单独计入冷捕获时间；未知 profile 仍可能编译，不把它算作稳态推理延迟。普通 FP16 模型的执行路径继续保留。

## 保存对齐

以先前已冻结的 W8A8 实验为参考，而不是要求量化输出与 FP16 一致：

| 检查 | 结果 |
| --- | --- |
| INT8 权重与对应 FP32 尺度 | 97 个矩阵逐项、逐位一致 |
| 其余权重与温度缓冲区 | 73 个张量逐位一致 |
| 冻结中英文标准答案小集 | 619 条请求、684 道题全部完成 |
| 同一 B16/L256/K6、compile + Graph 下的原始 logits | 最大差异 0 |
| 分类答案变化 | 0/659 |
| 概率最大绝对差异 | 2.5431e-7 |
| 连续评分最大绝对差异 | 7.2821e-8 |
| 保存后的分类正确数 | 526/659（79.82%），与先前 W8A8 相同 |

概率/评分的微小差异来自评测参考用 NumPy FP64 后处理，而主程序采用 FP32 logits 后处理。实际 INT8 Tensor Core PTX 也已记录。小测试集上的量化敏感性及 FP16 对照仍见 [accuracy/README.md](accuracy/README.md)。

## 主程序与服务验证

- `eager`、`cuda-graph`、`cuda-graph-compile`：中英文、27/512 token、choice/score/noul 的真实 HTTP 答案均与同后端离线加载结果完全一致，重复请求结果一致。
- 三个后端均测试 16 路并发，混合短长请求并插入一个非法输入；合法请求成功、非法请求返回 422，随后服务仍就绪。
- 本次并发输入相对各自 B1 对照的概率漂移为 0、choice 差异为 0。这只适用于本次输入；不同 runner/编译/shape 的浮点舍入与动态量化边界可能改变结果，不声明所有后端逐位一致。
- 两个 Graph 后端各观察到 4 次 LRU 淘汰，缓存保持 4 个 profile，覆盖 B1 与 B8、尾批占位和不同长度/选项数；编译 profile 包含 batch 维度。
- 三个测试服务正常关闭，Worker PID 全部消失；原服务与 Demo 仍就绪。
- 新格式拒绝 CPU/BF16/FP32、未知版本、错误模块/张量及非法尺度。模型路径缺失报错，参数与显式环境变量均可选路。

## 复现与记录

```bash
# 先用原 FP16 仓库重新生成本地实验参考，原逐题输出不再随仓库提交。
CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_COMPILE_THREADS=4 poetry run python -m tests.w8a8.accuracy.evaluate \
  --model-dir /workspace/model-bin/MigoXV/laya-multilingual
CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_COMPILE_THREADS=4 poetry run python -m tests.w8a8.check_saved \
  --model-dir /workspace/model-bin/MigoXV/laya-multilingual-w8a8 \
  --source-dir /workspace/model-bin/MigoXV/laya-multilingual
```

[check_saved.py](check_saved.py) 保留复现入口，运行时将权重校验、逐请求答案、缓存/批次指标及 INT8 指令证据写入本地 `saved-validation.json`。原逐题参考和验证输出已清理，由 `.gitignore` 忽略；原始产物已从 Git 历史移除。真实三路径验证执行于代码版本 `28214f2`；随后补充 eager W8A8 的代码指纹依赖、导出文件权限、GPU 0 验证环境约束及文档，未改变模型数学计算或权重。复现时用上述精度评测生成同输入、同实验实现的参考；报告中的历史对齐数值属于当时的执行结果。

针对性 Python 检查 `tests/test_quantization.py`、`tests/test_precision.py`、`tests/test_cuda_profiles.py`、`tests/test_runtime_batching.py`、`tests/test_sequence.py` 合计 **35 passed**；补充 FP32 架构 bias 加载检查后量化测试单独 **12 passed**。`ruff check src/laya scripts tests` 和 `git diff --check` 通过。Demo 的 TypeScript 检查及 Playwright 测试配置加载通过，仅列出四个测试，没有运行完整浏览器测试或项目完整测试套件。

上文的代码版本号为历史清理前的编号，仅记录当时的验证环境；Git 历史重写后提交编号已变化。
