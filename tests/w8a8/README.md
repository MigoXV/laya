# A100 W8A8 独立性能实验

仅在本目录实验，不修改主程序、默认精度和磁盘上的 FP16 权重。不检查模型精度是否可上线；记录输出漂移与有限值，后续再做校准或 QAT。

比较四条实际 INT8 路径：cuBLAS 动态逐 token 量化、cuBLAS 静态量化、Triton 动态逐 token 量化、Triton 静态融合量化。权重按输出通道对称量化。Triton 使用 `s8 × s8 -> s32` Tensor Core 指令，结果转回 FP16。静态融合路径在 GEMM 内量化 FP16 激活，避免额外的激活量化和反量化 kernel。

完整模型另测动态/静态 `LayerNorm + 激活量化` 融合：48 处归一化直接输出 INT8，避免先写 FP16 再读取量化。Torch 2.8 的 `_int_mm` 要求行数大于 16，小于等于 16 的选项打分输入会补到 32 行；padding 耗时计入。

进一步加入 `native` 对照：保留 FP16 权重，关闭 autocast，减少大张量的 FP32 中间结果与转换；LayerNorm 内部的归约仍使用 FP32。batch 16 还比较双方的完整模型 `torch.compile`，不启用主程序的严格舍入模拟，因为本次用户明确不要求精度对齐。末端概率等仍沿用原模型显式的 FP32 运算。

完整模型覆盖 22 层编码器、2 层决策 Transformer、选项打分和动作头。适合 INT8 的投影/FFN 换成 W8A8；embedding、LayerNorm、RoPE、SDPA、非线性和很小的末端输出层保留 FP16。展开决策头 MHA，量化其 QKV 和输出投影；同样展开的 FP16 是批处理对照。保留原服务的 8 流独立序列 CUDA Graph 作为另一条基线。

使用现有 `scripts/inputs/short.json`、`long.json`，分别为 27/512 token，测试 batch 1/16。**batch 16 是设备整批前向，不是 16 路 HTTP 端到端请求。** 全部路径包含完整前向和动作 softmax，激活量化计时，权重预量化和 Graph 捕获不计入稳态。输入已经在设备上，不包含分词、排队、网络和 H2D/D2H。

在 GPU 0 空闲时运行，脚本检查独占情况，不主动停止其他进程：

```bash
# 显式选择原 FP16 仓库，供下列实验入口使用。
export LAYA_MODEL_DIR=/workspace/model-bin/MigoXV/laya-multilingual
RUN_W8A8=1 poetry run pytest -q tests/w8a8/test_kernels.py
# 重新搜索矩阵 tile 并测量完整模型：
CUDA_VISIBLE_DEVICES=0 poetry run python -m tests.w8a8.benchmark
# 使用保留的精简 tile 配置，单独测完整模型：
CUDA_VISIBLE_DEVICES=0 poetry run python -m tests.w8a8.benchmark --no-matrices --tuning-file tests/w8a8/tuning.json
# 最终对照包含 batch 16 的 FP16/W8A8 编译：
CUDA_VISIBLE_DEVICES=0 poetry run python -m tests.w8a8.benchmark --no-matrices --tuning-file tests/w8a8/tuning.json --compiled --samples 30 --output tests/w8a8/results-native.json
```

默认每种 shape 预热后交错测量 3 轮，每轮 30 次；矩阵将 10 次运算捕获为一个 Graph，结果取每次平均，降低主机提交开销。矩阵的百分位是这类平均值，不代表请求尾延迟。完整模型每次 Graph replay 计时一次。原始 CUDA Event 时间、环境、进程盘点、量化覆盖、PTX 指令证据和 tile 选择写入 `results.json`。

INT8 内核检查包含不整齐的行数、bias、逐行量化和静态融合量化，与显式整数点积及反量化参考比较。核验 Tensor Core 指令，不以模型任务精度作为本次实验的通过条件。

长输入 batch 16 额外保存 FP16 和最快 INT8 路径的 CUPTI kernel 聚合耗时，以定位 GEMM 之外的瓶颈。计时结束之后才采集 profile，不把 profiler 开销混入基准。

## 实测结果

2026-10-03，A100-PCIE-40GB（SM80）、Torch `2.8.0+cu128`、Triton `3.4.0`、CUDA `12.8`、驱动 `595.71.05`。测速停止本项目在 GPU 0 的服务；GPU 1 的其他任务不动。允许本实验 Inductor 编译子进程创建 context，编译/预热完成后才开始计时。三轮交错测量，每轮 30 次。

下面比较本轮更快的 FP16 对照与 W8A8：双方同样展开决策头注意力、关闭 autocast、整批执行并使用 CUDA Graph；batch 16 双方还使用 compile，batch 1 没有做编译测试。W8A8 使用动态逐 token 量化、逐输出通道权重量化，以及 LayerNorm 量化融合。

| 输入 | batch | FP16 平均延迟 | W8A8 平均延迟 | FP16 吞吐 | W8A8 吞吐 | W8A8 吞吐倍率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 27 token | 1 | 2.264 ms | 2.341 ms | 441.8/s | 427.1/s | 0.967× |
| 27 token | 16 | 1.803 ms | 1.811 ms | 8,876.6/s | 8,832.8/s | 0.995× |
| 512 token | 1 | 3.981 ms | 3.730 ms | 251.2/s | 268.1/s | 1.067× |
| 512 token | 16 | 23.324 ms | 20.523 ms | 686.0/s | 779.6/s | **1.136×** |

长输入 batch 16 的 FP16/W8A8 P95 分别为 **23.534/20.659 ms**，三轮平均延迟分别为 `23.239/23.281/23.452` 与 `20.462/20.457/20.650 ms`。W8A8 平均延迟降低约 **12.0%**，吞吐提升约 **13.6%**。短输入 batch 16 基本持平，短输入 batch 1 略慢；本轮没有证明短输入的 INT8 独立收益。

同轮现有生产多流 Graph，短/长 batch 16 分别为 `9.508/37.027 ms`，W8A8 最快路径分别为 `1.811/20.523 ms`。这些差距还包含整批执行、native FP16 和编译，**不能全部算作 INT8 收益**。生产基线有明显轮间抖动，长 batch 16 三轮为 `35.129/35.855/40.096 ms`；上表使用更快且更稳定的 FP16 对照给出结论。

单纯换 `_int_mm` 未能加速完整模型。静态量化融合到 GEMM 中也不是本轮最快方案：不同 N tile 会重复读取 FP16 激活并执行量化，带来额外工作；提前一次动态量化，加上归一化融合和编译，整体更快。

97 个 Linear 的 **125,042,688 个权重**实际转成 INT8，覆盖编码器及决策 Transformer 的 QKV、输出投影和 FFN，以及 scorer 的大投影。大 embedding、SDPA 的 QK/AV、浮点归一化/位置运算和小末端层仍采用浮点计算。`int8_evidence` 保存 `mma.sync.aligned.m16n8k32...s32.s8.s8.s32` PTX 指令，保证不是反量化权重之后再做 FP16 GEMM。

矩阵层面，计入激活量化后的部分大投影达到约 **1.6×** FP16；纯 cuBLAS INT8 GEMM 最高约 **275 TOPS**。Triton tile 测试的 INT8 GEMM 加反量化最高约 **333 TOPS**（`M=8192,K=3072,N=768`，约 `115.968 μs`，不包含激活量化），该数字不是完整模型吞吐，也不能用于声称整个模型已占满 INT8 算力。

长 batch 16 的 profile 中，最快 W8A8 编译路径共 339 个 kernel，INT8 GEMM 约 `6.785 ms`，仍为浮点的 SDPA 约 `6.256 ms`，融合归一化量化约 `0.999 ms`、其他激活量化约 `0.972 ms`。这些是另一次带 profiler 的 kernel 聚合时间，与无 profiler 的 Graph 延迟不能直接相加对齐。后续提速需要继续减少注意力与非 GEMM 开销，仅继续换更快的 Linear 有收益上限。

首次捕获/预量化/编译不计入稳态：短 batch 16 的 FP16/W8A8 约 `14.87/31.54 s`，长 batch 16 约 `13.99/32.20 s`。这不包含最初的模型加载，且编译缓存会影响冷启动时间。

按用户要求没有精度门槛。全部记录的输出有限，但未校准的 W8A8 在短输入上会改变选项 argmax；这里只证明速度可行性，任务精度仍需后续校准/QAT 实验。

后续已增加[带标准答案的中英文精度评测](accuracy/README.md)：619 条请求、684 个问题，659 个分类问题的原始 FP16/W8A8 准确率为 **79.97% / 79.82%**，净下降 **0.15 个百分点**；同时 **23 个答案（3.49%）发生变化**。这是冻结小测试集上的结果，不是业务精度保证，分项和概率偏移见该报告。

## 验证与复现

- `RUN_W8A8=1 poetry run pytest -q tests/w8a8/test_kernels.py`：**6 passed**，没有运行项目完整测试套件。
- `poetry run ruff check tests/w8a8`：通过。
- [benchmark.py](benchmark.py)：矩阵与完整模型的复现入口。
- [tuning.json](tuning.json)：只保留 40 组 shape / tile 配置，不含候选计时与原始样本，供测速和精度评测加载。

逐次计时、探索结果和服务快照 JSON 已清理；上面的命令可重新生成本地结果，产物由 `.gitignore` 忽略。性能结论与限制保留在本报告。

主程序、默认 FP16 权重和依赖均未修改。服务恢复为 `0.0.0.0:10002 / fp16 / cuda-graph / max_batch=16 / streams=8`，Demo 保持 `10013`。

参考：[Triton 官方 `tl.dot` 类型与累加规则](https://triton-lang.org/main/python-api/generated/triton.language.dot.html)。

后续已完成[量化仓库保存与主程序自动加载](SAVED_MODEL.md)，97 个矩阵与尺度逐位一致，684 道题的保存对齐通过；三个 runner 的真实 HTTP 和 16 路并发验证通过。
