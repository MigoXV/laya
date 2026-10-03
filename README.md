# Laya 本地决策服务

从本地权重执行 `choice`（选择）、`score`（评分）和 `noul`（真假概率）决策，提供 CLI 与 HTTP 服务。模型在独立 Worker 中运行，服务负责有界队列、超时、取消、健康检查和指标。

本项目的发行包、Python 模块和命令均叫 `laya`，仅用于本地开发与运行。无需安装 PyPI 上的同名包。

## 安装

使用 Python 3.10、Poetry；可视化 Demo 另外需要 Node.js 24 和 pnpm 11.24.0。项目默认使用 `.venv`。

```bash
poetry env use python3.10
poetry install
```

Torch 由环境单独管理，不写入 Poetry 依赖和锁文件。请在同一虚拟环境手动安装 **2.9.1**，按设备选择一条命令：

```bash
# CPU
poetry run python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu

# NVIDIA CUDA 12.8
poetry run python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128

poetry run python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'
```

安装源见 [PyTorch 2.9.1 官方说明](https://pytorch.org/get-started/previous-versions/#v291)。当前开发环境使用 `2.9.1+cu128`，并手动安装 vLLM `0.14.1`；vLLM 要求 Transformers `<5`，项目约束已同步为 `>=4.57.6,<5`。使用 `poetry install` 安装项目依赖，避免使用会清理手动依赖的 `poetry sync`。

## 启动服务

默认模型目录为 `/workspace/model-bin/MigoXV/laya-multilingual`，只使用多语言版本。推理只需根目录的 `config.json`、`model.safetensors` 和 `tokenizer.json`（合计约 678 MB），另附中文说明、Apache-2.0 许可证和来源／SHA-256 清单。权重和分词器文件保持原始字节；全部推理配置合并到一份 `config.json`，移除不参与推理的训练字段。本项目不自动下载模型，也不执行模型目录里的 Python 文件。

```text
laya-multilingual/
├── model.safetensors
├── config.json
├── tokenizer.json
├── README.md
├── LICENSE
└── manifest.json
```

`config.json` 使用 `format_version=1`，包含 `encoder`（编码器完整结构）、`decision_head`（层数与动作数）、`input_limits`（Token 限制）、`calibration`（温度校准）和 `tokenizer`（特殊 Token 与分词器设置）。设备、端口、队列和超时仍由项目的服务配置管理。

```bash
poetry run laya serve

# 默认第一张 CUDA 卡，FP16 / eager
poetry run laya serve --device cuda:0 --dtype fp16

# CPU 显式使用 FP32 数值基线
poetry run laya serve --device cpu --dtype fp32

# 完整模型 CUDA Graph，跨请求合并调度，8 条 CUDA stream 执行独立序列
poetry run laya serve --runner cuda-graph --dtype fp16 --max-batch-size 16 --graph-streams 8

# 完整 CUDA Graph + 静态形状 torch.compile（初次编译耗时另计）
poetry run laya serve --runner cuda-graph-compile --dtype fp16 --max-batch-size 16 --graph-streams 8

# vLLM 原生 ModernBERT + 完整决策头／动作头
poetry run laya serve --runner vllm --dtype fp16

# vLLM 关闭编译和 CUDA Graph，使用相同模型做 eager 对照
poetry run laya serve --runner vllm-eager --dtype fp16

# 查看实际解析到的目录、配置和权重大小
poetry run laya inspect
```

默认使用 `cuda:0 / fp16 / eager`，监听 `0.0.0.0:10002`，本机访问 `http://127.0.0.1:10002`，其他设备使用服务器 IP。全部模型参数使用 FP16，前向启用 FP16 autocast；矩阵运算使用 FP16，RoPE 位置频率、softmax、校准与概率保留必要的 FP32 精度。FP16 推理采用现有 eager FP16 作为行为基线，FP32 用于独立观察跨精度漂移。FP16 权重转成 FP32 不会恢复已丢失的权重精度，但会改变运算舍入；因此跨精度差异不直接代表实现错误。321,908,995 个参数在 FP16 下占 643,817,990 字节，是 FP32 参数占用的一半；这不等于进程总内存或显存。

模块入口等价于 `poetry run python -m laya.commands.app`。模型路径优先级为 `--model-dir` 参数、进程环境变量 `LAYA_MODEL_DIR`、`.env` 中的 `LAYA_MODEL_DIR`、内置默认路径；设备与精度同样支持 `--device`／`LAYA_DEVICE` 和 `--dtype`／`LAYA_DTYPE`，`serve` 和 `infer` 共用这一规则。Runtime 显式传入所选 dtype 构造编码器，并统一转换模型；执行精度覆盖上游编码器配置里的 dtype 元信息。`/v1/info` 返回执行精度、autocast 状态、参数数量及参数字节数。Transformers 4 通过兼容转换读取权重配置中的全局／局部 RoPE theta，模型目录仍保留统一配置。CUDA 不可用时明确报错，需要 CPU 时显式指定 `--device cpu --dtype fp32`；FP16 当前仅支持 CUDA，CPU FP16 明确拒绝，不会静默切换设备或精度。监听地址和端口通过 `LAYA_HOST`、`LAYA_PORT` 等进程环境变量配置；其他配置参见 `.env.example`。

| 接口 | 用途 |
| --- | --- |
| `GET /health/live` | API 存活 |
| `GET /health/ready` | 模型加载和预热完成 |
| `GET /v1/info` | 模型指纹、Torch、设备、执行策略 |
| `POST /v1/decisions` | 提交决策请求 |
| `GET /metrics` | 队列、计数和延迟统计 |

`serve`、`infer` 支持 `--runner`／`LAYA_RUNNER`，可选 `eager`、`cuda-graph`、`cuda-graph-compile`、`vllm`、`vllm-eager`。后两者要求已安装 Torch 2.9.1、vLLM 0.14.1；使用 `cuda:0`，可通过 `CUDA_VISIBLE_DEVICES` 选择物理卡。安装当前包时会注册 vLLM 插件（已有环境可运行 `poetry install --only-root`，不会改动手工安装的 Torch）。完整权重仍从统一模型目录加载，不需另建 encoder 仓库。vLLM 的 pooling 接口只运输完整决策／动作 logits，公共 Runtime 负责相同的分词、校准和领域输出，不对 logits 进行 embedding 归一化。

适配器保留全局／局部 RoPE theta=160000，在编码器和完整决策头使用相同的 autocast 策略。注意力采用与 eager 基线相同的 PyTorch SDPA 与闭区间局部 mask；vLLM 负责原生 ModernBERT 权重映射、执行调度和编码器编译。编译模式开启 `emulate_precision_casts`，保留 eager 的 FP16 中间舍入。`/v1/info` 报告实际加载的参数数量、字节数、参数 dtype、注意力后端、线程数、编译模式与 CUDA Graph 捕获尺寸。vLLM 的 GPU scheduler 仍限制为单序列；即使一次向它提交多个问题，也不会宣称启用了 GPU 张量合批。

`cuda-graph` 捕获编码器、完整决策／动作头和动作 softmax，将多个独立序列分配到最多 8 条 CUDA stream，再用一次完整 Graph replay 执行整个调度批次。每条序列保持 eager batch=1 的计算形状；序列长度与选项数按精确值分组，不补齐它们，只把批次容量向上取到 1/2/4/8/16，额外行使用有效输入副本且丢弃其输出。这样既减少 CPU kernel 派发开销，也保留该模型敏感的 FP16 舍入行为。RoPE 缓存由原 HF 实现一次计算，参数仍是原始 FP16 权重。

`cuda-graph-compile` 额外编译整个模型，使用静态序列／选项形状与 `emulate_precision_casts`。`LAYA_COMPILE_CACHE_SIZE` 限制编译形状数量；超出后该形状采用未编译的完整 CUDA Graph，原因通过 `/metrics` 的 `compile_fallback_shapes` 显式报告。Graph 的 LRU 缓存由 `LAYA_GRAPH_CACHE_SIZE` 限制，首次出现或被淘汰的形状会先预热再捕获；因此冷捕获／初次编译延迟与稳态延迟分开记录。静态缓冲区由唯一 Worker 所有，每次正确复制输入、读回结果后再复用。

已知业务形状可用 `LAYA_GRAPH_PREWARM_PROFILES='[[1,27,2],[16,27,2]]'` 在服务就绪前预热；每项为 `[批次容量, 精确序列长度, 选项数]`，批次容量只支持 1/2/4/8/16，数量不能超过 Graph 缓存容量，也受批次和 token 上限约束。输入仍会完整复制并重新推理。预处理复用一次 Rust 分词结果，避免参考构建器重复分词；默认 `TOKENIZERS_PARALLELISM=false`，可通过环境变量显式覆盖。

服务默认仍为 eager FP16、单请求，用作稳定的行为基线。显式使用 `--max-batch-size 16`／`LAYA_MAX_BATCH_SIZE=16` 开启跨请求调度，最多等待 `--batch-wait-ms`／`LAYA_BATCH_WAIT_MS`（默认 2 ms），总设备 token 容量由 `LAYA_MAX_BATCH_TOKENS` 控制，包含额外占位行。`--graph-streams`／`LAYA_GRAPH_STREAMS` 设置 Graph 的 CUDA stream 数。每个请求保持独立的结果、校验错误、取消与 deadline，Worker 协议用版本和 ID 关联响应；API 仍不执行模型计算。`/metrics` 返回请求／模型调度批次分布、token 填充率、Graph 命中／淘汰和编译回退信息；批次的 `inference_ms` 表示共享执行耗时。

```bash
curl http://127.0.0.1:10002/v1/decisions \
  -H 'Content-Type: application/json' \
  -d '{"state":"小李负责测试，小王负责发布。","questions":{"owner":{"type":"choice","instructions":"谁负责测试？","criteria":["小李","小王"]}}}'
```

`choice` 的选项可用列表或标签到说明的映射；`score` 的等级从 0 起、由低到高排列；`noul` 返回命题成立概率。问题和选项受模型 Token 预算限制，超长输入明确拒绝，不静默截断。返回的 `confidence` 是分布集中度，不是准确率。

## 可视化 Demo

先启动上面的服务，再构建并启动 Demo：

```bash
pnpm --dir examples/demo01/web install
pnpm --dir examples/demo01/web build
poetry run python -m examples.demo01.commands.app serve
```

Demo 默认监听 `0.0.0.0:10013`。本机访问 **http://127.0.0.1:10013**，其他设备访问 `http://服务器IP:10013`，编辑文本、问题和选项，查看真实推理结果、概率分布及耗时。Demo 后端统一托管前端，并通过 HTTP 连接推理服务；不会加载第二份模型。

默认连接 `http://127.0.0.1:10002`，可用 `LAYA_DEMO_SERVICE_URL` 修改。VS Code 的“Laya Demo 01”调试会自动构建前端；推理服务需要单独启动。详见 [Demo 使用说明](examples/demo01/README.md)。

## 验证

```bash
poetry check --lock
poetry run ruff check src tests examples
poetry run pytest -q

# CPU FP32、CUDA FP32 / FP16 的真实服务与原始快照按相同推理精度对齐
LAYA_RUN_E2E=1 poetry run pytest tests/test_e2e.py -q -s

# 多流／编译 Graph 的 32 路 HTTP 并发、错误隔离与 Worker 故障后重启
LAYA_RUN_OPTIMIZED_E2E=1 poetry run pytest tests/test_optimized_e2e.py -q -s

# eager FP16 双头原始 logits 门槛；通过后执行 8 种策略的交错稳态矩阵
TOKENIZERS_PARALLELISM=false poetry run python -m scripts.check_optimized
poetry run python -m scripts.benchmark_optimized

pnpm --dir examples/demo01/web lint
pnpm --dir examples/demo01/web build
pnpm --dir examples/demo01/web exec playwright install chromium
pnpm --dir examples/demo01/web test:e2e
```

真实测试默认使用 `/workspace/model-bin/MigoXV/laya-multilingual`，可设置 `LAYA_E2E_MODEL_DIR`。对齐的 Oracle 独立加载 `model-bin/convaiinnovations/laya/multilingual` 原始快照，并采用与服务相同的 FP16／FP32 参数和 autocast 策略；可通过 `LAYA_E2E_SNAPSHOT_DIR` 指定含原始 Python 文件与 `multilingual/` 的快照根目录。检查分词器、模型输入、决策／动作 logits、答案及概率，同精度决策 logits 使用绝对容差 `1e-5`，动作 logits 使用 `rtol=1e-6, atol=1e-5`；原始接口概率四位小数的舍入误差限为 `0.000051`，详见 [验证记录](examples/demo01/VALIDATION.md)。浏览器测试自动启动隔离服务与 Demo（端口 11002/11003），默认使用 CUDA FP16，界面核对实际设备与精度。可通过 `LAYA_E2E_DEVICE` 和 `LAYA_E2E_DTYPE` 修改；断连测试只模拟网络错误。默认 pytest 跳过需权重的 E2E。

完整请求延迟与吞吐可用 `poetry run laya benchmark --concurrency 16 --samples 128 --rounds 3` 测量；使用 `--input-path` 指定请求 JSON，`--concurrency` 可重复。每种输入先串行预热 16 次，再按目标并发预热；输出逐请求原始延迟、错误数、排队／推理耗时及汇总 JSONL。接口返回完整 JSON，统计从发起 POST 到收完响应的延迟，不使用首包或 token 吞吐指标。CUDA FP32／FP16 × eager／vLLM eager／vLLM 编译的完整 logits 对齐、9216 次请求及资源数据见 [最终测试记录](benchmarks/vllm_16/README.md)。

完整模型优化的 [验证与性能记录](benchmarks/optimized_16/README.md) 包含 2,400 个双头原始 logits 检查和 24,576 次成功 HTTP 请求。A100 上，16 路并发的 27／512 token 输入，8 流 CUDA Graph 的吞吐为 eager FP16 的 **6.67／3.65 倍**，Graph + 静态编译为 **7.41／4.54 倍**；P95 也下降。交互服务推荐 `--runner cuda-graph --max-batch-size 16 --graph-streams 8`，保留 eager 作为默认行为基线；固定形状且提前预热的服务可使用编译模式。新形状的冷捕获／初次编译另计，编译可能耗时十几秒／形状。全部探索数据保留在 [首次探索记录](benchmarks/optimized_exploratory_16/README.md)。

第三方实现来源和许可证见 [THIRD_PARTY.md](THIRD_PARTY.md)。
