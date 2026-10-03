# Laya 本地决策服务

从本地权重执行 `choice`（选择）、`score`（评分）和 `noul`（真假概率）决策，提供 CLI 与 HTTP 服务。模型在独立 Worker 中运行，服务负责有界队列、超时、取消、健康检查和指标。

本项目的发行包、Python 模块和命令均叫 `laya`，仅用于本地开发与运行。无需安装 PyPI 上的同名包。

## 代码结构

主程序按职责组织为功能子包：

```text
src/laya/
├── commands/       Typer 命令、日志与启动入口
├── configs/        类型化运行与服务配置
├── api/            HTTP 装配、入口容量、Laya/Jev 协议适配
├── engine/         队列、调度、IPC 与独立 Worker
├── inferencers/    内部请求、预处理、问题组批与结果后处理
├── runtime/        模型与 tokenizer 加载、设备资源、身份指纹
├── runners/        eager、CUDA Graph、编译与 RoPE 缓存
├── models/         模型 forward、权重配置兼容与精度转换
└── quantization/   W8A8 格式、模块替换、导出与 CUDA 内核
```

线上调用链为 `api → engine → worker → DecisionInferencer → Runtime.runner → model`。CLI 离线推理直接创建 `DecisionInferencer`；它持有 `Runtime`，负责问题预处理、组批和答案处理。`Runtime` 管理加载、设备、tokenizer、runner 和关闭，不解析外部协议。SDK 类型集中在 `api/jev.py`，内部请求定义在 `inferencers/contracts.py`。API 和 CLI 模块导入不加载 Torch、Transformers 或 Triton，计算依赖在实际推理进程中加载。

Python 导入已直接迁移到新路径，例如 `laya.configs.settings.Config`、`laya.api.app.create_app` 和 `laya.inferencers.decision.DecisionInferencer`，不保留旧路径转发。命令入口仍为 `laya.commands.app`，Worker 入口为 `laya.engine.worker`。上游序列构造和快照对齐工具放在 `tests/support/`。源码指纹覆盖拆分后的计算模块，拆包会改变源码及运行身份指纹，不修改模型权重和配置文件。

## 安装

使用 Python 3.10、Poetry；可视化 Demo 另外需要 Node.js 24 和 pnpm 11.24.0。项目默认使用 `.venv`。

```bash
poetry env use python3.10
poetry install
```

Torch 由环境单独管理，不写入 Poetry 依赖和锁文件。请在同一虚拟环境手动安装 **2.8.0**，按设备选择一条命令：

```bash
# CPU
poetry run python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu

# NVIDIA CUDA 12.8
poetry run python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

poetry run python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'
```

安装源见 [PyTorch 2.8.0 官方说明](https://pytorch.org/get-started/previous-versions/#v280)。当前开发环境使用 `2.8.0+cu128`，推理后端仅依赖 PyTorch 与 Transformers，无需其他推理引擎。现有 Transformers 约束为 `>=4.57.6,<5`。使用 `poetry install` 安装项目依赖，避免使用会清理手动安装 Torch 的 `poetry sync`。

## 启动服务

下面的示例使用仓库外的 `/workspace/model-bin/MigoXV/laya-multilingual`，只使用多语言版本；模型目录必须显式指定。推理只需根目录的 `config.json`、`model.safetensors` 和 `tokenizer.json`（合计约 678 MB），另附中文说明、Apache-2.0 许可证和来源／SHA-256 清单。权重和分词器文件保持原始字节；全部推理配置合并到一份 `config.json`，移除不参与推理的训练字段。本项目不自动下载模型，也不执行模型目录里的 Python 文件。

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
poetry run laya serve --model-dir /workspace/model-bin/MigoXV/laya-multilingual

# 默认第一张 CUDA 卡，FP16 / eager
poetry run laya serve --model-dir /workspace/model-bin/MigoXV/laya-multilingual --device cuda:0 --dtype fp16

# CPU 显式使用 FP32 数值基线
poetry run laya serve --model-dir /workspace/model-bin/MigoXV/laya-multilingual --device cpu --dtype fp32

# 完整模型 CUDA Graph，跨请求合并调度，8 条 CUDA stream 执行独立序列
poetry run laya serve --model-dir /workspace/model-bin/MigoXV/laya-multilingual --runner cuda-graph --dtype fp16 --max-batch-size 16 --graph-streams 8

# 完整 CUDA Graph + 静态形状 torch.compile（初次编译耗时另计）
poetry run laya serve --model-dir /workspace/model-bin/MigoXV/laya-multilingual --runner cuda-graph-compile --dtype fp16 --max-batch-size 16 --graph-streams 8

# 查看实际解析到的目录、配置和权重大小
poetry run laya inspect --model-dir /workspace/model-bin/MigoXV/laya-multilingual
```

默认使用 `cuda:0 / fp16 / eager`，监听 `0.0.0.0:10002`，本机访问 `http://127.0.0.1:10002`，其他设备使用服务器 IP。普通 FP16 仓库的模型参数使用 FP16，前向启用 FP16 autocast；矩阵运算使用 FP16，RoPE 位置频率、softmax、校准与概率保留必要的 FP32 精度。FP16 推理采用现有 eager FP16 作为行为基线，FP32 用于独立观察跨精度漂移。FP16 权重转成 FP32 不会恢复已丢失的权重精度，但会改变运算舍入；因此跨精度差异不直接代表实现错误。321,908,995 个参数在 FP16 下占 643,817,990 字节，是 FP32 参数占用的一半；这不等于进程总内存或显存。

模块入口等价于 `poetry run python -m laya.commands.app`。模型路径优先级为 `--model-dir` 参数、进程环境变量 `LAYA_MODEL_DIR`、`.env` 中的 `LAYA_MODEL_DIR`；没有内置默认路径，缺失时启动报错；设备与精度同样支持 `--device`／`LAYA_DEVICE` 和 `--dtype`／`LAYA_DTYPE`，`serve` 和 `infer` 共用这一规则。Runtime 显式传入所选 dtype 构造编码器，并统一转换模型；执行精度覆盖上游编码器配置里的 dtype 元信息。`/v1/info` 返回执行精度、autocast 状态、参数数量及参数字节数。Transformers 4 通过兼容转换读取权重配置中的全局／局部 RoPE theta，模型目录仍保留统一配置。CUDA 不可用时明确报错，需要 CPU 时显式指定 `--device cpu --dtype fp32`；FP16 当前仅支持 CUDA，CPU FP16 明确拒绝，不会静默切换设备或精度。监听地址和端口通过 `LAYA_HOST`、`LAYA_PORT` 等进程环境变量配置；其他配置参见 `.env.example`。

## W8A8 模型仓库

模型目录必须显式指定，可通过 `--model-dir`、`LAYA_MODEL_DIR` 或主动配置的 `.env` 提供。程序根据统一 `config.json` 的 `quantization` 自动识别格式，目录后缀不参与判断；没有该字段的模型保留原浮点加载方式。

```bash
# 从原 FP16 权重导出独立仓库；目标目录必须不存在。
poetry run laya quantize --model-dir /workspace/model-bin/MigoXV/laya-multilingual \
  --output-dir /workspace/model-bin/MigoXV/laya-multilingual-w8a8 --device cuda:0

# 只需指定新目录，无须额外量化开关。
LAYA_REQUEST_TIMEOUT=300 poetry run laya serve --model-dir /workspace/model-bin/MigoXV/laya-multilingual-w8a8 \
  --runner cuda-graph-compile --max-batch-size 16
```

`laya_w8a8` v1 的量化信息仍放在 `config.json` 中，不另设配置文件。97 个大 Linear 的 125,042,688 个权重保存为 `[out,in]` INT8 `qweight`，每个输出通道对应 FP32 `weight_scales`；其余训练参数为 FP16，原温度缓冲区保留 FP32。对称量化采用 `absmax.clamp_min(1e-8)/127`、ties-to-even 舍入及 `[-127,127]` 范围。激活逐 token 动态量化，运行时产生尺度，不保存静态激活尺度，不在加载时重复量化权重。决策头显式 QKV，LayerNorm 与量化在加载后融合；模型仓库只包含配置、权重、分词器、来源和许可，代码由项目负责。

W8A8 首版使用 CUDA SM80+ 与 FP16 浮点计算，现有 A100 已验证。CPU、BF16、FP32 或未知量化格式明确拒绝。三种 runner 均可使用；W8A8 关闭 autocast，Graph 整批执行，compile 关闭 eager 舍入模拟。此时 `--graph-streams` 不控制逐条并行，实际为一条执行 stream，`/v1/info` 如实报告。普通模型保留原来的执行策略。首次编译不计入稳态，建议为编译服务显式设置 `LAYA_REQUEST_TIMEOUT=300`，或预热已知 profile 后再接流量；未知形状仍可能触发首次编译。

`/v1/info` 与推理响应的模型信息包含 `quantization`（普通模型为 null），报告格式、量化模块数与尺度字节数；`parameter_count` 保留逻辑参数数，`parameter_bytes` 报告实际混合权重字节数、排除尺度和温度，不等于文件大小或进程显存。来源与文件哈希见模型 `manifest.json`；精度实验见 [tests/w8a8/accuracy/README.md](tests/w8a8/accuracy/README.md)。

| 接口 | 用途 |
| --- | --- |
| `GET /health/live` | API 存活 |
| `GET /health/ready` | 模型加载和预热完成 |
| `GET /v1/info` | 模型指纹、Torch、设备、执行策略 |
| `POST /v1/decisions` | 原有 Laya 决策接口 |
| `POST /v1/systemone` | Jev 兼容决策接口 |
| `GET /v1/models` | 官方 SDK 可读取的本地模型与别名 |
| `GET /metrics` | 队列、计数和延迟统计 |

`serve`、`infer` 支持 `--runner`／`LAYA_RUNNER`，可选 `eager`、`cuda-graph`、`cuda-graph-compile`。三个后端共用完整权重、相同的分词、校准与领域输出。编码器保留全局／局部 RoPE theta=160000，注意力使用 PyTorch SDPA 与闭区间局部 mask。

对于普通 FP16 仓库，`cuda-graph` 捕获编码器、完整决策／动作头和动作 softmax，将多个独立序列分配到最多 8 条 CUDA stream，再用一次完整 Graph replay 执行整个调度批次。每条序列保持 eager batch=1 的计算形状；序列长度与选项数按精确值分组，不补齐它们，只把批次容量向上取到 1/2/4/8/16，额外行使用有效输入副本且丢弃其输出。这样既减少 CPU kernel 派发开销，也保留该模型敏感的 FP16 舍入行为。RoPE 缓存由原 HF 实现一次计算，参数仍是原始 FP16 权重。

普通 FP16 仓库的 `cuda-graph-compile` 额外编译整个模型，使用静态序列／选项形状与 `emulate_precision_casts`。`LAYA_COMPILE_CACHE_SIZE` 限制编译形状数量；超出后该形状采用未编译的完整 CUDA Graph，原因通过 `/metrics` 的 `compile_fallback_shapes` 显式报告。Graph 的 LRU 缓存由 `LAYA_GRAPH_CACHE_SIZE` 限制，首次出现或被淘汰的形状会先预热再捕获；因此冷捕获／初次编译延迟与稳态延迟分开记录。静态缓冲区由唯一 Worker 所有，每次正确复制输入、读回结果后再复用。

已知业务形状可用 `LAYA_GRAPH_PREWARM_PROFILES='[[1,27,2],[16,27,2]]'` 在服务就绪前预热；每项为 `[批次容量, 精确序列长度, 选项数]`，批次容量只支持 1/2/4/8/16，数量不能超过 Graph 缓存容量，也受批次和 token 上限约束。输入仍会完整复制并重新推理。预处理复用一次 Rust 分词结果，避免参考构建器重复分词；默认 `TOKENIZERS_PARALLELISM=false`，可通过环境变量显式覆盖。

服务默认仍为 eager FP16、单请求，用作稳定的行为基线。显式使用 `--max-batch-size 16`／`LAYA_MAX_BATCH_SIZE=16` 开启跨请求调度，最多等待 `--batch-wait-ms`／`LAYA_BATCH_WAIT_MS`（默认 2 ms），总设备 token 容量由 `LAYA_MAX_BATCH_TOKENS` 控制，包含额外占位行。`--graph-streams`／`LAYA_GRAPH_STREAMS` 设置 Graph 的 CUDA stream 数。每个请求保持独立的结果、校验错误、取消与 deadline，Worker 协议用版本和 ID 关联响应；API 仍不执行模型计算。`/metrics` 返回请求／模型调度批次分布、token 填充率、Graph 命中／淘汰和编译回退信息；批次的 `inference_ms` 表示共享执行耗时。

```bash
curl http://127.0.0.1:10002/v1/decisions \
  -H 'Content-Type: application/json' \
  -d '{"state":"小李负责测试，小王负责发布。","questions":{"owner":{"type":"choice","instructions":"谁负责测试？","criteria":["小李","小王"]}}}'
```

`choice` 的选项可用列表或标签到说明的映射；`score` 的等级从 0 起、由低到高排列；`noul` 返回命题成立概率。问题和选项受模型 Token 预算限制，超长输入明确拒绝，不静默截断。返回的 `confidence` 是分布集中度，不是准确率。

## Jev 接口与官方 SDK

服务同时提供 `POST /v1/systemone` 和 `GET /v1/models`，与旧接口共用同一个 Engine、Worker 和模型。协议基准为 **typesafe-sdk 0.7.2** 及其对应的 [TypeSafe OpenAPI](https://api.typesafe.ai/openapi.json)，依赖已锁定；confidence 遵循 [TypeSafe 官方定义](https://docs.typesafe.ai/confidence)。SDK 的请求 wire schema 集中在 `src/laya/api/jev.py` 引用，回答、响应、usage 和模型列表直接复用 SDK 类型。升级 SDK 时需要重跑协议及端到端测试。

**服务端不使用 API Key，也不验证 Authorization。** 官方 SDK 自身要求非空 `api_key`，示例中的 `"local"` 只是客户端占位值；直接 HTTP 请求无需任何凭据。该接口兼容 Jev 的协议和返回语义，实际模型仍为本地 Laya，不保证与 Jev 的预测结果或容量一致。

启动上面的服务后，在安装本项目依赖的环境运行：

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

with TypeSafeClient(base_url="http://127.0.0.1:10002", api_key="local") as client:
    print(client.models.list().models)
    result = client.system_one(
        state="小李负责测试，小王负责发布。",
        questions={
            "owner": Choice(
                instructions="谁负责测试？",
                criteria={"小李": None, "小王": None},
            ),
            "importance": Score(
                instructions="测试的重要程度？", criteria=["低", "中", "高"],
            ),
            "holds": Noul(instructions="小李负责测试吗？"),
        },
    )
    print(result.choices["owner"].choice)
    print(result.scores["importance"].legend)
    print(result.nouls["holds"].noul)
    print(result.request_id)
```

异步后端可以复用 `AsyncTypeSafeClient`：

```python
import asyncio
from typesafe_sdk import AsyncTypeSafeClient, Noul

async def main():
    async with AsyncTypeSafeClient(
        base_url="http://127.0.0.1:10002", api_key="local",
    ) as client:
        replies = await asyncio.gather(*(
            client.system_one(
                state=text,
                questions={"holds": Noul(instructions="小李负责测试吗？")},
            )
            for text in ["小李负责测试。", "小王负责测试。"]
        ))
        print([reply.nouls["holds"].noul for reply in replies])

asyncio.run(main())
```

SDK 默认发送 `model="jev-latest"`。服务接受该兼容别名、`laya`、`laya-latest` 和当前模型目录名，统一返回实际目录名；未登记的名称返回 422，不假装加载 Jev 检查点。`GET /v1/models` 的描述明确指出本地模型和别名关系，`release_date` 为本地兼容接口的发布日期，不是上游权重训练日期。`/docs` 和 `/openapi.json` 提供请求与响应定义。

Jev 接口的具体约定：

- `state` 支持字符串、对象和数组；instructions 和选项描述也支持结构化 JSON。省略／null instructions 转为空文本；对象和数组以保留中文及输入顺序的 JSON 转为推理文本。Score 的 legend 返回原始结构化描述。
- Choice criteria 必须是映射，Score criteria 必须是有序数组。未知字段按照 SDK wire schema 忽略；这是与旧接口拒绝未知字段的区别。
- 顶层仅返回 `model`、`answers`、`usage`。Choice 返回 `type/choice/probabilities/confidence`；Score 返回 `type/score/legend/probabilities/confidence`；Noul 仅返回 `type/noul`。模型配置和 timings 通过旧接口、`/v1/info`、`/metrics` 查看。
- Choice confidence 为 `(max(p) - 1/n) / (1 - 1/n)`；Score confidence 为 `max(0, 1 - sum(p[i] * abs(i-m)) / mean(abs(i-(n-1)/2)))`，其中 `m` 是首个最高概率等级。旧接口保留原有的熵公式。confidence 不是预测准确率。
- Score 的分数为等级索引的概率加权期望；JSON 中 legend/probabilities 的键为字符串，SDK 解码后为整数。`input_tokens` 为实际模型输入统计（各问题完整序列之和），`output_tokens=0`，因为本模型不生成文本。
- 新接口的所有响应带 `x-typesafe-request-id`。422 使用官方 OpenAPI 的 `detail` 列表，包含字段位置和错误说明，不回显完整输入。饱和、未就绪和超时返回 529，并带 `Retry-After: 1`；SDK 默认可重试。意外内部错误返回脱敏 500。请求正文超过配置的字节上限返回 413。

**本地容量限制仍适用**：每次 1–16 个问题，Choice 2–16 个非空唯一选项，Score 2–10 个等级；序列和问题预算由模型配置决定，当前多语言模型为 1024／256 Token，单个选项文本最多 48 Token，序列化后的 instructions 最多 8192 字符。模型保留的 MASK 标记不能出现在输入里。超限或不可处理的输入返回 422，不静默截断。默认正文上限为 262144 字节。Jev 的更大选项集、长上下文以及 SDK schema 接受的单选项／单等级请求，不在该本地模型的支持范围内。

协议测试使用真实 HTTP 和未经修改的同步／异步官方 SDK，覆盖混合题型、结构化输入、可选字段、原始 JSON 字段集合、模型别名、自定义响应模型、错误解析、重试、取消及请求关联。真实权重测试默认另启动 CPU FP32 服务，可显式选择 GPU 0，验证旧接口概率对齐、并发错误隔离、完整 Worker 链及退出后的进程／端口清理：

```bash
poetry run pytest tests/test_jev.py -q
LAYA_RUN_JEV_E2E=1 \
LAYA_E2E_MODEL_DIR=/workspace/model-bin/MigoXV/laya-multilingual \
poetry run pytest tests/test_jev_e2e.py -q

# 在 GPU 0 上验证官方 SDK、HTTP 和独立 Worker
LAYA_RUN_JEV_E2E=1 \
LAYA_E2E_MODEL_DIR=/workspace/model-bin/MigoXV/laya-multilingual \
LAYA_JEV_E2E_DEVICE=cuda:0 LAYA_JEV_E2E_DTYPE=fp16 \
LAYA_JEV_E2E_RUNNER=cuda-graph \
poetry run pytest tests/test_jev_e2e.py -q
```

`LAYA_JEV_E2E_RUNNER` 默认是 `eager`。GPU FP16 的严格并发概率对照使用 `cuda-graph`，保持精确序列形状；eager 组批 padding 会改变浮点运算结果，本轮样例观察到约 `0.00101945` 的概率差，重构前后均可复现，超过该 SDK 测试沿用的 `1e-5` 容差。

拆包验收环境为 Python 3.10、PyTorch 2.8.0+cu128、SDK 0.7.2：默认测试集 **94 passed、18 skipped**；GPU 0 FP16 CUDA Graph 的真实 SDK 测试 **3 passed**。固定两份输入、三类问题，对照拆包前后 CPU FP32，以及 GPU 0 上 FP16/W8A8 的 eager、CUDA Graph、CUDA Graph + compile，共 7 组配置：输入 Token、markers、答案、用量和超长错误一致，决策及动作 logits 最大绝对差均为 **0**。需要显式启用的模型／设备测试不计为默认通过。原服务继续运行，本次仅验证正确性，不作性能结论。

GPU 0 CUDA Graph 的 32 并发、Worker 故障检测、同端口重启及资源清理测试通过；该生命周期测试的 compile 参数本轮未启用，compile 数值路径已包含在上述 7 组对照中。W8A8 重新导出的 267 个张量与现有量化仓库逐项一致。Ruff、Poetry 配置检查和 wheel 打包通过；Poetry 仍提示已有的 scripts 配置弃用警告。

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
export LAYA_MODEL_DIR=/workspace/model-bin/MigoXV/laya-multilingual
poetry check --lock
poetry run ruff check src tests examples
poetry run pytest -q

# CPU FP32、CUDA FP32 / FP16 的真实服务与原始快照按相同推理精度对齐
LAYA_RUN_E2E=1 poetry run pytest tests/test_e2e.py -q -s

# 多流／编译 Graph 的 32 路 HTTP 并发、错误隔离与 Worker 故障后重启
LAYA_RUN_OPTIMIZED_E2E=1 poetry run pytest tests/test_optimized_e2e.py -q -s

# eager FP16 双头原始 logits 门槛；通过后执行当前 FP16 后端的交错稳态矩阵
TOKENIZERS_PARALLELISM=false poetry run python -m scripts.check_optimized
poetry run python -m scripts.benchmark_optimized

pnpm --dir examples/demo01/web lint
pnpm --dir examples/demo01/web build
pnpm --dir examples/demo01/web exec playwright install chromium
pnpm --dir examples/demo01/web test:e2e
```

真实测试必须通过 `LAYA_E2E_MODEL_DIR` 或 `LAYA_MODEL_DIR` 显式指定模型目录。对齐的 Oracle 需要另行准备上游多语言快照，并通过 `LAYA_E2E_SNAPSHOT_DIR` 显式指定仓库外含原始 Python 文件与 `multilingual/` 的快照根目录。Oracle 独立加载原始快照，采用与服务相同的 FP16／FP32 参数和 autocast 策略；项目不再保留旧快照下载。检查分词器、模型输入、决策／动作 logits、答案及概率，同精度决策 logits 使用绝对容差 `1e-5`，动作 logits 使用 `rtol=1e-6, atol=1e-5`；原始接口概率四位小数的舍入误差限为 `0.000051`，详见 [验证记录](examples/demo01/VALIDATION.md)。浏览器测试自动启动隔离服务与 Demo（端口 11002/11003），默认使用 CUDA FP16，界面核对实际设备与精度。可通过 `LAYA_E2E_DEVICE` 和 `LAYA_E2E_DTYPE` 修改；断连测试只模拟网络错误。默认 pytest 跳过需权重的 E2E。

完整请求延迟与吞吐可用 `poetry run laya benchmark --concurrency 16 --samples 128 --rounds 3` 测量；使用 `--input-path` 指定请求 JSON，`--concurrency` 可重复。每种输入先串行预热 16 次，再按目标并发预热；输出逐请求原始延迟、错误数、排队／推理耗时及汇总 JSONL。接口返回完整 JSON，统计从发起 POST 到收完响应的延迟，不使用首包或 token 吞吐指标。

当前 Torch 2.8.0 的保留报告见 [基准索引](benchmarks/README.md)，包括基础冒烟、W8A8 性能、标准答案精度和量化保存加载验证。旧 Torch 2.9.1 / vLLM 产物和新实验的原始输出已从工作树及 Git 历史移除；复现代码、冻结标准答案和必要配置继续保留，新生成的基准产物不提交。

第三方实现来源和许可证见 [THIRD_PARTY.md](THIRD_PARTY.md)。
