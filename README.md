# Laya 本地决策服务

从本地权重执行 `choice`（选择）、`score`（评分）和 `noul`（真假概率）决策，提供 CLI 与 HTTP 服务。模型在独立 Worker 中运行，服务负责有界队列、超时、取消、健康检查和指标。

本项目的发行包、Python 模块和命令均叫 `laya`，仅用于本地开发与运行。无需安装 PyPI 上的同名包。

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

安装源见 [PyTorch 2.8.0 官方说明](https://pytorch.org/get-started/previous-versions/#v280)。当前开发环境复用了改名前的虚拟环境，保留 `2.8.0+cu128`；不必重新下载 Torch。使用 `poetry install` 安装项目依赖，避免使用会清理手动依赖的 `poetry sync`。

## 启动服务

模型目录需要包含权重、决策配置、encoder 配置和 tokenizer。本项目不自动下载模型，也不执行模型目录里的 Python 文件。

```bash
poetry run laya serve --model-dir model-bin/convaiinnovations/laya/multilingual

# 显式使用第一张 CUDA 卡，FP32 / eager
poetry run laya serve --model-dir model-bin/convaiinnovations/laya/multilingual --device cuda:0
```

默认监听 `0.0.0.0:10002`，本机访问 `http://127.0.0.1:10002`，其他设备使用服务器 IP。模块入口等价于 `poetry run python -m laya.commands.app`。参数通过 `LAYA_MODEL_DIR`、`LAYA_DEVICE`、`LAYA_HOST`、`LAYA_PORT` 等环境变量配置；其他配置参见 `.env.example`。CLI 的必填模型路径须用参数或进程环境变量传入；VS Code 的“Laya 服务”配置会读取 `.env`。

| 接口 | 用途 |
| --- | --- |
| `GET /health/live` | API 存活 |
| `GET /health/ready` | 模型加载和预热完成 |
| `GET /v1/info` | 模型指纹、Torch、设备、执行策略 |
| `POST /v1/decisions` | 提交决策请求 |
| `GET /metrics` | 队列、计数和延迟统计 |

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

# 真实 CPU 和 cuda:0 服务测试；无 CUDA 时对应项跳过
LAYA_RUN_E2E=1 poetry run pytest tests/test_e2e.py -q -s

pnpm --dir examples/demo01/web lint
pnpm --dir examples/demo01/web build
pnpm --dir examples/demo01/web exec playwright install chromium
pnpm --dir examples/demo01/web test:e2e
```

真实测试默认使用本地 multilingual 权重，可设置 `LAYA_E2E_MODEL_DIR`。浏览器测试自动启动隔离服务与 Demo（端口 11002/11003），主流程调用真实模型；断连测试只模拟网络错误。设置 `LAYA_E2E_DEVICE=cuda:0` 可让浏览器测试使用 GPU。默认 pytest 跳过需权重的 E2E。

第三方实现来源和许可证见 [THIRD_PARTY.md](THIRD_PARTY.md)。
