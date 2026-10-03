# Demo 01：推理测试工作台

一个连接当前项目 HTTP 服务的轻量界面，用真实响应观察 Laya 的决策效果。采用 MANAS 工作台结构与苍渊·白垣的白垣主题。

## 启动

在项目根目录执行，两个服务分别运行在两个终端中：

```bash
# 终端一：启动推理服务
poetry run laya serve

# 终端二：首次安装、构建并启动 Demo
pnpm --dir examples/demo01/web install
pnpm --dir examples/demo01/web build
poetry run python -m examples.demo01.commands.app serve
```

推理服务和 Demo 分别默认监听 `0.0.0.0:10002` 和 `0.0.0.0:10013`。本机打开 http://127.0.0.1:10013 ，其他设备使用 `http://服务器IP:10013`。模型加载完成后，界面显示“服务已就绪”。服务未启动时，输入仍可编辑，启动服务后点击“重新连接”。

服务连接和超时配置可从 `.env` 或进程环境读取；启动入口的监听地址和端口通过进程环境或 CLI 参数配置：

| 配置 | 默认值 | 用途 |
| --- | --- | --- |
| `LAYA_DEMO_SERVICE_URL` | `http://127.0.0.1:10002` | 推理服务地址 |
| `LAYA_DEMO_TIMEOUT` | `60` | 等待上游响应的秒数 |
| `LAYA_DEMO_HOST` | `0.0.0.0` | Demo 监听地址，可用 `--host` 覆盖 |
| `LAYA_DEMO_PORT` | `10013` | Demo 监听端口，可用 `--port` 覆盖 |

Demo 只代理就绪、模型信息和决策接口，不导入运行时或加载权重。前端通过同源接口访问，无需给推理服务增加 CORS。

## 操作

1. 编辑待判断文本。
2. 选择问题类型并填写问题描述；选择题每行一个候选项，评分题每行一个等级、从低到高排列。
3. 点击“运行推理”，观察结果与概率条形图。也可载入三个内置中文示例。
4. 修改输入后重新运行；旧结果会明确标为上次结果。

选择题展示模型选项；评分题展示从 0 起的期望等级，可为小数；真假题展示命题成立的概率。模型不生成解释文本，条形图展示各候选结果的概率。

“分布集中度”越高，说明概率分布越集中；它不是准确率。服务端耗时和浏览器往返耗时分别展示，后者包括网络及代理开销。“查看请求与响应 JSON”可核对本次运行的原始数据。

运行时禁用重复提交。连接错误、超时、超长文本或无效选项都有可恢复提示；失败保留上次成功结果及本次输入。

## 开发与测试

```bash
pnpm --dir examples/demo01/web lint
pnpm --dir examples/demo01/web build
pnpm --dir examples/demo01/web exec playwright install chromium
pnpm --dir examples/demo01/web test:e2e
```

测试自动启动本项目真实推理服务与 Demo，覆盖三类问题、结果与图表一致性、错误恢复、重复提交、窄屏、键盘、对比度及长文本布局。测试失败的 trace 和截图位于 `web/test-results`，报告位于 `web/playwright-report`，均不进入 Git。

本次真实 CPU、CUDA、参考实现对照与浏览器结果见 [验证记录](VALIDATION.md)。

默认测试端口为 11002/11003，可通过 `LAYA_E2E_SERVICE_PORT`、`LAYA_E2E_DEMO_PORT` 更改；`LAYA_E2E_MODEL_DIR` 可更改本地模型路径。默认连接 CUDA FP16 模型服务；通过 `LAYA_E2E_DEVICE` 和 `LAYA_E2E_DTYPE` 可选择 CUDA FP16／FP32，或同时设置 `LAYA_E2E_DEVICE=cpu LAYA_E2E_DTYPE=fp32` 使用 CPU 基线。界面显示并验证实际设备与精度。

需要热更新时先启动 Demo 后端，再运行 `pnpm --dir examples/demo01/web dev`；Vite 会把 `/api` 转发到 Demo 后端。正常使用由 Demo 后端托管构建产物，不需要常驻 Vite。

## 界面状态与控件约定

主范式为实时工作台，输入表单作为辅助任务；预设和问题类型是有限选择，文本和问题是长内容，“运行推理”是唯一主动作。遵循锁定 Token、40px 常规控件、8px 间距基准和低卡片化表面。主题 Token 来自 `abyssus-vallum` 技能的 v1.1 CSS，主按钮使用墨骨背景和石素文字。

| 状态 | 显示 | 恢复路径 |
| --- | --- | --- |
| 正在连接 | 正在连接，禁止运行 | 就绪或连接失败 |
| 未连接 | 具体连接错误，保留输入 | 启动服务并重新连接 |
| 运行中 | 正在推理，禁用输入和重复提交 | 响应成功或错误 |
| 成功 | 答案、概率、耗时与 JSON | 编辑并再次运行 |
| 输入已变更 | 上次成功结果，不冒充当前结果 | 重新运行 |
| 失败 | 错误提示、保留上次结果 | 修改输入或重试 |
