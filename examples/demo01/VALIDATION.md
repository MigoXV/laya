# Demo 01 验证记录

验证日期：2026-10-03。使用本地 multilingual 权重，未安装 PyPI 同名 `laya` 依赖；参考对照入口显式读取仓库模型快照，线上和 Demo 均不执行快照源码。

## 环境与配置

| 项目 | 实际值 |
| --- | --- |
| Python | 3.10.18 |
| Torch | 2.8.0+cu128，手动安装并保留 |
| Transformers | 5.18.0 |
| GPU | NVIDIA A100-PCIE-40GB，测试使用 cuda:0 |
| 模型目录 | model-bin/convaiinnovations/laya/multilingual |
| 执行 | FP32 / eager / batch 1 |
| Token 预算 | max_len=1024，head_max_len=256 |
| Node.js / pnpm | 24.18.0 / 11.24.0 |
| 浏览器 | Playwright Chromium |

CPU 模型及运行时指纹：`c5d5e84174ece910fa1b46d6a7b7320d58cc603345ddba2cb64fadfe0a3985be`。

CUDA 模型及运行时指纹：`ff4b7e826d5dd670b8987293ccdc28011b7216cf3eb54b20a3d3b995606ab50a`。指纹包含设备，CPU 与 CUDA 不同是预期行为。

## 已通过

| 验证 | 结果 |
| --- | --- |
| `poetry check --lock` | 通过，保留既有 Poetry scripts 格式，有弃用提示 |
| `poetry run ruff check src tests examples` | 通过 |
| `poetry run pytest -q` | 13 通过，2 个真实模型测试默认跳过 |
| `LAYA_RUN_E2E=1 poetry run pytest tests/test_e2e.py -q -s` | CPU 与 cuda:0 共 2 通过 |
| `pnpm --dir examples/demo01/web lint` | ESLint 与 TypeScript 检查通过 |
| `pnpm --dir examples/demo01/web build` | 构建通过 |
| `pnpm --dir examples/demo01/web test:e2e` | 4 个浏览器测试全部通过 |

真实 HTTP 测试覆盖三种问题、重复请求一致性、概率归一化、模型指纹、422 非法请求／超长文本、413 超大正文，以及错误后的就绪状态。关闭后检查 Worker PID 已不存在。

参考实现对照在每个设备执行 9 个输入用例和 3 个截断拒绝场景。既有容差为 logits 最大绝对差不超过 `1e-5`、评分／真假结果差不超过 `0.000051`；CPU logits 差为 **0**，CUDA logits 差为 **8.344650268554688e-7**，均通过。

可复现对照命令（仓库根目录）：

```bash
poetry run python - <<'PY'
from pathlib import Path
from laya.checks import check_reference
import json

root = Path("model-bin/convaiinnovations/laya")
for device in ("cpu", "cuda:0"):
    print(json.dumps(check_reference(root / "multilingual", root, device)))
PY
```

浏览器主流程经过真实 Demo 代理、服务、Worker 和 CPU 模型，三种结果的文字、概率条长度、百分比和 JSON 均与服务响应对照。仅断连恢复场景模拟网络错误；推理结果均来自真实服务。

界面检查覆盖 320/768/1280px，无页面横向溢出；CSS 200% 缩放布局、长中文、键盘 Tab 焦点和 Enter 提交、减少动效偏好均通过。axe-core 在成功结果页执行 WCAG A/AA 检查，未发现违规。已人工查看桌面与窄屏截图。

## 验证边界

本次是正确性、交互与生命周期验收，不是吞吐基准或模型准确率评测。浏览器测试使用 CPU；CUDA 在真实 HTTP 和参考实现对照中验证。未验证 Firefox、Safari 或高并发长时间运行。

浏览器报告和截图保存在 `web/playwright-report`、`web/test-results`，真实 HTTP 服务日志位于 pytest 临时目录；这些运行产物不进入 Git。
