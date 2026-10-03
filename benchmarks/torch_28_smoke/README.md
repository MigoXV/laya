# 移除旧后端后的极简冒烟

本次按要求只做小规模测试，没有运行完整端到端套件、完整数值矩阵或性能矩阵。

环境为 Torch `2.8.0+cu128`。当前实现仅保留 eager、cuda-graph、cuda-graph-compile；插件注册、旧后端实现、专用测试和旧矩阵脚本均已移除，基准辅助函数及输入改为独立文件。环境中已无旧推理引擎包。

降级 Torch 后残留的 torchvision 0.24.1 会触发 `operator torchvision::nms does not exist`，阻止 Transformers 加载文本编码器。项目不使用视觉模块，已移除这项未使用且不兼容的包，未修改用户安装的 Torch。

- `poetry run ruff check src tests scripts examples`：通过。
- `poetry check --lock`：通过，保留现有 Poetry scripts 表的弃用提示。
- `poetry run pytest -q tests/test_precision.py::test_decision_head_precision tests/test_cuda_profiles.py::test_cached_rotary_preserves_original_prefix_and_dtype`：4 项通过。
- 一个 27 token 的真实 choice 输入，以 eager FP16 为基线，分别用两个独立序列执行 Graph 与 compile 各两次，检查原始决策／动作 logits 及动作概率。Graph 最大绝对差均为 0；compile 决策 logits 最大绝对差约 0.000824，动作 logits 差为 0。均通过已有门槛。
- 服务已在 `0.0.0.0:10002` 以 Torch 2.8.0、FP16、8 流 Graph 恢复。Demo `10013` 的一次三类问题代理请求及页面／模型 info 检查通过。

`results.json` 保存小样本数值，`service.json` 保存恢复的实际服务信息与响应。历史性能记录保持原始数据，并标明为历史结论；本次未重新测量吞吐。
