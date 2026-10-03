# 基准报告索引

仅保留报告，逐请求响应、逐题 logits、CUDA Event 时间、服务快照和重复实验 JSON 不提交。旧 Torch 2.9.1 / vLLM 实验已移除，历史产物可从 Git 查询。

| 报告 | 内容 |
| --- | --- |
| [Torch 2.8 冒烟](torch_28_smoke/README.md) | 移除旧后端后的基础推理与服务检查 |
| [A100 W8A8 性能](../tests/w8a8/README.md) | 当前 Torch 2.8 的 FP16 / INT8 设备前向比较及复现命令 |
| [中英文精度](../tests/w8a8/accuracy/README.md) | 684 道带标准答案问题的准确率、答案翻转与概率偏移 |
| [量化保存与加载验证](../tests/w8a8/SAVED_MODEL.md) | 权重保存对齐、三种 runner 的 HTTP 与并发验证 |

复现代码、冻结标准答案测试集和必要的内核调优配置保留在 `tests/w8a8/`；重新运行产生的结果 JSON 和临时基准目录由 `.gitignore` 忽略。
