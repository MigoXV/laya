# 第三方来源

`reference.py` 摘自本地 `convaiinnovations/laya` 模型快照中的 `rl_common.py`，模型卡声明 Apache-2.0。上游：https://github.com/NandhaKishorM/laya 。

保留模型 forward、序列构建、分数校准与批构建数学；移除了训练及评估入口。Runtime 自行管理离线加载、FP32、长度检查和返回结构。不执行模型目录中的 Python 文件。

本地下载元数据对应快照：`55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`。服务启动还计算实际权重、配置和 Tokenizer 的 SHA-256 指纹，防止同路径替换无法追踪。权重不纳入 Git。
