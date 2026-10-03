# 第三方来源

`reference.py` 摘自本地 `convaiinnovations/laya` 模型快照中的 `rl_common.py`，模型卡声明 Apache-2.0。上游：https://github.com/NandhaKishorM/laya 。

保留模型 forward、序列构建、分数校准与批构建数学；移除了训练及评估入口。为 FP16 推理将动作头输入转换到其权重 dtype；位置频率、分布与概率计算保留 FP32，前向使用 FP16 autocast 处理精度敏感算子。Runtime 自行管理离线加载、FP16／FP32 精度选择、长度检查和返回结构。不执行模型目录中的 Python 文件。

本地下载元数据对应快照：`55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`。服务启动还计算实际权重、配置和 Tokenizer 的 SHA-256 指纹，防止同路径替换无法追踪。权重不纳入 Git。

当前默认加载 `/workspace/model-bin/MigoXV/laya-multilingual`，整理自上述快照的 `multilingual/` 子目录。权重与分词器 JSON 保持原始字节；原始编码器、决策与分词器配置合并为根目录的 `config.json`，仅保留模型结构与推理参数，删除训练记录、训练成本等未使用字段。`manifest.json` 记录当前 3 个推理文件的 SHA-256、大小、来源及整理方式，`LICENSE` 保留 Apache-2.0 全文。推理代码由当前项目提供；原始快照独立保留用于参考对照。
