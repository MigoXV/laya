"""延迟注册；eager 启动不需要导入或安装 vLLM。"""


def register():
    from vllm import ModelRegistry

    if "LayaForDecisions" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("LayaForDecisions", "laya.vllm_model:LayaForDecisions")


class LayaWorkerExtension:
    def laya_model_info(self):
        from .vllm_model import describe_model

        return describe_model(self.model_runner.get_model())
