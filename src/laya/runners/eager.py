"""Eager 执行策略。"""
import torch

class EagerRunner:
    def __init__(self, model, device, dtype, quantized=False, max_len=1024):
        self.model, self.device, self.dtype = model, device, dtype
        self.raw_observer = None
        self.autocast = not quantized and dtype in (torch.float16, torch.bfloat16)
        self.function = model
        if quantized:
            from laya.runners.prepared import PreparedModel

            self.function = PreparedModel(model, max_len).eval()

    @torch.inference_mode()
    def execute(self, batch):
        tensors = [
            batch[k].to(self.device)
            for k in (
                "input_ids",
                "attention_mask",
                "marker_pos",
                "marker_mask",
                "qtype",
            )
        ]
        with torch.autocast(
            device_type=self.device.type, dtype=self.dtype,
            enabled=self.autocast,
        ):
            logits, acts = self.function(*tensors)
        if self.raw_observer is not None:
            self.raw_observer(batch, logits, acts)
        return logits.float().cpu().numpy(), torch.softmax(
            acts.float(), -1
        ).cpu().numpy()
