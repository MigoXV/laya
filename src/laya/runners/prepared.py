"""可捕获的模型前向与 RoPE 缓存。"""
import torch
from torch import nn

class CachedRotary(nn.Module):
    """固定位置范围的 RoPE；沿用 HF 计算缓存，避免重复三角函数与编译舍入漂移。"""

    def __init__(self, original, positions, dtype):
        super().__init__()
        self.original = original
        with torch.inference_mode():
            cos, sin = original(torch.empty(1, device=positions.device, dtype=dtype), positions)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x, position_ids):
        return self.cos[:, :x.shape[1]].to(x.dtype), self.sin[:, :x.shape[1]].to(x.dtype)


class PreparedModel(nn.Module):
    """复用原有每一层，只把 CPU mask 构建移到设备，便于捕获完整前向。"""

    def __init__(self, model, max_len):
        super().__init__()
        self.model = model
        positions = torch.arange(max_len, device=next(model.parameters()).device)
        self.register_buffer("positions", positions[None], persistent=False)
        self.register_buffer("local_allowed", (positions[:, None] - positions[None, :]).abs()
                             <= model.encoder.config.local_attention // 2, persistent=False)
        for layer in model.encoder.layers:
            layer.attn.rotary_emb = CachedRotary(layer.attn.rotary_emb, self.positions, model.encoder.dtype)

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        encoder = self.model.encoder
        length = input_ids.shape[1]
        mask = (1.0 - attention_mask[:, None, None, :].to(encoder.dtype)).expand(-1, 1, length, -1)
        mask = mask.masked_fill(mask.bool(), torch.finfo(encoder.dtype).min)
        local = mask.masked_fill(~self.local_allowed[:length, :length], torch.finfo(encoder.dtype).min)
        h = encoder.embeddings(input_ids=input_ids)
        for layer in encoder.layers:
            h = layer(h, attention_mask=mask, sliding_window_mask=local,
                      position_ids=self.positions[:, :length])[0]
        h = encoder.final_norm(h)
        return self.model.score_hidden(h, attention_mask, marker_pos, marker_mask, qtype)
