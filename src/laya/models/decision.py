"""决策模型；源自 convaiinnovations/laya，Apache-2.0，见 THIRD_PARTY.md。"""
import torch
from torch import nn
import torch.utils.checkpoint

class DecisionModel(nn.Module):
    """Pretrained bidirectional encoder (no LLM, no LoRA) + from-scratch decision head.

    Each option gets a [MASK] marker; the head scores markers -> softmax over the question's options.
    """

    def __init__(
        self,
        encoder: nn.Module,
        head_layers: int = 2,
        n_act: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        nhead = max(1, d // 64)
        layer = nn.TransformerEncoderLayer(
            d, nhead, 4 * d, dropout, batch_first=True, norm_first=True
        )
        self.head = (
            nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False)
            if head_layers > 0
            else None
        )
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1)
        )
        self.act_head = nn.Sequential(
            nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act)
        )
        self.register_buffer(
            "temperature", torch.ones(3)
        )  # per qtype, fitted post-hoc in evaluate.py
        self.head_checkpointing = False

    def forward(
        self,
        input_ids,
        attention_mask,
        marker_pos,
        marker_mask,
        qtype,
        detach_encoder: bool = False,
    ):
        h = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        if detach_encoder:
            h = h.detach()
        return self.score_hidden(h, attention_mask, marker_pos, marker_mask, qtype)

    def score_hidden(self, h, attention_mask, marker_pos, marker_mask, qtype):
        """完整决策与动作头；供 eager 和 CUDA Graph 共用同一数学定义。"""
        h = h + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                if (
                    self.head_checkpointing
                    and self.training
                    and torch.is_grad_enabled()
                ):
                    h = torch.utils.checkpoint.checkpoint(
                        layer, h, None, pad, use_reentrant=False
                    )
                else:
                    h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)
        # act head sees the pooled sequence + detached summary of its own answer distribution
        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        top2 = p.topk(2, -1).values
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        pooled = h[:, 0].float()
        act_input = torch.cat([pooled, feats], -1).to(self.act_head[0].weight.dtype)
        act_logits = self.act_head(act_input)
        return logits, act_logits
