"""上游模型构造器，见 THIRD_PARTY.md。"""
from typing import Dict, Optional
from laya.models.decision import DecisionModel

def build_model(cfg: Dict, encoder_dir: Optional[str] = None) -> DecisionModel:
    from transformers import AutoConfig, AutoModel

    if (
        encoder_dir
    ):  # offline: architecture only, weights come from the saved state dict
        ecfg = AutoConfig.from_pretrained(encoder_dir)
        enc = AutoModel.from_config(ecfg, attn_implementation="sdpa")
    else:
        enc = AutoModel.from_pretrained(cfg["encoder"], attn_implementation="sdpa")
    return DecisionModel(enc, cfg["head_layers"], len(cfg["act_costs"]) + 1)
