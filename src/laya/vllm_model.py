"""vLLM 0.14.1 原生 ModernBERT 编码器与完整 Laya 决策头。"""

from types import SimpleNamespace

import torch
from torch import nn
from vllm.model_executor.layers.pooler.abstract import Pooler
from vllm.model_executor.layers.pooler.common import PoolingParamsUpdate
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces_base import attn_type, default_pooling_type
from vllm.model_executor.models.modernbert import ModernBertModel

from .reference import DecisionModel


class CanonicalSDPA(nn.Module):
    """batch=1 的双向注意力，与行为基线使用相同的 additive mask / SDPA。"""
    def __init__(self, num_heads, head_size, radius):
        super().__init__()
        self.num_heads, self.head_size, self.radius = num_heads, head_size, radius

    def forward(self, query, key, value):
        length = query.shape[0]
        q, k, v = [tensor.reshape(1, length, self.num_heads, self.head_size).transpose(1, 2)
                   for tensor in (query, key, value)]
        mask = torch.zeros((1, 1, length, length), device=query.device, dtype=query.dtype)
        if self.radius is not None:
            positions = torch.arange(length, device=query.device)
            outside = (positions[:, None] - positions[None, :]).abs() > self.radius
            mask = mask.masked_fill(outside, torch.finfo(query.dtype).min)
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0,
        ).transpose(1, 2).contiguous().reshape(length, -1)


def describe_model(model):
    attention = model.encoder.encoder_layer.layers[1].attn.attn
    return {
        "loaded_parameter_count": sum(p.numel() for p in model.parameters()),
        "loaded_parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
        "attention_backend": "TORCH_SDPA",
        "local_attention_radius": attention.radius,
        "worker_threads": torch.get_num_threads(),
        "global_rope_theta": model.config.global_rope_theta,
        "local_rope_theta": model.config.local_rope_theta,
    }


class DecisionPooler(Pooler):
    def __init__(self, config):
        super().__init__()
        encoder = nn.Identity()
        encoder.config = SimpleNamespace(hidden_size=config.hidden_size)
        self.model = DecisionModel(
            encoder, config.laya_head_layers, config.laya_num_actions
        ).eval()
        self.model.temperature = self.model.temperature.float()
        self.mask_id = config.laya_mask_id
        self.prefixes = config.laya_type_prefixes

    def get_supported_tasks(self):
        # encode() 的输出传输接口；内容是原始决策/动作 logits，不是文本 embedding。
        return {"embed"}

    def get_pooling_updates(self, task):
        return PoolingParamsUpdate(requires_token_ids=True)

    def question_layout(self, ids):
        tokens = ids.tolist()
        if all(token == 0 for token in tokens):
            # vLLM 内存预演的 dummy token；也执行完整决策头。
            return 0, [0, len(tokens) - 1]
        for qtype, prefix in enumerate(self.prefixes):
            if tokens[1:1 + len(prefix)] == prefix:
                markers = [i for i, token in enumerate(tokens) if token == self.mask_id]
                if len(markers) < 2:
                    raise ValueError("laya_missing_option_markers")
                return qtype, markers
        raise ValueError("laya_unknown_question_type")

    def forward(self, hidden_states, pooling_metadata):
        cursor = pooling_metadata.get_pooling_cursor()
        if cursor.is_partial_prefill():
            raise ValueError("Laya 不支持分块预填充")
        results, start = [], 0
        device, dtype = hidden_states.device, self.model.type_emb.weight.dtype
        for ids in pooling_metadata.get_prompt_token_ids():
            length = ids.numel()
            qtype, markers = self.question_layout(ids)
            h = hidden_states[start:start + length].unsqueeze(0)
            start += length
            positions = torch.tensor([markers], device=device)
            with torch.autocast(
                device_type=device.type, dtype=dtype,
                enabled=dtype in (torch.float16, torch.bfloat16),
            ):
                logits, acts = self.model.score_hidden(
                    h, torch.ones((1, length), device=device, dtype=torch.long),
                    positions, torch.ones_like(positions, dtype=torch.bool),
                    torch.tensor([qtype], device=device),
                )
            results.append(torch.cat((logits[0].float(), acts[0].float())))
        return results


@attn_type("encoder_only")
@default_pooling_type(seq_pooling_type="CLS")
class LayaForDecisions(nn.Module):
    is_pooling_model = True

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        if vllm_config.scheduler_config.max_num_seqs != 1:
            raise ValueError("Laya 当前 vLLM 执行要求 max_num_seqs=1")
        torch.set_num_threads(self.config.laya_threads)
        self.encoder = ModernBertModel(vllm_config=vllm_config, prefix=prefix + ".encoder")
        for i, layer in enumerate(self.encoder.encoder_layer.layers):
            radius = None if i % self.config.global_attn_every_n_layers == 0 else self.config.laya_local_attention // 2
            layer.attn.attn = CanonicalSDPA(
                self.config.num_attention_heads,
                self.config.hidden_size // self.config.num_attention_heads,
                radius,
            )
            # 原生 ModernBERT 把 RoPE 缓存硬编码为 FP16；FP32 保留 FP32 三角值。
            if vllm_config.model_config.dtype == torch.float32:
                rope = layer.attn.rotary_emb
                rope.cos_sin_cache = rope._compute_cos_sin_cache().float()
        self.pooler = DecisionPooler(self.config)

    def embed_input_ids(self, input_ids):
        return self.encoder.embed_input_ids(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        dtype = self.pooler.model.type_emb.weight.dtype
        # eager FP16 的 autocast 使 LayerNorm 和残差保持 FP32；参数仍为 FP16。
        # 省略这一上下文会改变 22 层残差累加的精度和最终决策。
        with torch.autocast("cuda", dtype=dtype, enabled=dtype in (torch.float16, torch.bfloat16)):
            return self.encoder(input_ids, positions, intermediate_tensors, inputs_embeds)

    def load_weights(self, weights):
        heads = []

        def encoder_weights():
            for name, tensor in weights:
                if name.startswith("encoder."):
                    yield name[len("encoder."):], tensor
                else:
                    heads.append((name, tensor))

        loaded = {"encoder." + name for name in self.encoder.load_weights(encoder_weights())}
        parameters = dict(self.pooler.model.named_parameters())
        targets = {**parameters, **dict(self.pooler.model.named_buffers())}
        for name, tensor in heads:
            if name not in targets:
                raise ValueError(f"未知 Laya 权重: {name}")
            default_weight_loader(targets[name], tensor)
            loaded.add("pooler.model." + name)
        missing = set(dict(self.named_parameters())) - loaded
        if missing:
            raise ValueError(f"Laya 权重不完整: {sorted(missing)}")
        return loaded
