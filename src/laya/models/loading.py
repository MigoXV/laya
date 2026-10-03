"""模型构造兼容性与参数精度。"""
from transformers import AutoConfig

def move_model(model, device, dtype):
    """转换参数精度，保留 RoPE 频率等浮点缓冲区的原始精度。"""
    buffers = {
        name: buffer for name, buffer in model.named_buffers()
        if buffer.is_floating_point()
    }
    model.to(device=device, dtype=dtype)
    for name, buffer in buffers.items():
        owner, _, attribute = name.rpartition(".")
        setattr(model.get_submodule(owner), attribute, buffer.to(device=device))
    return model


def encoder_config_for_runtime(raw):
    encoder_config = dict(raw)
    # Transformers 4.x reads legacy theta fields; 5.x writes rope_parameters.
    # Preserve the asset's position encoding instead of silently using defaults.
    rope = encoder_config.get("rope_parameters", {})
    for kind, legacy in (("full_attention", "global_rope_theta"), ("sliding_attention", "local_rope_theta")):
        if kind in rope:
            if rope[kind].get("rope_type", "default") != "default":
                raise ValueError("unsupported native RoPE scaling")
            encoder_config[legacy] = rope[kind]["rope_theta"]
    return AutoConfig.for_model(**encoder_config)
