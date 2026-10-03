"""配置在资源创建前校验。"""

from pathlib import Path
from typing import Literal
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LAYA_", env_file=".env", extra="ignore"
    )
    model_dir: Path
    device: str = "cuda:0"
    dtype: Literal["fp16", "bf16", "fp32"] = "fp16"
    runner: Literal["eager", "cuda-graph", "cuda-graph-compile"] = "eager"
    host: str = "0.0.0.0"
    port: int = Field(default=10002, ge=1, le=65535)
    max_inflight: int = Field(default=64, ge=1, le=256)
    queue_size: int = Field(default=32, ge=1, le=256)
    max_batch_size: int = Field(default=1, ge=1, le=16)
    batch_wait_ms: float = Field(default=2, ge=0, le=10)
    max_batch_tokens: int = Field(default=8192, ge=1024, le=16384)
    graph_cache_size: int = Field(default=16, ge=1, le=32)
    graph_streams: int = Field(default=8, ge=1, le=8)
    graph_prewarm_profiles: list[tuple[int, int, int]] = Field(default_factory=list, max_length=32)
    compile_cache_size: int = Field(default=32, ge=1, le=64)
    request_timeout: float = Field(default=30, gt=0, le=300)
    startup_timeout: float = Field(default=180, gt=0)
    shutdown_timeout: float = Field(default=5, gt=0)
    max_body_bytes: int = Field(default=262144, ge=1024, le=1048576)
    threads: int = Field(default=4, ge=1, le=64)

    @model_validator(mode="after")
    def valid_source(self):
        for filename in (
            "model.safetensors",
            "config.json",
            "tokenizer.json",
        ):
            if not (self.model_dir / filename).is_file():
                raise ValueError(f"本地模型缺少 {filename}: {self.model_dir}")
        if self.device != "cpu" and not (
            self.device.startswith("cuda:") and self.device[5:].isdigit()
        ):
            raise ValueError("device 必须显式指定 cpu 或 cuda:N")
        if self.device == "cpu" and self.dtype == "fp16":
            raise ValueError("FP16 推理使用 CUDA；CPU 推理请显式指定 dtype=fp32")
        if self.runner.startswith("cuda-graph") and self.device == "cpu":
            raise ValueError("cuda-graph 需要 CUDA 设备")
        if self.graph_prewarm_profiles:
            if not self.runner.startswith("cuda-graph"):
                raise ValueError("graph_prewarm_profiles 仅用于 CUDA Graph runner")
            if len(self.graph_prewarm_profiles) > self.graph_cache_size:
                raise ValueError("预热 profile 数量超过 Graph 缓存容量")
            for batch, length, options in self.graph_prewarm_profiles:
                if (batch not in (1, 2, 4, 8, 16) or batch > self.max_batch_size
                        or not 2 <= options <= 16 or length < options
                        or batch * length > self.max_batch_tokens):
                    raise ValueError("invalid_graph_prewarm_profile")
        return self
