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
    device: str = "cpu"
    dtype: Literal["fp32"] = "fp32"
    host: str = "0.0.0.0"
    port: int = Field(default=10002, ge=1, le=65535)
    max_inflight: int = Field(default=64, ge=1, le=256)
    queue_size: int = Field(default=32, ge=1, le=256)
    request_timeout: float = Field(default=30, gt=0, le=300)
    startup_timeout: float = Field(default=180, gt=0)
    shutdown_timeout: float = Field(default=5, gt=0)
    max_body_bytes: int = Field(default=262144, ge=1024, le=1048576)
    threads: int = Field(default=4, ge=1, le=64)

    @model_validator(mode="after")
    def valid_source(self):
        for filename in (
            "model.safetensors",
            "rl_agent_config.json",
            "encoder/config.json",
            "tokenizer/tokenizer.json",
        ):
            if not (self.model_dir / filename).is_file():
                raise ValueError(f"本地模型缺少 {filename}: {self.model_dir}")
        if self.device != "cpu" and not (
            self.device.startswith("cuda:") and self.device[5:].isdigit()
        ):
            raise ValueError("device 必须显式指定 cpu 或 cuda:N")
        return self
