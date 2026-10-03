"""W8A8 权重格式。"""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

class QuantizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: Literal["laya_w8a8"] = "laya_w8a8"
    version: Literal[1] = 1
    weight_dtype: Literal["int8"] = "int8"
    weight_scheme: Literal["symmetric_per_output_channel"] = "symmetric_per_output_channel"
    weight_layout: Literal["out_in"] = "out_in"
    activation_dtype: Literal["int8"] = "int8"
    activation_scheme: Literal["symmetric_dynamic_per_token"] = "symmetric_dynamic_per_token"
    compute_dtype: Literal["fp16"] = "fp16"
    quantized_modules: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_modules(self):
        if len(self.quantized_modules) != len(set(self.quantized_modules)):
            raise ValueError("duplicate_quantized_modules")
        return self


def read_quantization(config):
    return QuantizationConfig.model_validate(config["quantization"]) if "quantization" in config else None
