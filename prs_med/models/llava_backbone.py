from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from peft import LoraConfig, TaskType, get_peft_model

from prs_med.configs.base import ModelConfig


def _custom_lora_init(module: nn.Module) -> None:
    if hasattr(module, "lora_A"):
        nn.init.kaiming_uniform_(module.lora_A.weight, a=5 ** 0.5)
    if hasattr(module, "lora_B"):
        nn.init.zeros_(module.lora_B.weight)


def build_llava_backbone(
    model_config: ModelConfig,
) -> Tuple:
    """Create tokenizer, PEFT-wrapped model, image processor, and context length."""
    disable_torch_init()
    device = torch.device(model_config.device)

    model_name = get_model_name_from_path(model_config.model_path)
    tokenizer, base_model, image_processor, context_len = load_pretrained_model(
        model_config.model_path,
        model_config.base_model,
        model_name,
        model_config.load_8bit,
        model_config.load_4bit,
        device=device,
    )

    lora_kwargs = {}
    if model_config.lora_modules_to_save:
        lora_kwargs["modules_to_save"] = model_config.lora_modules_to_save

    lora_config = LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        lora_dropout=model_config.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
        target_modules=model_config.lora_target_modules,
        **lora_kwargs,
    )

    peft_model = get_peft_model(base_model, lora_config)
    peft_model.apply(_custom_lora_init)
    return tokenizer, peft_model, image_processor, context_len, base_model


