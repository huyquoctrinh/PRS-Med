"""Version-dispatch model builder.

Reads model_config.version and constructs the correct model variant.
Both versions return the same 4-tuple (model, tokenizer, image_processor, config).
"""
from __future__ import annotations

from typing import Tuple

from prs_med.configs.base import ModelConfig


def build_model(model_config: ModelConfig) -> Tuple:
    """Build the model for the configured version."""
    if model_config.version == "v1":
        from prs_med.models.llm_seg import build_llm_seg
        return build_llm_seg(model_config)
    elif model_config.version == "v1.5":
        from prs_med.models.llm_seg_chat import build_llm_seg_chat
        return build_llm_seg_chat(
            model_path=model_config.model_path,
            model_base=model_config.base_model,
            load_8bit=model_config.load_8bit,
            load_4bit=model_config.load_4bit,
            device=model_config.device,
            is_training=True,
            freeze_image_encoder=model_config.freeze_image_encoder,
        )
    else:
        raise ValueError(f"Unknown model version: {model_config.version!r}. "
                         f"Supported: 'v1', 'v1.5'")
