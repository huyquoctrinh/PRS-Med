from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from peft import PeftModel

from prs_med.configs.base import ModelConfig
from prs_med.models.llava_backbone import build_llava_backbone
from prs_med.models.mask_decoder import PromptedMaskDecoder
from prs_med.models.sam_encoder import SamImageEncoder
from prs_med.models.seg_head import SegmentationClassifier


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


class LLMSeg(nn.Module):
    """Unified segmentation model combining LLaVA, SAM encoder, and mask decoder."""

    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()
        self.config = model_config
        (
            tokenizer,
            llava_model,
            image_processor,
            context_len,
            base_model,
        ) = build_llava_backbone(model_config)

        self.tokenizer = tokenizer
        self.llava_model = llava_model
        self.image_processor = image_processor
        self.context_len = context_len
        self.base_model = base_model

        self.target_dtype = _resolve_dtype(model_config.dtype)
        self.llava_model.to(dtype=self.target_dtype)

        self.mask_decoder = PromptedMaskDecoder()
        self.sam_encoder = SamImageEncoder(model_config.sam_model_type, model_config.sam_checkpoint)
        self.classifier = SegmentationClassifier(in_channels=256, num_classes=model_config.num_classes)

    def get_model_utils(self):
        return self.tokenizer, self.image_processor, self.context_len, self.base_model.config

    def save_checkpoint(self, output_path: str) -> None:
        path = Path(output_path)
        path.mkdir(parents=True, exist_ok=True)

        self.llava_model.save_pretrained(path / "lora_adapter")
        self.tokenizer.save_pretrained(path / "tokenizer")
        torch.save(self.sam_encoder.state_dict(), path / "image_encoder.pth")
        torch.save(self.mask_decoder.state_dict(), path / "mask_decoder.pth")
        torch.save(self.classifier.state_dict(), path / "classifier.pth")

    def load_checkpoint(self, checkpoint_path: str, merge_lora: bool = False) -> None:
        checkpoint_dir = Path(checkpoint_path)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

        self.tokenizer = self.tokenizer.from_pretrained(checkpoint_dir / "tokenizer")
        self.mask_decoder.load_state_dict(torch.load(checkpoint_dir / "mask_decoder.pth"))
        self.sam_encoder.load_state_dict(torch.load(checkpoint_dir / "image_encoder.pth"))
        self.classifier.load_state_dict(torch.load(checkpoint_dir / "classifier.pth"))

        self.llava_model = PeftModel.from_pretrained(self.llava_model, checkpoint_dir / "lora_adapter")
        self.llava_model.to(dtype=self.target_dtype)
        self.llava_model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        if merge_lora:
            self.llava_model = self.llava_model.merge_and_unload()
            self.llava_model.to(dtype=self.target_dtype)
            self.base_model = self.llava_model
        else:
            self.base_model = self.llava_model.base_model

    def forward(
        self,
        input_ids: torch.Tensor,
        image_tensor_for_vlm: torch.Tensor,
        image_tensor_for_image_enc: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        answers: Optional[torch.Tensor] = None,
    ):
        self.llava_model.to(dtype=self.target_dtype)
        if self.training:
            self.llava_model.train()
            self.sam_encoder.train()
            self.mask_decoder.train()
            self.classifier.train()
        else:
            self.llava_model.eval()
            self.sam_encoder.eval()
            self.mask_decoder.eval()
            self.classifier.eval()

        outputs = self.llava_model(
            input_ids=answers if answers is not None else input_ids,
            attention_mask=attention_mask,
            images=image_tensor_for_vlm,
            use_cache=False,
            labels=answers if answers is not None else input_ids,
            return_dict=True,
            output_hidden_states=True,
        )

        prompt_embedding = outputs.hidden_states[-1]
        image_embedding = self.sam_encoder(image_tensor_for_image_enc)
        cls_logits = self.classifier(image_embedding)
        masks = self.mask_decoder(image_embedding, prompt_embedding)

        if self.training:
            return masks, cls_logits, outputs.loss

        return masks, outputs.logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        image_tensor_for_vlm: torch.Tensor,
        image_tensor_for_image_enc: torch.Tensor,
        input_ids_for_seg: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 0.1,
        max_new_tokens: int = 512,
        top_p: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        device = next(self.parameters()).device

        generated_ids = self.llava_model.generate(
            inputs=input_ids.to(device),
            images=image_tensor_for_vlm.to(device),
            attention_mask=attention_mask.to(device) if attention_mask is not None else None,
            do_sample=temperature > 0,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
        )

        embedding_input = input_ids_for_seg if input_ids_for_seg is not None else input_ids
        prompt_embedding = self.base_model.extract_last_hidden_state(
            input_ids=embedding_input.to(device),
            images=image_tensor_for_vlm.to(device),
            do_sample=False,
            temperature=0,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
        )["hidden_states"][-1]

        image_embedding = self.sam_encoder(image_tensor_for_image_enc.to(device))
        masks = self.mask_decoder(image_embedding, prompt_embedding)

        return masks, generated_ids


def build_llm_seg(model_config: ModelConfig) -> Tuple[LLMSeg, Any, Any, Any]:
    model = LLMSeg(model_config)
    tokenizer, image_processor, context_len, base_config = model.get_model_utils()
    return model, tokenizer, image_processor, base_config

