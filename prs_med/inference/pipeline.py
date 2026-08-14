from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional, Tuple

import torch
from torchvision import transforms
from llava.mm_utils import process_images, tokenizer_image_token

from prs_med.configs.base import ModelConfig
from prs_med.data.utils import load_image
from prs_med.models.llm_seg import build_llm_seg


class InferencePipeline:
    """Reusable inference helper wrapping preprocessing and generation."""

    def __init__(
        self,
        model_config: ModelConfig,
        checkpoint_path: str,
        device: Optional[str] = None,
        merge_lora: bool = True,
    ) -> None:
        device_str = device or model_config.device
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        config = replace(model_config, device=str(self.device))

        self.model, self.tokenizer, self.image_processor, self.processor_config = build_llm_seg(config)
        self.model.to(self.device)
        self.model.load_checkpoint(checkpoint_path, merge_lora=merge_lora)
        self.model.eval()

        self.mask_size = (1024, 1024)
        self.sam_transform = transforms.Compose(
            [
                transforms.Resize(self.mask_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _process_for_vlm(self, image_path: str) -> torch.Tensor:
        image = load_image(image_path)
        tensor = process_images([image], self.image_processor, self.processor_config)
        return tensor.to(self.device, dtype=torch.float16)

    def _process_for_sam(self, image_path: str) -> torch.Tensor:
        image = load_image(image_path)
        tensor = self.sam_transform(image)
        return tensor.unsqueeze(0).to(self.device, dtype=torch.float32)

    def _tokenize(self, prompt: str) -> torch.Tensor:
        tokens = tokenizer_image_token(
            prompt,
            self.tokenizer,
            -200,
            return_tensors="pt",
        )
        return tokens.to(self.device, dtype=torch.int64)

    def predict(
        self,
        prompt: str,
        image_path: str,
        segmentation_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_new_tokens: int = 512,
        top_p: float = 0.95,
    ) -> Tuple[torch.Tensor, str]:
        image_tensor = self._process_for_vlm(image_path)
        sam_tensor = self._process_for_sam(image_path)

        prompt_tokens = self._tokenize(f"<image>\n### User: {prompt}\n")
        seg_prompt = segmentation_prompt or prompt
        seg_tokens = self._tokenize("<image>\n" + seg_prompt)

        masks, generated_ids = self.model.generate(
            input_ids=prompt_tokens,
            input_ids_for_seg=seg_tokens,
            image_tensor_for_vlm=image_tensor,
            image_tensor_for_image_enc=sam_tensor,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
        )

        decoded_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        normalized_mask = self._normalize_mask(masks)
        return normalized_mask, decoded_text

    @staticmethod
    def _normalize_mask(mask_tensor: torch.Tensor) -> torch.Tensor:
        mask = mask_tensor.sigmoid().detach().cpu()
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        return mask.squeeze(0)

