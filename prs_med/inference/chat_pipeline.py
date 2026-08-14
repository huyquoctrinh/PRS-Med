"""Multi-turn inference pipeline for v1.5 LLMSegChat."""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from PIL import Image
from torchvision import transforms

from llava.mm_utils import process_images, tokenizer_image_token
from prs_med.models.llm_seg_chat import build_llm_seg_chat

IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"

SAM_TF = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class ChatInferencePipeline:
    """Wraps LLMSegChat with conversation state for multi-turn inference."""

    def __init__(
        self,
        model_path: str,
        checkpoint_path: str,
        device: str = "cuda:0",
        model_base: Optional[str] = None,
    ):
        self.device = device
        self.model, self.tokenizer, self.image_processor, self.config = \
            build_llm_seg_chat(
                model_path=model_path,
                model_base=model_base,
                device=device,
                is_training=False,
            )
        self.tokenizer = self.model.load_checkpoint(checkpoint_path)
        self.model.eval()
        self.history: List[Tuple[str, str]] = []
        self._cache = None

    def set_image(self, image: Image.Image):
        """Encode an image once; reuse across turns."""
        pil = image.convert("RGB")
        vlm = process_images([pil], self.image_processor, self.config).to(
            self.device, torch.float16)
        sam = SAM_TF(pil).unsqueeze(0).to(self.device, torch.float16)
        self._cache = (vlm, sam, pil)
        self.history = []

    def _build_prompt(self, user_msg: str) -> str:
        parts = []
        for i, (q, a) in enumerate(self.history):
            img = DEFAULT_IMAGE_TOKEN + "\n" if i == 0 else ""
            parts.append(f"[INST] {img}{q} [/INST] {a}</s>")
        img = DEFAULT_IMAGE_TOKEN + "\n" if not self.history else ""
        parts.append(f"[INST] {img}{user_msg} [/INST]")
        return "".join(parts)

    def chat(
        self,
        user_msg: str,
        temperature: float = 0.2,
        max_new_tokens: int = 512,
    ) -> Tuple[Optional[torch.Tensor], Optional[list], str]:
        """Generate a reply (and optional masks). Returns (masks, token_ids, text)."""
        if self._cache is None:
            raise RuntimeError("Call set_image() before chat()")
        vlm, sam, _ = self._cache

        prompt = self._build_prompt(user_msg)
        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.device)

        masks, token_ids, output_ids = self.model.generate(
            input_ids=input_ids,
            image_tensor_for_vlm=vlm,
            image_tensor_for_image_enc=sam,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        raw = self.tokenizer.decode(output_ids[0], skip_special_tokens=False)
        for tok in (self.tokenizer.eos_token, self.tokenizer.bos_token,
                    "<unk>", "<s>", "</s>"):
            if tok:
                raw = raw.replace(tok, "")
        reply = raw.strip()

        self.history.append((user_msg, reply))
        return masks, token_ids, reply

    def segment_forced(
        self,
        user_msg: str,
        seg_tokens: List[str],
        shown: str = "",
    ) -> Tuple[Optional[torch.Tensor], List[int]]:
        """Force segmentation by appending [SEG] token(s)."""
        if self._cache is None:
            raise RuntimeError("Call set_image() before segment_forced()")
        vlm, sam, _ = self._cache

        full = self._build_prompt(user_msg) + " " + shown
        ids = tokenizer_image_token(
            full, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            return self.model.segment_forced(ids, vlm, sam, seg_tokens)

    def reset(self):
        """Clear conversation history (keeps the image)."""
        self.history = []
