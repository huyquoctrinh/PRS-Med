from __future__ import annotations

import torch.nn as nn
from tinysam import sam_model_registry


class SamImageEncoder(nn.Module):
    """Thin wrapper around TinySAM image encoder (v1)."""

    def __init__(self, model_type: str, checkpoint_path: str) -> None:
        super().__init__()
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.image_encoder = sam.image_encoder

    def forward(self, inputs):
        return self.image_encoder(inputs)


class MedSamImageEncoder(nn.Module):
    """MedSAM vision encoder (v1.5): (B, 3, 1024, 1024) -> (B, 256, 64, 64)."""

    def __init__(self, model_id: str = "wanglab/medsam-vit-base") -> None:
        super().__init__()
        from transformers import SamModel
        sam = SamModel.from_pretrained(model_id)
        self.image_encoder = sam.vision_encoder

    def forward(self, inputs):
        out = self.image_encoder(inputs)
        if isinstance(out, (tuple, list)):
            return out[0]
        return getattr(out, "last_hidden_state", out)

