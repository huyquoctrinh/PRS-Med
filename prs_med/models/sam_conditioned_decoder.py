"""SAM-style conditioning decoder for [SEG]-driven segmentation (LISA-style).

Replaces PromptedMaskDecoder (mask_decoder_v5), whose output went entirely NaN
under bf16 autocast -- it ran a TransformerEncoderLayer over 4096 image tokens,
which is both unstable and not what SAM does.

Here the [SEG] hidden state is projected to a SAM *sparse prompt token* and fed
to MedSAM's own pretrained two-way-attention mask decoder, which is designed
exactly for this: sparse prompt + dense image embedding -> mask.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SamConditionedMaskDecoder(nn.Module):
    """(B,256,64,64) image embedding + (B,1,4096) [SEG] state -> (B,1,1024,1024).

    The only new parameters are the 4096->256 projection; the decoder itself
    starts from MedSAM's pretrained weights.
    """

    def __init__(self, sam_model_id="wanglab/medsam-vit-base",
                 prompt_dim=4096, sam_dim=256, out_size=1024):
        super().__init__()
        from transformers import SamModel

        sam = SamModel.from_pretrained(sam_model_id)
        self.prompt_encoder = sam.prompt_encoder
        self.mask_decoder = sam.mask_decoder
        self.out_size = out_size

        # Positional embedding is fixed for a 64x64 grid -- compute once so we
        # don't have to keep the whole SamModel alive just for this method.
        with torch.no_grad():
            pe = sam.get_image_wide_positional_embeddings()   # (1, 256, 64, 64)
        self.register_buffer("image_pe", pe, persistent=False)

        # The trainable bridge from LLM hidden space to SAM prompt space.
        self.text_proj = nn.Sequential(
            nn.Linear(prompt_dim, sam_dim),
            nn.ReLU(inplace=True),
            nn.Linear(sam_dim, sam_dim),
        )
        nn.init.xavier_uniform_(self.text_proj[0].weight)
        nn.init.zeros_(self.text_proj[0].bias)
        nn.init.xavier_uniform_(self.text_proj[2].weight)
        nn.init.zeros_(self.text_proj[2].bias)

    def forward(self, image_embeddings, seg_embeds):
        """image_embeddings: (B, 256, 64, 64); seg_embeds: (B, 1, prompt_dim)."""
        # Run in fp32 regardless of the surrounding autocast: the previous
        # decoder's NaNs came from bf16 attention, and this path is cheap.
        with torch.autocast(device_type=image_embeddings.device.type, enabled=False):
            image_embeddings = image_embeddings.float()
            seg = seg_embeds.float()
            if seg.dim() == 2:                      # (B, D) -> (B, 1, D)
                seg = seg.unsqueeze(1)

            b = image_embeddings.shape[0]
            # SAM expects sparse prompts as (B, point_batch, n_points, sam_dim)
            sparse = self.text_proj(seg).unsqueeze(1)          # (B, 1, 1, 256)
            dense = self.prompt_encoder.no_mask_embed.weight.reshape(
                1, -1, 1, 1).expand(b, -1, *image_embeddings.shape[-2:]).float()

            decoder_out = self.mask_decoder(
                image_embeddings=image_embeddings,
                image_positional_embeddings=self.image_pe.to(
                    image_embeddings.dtype),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )
            # Return arity varies across transformers versions; masks are first.
            low_res = decoder_out[0] if isinstance(
                decoder_out, (tuple, list)) else decoder_out
            # (B, point_batch, num_masks, 256, 256) -> (B, 1, 256, 256)
            low_res = low_res[:, 0]
            masks = F.interpolate(
                low_res, size=(self.out_size, self.out_size),
                mode="bilinear", align_corners=False,
            )
        return masks                                # logits, (B, 1, 1024, 1024)
