import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
class MaskRefinerWithDeepSupervision(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(1, 64, kernel_size=4, stride=2, padding=1)   # 64 → 128
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # 128 → 256
        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)  # 256 → 512
        self.final = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x1 = self.relu(self.up1(x))  # 64 → 128
        x2 = self.relu(self.up2(x1))  # 128 → 256
        x3 = self.relu(self.up3(x2))  # 256 → 512
        x4 = self.final(x3)  # final 1-channel output
        return self.sigmoid(x4)  # Apply sigmoid to get the final mask


# class PromptedMaskDecoder(nn.Module):
#     def __init__(self, 
#                  image_embed_dim=256,
#                  prompt_embed_dim=4096,
#                  common_dim=256,
#                  num_heads=8,
#                  num_layers=4,
#                  target_mask_size=(512, 512)):
#         super().__init__()
#         self.common_dim = common_dim
#         self.target_mask_size = target_mask_size

#         # Project image embedding to common dimension
#         self.image_proj = nn.Conv2d(image_embed_dim, common_dim, kernel_size=1)

#         # Project prompt tokens to common dimension
#         self.prompt_proj = nn.Linear(prompt_embed_dim, common_dim)

#         self.mask_token = nn.Parameter(torch.randn(1, 1, common_dim), requires_grad=True)

#         # Transformer layers for fusion
#         self.transformer_layers = nn.ModuleList([
#             nn.TransformerEncoderLayer(d_model=common_dim, nhead=num_heads)
#             for _ in range(num_layers)
#         ])

#         # Predict coarse mask logits from the image tokens
#         self.mask_head = nn.Sequential(
#             nn.Linear(common_dim, common_dim),
#             nn.GELU(),
#             nn.Linear(common_dim, 1)
#         )

#         # Refinement CNN to enhance the mask resolution (64→128→256→512)
#         self.mask_refiner = nn.Sequential(
#             nn.ConvTranspose2d(1, 64, kernel_size=4, stride=2, padding=1),  # 64→128
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 128→256
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 256→512
#             nn.ReLU(inplace=True),
#             nn.Conv2d(16, 1, kernel_size=3, padding=1)  # final 1-channel output
#         )

#     def forward(self, image_embedding, prompt_features):
#         B, C, H, W = image_embedding.shape  # (B, 256, 64, 64)
#         T = prompt_features.size(1)
#         image_embedding = image_embedding.to(torch.float32)  # Ensure image embedding is float32
#         prompt_features = prompt_features.to(torch.float32)  # Ensure prompt features are float32
#         print(image_embedding.dtype)
#         print("=============")
#         print(prompt_features.dtype)
#         image_proj = self.image_proj(image_embedding)  # (B, common_dim, H, W)
#         image_tokens = image_proj.flatten(2).transpose(1, 2)  # (B, H*W, common_dim)
#         print(image_tokens)
#         prompt_proj = self.prompt_proj(prompt_features)  # (B, T, common_dim)
#         print(prompt_proj)
#         # Expand learnable mask token for each batch
#         mask_token = self.mask_token.expand(B, -1, -1)  # (B, 1, common_dim)
#         print(mask_token)
#         tokens = torch.cat([image_tokens, prompt_proj, mask_token], dim=1)  # (B, N+T+1, D)
#         # print(tokens)
#         # Transformer fusion
#         for layer in self.transformer_layers:
#             tokens = layer(tokens)

#         # Extract only the image tokens back
#         image_tokens_out = tokens[:, :H * W, :]  # (B, H*W, common_dim)
#         # print(image_tokens_out)
#         # Predict coarse mask
#         mask_logits = self.mask_head(image_tokens_out)  # (B, H*W, 1)
#         coarse_mask = mask_logits.view(B, 1, H, W)  # (B, 1, 64, 64)

#         # Refine to target resolution
#         refined_mask = self.mask_refiner(coarse_mask)  # (B, 1, 512, 512)

#         return refined_mask

class SkipConnection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SkipConnection, self).__init__()
        self.ln1 = nn.Linear(in_channels, out_channels)
        self.ln2 = nn.Linear(out_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        # self.batch_norm = nn.BatchNorm1d(out_channels)
        self.skip = nn.Linear(out_channels, out_channels)
    def forward(self, x):
        x = self.ln1(x)
        x = self.relu(x)
        x = self.ln2(x)
        # x = self.batch_norm(x)
        x = self.relu(x)
        skip = self.skip(x)
        x = x + skip
        x = self.relu(x)
        return x

class PromptedMaskDecoder(nn.Module):
    def __init__(self, prompt_dim=4096, image_dim=256, hidden_dim=512):
        super().__init__()

        self.prompt_projection = nn.Sequential(
            nn.Linear(prompt_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps = 1e-5),
            nn.ReLU(),
            nn.Linear(hidden_dim, image_dim),
            nn.LayerNorm(image_dim, eps = 1e-5),
            nn.ReLU()
        )

        for layer in self.prompt_projection:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.LayerNorm):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

        # cross-attention: image tokens attend to prompt
        self.attn = nn.MultiheadAttention(embed_dim=image_dim, num_heads=8, batch_first=True)

        nn.init.xavier_uniform_(self.attn.in_proj_weight)
        nn.init.xavier_uniform_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.in_proj_bias)
        nn.init.zeros_(self.attn.out_proj.bias)

        self.decoder = nn.Sequential(
            nn.Conv2d(image_dim, image_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(image_dim // 2, 1, kernel_size=1)
        )

    def forward(self, image_feat, prompt_feat):
        """
        image_feat: (B, 256, 64, 64) - float32
        prompt_feat: (B, T, 2048) - float16
        """
        B, _, H, W = image_feat.shape
        T = prompt_feat.shape[1]

        prompt_feat = prompt_feat.float()
        # print("prompt_proj before nan or inf:", torch.isnan(prompt_feat).any(), torch.isinf(prompt_feat).any())
        
        prompt_proj = self.prompt_projection(prompt_feat)  # (B, T, hidden_dim)
        # print("prompt_proj nan or inf:", torch.isnan(prompt_proj).any(), torch.isinf(prompt_proj).any())
        # print("position of nan:", torch.where(torch.isnan(prompt_proj)))
        image_flat = image_feat.flatten(2).transpose(1, 2)  # (B, H*W, hidden_dim)
        attn_out, _ = self.attn(image_flat, prompt_proj, prompt_proj)  # (B, H*W, hidden_dim)
        # print("image_proj nan or inf:", torch.isnan(image_feat).any(), torch.isinf(image_feat).any())

        # print("attn_out nan or inf:", torch.isnan(attn_out).any(), torch.isinf(attn_out).any())

        attn_map = attn_out.transpose(1, 2).reshape(B, -1, H, W)  # (B, hidden_dim, H, W)

        mask = self.decoder(attn_map)  # (B, 1, H, W)
        mask = F.interpolate(mask, scale_factor=8, mode='bilinear')
        # print(mask.shape)
        return mask

# class PromptedMaskDecoder(nn.Module):
#     def __init__(self, 
#                  image_embed_dim=256,
#                  prompt_embed_dim=768,
#                  common_dim=256,
#                  num_heads=8,
#                  num_layers=4,
#                  target_mask_size=(512, 512)):
#         super().__init__()
#         self.common_dim = common_dim
#         self.target_mask_size = target_mask_size

#         # Project image and prompt into shared token space
#         self.image_proj = nn.Sequential(
#             nn.Conv2d(image_embed_dim, common_dim, kernel_size=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(common_dim, common_dim, kernel_size=1),
#         )
        
#         self.prompt_proj = nn.Sequential(
#             nn.Linear(prompt_embed_dim, common_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(common_dim, common_dim)
#         )
#         # Transformer layers (no mask token)
#         decoder_layer = nn.TransformerDecoderLayer(
#             d_model=common_dim,
#             nhead=num_heads,
#             dim_feedforward=512,
#             dropout=0.1,
#             activation='relu',
#             batch_first=True
#         )
#         self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
#         self.ln1 = nn.LayerNorm(common_dim)
#         self.ln2 = nn.LayerNorm(common_dim)
#         # self.cross_attn = nn.MultiheadAttention(embed_dim=common_dim, num_heads=num_heads, batch_first=True)

#         # Predict coarse mask from image tokens
#         self.mask_head = nn.Sequential(
#             nn.Linear(common_dim, common_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(common_dim, 1)
#         )

#         # Refinement: upsample 64 → 128 → 256 → 512
#         # self.mask_refiner = nn.Sequential(
#         #     nn.ConvTranspose2d(1, 64, kernel_size=4, stride=2, padding=1),  # 128
#         #     nn.ReLU(inplace=True),
#         #     nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 256
#         #     nn.ReLU(inplace=True),
#         #     nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 512
#         #     nn.ReLU(inplace=True),
#         #     nn.Conv2d(16, 1, kernel_size=3, padding=1)
#         # )
#         self.batch_norm = nn.BatchNorm2d(image_embed_dim)
#     def forward(self, image_embedding, prompt_features):
#         B, C, H, W = image_embedding.shape
#         image_embedding = image_embedding.to(torch.float32)  
#         prompt_features = prompt_features.to(torch.float32)  
#         # prompt_features = F.normalize(prompt_features, p=2, dim=-1)  # Normalize prompt features
#         # print(prompt_features.shape)
#         # print(normalized_image_embedding)
#         # print("=============")
#         # print("=============")
#         image_tokens = self.image_proj(image_embedding)  # (B, common_dim, H, W)
#         # image_tokens = self.batch_norm(image_tokens)
#         # image_tokens = nn.ReLU()(image_tokens)
#         image_tokens = image_tokens.flatten(2).transpose(0, 2, 1)  # (B, H*W, common_dim)

#         prompt_proj = self.prompt_proj(prompt_features)  # (B, T, common_dim)
#         # prompt_proj = nn.ReLU()(prompt_proj)
#         # prompt_proj = nn.Tanh()(prompt_proj)
        
#         print("Prompt proj before:",torch.mean(prompt_proj, dim=1))
#         print("=======================")
#         image_tokens = self.ln1(image_tokens)
#         prompt_proj_norm = self.ln2(prompt_proj)
#         # prompt_proj_norm = torch.clamp(prompt_proj, -1e4, 1e4)
#         print("=============")
#         print("Mean image token:",torch.mean(image_tokens, dim=1))
#         print("Mean image token type:",image_tokens.dtype)
#         print("=============")
#         print("Mean prompt token:",torch.mean(prompt_proj_norm, dim=1))
#         print("Mean prompt token type:",prompt_proj_norm.dtype)
#         decoded_tokens = self.decoder(tgt=image_tokens, memory=prompt_proj_norm)
#         print("Mean decoded token:",torch.mean(decoded_tokens, dim=1))
#         print("decoded_tokens type:",decoded_tokens.dtype)
#         mask_logits = self.mask_head(decoded_tokens)  # (B, H*W, 1)
#         coarse_mask = mask_logits.transpose(1, 2).view(B, 1, H, W)
#         refined_mask = F.interpolate(coarse_mask, scale_factor=8, mode='bilinear')
#         return refined_mask

if __name__ == "__main__":
    image_embedding = torch.randn(1, 256, 64, 64)  # Example image embedding
    prompt_embedding = torch.randn(1, 512, 1024)  # Example prompt features
    model = PromptedMaskDecoder()
    mask = model(image_embedding, prompt_embedding)
    print(mask.shape)  # Should be (1, 1, 64, 64)