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