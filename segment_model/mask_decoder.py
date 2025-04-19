import torch 
import torch.nn as nn 

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


class PromptedMaskDecoder(nn.Module):
    def __init__(self, 
                 image_embed_dim=256,
                 prompt_embed_dim=1024,
                 common_dim=256,
                 num_heads=8,
                 num_layers=4,
                 target_mask_size=(512, 512)):
        super().__init__()
        self.common_dim = common_dim
        self.target_mask_size = target_mask_size

        # Project image embedding to common dimension
        self.image_proj = nn.Conv2d(image_embed_dim, common_dim, kernel_size=1)

        # Project prompt tokens to common dimension
        self.prompt_proj = nn.Linear(prompt_embed_dim, common_dim)

        # Learnable mask token (used for attention fusion like SAM)
        self.mask_token = nn.Parameter(torch.randn(1, 1, common_dim))

        # Transformer layers for fusion
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=common_dim, nhead=num_heads)
            for _ in range(num_layers)
        ])

        # Predict coarse mask logits from the image tokens
        self.mask_head = nn.Sequential(
            nn.Linear(common_dim, common_dim),
            nn.GELU(),
            nn.Linear(common_dim, 1)
        )

        # Refinement CNN to enhance the mask resolution (64→128→256→512)
        self.mask_refiner = nn.Sequential(
            nn.ConvTranspose2d(1, 64, kernel_size=4, stride=2, padding=1),  # 64→128
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 128→256
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 256→512
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)  # final 1-channel output
        )

    def forward(self, image_embedding, prompt_features):
        B, C, H, W = image_embedding.shape  # (B, 256, 64, 64)
        T = prompt_features.size(1)

        # Project image embedding to tokens: (B, 64x64, common_dim)
        image_proj = self.image_proj(image_embedding)  # (B, common_dim, H, W)
        image_tokens = image_proj.flatten(2).transpose(1, 2)  # (B, H*W, common_dim)

        # Project prompts
        prompt_proj = self.prompt_proj(prompt_features)  # (B, T, common_dim)

        # Expand learnable mask token for each batch
        mask_token = self.mask_token.expand(B, -1, -1)  # (B, 1, common_dim)

        # Concatenate all tokens: [image tokens | prompt tokens | mask token]
        tokens = torch.cat([image_tokens, prompt_proj, mask_token], dim=1)  # (B, N+T+1, D)

        # Transformer fusion
        for layer in self.transformer_layers:
            tokens = layer(tokens)

        # Extract only the image tokens back
        image_tokens_out = tokens[:, :H * W, :]  # (B, H*W, common_dim)

        # Predict coarse mask
        mask_logits = self.mask_head(image_tokens_out)  # (B, H*W, 1)
        coarse_mask = mask_logits.view(B, 1, H, W)  # (B, 1, 64, 64)

        # Refine to target resolution
        refined_mask = self.mask_refiner(coarse_mask)  # (B, 1, 512, 512)

        return refined_mask

if __name__ == "__main__":
    image_embedding = torch.randn(1, 256, 64, 64)  # Example image embedding
    prompt_embedding = torch.randn(1, 512, 1024)  # Example prompt features
    model = PromptedMaskDecoder()
    mask = model(image_embedding, prompt_embedding)
    print(mask.shape)  # Should be (1, 1, 64, 64)