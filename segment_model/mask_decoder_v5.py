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
        self.ln1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.ln2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.ln1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.ln2(x)
        x = self.bn2(x)
        # x = self.batch_norm(x)
        x = self.relu(x)
        skip = self.skip(x)
        x = x + skip
        # x = self.relu(x)
        return x
    
class MaskDecoder(nn.Module):
    def __init__(self, in_channels):
        super(MaskDecoder, self).__init__()
        self.upconv1 = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=4, stride=2, padding=1)  # 64 → 128
        self.upconv2 = nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=4, stride=2, padding=1)  # 128 → 256
        self.upconv3 = nn.ConvTranspose2d(in_channels // 4, in_channels // 8, kernel_size=4, stride=2, padding=1)  # 256 → 512
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.norm2 = nn.BatchNorm2d(in_channels // 2)
        self.norm3 = nn.BatchNorm2d(in_channels // 4)
        self.final_conv = nn.Conv2d(in_channels // 8, 1, kernel_size=3, padding=1)  # final 1-channel output
        self.relu = nn.ReLU(inplace=True)
        self.skip1 = SkipConnection(in_channels, in_channels)
        self.skip2 = SkipConnection(in_channels // 2, in_channels // 2)
        self.skip3 = SkipConnection(in_channels // 4, in_channels // 4)

    def forward(self, x):
        x = self.skip1(x)
        x = self.relu(self.norm1(x))
        x = self.upconv1(x)  # 64 → 128
        x = self.skip2(x)
        x = self.relu(self.norm2(x))
        x = self.upconv2(x)  # 128 → 256
        x = self.skip3(x)
        x = self.relu(self.norm3(x))
        x = self.upconv3(x)  # 256 → 512
        x = self.final_conv(x)  # final 1-channel output
        return x 

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.InstanceNorm2d(in_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=1)
        self.bn2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0)

    def weight_init(self):
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        if self.conv1.bias is not None:
            torch.nn.init.zeros_(self.conv1.bias)
        if self.conv2.bias is not None:
            torch.nn.init.zeros_(self.conv2.bias)
        if self.conv3.bias is not None:
            torch.nn.init.zeros_(self.conv3.bias)
        torch.nn.init.ones_(self.bn1.weight)
        torch.nn.init.zeros_(self.bn1.bias)
        torch.nn.init.ones_(self.bn2.weight)
        torch.nn.init.zeros_(self.bn2.bias)

    def forward(self, x):
        identity = x
        # print("X shape:",x.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = out + identity
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv3(out)
        # out = self.relu(out)
        return out

class Adapter(nn.Module):
    def __init__(self, in_features, out_features, hidden_feature):
        super(Adapter, self).__init__()
        self.conv1 = nn.Conv2d(in_features, hidden_feature, kernel_size=1)
        self.norm1 = nn.InstanceNorm2d(hidden_feature, affine=True)
        self.conv2 = nn.Conv2d(hidden_feature, hidden_feature, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(hidden_feature, affine=True)
        self.conv3 = nn.Conv2d(hidden_feature, out_features, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        # self.weight_init()

    def weight_init(self):
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        if self.conv1.bias is not None:
            nn.init.zeros_(self.conv1.bias)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)
        if self.conv3.bias is not None:
            nn.init.zeros_(self.conv3.bias)
        nn.init.ones_(self.norm1.weight)
        nn.init.zeros_(self.norm1.bias)
        nn.init.ones_(self.norm2.weight)
        nn.init.zeros_(self.norm2.bias)
        nn.init.ones_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)
        # nn.init.xavier_uniform_(self.conv3.weight)
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        identity = x 
        x1 = self.conv2(x)
        x1 = self.norm2(x1)
        x1 = self.relu(x1)
        x1 = x1 + identity
        x2 = self.conv3(x1)
        return x2

class FFN(nn.Module):
    def __init__(self, in_features, out_features):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(out_features, in_features)
        self.norm = nn.LayerNorm(in_features, eps=1e-5)
        self.norm1 = nn.LayerNorm(in_features, eps=1e-5)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)

    def forward(self, x):
        identity = x
        x = self.fc1(x)        
        x = self.act(x)
        x = x + identity
        x = self.fc2(x)
        x = self.act(x)
        x= self.norm(x)
        # x = x + norm_out
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
                    nn.init.ones_(layer.bias)
            elif isinstance(layer, nn.LayerNorm):
                nn.init.ones_(layer.weight)
                nn.init.ones_(layer.bias)

        # cross-attention: image tokens attend to prompt
        self.attn = nn.MultiheadAttention(embed_dim=image_dim, num_heads=8, batch_first=True)
        self.norm1 = nn.LayerNorm(image_dim, eps=1e-5)
        nn.init.xavier_uniform_(self.attn.in_proj_weight)
        nn.init.xavier_uniform_(self.attn.out_proj.weight)
        nn.init.ones_(self.attn.in_proj_bias)
        nn.init.ones_(self.attn.out_proj.bias)

        # self.adapter = Adapter(image_dim, image_dim, image_dim)
        self.ffn = FFN(image_dim, image_dim)

        torch.nn.init.xavier_uniform_(self.ffn.fc1.weight)
        torch.nn.init.ones_(self.ffn.fc1.bias)

        # self.decoder = nn.Sequential(
        #     BasicBlock(image_dim, image_dim // 2),
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #     BasicBlock(image_dim //2, image_dim // 4),
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #     BasicBlock(image_dim // 4, image_dim // 8),
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #     BasicBlock(image_dim // 8, image_dim // 16),
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        # )

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=16, batch_first=True, activation = "relu", dim_feedforward=512)
        torch.nn.init.xavier_uniform_(self.encoder_layer.self_attn.in_proj_weight)
        torch.nn.init.xavier_uniform_(self.encoder_layer.self_attn.out_proj.weight)
        torch.nn.init.ones_(self.encoder_layer.self_attn.in_proj_bias)
        torch.nn.init.ones_(self.encoder_layer.self_attn.out_proj.bias)
        torch.nn.init.xavier_uniform_(self.encoder_layer.linear1.weight)
        torch.nn.init.ones_(self.encoder_layer.linear1.bias)
        torch.nn.init.xavier_uniform_(self.encoder_layer.linear2.weight)
        torch.nn.init.ones_(self.encoder_layer.linear2.bias)

        self.mask_generation = nn.Sequential(
            BasicBlock(image_dim, image_dim),
            BasicBlock(image_dim, image_dim // 2),
            BasicBlock(image_dim // 2, image_dim // 4),
        )

        # self.conv1 = nn.Conv2d(image_dim, image_dim, kernel_size=3, padding=1, stride = 1)
        # self.gen1 = nn.Sequential(
        #     nn.Linear(image_dim, image_dim),
        #     nn.ReLU(),
        #     nn.Linear(image_dim, image_dim),
        # )
        # self.up1 = 
        self.relu = nn.ReLU(inplace=True)
        self.out_dec = nn.Conv2d(image_dim // 4, 1, 1)

        # self.decoder = MaskDecoder(image_dim)

    def forward(self, image_feat, prompt_feat):
        """
        image_feat: (B, 256, 64, 64) - float32
        prompt_feat: (B, T, 2048) - float16
        """
        B, _, H, W = image_feat.shape
        T = prompt_feat.shape[1]

        prompt_feat = prompt_feat.float()
        # print("prompt_proj before nan or inf:", torch.isnan(prompt_feat).any(), torch.isinf(prompt_feat).any())
        image_identity = image_feat
        # image_feat = self.adapter(image_feat)
        prompt_proj = self.prompt_projection(prompt_feat)  # (B, T, hidden_dim)
        # print("prompt_proj nan or inf:", torch.isnan(prompt_proj).any(), torch.isinf(prompt_proj).any())
        # print("position of nan:", torch.where(torch.isnan(prompt_proj)))
        image_flat = image_feat.flatten(2).transpose(1, 2)  # (B, H*W, hidden_dim)
        attn_out, _ = self.attn(image_flat, prompt_proj, prompt_proj)  # (B, H*W, hidden_dim)
        # print("image_proj nan or inf:", torch.isnan(image_feat).any(), torch.isinf(image_feat).any())
         # (B, H*W, hidden_dim)
        # print("attn_out nan or inf:", torch.isnan(attn_out).any(), torch.isinf(attn_out).any())
        attn_out = self.norm1(attn_out)  # (B, H*W, hidden_dim)
        attn_out = self.encoder_layer(attn_out) 
        attn_out = self.ffn(attn_out)  # (B, H*W, hidden_dim)
        attn_map = attn_out.transpose(1, 2).reshape(B, -1, H, W)  # (B, hidden_dim, H, W)
        # print(attn_map.shape)
        # attn_map = self.gen1(attn_map)
        attn_map = attn_map + image_identity
        # attn_map = torch.cat([attn_map, image_identity], dim=1)  # (B, hidden_dim + 256, H, W)
          # (B, hidden_dim, H, W)

        # mask = self.decoder(attn_map)  # (B, 1, H, W)
        # mask = F.interpolate(mask, scale_factor=8, mode='bilinear')
        # mask = self.up1(mask)  # (B, 1, 128, 128)
        attn_map = self.mask_generation(attn_map)  # (B, hidden_dim // 4, H, W)
        # attn_map = self.relu(attn_map)
        mask = self.out_dec(attn_map)  # (B, 1, 128, 128)
        mask = F.interpolate(mask, scale_factor=16, mode='bilinear', align_corners=True)  # (B, 1, 64, 64)
        # image_encoder_mask = F.interpolate(image_identity, scale_factor=16, mode='bilinear', align_corners=True)  # (B, 256, 64, 64)
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