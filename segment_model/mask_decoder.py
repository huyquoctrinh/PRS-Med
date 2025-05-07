import torch
import torch.nn as nn
import torch.nn.functional as F

class PromptedMaskDecoder(nn.Module):
    def __init__(self, 
                 feature_dim=256, 
                 prompt_dim=4096, 
                 transformer_dim=512, 
                 num_transformer_layers=4, 
                 num_heads=8, 
                 mask_resolution=1024):
        super(PromptedMaskDecoder, self).__init__()

        self.feature_dim = feature_dim
        self.transformer_dim = transformer_dim
        self.mask_resolution = mask_resolution

        # Project SAM features to transformer dimension
        self.img_proj = nn.Linear(feature_dim, transformer_dim)
        torch.nn.init.xavier_uniform_(self.img_proj.weight)
        torch.nn.init.zeros_(self.img_proj.bias)

        # Project prompt embeddings to transformer dimension
        self.prompt_proj = nn.Linear(prompt_dim, transformer_dim)
        torch.nn.init.xavier_uniform_(self.prompt_proj.weight)
        torch.nn.init.zeros_(self.prompt_proj.bias)
        # Positional embeddings for spatial features (16x16)
        self.positional_embedding = nn.Parameter(torch.randn(1, 64*64, transformer_dim))

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            batch_first=True,
            activation='relu'
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_transformer_layers
        )

        # Mask prediction head
        self.mask_head = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 2, kernel_size=2, stride=2),  # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(transformer_dim // 2, transformer_dim // 4, kernel_size=2, stride=2),  # 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),  # 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(transformer_dim // 8, transformer_dim // 16, kernel_size=2, stride=2),  # 256x256
            nn.ReLU(),
            nn.ConvTranspose2d(transformer_dim // 16, transformer_dim // 32, kernel_size=2, stride=2),  # 512x512
            nn.ReLU(),
            nn.ConvTranspose2d(transformer_dim // 32, 1, kernel_size=2, stride=2),  # 1024x1024
        )
        self.num_layers = num_transformer_layers
        self.__init_weights()

    def __init_weights(self):
        torch.nn.init.xavier_uniform_(self.img_proj.weight)
        torch.nn.init.xavier_uniform_(self.prompt_proj.weight)
        torch.nn.init.xavier_uniform_(self.mask_head[0].weight)
        torch.nn.init.xavier_uniform_(self.mask_head[2].weight)
        torch.nn.init.xavier_uniform_(self.mask_head[4].weight)
        torch.nn.init.xavier_uniform_(self.mask_head[6].weight)
        torch.nn.init.xavier_uniform_(self.mask_head[8].weight)
        torch.nn.init.xavier_uniform_(self.mask_head[10].weight)
        for i in range(self.num_layers):
            torch.nn.init.xavier_uniform_(self.transformer_decoder.layers[i].self_attn.in_proj_weight)
            torch.nn.init.xavier_uniform_(self.transformer_decoder.layers[i].multihead_attn.in_proj_weight)
            torch.nn.init.xavier_uniform_(self.transformer_decoder.layers[i].linear1.weight)
            torch.nn.init.xavier_uniform_(self.transformer_decoder.layers[i].linear2.weight)

    def forward(self, image_features, prompt_embeddings):
        """
        image_features: (B, 256, 16, 16)
        prompt_embeddings: (B, T, 2048)
        """
        B = image_features.size(0)

        # Flatten spatial dimensions and project
        # print(image_features.shape)
        image_features = image_features.float()
        img_feats_flat = image_features.view(B, self.feature_dim, -1).permute(0, 2, 1)  # (B, 256, 16x16) -> (B, 256, 256)
        # print(img_feats_flat.shape)
        img_feats_proj = self.img_proj(img_feats_flat)  # (B, 256, transformer_dim)

        # Add positional embedding
        # print(img_feats_proj.shape)
        img_feats_proj += self.positional_embedding  # (B, 256, transformer_dim)

        # Project prompt embeddings
        prompt_proj = self.prompt_proj(prompt_embeddings)  # (B, T, transformer_dim)

        # Transformer decoding: Conditioned on image features (memory)
        decoder_output = self.transformer_decoder(tgt=prompt_proj, memory=img_feats_proj)
        # (B, T, transformer_dim)

        # Pool transformer output (e.g., mean pooling over tokens)
        decoder_pooled = decoder_output.mean(dim=1)  # (B, transformer_dim)

        # Reshape for convolutional upsampling
        x = decoder_pooled.view(B, self.transformer_dim, 1, 1)

        # Generate mask via upsampling
        mask = self.mask_head(x)  # (B, 1, 1024, 1024)

        # Optionally apply sigmoid for binary mask prediction
        mask = torch.sigmoid(mask)

        return mask
