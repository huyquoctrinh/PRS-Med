import cv2 
import numpy as np
import torch
from tinysam import sam_model_registry, SamHierarchicalMaskGenerator

model_type = "vit_t"
sam = sam_model_registry[model_type](checkpoint="/home/mamba/ML_project/Testing/Huy/llm_seg/weight/sam_ckpts/tinysam_42.3.pth")
device = "cuda" if torch.cuda.is_available() else "cpu"

sam.eval()
# print(sam.image_encoder)
image_encoder = sam.image_encoder.to(device)
inputs = torch.randn(1, 3, 1024, 1024).to(device)
out = image_encoder(inputs)
print(out.shape)