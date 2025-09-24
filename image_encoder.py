import cv2 
import numpy as np
import torch
# from tinysam import sam_model_registry, SamHierarchicalMaskGenerator
from sam_med.segment_anything import sam_model_registry
from argparse import Namespace
args = Namespace()
args.image_size = 256
args.encoder_adapter = True
args.sam_checkpoint = "/home/mamba/ML_project/Testing/Huy/gaussian_splatting/SAM-Med2D/pretrained/sam-med2d_b.pth"

sam = sam_model_registry["vit_b"](args)
# sam = sam_model_registry["vit_b"](checkpoint="/home/mamba/ML_project/Testing/Huy/llm_seg/weight/sam_ckpts/sam_vit_b_01ec64.pth")

# model_type = "vit_t"
# sam = sam_model_registry[model_type](checkpoint="/home/mamba/ML_project/Testing/Huy/llm_seg/weight/sam_ckpts/tinysam_42.3.pth")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

sam.train()
# print(sam.image_encoder)
image_encoder = sam.image_encoder.to(device)
inputs = torch.randn(1, 3, 256, 256).to(device)
out = image_encoder(inputs)
print(out.shape)
i = 0
for name, param in image_encoder.named_parameters():
    i+=1
    if i==10:
        break
    print(name, param.requires_grad)
# print(len(list(image_encoder.named_parameters())))
# print(out.shape)
# print(len(list_output))
# print(list_output[0].shape)   