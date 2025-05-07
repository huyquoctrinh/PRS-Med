import cv2 
import pandas as pd
import numpy as np
import os
def visualize_mask_on_image(image, mask):
    image = cv2.resize(image, (512, 512))
    mask = cv2.resize(mask, (512, 512))
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    # Convert mask to binary
    # overlay  = cv2.addWeighted(image, 0.6, mask, 0.5, 0)
    color_mask = np.zeros_like(image)
    color_mask[mask>0] = [188, 145, 210]  # Red color for label 1
    alpha = 0.7  # transparency for original image
    beta = 0.3   # transparency for mask overlay
    gamma = 0.1   # scalar added to each sum
    print("image.shape:", image.shape, "mask.shape:", color_mask.shape)
    overlay = cv2.addWeighted(image, alpha, color_mask, beta, gamma)
    return overlay

df = pd.read_csv("/home/mamba/ML_project/Testing/Huy/llm_seg/dataset/annotation1/annotation_v1/breast_tumors_ct_scan.csv")

image_path = "/home/mamba/ML_project/Testing/Huy/llm_seg/dataset/data/" + df.iloc[0]["image_path"].replace("train_masks", "train_images")
mask_path = "/home/mamba/ML_project/Testing/Huy/llm_seg/dataset/data/" + df.iloc[0]["image_path"]
print(mask_path)
print(image_path)
image = cv2.imread(image_path)
mask = cv2.imread(mask_path, 0)

overlay = visualize_mask_on_image(image, mask)

cv2.imwrite("visualized_overlay.png", overlay)
description = df["description"].iloc[0]
question = df["question"].iloc[0]
answers = df["position"].iloc[0]
print("Description:", description)
print("Question:", question)
print("Answers:", answers)