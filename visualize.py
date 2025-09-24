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
    color_mask[mask>100] = [188, 145, 210]  # Red color for label 1
    alpha = 0.7  # transparency for original image
    beta = 0.3   # transparency for mask overlay
    gamma = 0.1   # scalar added to each sum
    print("image.shape:", image.shape, "mask.shape:", color_mask.shape)
    overlay = cv2.addWeighted(image, alpha, color_mask, beta, gamma)
    return overlay
# /home/mamba/ML_project/Testing/Huy/llm_seg/results/lm_seg_test_3_full_6_classes_16_benchmark_3/results_lung_CT.csv
# /home/mamba/ML_project/Testing/Huy/llm_seg/results/lm_seg_test_3_full_6_classes_16_benchmark_3/results_lung_Xray.csv
idx = 30
df = pd.read_csv("/home/mamba/ML_project/Testing/Huy/llm_seg/results/lm_seg_test_3_full_6_classes_16_benchmark_3/results_lung_CT.csv")
modal = "lung_CT"
image_path = "/home/mamba/ML_project/Testing/Huy/llm_seg/dataset/data/" + df.iloc[idx]["image_path"].replace("test_masks", "test_images")
mask_path = f"/home/mamba/ML_project/Testing/Huy/llm_seg/results/lm_seg_test_3_full_6_classes_16_benchmark_3/masks/{modal}/" + df.iloc[idx]["mask_path"].split("/")[-1]
# "/home/mamba/ML_project/Testing/Huy/llm_seg/dataset/data/"
gt_mask = "/home/mamba/ML_project/Testing/Huy/llm_seg/dataset/data/"
print(image_path)
print(mask_path)
# image_path = image_path.replace("_Segmentation", "")
# image_path = image_path.replace(".png", ".jpg")
image = cv2.imread(image_path)
mask = cv2.imread(mask_path, 0)
gt_mask = cv2.imread(gt_mask + df.iloc[idx]["image_path"].replace("test_masks", "test_masks"), 0)
overlay = visualize_mask_on_image(image, mask)
overlay_gt = visualize_mask_on_image(image, gt_mask)

cv2.imwrite("visualized_overlay_gt.png", overlay_gt)

image = cv2.resize(image, (512, 512))
cv2.imwrite("visualized_overlay.png", overlay)
cv2.imwrite("image.png", image)
description = df["results"].iloc[idx]
question = df["prompt"].iloc[idx]
# answers = df["position"].iloc[0]
print("Description:", description)
print("Question:", question)
# print("Answers:", answers)