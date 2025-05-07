# import cv2
# import numpy as np
# def remove_redundant(mask):
#     kernel = np.ones((,5), np.uint8)
#     opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#     return opened

# mask = cv2.imread("/home/mamba/ML_project/Testing/Huy/llm_seg/results/lm_seg8/masks/brain_tumors_ct_scan/1283.png", 0)
# out = remove_redundant(mask)
# cv2.imwrite("mask_out_1283.png", out)

import cv2
import numpy as np

# Load image in grayscale
img = cv2.imread('/home/mamba/ML_project/Testing/Huy/llm_seg/results/lm_seg8/masks/brain_tumors_ct_scan/1596.png', cv2.IMREAD_GRAYSCALE)

# Threshold to binary image
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Find connected components
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

# Filter by area (keep only large regions, e.g., area > 500)
min_area = 400
filtered = np.zeros_like(binary)
for i in range(1, num_labels):  # skip background
    if stats[i, cv2.CC_STAT_AREA] > min_area:
        filtered[labels == i] = 255

cv2.imwrite('filtered_output.png', filtered)
