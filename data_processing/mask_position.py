import os
import random
import cv2
import numpy as np
from tqdm import tqdm

def get_mask_position(mask):
    h, w = mask.shape
    center_x, center_y = w // 2, h // 2
    mask_coords = np.argwhere(mask > 0)
    if mask_coords.size == 0:
        return "no mask", False

    min_y, min_x = mask_coords.min(axis=0)
    max_y, max_x = mask_coords.max(axis=0)

    # Check if tumor crosses the center point
    crosses_center = (min_x <= center_x <= max_x) and (min_y <= center_y <= max_y)
    if crosses_center:
        return "center", True

    # Calculate distance to center
    tumor_center_x = (min_x + max_x) // 2
    tumor_center_y = (min_y + max_y) // 2
    distance_to_center = np.sqrt((tumor_center_x - center_x) ** 2 + (tumor_center_y - center_y) ** 2)
    near_center = distance_to_center < 0.2 * w  # 20% of image width

    # Determine position
    if tumor_center_x < center_x and tumor_center_y < center_y:
        position = "top left"
    elif tumor_center_x >= center_x and tumor_center_y < center_y:
        position = "top right"
    elif tumor_center_x < center_x and tumor_center_y >= center_y:
        position = "bottom left"
    else:
        position = "bottom right"

    return position, near_center

def load_random_sentence(file_path, position):
    with open(file_path, 'r') as f:
        sentences = f.readlines()
    sentence = random.choice(sentences).strip()
    return sentence.replace("{position}", position)

def load_random_description(file_path):
    with open(file_path, 'r') as f:
        descriptions = f.readlines()
    return random.choice(descriptions).strip()

def process_images(base_folder, output_file):
    splits = ["train", "val", "test"]
    template_near = "position.txt"
    template_far = "position_far.txt"
    description_file = f"descriptions/{base_folder}.txt"

    with open(output_file, 'w') as out_file:
        out_file.write("image_path,image_name,split,description,position\n")
        for split in splits:
            split_folder = os.path.join(base_folder, f"{split}_masks")
            for image_name in tqdm(os.listdir(split_folder)):
                if not image_name.endswith(('.png', '.jpg', '.jpeg')):
                    continue

                mask_path = os.path.join(split_folder, image_name)

                # Load image and mask
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue

                position, near_center = get_mask_position(mask)
                if position == "no mask":
                    continue

                # Load appropriate template
                template_file = template_near if near_center else template_far
                position_sentence = load_random_sentence(template_file, position)
                description = load_random_description(description_file)

                # Write to output file
                out_file.write(f"{mask_path},{image_name},{split},{description},{position_sentence}\n")

# Define paths and process
base_folder = "brain_tumors_ct_scan"
output_file = "brain_tumors_ct_scan.csv"
process_images(base_folder, output_file)

# Define paths and process
base_folder = "breast_tumors_ct_scan"
output_file = "breast_tumors_ct_scan.csv"
process_images(base_folder, output_file)

# Define paths and process
base_folder = "lung_CT"
output_file = "lung_CT.csv"
process_images(base_folder, output_file)

# Define paths and process
base_folder = "lung_Xray"
output_file = "lung_Xray.csv"
process_images(base_folder, output_file)