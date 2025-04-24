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
        return ["no mask"]

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    positions = []
    for i in range(1, num_labels):  # Skip background (label 0)
        min_x, min_y, width, height, _ = stats[i]
        max_x = min_x + width
        max_y = min_y + height

        # Check if tumor crosses the center point
        crosses_center = (min_x <= center_x <= max_x) and (min_y <= center_y <= max_y)

        # Determine position
        tumor_center_x = centroids[i][0]
        tumor_center_y = centroids[i][1]

        if tumor_center_x < center_x and tumor_center_y < center_y:
            position = "top left"
        elif tumor_center_x >= center_x and tumor_center_y < center_y:
            position = "top right"
        elif tumor_center_x < center_x and tumor_center_y >= center_y:
            position = "bottom left"
        else:
            position = "bottom right"

        if crosses_center:
            position += " center"

        positions.append(position)

    return [positions[0], positions[-1]]  # Return only the first two positions

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
    template_first = "2_pos.txt"
    template_second = "2_pos_2nd.txt"
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

                position = get_mask_position(mask)
                if "no mask" in position:
                    continue

                # Generate sentences for each tumor position
                position_sentences = []
                for idx, pos in enumerate(position):
                    template_file = template_first if idx == 0 else template_second
                    position_sentences.append(load_random_sentence(template_file, pos))

                # Concatenate sentences
                position_sentence = ". ".join(sentence.replace(",", "") for sentence in position_sentences)
                description = load_random_description(description_file).replace(",", "")

                # Write to output file
                out_file.write(f"{mask_path},{image_name},{split},{description},{position_sentence}\n")

# # Define paths and process
# base_folder = "brain_tumors_ct_scan"
# output_file = "brain_tumors_ct_scan.csv"
# process_images(base_folder, output_file)

# # Define paths and process
# base_folder = "breast_tumors_ct_scan"
# output_file = "breast_tumors_ct_scan.csv"
# process_images(base_folder, output_file)

# Define paths and process
base_folder = "lung_CT"
output_file = "lung_CT.csv"
process_images(base_folder, output_file)

# # Define paths and process
# base_folder = "lung_Xray"
# output_file = "lung_Xray.csv"
# process_images(base_folder, output_file)