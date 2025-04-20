import pandas as pd
import numpy as np
import os
from random import shuffle
import random
from PIL import Image
import requests
from io import BytesIO

def load_annotation(annotation_path):
    list_df_path = os.listdir(annotation_path)
    list_df_path = [os.path.join(annotation_path, df) for df in list_df_path]
    list_df = []
    for df_path in list_df_path:
        df = pd.read_csv(df_path)
        df = df.dropna()
        df = df.reset_index(drop=True)
        list_df.append(df)
    df = pd.concat(list_df, ignore_index=True)
    df = df.dropna()
    df = df.sample(frac=1, random_state=42)
    df = df.reset_index(drop=True)
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']
    return train_df, test_df

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def binary_loader(mask_path):
    with open(mask_path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')