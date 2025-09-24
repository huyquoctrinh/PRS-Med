import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt

def analyze_data(df):
    split = ["train", "test"]
    res_analyze = {}
    # print("Analyzing data from:", file_path)
    # df = pd.read_csv(file_path)
    for s in split:
        res_analyze[s] = {}
        res_analyze[s]["num_images"] = len(df[df["split"] == s])
        res_analyze[s]["num_masks"] = len(df[df["split"] == s])
    return res_analyze

if __name__ == "__main__":
    root = "/home/yangchen/llava/data/segment_data"
    list_df = ["/home/mamba/ML_project/Testing/Huy/llm_seg/dataset/annotation_v2/" + df_dir for df_dir in os.listdir("/home/mamba/ML_project/Testing/Huy/llm_seg/dataset/annotation_v2")]
    list_res = {}
    for df_path in list_df:
        df = pd.read_csv(df_path)
        modal = df_path.split("/")[-1].split(".")[0]
        list_res[modal] = analyze_data(df)
    # fig, ax = plt.subplots()
    print(list_res)