import pandas as pd
import numpy as np
import os 

df_gt_init = pd.read_csv("/home/mamba/ML_project/Testing/Huy/llm_seg/dataset/annotation1/annotation_v1/breast_tumors_ct_scan.csv")
df_res = pd.read_csv("/home/mamba/ML_project/Testing/Huy/llm_seg/results/lm_seg_test_3_norm_11/results_breast_tumors_ct_scan.csv")

df_gt = df_gt_init[df_gt_init["split"] == "test"]

df_res.head(5)

df_gt.head(5)

for i in range(len(df_gt)):
    answer = df_gt.iloc[i]["position"]
    predict = df_res.iloc[i]["results"]
    question = df_gt.iloc[i]["question"]
    print("Answer:", df_gt.iloc[i]["image_path"], "| Predict:", df_res.iloc[i]["image_path"])
    print("Question:", question, "| Answer:", answer, "| Predict:", predict)
    print("==================================")
    if i == 20:
        break
# df_gt = pd.read_csv("/home/mamba/ML_project/Testing/Huy/llm_seg/dataset/annotation1/annotation_v1/breast_tumors_ct_scan.csv"