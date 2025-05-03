import pandas as pd
import numpy as np
import os 

df_gt = pd.read_csv("/home/mamba/ML_project/Testing/Huy/llm_seg/dataset/annotation1/annotation_v1/breast_tumors_ct_scan.csv")
df_res = pd.read_csv("/home/mamba/ML_project/Testing/Huy/llm_seg/results/lm_seg_test_2_2/results_breast_tumors_ct_scan.csv")

for i in range(len(df_gt)):
    answer = df_gt.iloc[i]["position"]
    predict = df_res.iloc[i]["results"]
    print("Answer:", answer, "| Predict:", predict)
    if i == 2:
        break
# df_gt = pd.read_csv("/home/mamba/ML_project/Testing/Huy/llm_seg/dataset/annotation1/annotation_v1/breast_tumors_ct_scan.csv"