import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
import os 
import numpy as np

def infer(question, answer, groundtruth, model, tokenizer):
    system_prompt = """You are a the people who understand medical image"""
    prompt = f'''
    Given the following answer with the groundtruth, check if the assistant prediction have the content that describe the position that is similar with the position you think in the groundtruth, dont care about the tissue mentioned in the sentence. 
    Groundtruth: {groundtruth} 
    Assistant Prediction: {answer}
    #####
    Example of the failure case:
    1. Groundtruth: The tumor is located in the upper right quadrant.
    Assistant Prediction: The tumor is located in the lower left quadrant.
    2. Groundtruth: The tumor is located in the upper right quadrant.
    Assistant Prediction: The tumor is located in the upper left.
    ######
    Example of the success case:
    1. Groundtruth: The tumor is located in the upper right quadrant.
    Assistant Prediction: The tumor is located in the right region.
    2. Groundtruth: The tumor is located in the upper right quadrant.
    Assistant Prediction: The tumor is located in the top right.
    ######
    Return only "Yes" or "No", and do not have explanation.
    '''
    # print("GT:", groundtruth, "| Predict:", answer)
    # prompt = f'''
    # You are a doctor and you want to see the position of the tumor in the medical image.
    # Given the following question from you and answer with the groundtruth, check if the prediction has the position word that is similar with the position in the groundtruth, dont care about the tissue mentioned in the sentence.
    # Question: {question} | groundtruth: {groundtruth} | Predict: {answer}
    # Return only "Yes" or "No".
    # '''
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    tokenized_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda:1")
    generated_ids = model.generate(tokenized_chat, max_new_tokens=1024, temperature=1, do_sample = True, repetition_penalty=1.005, top_k=40, top_p=0.8)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(tokenized_chat, generated_ids)
    ]
    prompt = tokenizer.batch_decode(tokenized_chat)[0]
    # print(prompt)
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
    print("==================================")
    return response

def benchmark(df_path_res, df_path_gt, model, tokenizer):
    df_gt_init = pd.read_csv(df_path_gt)
    df_res = pd.read_csv(df_path_res, lineterminator='\n')

    df_gt = df_gt_init[df_gt_init["split"] == "test"]

    df_res.head(5)

    df_gt.head(5)

    # right = 0
    results = []
    for i in tqdm(range(len(df_gt))):
        answer = df_gt.iloc[i]["position"]
        predict = df_res.iloc[i]["results"]
        question = df_gt.iloc[i]["question"]
        benchmark_res = infer(
            question=question,
            answer=predict,
            groundtruth=answer,
            model=model,
            tokenizer=tokenizer
        )
        print("Answer:", df_gt.iloc[i]["image_path"], "| Predict:", df_res.iloc[i]["image_path"])
        print("Question:", question, "| Answer:", answer, "| Predict:", predict)
        print("Response:", benchmark_res)
        print("==================================")
        if benchmark_res == "Yes":
            results.append(1)
        else:
            results.append(0)
    return sum(results) / len(results)


if __name__ == "__main__":
    # Example usage

    model_dir = "/home/mamba/ML_project/Testing/Huy/llm_seg/weight/m/internlm3-8b"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda:1")
    model = model.eval()
    # question = "What is the position of the tumor in the image?"
    # answer = "The tumor is located in the upper right quadrant."
    # groundtruth = "The tumor is located in the upper right quadrant."
    df_res_path = "/home/mamba/ML_project/Testing/Huy/llm_seg/results/lm_seg_test_3_full_6_classes_16_benchmark_2"
    df_gt_path = "/home/mamba/ML_project/Testing/Huy/llm_seg/dataset/annotation_v2/"
    list_modal = ["breast_tumors_ct_scan", "brain_tumors_ct_scan", "lung_Xray", "lung_CT", "polyp_endoscopy", "skin_rgbimage"]

    acc_dict = {}
    for i in range(len(list_modal)):
        df_gt_path_modal = df_gt_path + "/" + list_modal[i] + ".csv" 
        df_predict_path_modal = df_res_path + "/results_" + list_modal[i] + ".csv"
        acc = benchmark(
            df_path_res=df_predict_path_modal,
            df_path_gt=df_gt_path_modal,
            model=model,
            tokenizer=tokenizer
        )
        modal_name = list_modal[i]
        acc_dict[modal_name] = acc
        print("==================================")
        print(f"{modal_name} - Acc:", acc)
    
    avg_acc = sum(acc_dict.values()) / len(acc_dict)
    print("Avgerage Acc:", avg_acc)
    # response = infer(question, answer, groundtruth, model, tokenizer)
    # print("Response:", response)

# system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
# - InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
# - InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文."""
# messages = [
#     {"role": "system", "content": system_prompt},
#     {"role": "user", "content": "Please tell me five scenic spots in Shanghai"},
#  ]
# tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

# generated_ids = model.generate(tokenized_chat, max_new_tokens=1024, temperature=1, repetition_penalty=1.005, top_k=40, top_p=0.8)

# generated_ids = [
#     output_ids[len(input_ids):] for input_ids, output_ids in zip(tokenized_chat, generated_ids)
# ]
# prompt = tokenizer.batch_decode(tokenized_chat)[0]
# print(prompt)
# response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(response)
