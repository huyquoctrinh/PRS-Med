from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import numpy as np
import os 
from glob import glob


model_name = "/home/mamba/ML_project/Testing/Huy/llm_seg/weight/llama3"  # or "meta-llama/Meta-Llama-3-70B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cuda:1"  # Use "cuda" or "auto" for GPU inference
)

def apply_chat_template_llama3(messages: list[dict], add_bot: bool = False) -> str:
    prompt = "<|begin_of_text|>" if add_bot else ""
    for msg in messages:
        if msg["role"] not in ["system", "user", "assistant"]:
            raise ValueError(f"Role {msg['role']} not recognized")
        prompt += f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt

def process_question(question: str, answer: str, groundtruth:str) -> str:
    # Process the question and answer to create a prompt
    prompt = f'''
    Given the following question and answer with the groundtruth, check if the prediction describe the position that is similar with the position in the groundtruth, dont care about the tissue mentioned in the sentence.
    Groundtruth: {groundtruth} | Prediction: {answer}
    Return only "Yes" or "No", and do not have explanation.
    '''
    message_llama = [{
        'role': 'user',
        'content': prompt
    }]
    prompt_dict = apply_chat_template_llama3(
        messages=message_llama
    )
    return prompt_dict

def process_response(response: str) -> str:
    # Process the response from the model
    response = response.replace("<|end_of_text|>", "").strip()
    return response[-1]

# Load the model and tokenizer

def benchmark(df_path_res, df_path_gt):
    df_gt_init = pd.read_csv(df_path_gt)
    df_res = pd.read_csv(df_path_res, lineterminator='\n')
    print(df_res.columns)
    df_gt = df_gt_init[df_gt_init["split"] == "test"]

    df_res.head(5)

    df_gt.head(5)

    right = 0

    for i in range(len(df_gt)):
        # print(df_gt.iloc[i].columns)
        answer = df_res["ground_truth"][i]
        predict = df_res["predict"][i]
        # question = df_gt.iloc[i]["question"]
        # question = "What is the position of the tumor in the image?"
        # print("Answer:", answer, "| Predict:", predict)
        # print("| Answer:", answer, "| Predict:", predict)
        # print("==================================")
        prompt_dict = process_question(
            question="question",
            answer=predict,
            groundtruth=answer
        )
        inputs = tokenizer(prompt_dict, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                top_p=0.95,
                temperature=0.7
            )

        # Decode and print
        token_len = inputs["input_ids"].shape[1]
        response = tokenizer.decode(outputs[0][token_len:], skip_special_tokens=True)
        # print(response)
        # response = process_response(response)
        print("====================")
        print("Response:", response)
        if "Yes" in response:
            right += 1
        
    acc = right / len(df_gt)
    print("Number of samples:", len(df_gt))
    print("Acc:", acc)
    return acc


if __name__ == "__main__":
    df_res_path = "/home/mamba/ML_project/Testing/Huy/gaussian_splatting/LISA/HuatuoGPT-Vision-main/res/"
    df_gt_path = "/home/mamba/ML_project/Testing/Huy/llm_seg/dataset/annotation_v2/"
    list_modal = ["breast_tumors_ct_scan", "brain_tumors_ct_scan", "lung_Xray", "lung_CT", "polyp_endoscopy", "skin_rgbimage"]
    # print(list_df_res)
    # print(list_df_gt)
    acc_dict = {}
    for i in range(len(list_modal)):
        df_gt_path_modal = df_gt_path + "/" + list_modal[i] + ".csv" 
        # df_predict_path_modal = df_res_path + "/results_" + list_modal[i] + ".csv"
        df_predict_path_modal = df_res_path + "/" + list_modal[i] + ".csv"
        acc = benchmark(
            df_path_res=df_predict_path_modal,
            df_path_gt=df_gt_path_modal
        )
        modal_name = list_modal[i]
        acc_dict[modal_name] = acc
        print("==================================")
        print(f"{modal_name} - Acc:", acc)
    
    avg_acc = sum(acc_dict.values()) / len(acc_dict)
    print("==================================")
    print("Average Acc:", avg_acc)
    acc_dict["Average"] = avg_acc
    for key, value in acc_dict.items():
        print(f"{key} - Acc:", value)
    
    df_res = pd.DataFrame.from_dict(acc_dict, orient='index', columns=[modal_name for modal_name in acc_dict.keys()])
    df_res.to_csv("/home/mamba/ML_project/Testing/Huy/llm_seg/evaluation/reasoning/reasoning_acc.csv", index=False)
        # Uncomment to break after 20 iterations
    # if i == 20:
        # break

