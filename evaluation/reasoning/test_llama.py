from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import numpy as np
import os 

df_gt_init = pd.read_csv("/home/mamba/ML_project/Testing/Huy/llm_seg/dataset/annotation1/annotation_v1/breast_tumors_ct_scan.csv")
df_res = pd.read_csv("/home/mamba/ML_project/Testing/Huy/llm_seg/results/lm_seg_test_3_norm_11/results_breast_tumors_ct_scan.csv")

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
    Given the following question and answer, check if the predict correct with the answers in the position content.
    Question: {question} | Answer: {answer} | Predict: ### AI: {groundtruth}
    Return only "Yes" or "No".
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
model_name = "/home/mamba/ML_project/Testing/Huy/llm_seg/weight/llama3"  # or "meta-llama/Meta-Llama-3-70B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cuda:2"  # Use "cuda" or "auto" for GPU inference
)

df_gt = df_gt_init[df_gt_init["split"] == "test"]

df_res.head(5)

df_gt.head(5)

right = 0

for i in range(len(df_gt)):
    answer = df_gt.iloc[i]["position"]
    predict = df_res.iloc[i]["results"]
    question = df_gt.iloc[i]["question"]
    print("Answer:", df_gt.iloc[i]["image_path"], "| Predict:", df_res.iloc[i]["image_path"])
    print("Question:", question, "| Answer:", answer, "| Predict:", predict)
    print("==================================")
    prompt_dict = process_question(
        question=question,
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

    # if i == 20:
        # break

