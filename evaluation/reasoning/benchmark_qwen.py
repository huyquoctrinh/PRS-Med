from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

model_name = "/home/mamba/ML_project/Testing/Huy/llm_seg/weight/qwen"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cuda:1"
)

# prepare the model input
def predict_fn(question: str, answer: str, groundtruth: str) -> str:
    # Process the question and answer to create a prompt
    # print(answer, groundtruth)
    prompt = f'''
    You are a doctor and you want to see the position of the tumor in the medical image.
    Given the following question from you and answer with the groundtruth, check if the prediction has the position word that is similar with the position in the groundtruth, dont care about the tissue mentioned in the sentence.
    Question: {question} | groundtruth: {groundtruth} | Predict: {answer}
    Return only "Yes" or "No".
    '''

    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking= False # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    # thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    # print(content)
    # print("thinking content:", thinking_content)
    # print("content:", content)
    return content

def benchmark(df_path_res, df_path_gt):
    df_gt_init = pd.read_csv(df_path_gt)
    df_res = pd.read_csv(df_path_res, lineterminator='\n')

    df_gt = df_gt_init[df_gt_init["split"] == "test"]

    df_res.head(5)

    df_gt.head(5)

    right = 0

    for i in tqdm(range(len(df_gt))):
        answer = df_gt.iloc[i]["position"]
        predict = df_res.iloc[i]["results"]
        question = df_gt.iloc[i]["question"]
        benchmark_res = predict_fn(
            question=question,
            answer=predict,
            groundtruth=answer
        )
        # print("Answer:", df_gt.iloc[i]["image_path"], "| Predict:", df_res.iloc[i]["image_path"])
        # print("Question:", question, "| Answer:", answer, "| Predict:", predict)
        # print("Response:", benchmark_res)
        # print("==================================")
        if benchmark_res == "Yes":
            right += 1
    return right / len(df_gt)

if __name__ == "__main__":
    df_res_path = "/home/mamba/ML_project/Testing/Huy/llm_seg/results/lm_seg_test_3_full_6_classes_16_benchmark_3"
    df_gt_path = "/home/mamba/ML_project/Testing/Huy/llm_seg/dataset/annotation_v2/"
    list_modal = ["breast_tumors_ct_scan", "brain_tumors_ct_scan", "lung_Xray", "lung_CT", "polyp_endoscopy", "skin_rgbimage"]
    # print(list_df_res)
    # print(list_df_gt)
    acc_dict = {}
    for i in range(len(list_modal)):
        df_gt_path_modal = df_gt_path + "/" + list_modal[i] + ".csv" 
        df_predict_path_modal = df_res_path + "/results_" + list_modal[i] + ".csv"
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
    
    # df_res = pd.DataFrame.from_dict(acc_dict, orient='index', columns=[modal_name for modal_name in acc_dict.keys()])
    # df_res.to_csv("/home/mamba/ML_project/Testing/Huy/llm_seg/evaluation/reasoning/reasoning_acc.csv", index=False)
