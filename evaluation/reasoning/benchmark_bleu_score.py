from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import numpy as np
import os 
from glob import glob
from rouge_score import rouge_scorer
import evaluate

meteor_class = evaluate.load("meteor")
bleu_class = evaluate.load("bleu")

def apply_chat_template_llama3(messages: list[dict], add_bot: bool = False) -> str:
    prompt = "<|begin_of_text|>" if add_bot else ""
    for msg in messages:
        if msg["role"] not in ["system", "user", "assistant"]:
            raise ValueError(f"Role {msg['role']} not recognized")
        prompt += f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

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

# def calculate_bleu_score(reference, hypothesis):
#     """
#     Calculate the BLEU score between a reference and a hypothesis with smoothing.

#     Args:
#         reference (str): The reference text.
#         hypothesis (str): The hypothesis text.

#     Returns:
#         float: The BLEU score.
#     """

#     decoded_preds, decoded_labels = postprocess_text(reference, hypothesis)
#     min_len = min(len(decoded_preds), len(decoded_labels))
#     decoded_preds = decoded_preds[:min_len]
#     decoded_labels = decoded_labels[:min_len]
#     bleu_score = bleu_class.compute(predictions=decoded_preds, references=decoded_labels, max_order=1)
#     return bleu_score['bleu']

def calculate_rough_score(reference, hypothesis):
    """
    Calculate the ROUGE score between a reference and a hypothesis.

    Args:
        reference (str): The reference text.
        hypothesis (str): The hypothesis text.

    Returns:
        float: The ROUGE score.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores['rouge1'].fmeasure

def benchmark(df_path_res, df_path_gt):
    df_gt_init = pd.read_csv(df_path_gt)
    df_res = pd.read_csv(df_path_res, lineterminator='\n')
    print(df_res.columns)
    df_gt = df_gt_init[df_gt_init["split"] == "test"]
    print(df_gt.columns)

    df_res.head(5)

    df_gt.head(5)

    bleu_scores = []
    rouge_scores = []
    meteor_scores = []

    for i in range(len(df_gt)):
        # print(df_gt.iloc[i].columns)
        
        answer = df_gt.iloc[i]["position"]
        predict = df_res.iloc[i]["results"]
        if predict is not np.nan:
            
            print("Answer:", answer, "Predict:", predict)
            # bleu_score = calculate_bleu_score(predict, answer)
            bleu_score = bleu_class.compute(predictions=[predict], references=[[answer]])['bleu']
            bleu_scores.append(bleu_score)
            rouge_score = calculate_rough_score(predict, answer)
            rouge_scores.append(rouge_score)
            meteor_score = meteor_class.compute(predictions=[predict], references=[[answer]])['meteor']
            meteor_scores.append(meteor_score)
    
    mean_rouge = np.mean(rouge_scores)
    mean_bleu = np.mean(bleu_scores)
    mean_meteor = np.mean(meteor_scores)
    return mean_bleu, mean_rouge, mean_meteor


if __name__ == "__main__":
    df_res_path = "/home/mamba/ML_project/Testing/Huy/llm_seg/results/lm_seg_test_3_full_6_classes_16_benchmark_4"
    df_gt_path = "/home/mamba/ML_project/Testing/Huy/llm_seg/dataset/annotation_v2/"
    list_modal = ["breast_tumors_ct_scan", "brain_tumors_ct_scan", "lung_Xray", "lung_CT", "polyp_endoscopy", "skin_rgbimage"]
    # print(list_df_res)
    # print(list_df_gt)
    bleu_dict = {}
    rouge_dict = {}
    meteor_dict = {}
    for i in range(len(list_modal)):
        df_gt_path_modal = df_gt_path + "/" + list_modal[i] + ".csv" 
        # df_predict_path_modal = df_res_path + "/results_" + list_modal[i] + ".csv"
        df_predict_path_modal = df_res_path + "/results_" + list_modal[i] + ".csv"
        bleu, rouge, meteor = benchmark(
            df_path_res=df_predict_path_modal,
            df_path_gt=df_gt_path_modal
        )
        modal_name = list_modal[i]
        bleu_dict[modal_name] = bleu
        meteor_dict[modal_name] = meteor
        rouge_dict[modal_name] = rouge

        print("==================================")
        print(f"{modal_name} - Bleu:", bleu, "Rouge:", rouge, "Meteor:", meteor)
    
    avg_bleu = np.mean(list(bleu_dict.values()))
    bleu_dict["Average"] = avg_bleu
    rouge_dict["Average"] = np.mean(list(rouge_dict.values()))
    meteor_dict["Average"] = np.mean(list(meteor_dict.values()))
    print("==================================")
    print("Average Bleu Score:", avg_bleu)
    print("Average Rouge Score:", rouge_dict["Average"])
    print("Average Meteor Score:", np.mean(list(meteor_dict.values())))
    # bleu_dict["Rouge"] = rouge_dict["Average"]
    # bleu_dict["Bleu"] = avg_bleu

    
    # df_res = pd.DataFrame.from_dict(acc_dict, orient='index', columns=[modal_name for modal_name in acc_dict.keys()])
    # df_res.to_csv("/home/mamba/ML_project/Testing/Huy/llm_seg/evaluation/reasoning/reasoning_acc.csv", index=False)
        # Uncomment to break after 20 iterations
    # if i == 20:
        # break

