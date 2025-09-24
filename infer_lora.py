from segment_model.model import build_llm_seg
import torch
from torch.cuda.amp import autocast
from PIL import Image
from data_utils.utils import load_image
from torchvision import transforms
from llava.mm_utils import process_images
from llava.mm_utils import tokenizer_image_token
import cv2
import numpy as np
import os
import random
from llava.mm_utils import get_model_name_from_path
from peft import LoraConfig, TaskType
from peft import get_peft_model
import pandas as pd
from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model
import requests
from io import BytesIO

IMAGE_TOKEN_INDEX = -200
def prepare_model(
    model_path,
    model_base,
    load_8bit,
    load_4bit,
    device
):
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, 
        model_base, 
        model_name, 
        load_8bit, 
        load_4bit, 
        device=device
    )
    return tokenizer, model, image_processor, context_len

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

# def load_image_for_vlm(image_path, image_processor, config):
#     image_pil = load_image(image_path)
#     image_tensor = process_images(
#         [image_pil],
#         image_processor,
#         config
#     )
#     return image_tensor.to(torch.float16)

def process_prompt(image_type, tokenizer):
    # prompt_for_vlm = "<image>" + " You are doing the segmentation." + prompt
    template = [
        f"Can you identify the location of the tumour in this {image_type} medical image?",
        f"Please describe the tumour’s position in this medical image of types {image_type}",
        f"What is the anatomical location of the tumour in this {image_type} medical image?",
        f"Based on this {image_type} medical image, can you provide the location of the tumour in this image?",
        f"Where is the tumour located in this {image_type} medical image?",
    ]
    prompt_for_vlm = "<image>\n" + f"### User: {random.choice(template)} \n"
    print("Prompt:", prompt_for_vlm)
    input_ids = tokenizer_image_token(
        prompt_for_vlm,
        tokenizer,
        -200,
        return_tensors="pt"
    )
    return input_ids.to(torch.int64).unsqueeze(0)

def infer(
    prompt,
    image_path,
    image_processor,
    model,
    tokenizer,
    config,
    device = "cuda:1",
):
    
    image = load_image(image_path)
    prompt = f"<image> \n #User: {prompt}"

    image_tensor = process_images([image], image_processor, config)

    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(device, dtype=torch.float16)

    input_ids = tokenizer_image_token(
        prompt, 
        tokenizer, 
        IMAGE_TOKEN_INDEX, 
        return_tensors='pt'
    ).unsqueeze(0).to(device)

    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        do_sample=True,
        temperature=0.0001,
        max_new_tokens=256,
        top_p=0.95
    )
    # output_ids = remove_pad_tokens(output_ids, pad_token_id=tokenizer.pad_token_id)
    res = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return res

def remove_pad_tokens(output_ids, pad_token_id=0):
    output_ids = output_ids[0]
    output_ids = output_ids[output_ids != pad_token_id]
    return output_ids

if __name__ == "__main__":
    model_path = "/home/mamba/ML_project/Testing/Huy/llm_seg/weight/llava-med-v1.5-mistral-7b"
    model_base = None
    load_8bit = False
    load_4bit = False
    device = "cuda:1"

    tokenizer, base_model, image_processor, context_len = prepare_model(
        model_path="/home/mamba/ML_project/Testing/Huy/llm_seg/weight/llava-med-v1.5-mistral-7b",
        model_base=None,
        load_8bit=False,
        load_4bit=False,
        device="cuda:1"
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], 
        inference_mode=False,
    )
    model = get_peft_model(base_model, lora_config)
    config = base_model.config
    model.from_pretrained(base_model, "/home/mamba/ML_project/Testing/Huy/llm_seg/training_results/ablation/llava_lora")
    model = model.merge_and_unload()
    model = model.to("cuda:1")
    model.eval()

    results_path = "results/lora_results_llava/"
    
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    # df_res = pd.DataFrame(columns=["image_path", "mask_path", "prompt", "results"])

    list_csv_path = ["/home/mamba/ML_project/Testing/Huy/llm_seg/dataset/annotation_v2/" + df_dir for df_dir in os.listdir("/home/mamba/ML_project/Testing/Huy/llm_seg/dataset/annotation_v2")]
    # df = pd.read_csv("/home/mamba/ML_project/Testing/Huy/llm_seg/dataset/annotation1/annotation_v1/lung_Xray.csv")
    # len(df)
     
    for csv_path in list_csv_path:
        df = pd.read_csv(csv_path)
        image_list = []
        mask_list = []
        prompt_list = []
        answer_list = []
        # cnt =0
        for i in range(len(df)):
            # if cnt > 10:
                # break
            if df.iloc[i]["split"] == "test":
                # cnt+=1
                image_path = "/home/mamba/ML_project/Testing/Huy/llm_seg/dataset/data/" + df.iloc[i]["image_path"]
                idx = df.iloc[i]["image_path"].split("/")[-1].split(".")[0]
                modal = df.iloc[i]["image_path"].split("/")[0]
                prompt = df.iloc[i]["question"]
                print("Image path:", image_path)
                # print("Prompt:", prompt)
                answer = infer(
                    prompt,
                    image_path,
                    image_processor,
                    model,
                    tokenizer,
                    config
                )
                print("Save mask path:", image_path, "| Answer:", answer)
                image_list.append(df.iloc[i]["image_path"])
                prompt_list.append(prompt)
                answer_list.append(answer.replace("\n","").replace("### Assistant: ","").replace("### User: ","").replace("You are doing the segmentation for the tumour with the condition: ",""))
                # df_res = pd.concat([df_res, pd.DataFrame({"image_path": [df.iloc[i]["image_path"]], "mask_path": [save_mask_path], "prompt": [prompt], "results": [answer]})], ignore_index=True)

        df_res = pd.DataFrame({
            "image_path": image_list,
            "prompt": prompt_list,
            "results": answer_list
        })
        print("Number of answer:", len(answer_list), "Number of image:", len(image_list), "Number of mask:", len(mask_list))
        df_res.to_csv(results_path + f"/results_{modal}.csv", index=False)