import argparse
import os

import torch
from torch.cuda.amp import autocast
import cv2
import pandas as pd
from torchvision import transforms

from segment_model.model import build_llm_seg
from data_utils.utils import load_image
from llava.mm_utils import process_images, tokenizer_image_token

IMAGE_TOKEN_INDEX = -200


def load_model(model_path, checkpoint_path, device="cuda"):
    model, tokenizer, image_processor, config = build_llm_seg(
        model_path=model_path,
        model_base=None,
        load_8bit=False,
        load_4bit=False,
        device=device
    )
    tokenizer = model.load_model(checkpoint_path)
    model = model.to(device)
    return model, tokenizer, image_processor, config


def transform_for_sam(image_path):
    image_sam_transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
    image = load_image(image_path)
    image_tensor = image_sam_transform(image)
    return image_tensor.to(torch.float32).unsqueeze(0)


def load_image_for_vlm(image_path, image_processor, config):
    image_pil = load_image(image_path)
    image_tensor = process_images(
        [image_pil],
        image_processor,
        config
    )
    return image_tensor.to(torch.float16)


def process_prompt(prompt, tokenizer):
    prompt_for_vlm = (
        "<image>\n" + f"### User: {prompt}\n"
    )
    input_ids = tokenizer_image_token(
        prompt_for_vlm,
        tokenizer,
        IMAGE_TOKEN_INDEX,
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
    device="cuda"
):
    image_tensor = load_image_for_vlm(
        image_path, image_processor, config
    )
    image_tensor_for_sam = transform_for_sam(image_path)
    image_tensor_for_sam = image_tensor_for_sam.to(device)
    image_tensor = image_tensor.to(device)
    input_ids = process_prompt(prompt, tokenizer)
    input_ids = input_ids.to(device)

    model.eval()
    model.to(device)

    with autocast(dtype=torch.float16):
        with torch.no_grad():
            output_mask, output_ids = model.generate(
                input_ids=input_ids,
                input_ids_for_seg=None,
                image_tensor_for_vlm=image_tensor,
                image_tensor_for_image_enc=image_tensor_for_sam,
                attention_mask=None,
                temperature=0.7,
                max_new_tokens=512,
                top_p=0.95
            )

            outputs_text = tokenizer.decode(
                output_ids[0, :], skip_special_tokens=True
            )
            res = output_mask.sigmoid().cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            return res, outputs_text


def parse_answer(raw_text):
    parts = raw_text.split("### Assistant:")
    if len(parts) > 1:
        answer = parts[1].split("### Student:")[0]
    else:
        answer = parts[0]
    answer = answer.replace("\n", "")
    answer = answer.replace("### Assistant: ", "")
    answer = answer.replace("### User: ", "")
    return answer.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str,
        default="/home/mamba/ML_project/Testing/Huy/llm_seg"
                "/weight/llava-med-v1.5-mistral-7b"
    )
    parser.add_argument(
        "--checkpoint_path", type=str,
        default="/home/mamba/ML_project/Testing/Huy/llm_seg"
                "/training_results/weights3_full_tuning_6_classes"
                "/llm_seg_20"
    )
    parser.add_argument(
        "--data_path", type=str,
        default="/home/mamba/ML_project/Testing/Huy/llm_seg"
                "/dataset/data"
    )
    parser.add_argument(
        "--annotation_path", type=str,
        default="/home/mamba/ML_project/Testing/Huy/llm_seg"
                "/dataset/annotation_v2"
    )
    parser.add_argument(
        "--results_path", type=str,
        default="results/llm_seg_20_clean/"
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device
    if "cuda" in device:
        gpu_id = int(device.split(":")[-1]) if ":" in device else 0
        torch.cuda.set_device(gpu_id)

    model, tokenizer, image_processor, config = load_model(
        args.model_path, args.checkpoint_path, device
    )

    results_path = args.results_path
    mask_path = os.path.join(results_path, "masks")
    os.makedirs(mask_path, exist_ok=True)

    csv_files = [
        os.path.join(args.annotation_path, f)
        for f in os.listdir(args.annotation_path)
        if f.endswith(".csv")
    ]

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        image_list, mask_list, prompt_list, answer_list = [], [], [], []

        for i in range(len(df)):
            if df.iloc[i]["split"] != "test":
                continue

            image_path = os.path.join(
                args.data_path,
                df.iloc[i]["image_path"].replace(
                    "test_masks", "test_images"
                )
            )
            if "ISIC" in image_path:
                image_path = image_path.replace("_Segmentation", "")
                image_path = image_path.replace(".png", ".jpg")

            idx = df.iloc[i]["image_path"].split("/")[-1].split(".")[0]
            modal = df.iloc[i]["image_path"].split("/")[0]
            prompt = df.iloc[i]["question"]

            print(f"Processing: {image_path}")

            mask, raw_answer = infer(
                prompt,
                image_path,
                image_processor,
                model,
                tokenizer,
                config,
                device=device
            )

            mask_out = mask * 255
            modal_mask_dir = os.path.join(mask_path, modal)
            os.makedirs(modal_mask_dir, exist_ok=True)
            save_mask_path = os.path.join(
                modal_mask_dir, f"{idx}.png"
            )
            cv2.imwrite(save_mask_path, mask_out)

            answer = parse_answer(raw_answer)
            print(f"  Saved: {save_mask_path} | Answer: {answer}")

            image_list.append(df.iloc[i]["image_path"])
            mask_list.append(save_mask_path)
            prompt_list.append(prompt)
            answer_list.append(answer)

        if image_list:
            df_res = pd.DataFrame({
                "image_path": image_list,
                "mask_path": mask_list,
                "prompt": prompt_list,
                "results": answer_list
            })
            df_res.to_csv(
                os.path.join(results_path, f"results_{modal}.csv"),
                index=False
            )


if __name__ == "__main__":
    main()
