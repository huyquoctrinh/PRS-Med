from segment_model.model import build_llm_seg
import torch
from torch.cuda.amp import autocast
from PIL import Image
from data_utils.utils import load_image
from torchvision import transforms
from llava.mm_utils import process_images
from llava.mm_utils import tokenizer_image_token
import cv2

def load_model():
    model, tokenizer, image_processor, config = build_llm_seg(
        model_path="/home/mamba/ML_project/Testing/Huy/llm_seg/weight/llava-med-v1.5-mistral-7b",
        model_base=None,
        load_8bit=False,
        load_4bit=False,
        device="cuda:1"
    )
    tokenizer = model.load_model("/home/mamba/ML_project/Testing/Huy/llm_seg/weights/llm_seg_5")
    return model, tokenizer, image_processor, config

def transform_for_sam(image_path):
    image_sam_transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
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
    prompt_for_vlm = "<image>" + " You are doing the segmentation." + prompt
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
    device = "cuda:1"
):
    image_tensor = load_image_for_vlm(image_path, image_processor, config)
    image_tensor_for_sam = transform_for_sam(image_path)
    image_tensor_for_sam = image_tensor_for_sam.to(device)
    image_tensor = image_tensor.to(device)
    input_ids = process_prompt(prompt, tokenizer)
    input_ids = input_ids.to(device)
    # print(input_ids.shape)
    # print(image_tensor.shape)
    # print(image_tensor.dtype)
    print(input_ids)
    model.eval()
    model.to(device)
    with autocast(dtype=torch.float16):
        with torch.no_grad():
            output_mask, output_ids = model.generate(
                input_ids = input_ids,
                image_tensor_for_vlm = image_tensor,
                image_tensor_for_image_enc = image_tensor_for_sam,
                attention_mask = None,
                temperature=0.2,
                max_new_tokens=512,
                top_p=0.95
            )
            print(output_mask.shape, output_ids.shape)
            print(output_ids)
            print(output_ids[0, input_ids.shape[1]:])
            print(output_ids[0, :])
            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
            ouptuts_1 = tokenizer.decode(output_ids[0, :], skip_special_tokens=True)
            print("Output:", outputs)
            print("Output 1:", ouptuts_1)
            masks = output_mask.sigmoid().squeeze(0).cpu().numpy()
            res = masks.transpose(1, 2, 0)
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            cv2.imwrite("output_mask.png", res* 255)

            # outputs_answers = tokenizer.batch_decode(
            #     output_ids[:, input_ids.shape[1]:],
            #     skip_special_tokens=True
            # )[0]
            # print(outputs_answers)

if __name__ == "__main__":
    model, tokenizer, image_processor, config = load_model()
    image_path = "/home/mamba/ML_project/Testing/Huy/llm_seg/dataset/data/lung_CT/test_images/ID00010637202177584971671_25.jpg"
    prompt = "There is the tumour inside the image, where is it position? "
    infer(
        prompt,
        image_path,
        image_processor,
        model,
        tokenizer,
        config
    )


# model.to("cuda:1")
