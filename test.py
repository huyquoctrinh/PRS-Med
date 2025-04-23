import requests
from PIL import Image
from io import BytesIO
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import torch 

IMAGE_TOKEN_INDEX = -200

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

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

if __name__ == "__main__":
    tokenizer, model, image_processor, context_len = prepare_model(
        model_path="/home/mamba/ML_project/Testing/Huy/llm_seg/weight/llava-med-v1.5-mistral-7b",
        model_base=None,
        load_8bit=False,
        load_4bit=False,
        device="cuda:1"
    )

    # print(tokenizer.IGNORE_INDEX)
    print(tokenizer.pad_token_id)
    # print(tokenizer.IMAGE_TOKEN_ID)

    image = load_image("/home/mamba/ML_project/Testing/Huy/llm_seg/dataset/data/lung_CT/test_images/ID00010637202177584971671_25.jpg")
    prompt = "<image> \n There is a nodule in the ct image. What is inside this image?"

    image_tensor = process_images([image], image_processor, model.config)

    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    input_ids = tokenizer_image_token(
        prompt, 
        tokenizer, 
        IMAGE_TOKEN_INDEX, 
        return_tensors='pt'
    ).unsqueeze(0).to(model.device)

    batch_size = 2
    # batch_input_ids = input_ids.repeat(batch_size, 1)
    # batch_input_ids = batch_input_ids.to(model.device)
    # print(batch_input_ids.shape)
    print(input_ids)
    print(image_tensor.shape)
    model.eval()
    output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature= 0.2,
            max_new_tokens=512,
            use_cache = True,
            top_p=0.95
    )
    print(output_ids)
    print(output_ids.shape)
    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
    print(outputs)
    # print(output_ids)
    # print(len(output_ids['hidden_states']))
    # print(output_ids['hidden_states'][-1].shape)