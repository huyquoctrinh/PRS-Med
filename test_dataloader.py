from llava.mm_utils import get_model_name_from_path
from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model
from data_utils.dataset import create_dataloader
import torch 
import numpy as np
import cv2

# def convert_mask_back(mask_path, mask_tensor):
#     mask_tensor = mask_tensor.view(1, 512, 512, 1).cpu().numpy()
#     mask_tensor = np.squeeze(mask_tensor)
#     # mask_tensor = np.clip(mask_tensor, 0, 1)
#     mask_tensor = mask_tensor.astype(np.uint8)
#     mask_tensor = cv2.resize(mask_tensor, (512, 512), interpolation=cv2.INTER_NEAREST)
#     cv2.imwrite(mask_path, mask_tensor*255)

disable_torch_init()
model_name = get_model_name_from_path("/home/mamba/ML_project/Testing/Huy/llm_seg/weight/llava-med-v1.5-mistral-7b")
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path="/home/mamba/ML_project/Testing/Huy/llm_seg/weight/llava-med-v1.5-mistral-7b",
    model_name = model_name,
    model_base=None,
    load_8bit=False,
    load_4bit=False,
    device="cuda:0"
)

dataloader = create_dataloader(
    data_path="/home/mamba/ML_project/Testing/Huy/llm_seg/dataset/data",
    annotation_path="/home/mamba/ML_project/Testing/Huy/llm_seg/dataset/annotation",
    data_config=model.config,
    image_processor=image_processor,
    tokenizer=tokenizer,
    batch_size=1,
    mode="train"
)

cnt = 0
mask_test = "./mask_test/"
for batch in dataloader["train"]:
    cnt+=1
    # print(batch)
    print(batch['input_ids'])
    print(batch['input_ids'].shape)
    print(batch['image_tensor'].shape)
    print(batch['attention_masks'].shape)
    print(batch['mask_tensor'].shape)
    print(batch['answers_ids'].shape)
    print("====================")
    # mask_path = mask_test + str(cnt) + ".png"
    # convert_mask_back(mask_path, batch['mask_tensor'])
