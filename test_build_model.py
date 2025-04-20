from segment_model.model import build_llm_seg
import torch
from torch.cuda.amp import autocast

model = build_llm_seg(
    model_path="/home/mamba/ML_project/Testing/Huy/llm_seg/weight/llava-med-v1.5-mistral-7b",
    model_base=None,
    load_8bit=False,
    load_4bit=False,
    device="cuda:0"
)
model.to("cuda:0")
# print(model)
input_ids = torch.randint(0, 100, (2, 21)).to("cuda:0")
image_tensor_1 = torch.randn(2, 3, 336, 336).to("cuda:0")
image_tensor_2 = torch.randn(2, 3, 1024, 1024).to("cuda:0")
with autocast(dtype=torch.float16):
    output = model(input_ids, image_tensor_1, image_tensor_2)
    print(output.shape)