import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
from data_utils.dataset import PromptSegmentDataset
from segment_model.model import build_llm_seg
from torch.amp import autocast
from llava.mm_utils import get_model_name_from_path
from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model
from data_utils.dataset import create_dataloader
from loss import structure_loss
from tqdm import tqdm
def train(
    model,
    dataloader,
    optimizer,
    num_epochs=10,
    device="cuda:0"
):
    
    for epoch in range(num_epochs):
        model.train()
        model.to(device)
        for batch in tqdm(dataloader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            image_tensor = batch['image_tensor'].to(device)
            mask_tensor = batch['mask_tensor'].to(device)
            # answers_ids = batch['answers_ids'].to(device)
            image_sam_tensor = batch['image_sam'].to(device)
            # with autocast(dtype=torch.float16, device_type=device):

            outputs = model(input_ids, image_tensor, image_sam_tensor)
            print("=====================")
            print("outputs:", outputs)
            loss = structure_loss(outputs, mask_tensor)
            print("======================")
            print("mask_tensor:", mask_tensor)
            print("loss:", loss)
        # print("loss:", loss.item())
            loss.backward()
            optimizer.step()
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
            break
        break
        # model.eval()

device = "cuda:0"
model, tokenizer, image_processor, config = build_llm_seg(
    model_path="/home/mamba/ML_project/Testing/Huy/llm_seg/weight/llava-med-v1.5-mistral-7b",
    model_base=None,
    load_8bit=False,
    load_4bit=False,
    device=device
)

dataloader = create_dataloader(
    data_path="/home/mamba/ML_project/Testing/Huy/llm_seg/dataset/data",
    annotation_path="/home/mamba/ML_project/Testing/Huy/llm_seg/dataset/annotation",
    data_config=config,
    image_processor=image_processor,
    tokenizer=tokenizer,
    batch_size=1,
    mode="train"
)

model.to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01
)

train(
    model=model,
    dataloader=dataloader,
    optimizer=optimizer,
    num_epochs=10,
    device=device
)
