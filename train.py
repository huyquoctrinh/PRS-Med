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
from loss import structure_loss, dice_score
from tqdm import tqdm
import logging

logging.basicConfig(
    filename='logs/training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def count_train_parameters(model):
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([p.numel() for p in trainable_params])
    print(f"Number of trainable parameters: {num_params / 1e6:.2f}M")
    return num_params

def evaluate(model, val_loader, device="cuda:0"): 
    dice_score_list = []
    print("Number of val sample", len(val_loader))
    for batch in tqdm(val_loader, desc="Evaluating"):
        model.eval()
        model.to(device)
        input_ids = batch['input_ids'].to(device)
        image_tensor = batch['image_tensor'].to(device)
        mask_tensor = batch['mask_tensor'].to(device)
        image_sam_tensor = batch['image_sam'].to(device)
        with torch.no_grad():
            outputs = model(input_ids, image_tensor, image_sam_tensor)
            # print("outputs:", outputs)
            dice_score_value = dice_score(outputs, mask_tensor)
            # print("Dice score value:", dice_score_value)
            # print("loss:", loss.item())
            # print("dice_score_value:", dice_score_value)
            dice_score_list.append(dice_score_value.item())
    mean_dice = sum(dice_score_list) / len(dice_score_list)
    return mean_dice
        
def train(
    model,
    full_loader,
    optimizer,
    num_epochs=10,
    device="cuda:0"
):
    
    dataloader = full_loader["train"]
    val_dataloader = full_loader["val"]
    for epoch in range(num_epochs):
        model.train()
        model.to(device)
        ep_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        cnt = 0
        for batch in progress_bar:
            # cnt +=1
            # if cnt > 10:
                # break
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            image_tensor = batch['image_tensor'].to(device)
            mask_tensor = batch['mask_tensor'].to(device)
            image_sam_tensor = batch['image_sam'].to(device)
            with autocast(dtype=torch.float16, device_type=device):
                outputs = model(input_ids, image_tensor, image_sam_tensor)
            # print("============/=========")
            # print("outputs:", outputs)
            loss = structure_loss(outputs, mask_tensor)
            # print("======================")
            # print("mask_tensor:", mask_tensor)
            # print("loss:", loss)
        # print("loss:", loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ep_loss += loss.item()
            avg_loss = ep_loss / (progress_bar.n + 1)
            progress_bar.set_postfix(loss=avg_loss)
            # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
            # break
        # break
        # model.eval()
        ep_loss /= len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {ep_loss}")
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {ep_loss}")
        model.eval()
        mean_dice = evaluate(model, val_dataloader, device=device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Val mean Dice Score: {mean_dice}")
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Val mean Dice Score: {mean_dice}")

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
    batch_size=4,
    mode="train"
)

model.to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01
)

train_params = count_train_parameters(model)
print("Trainable parameters:", train_params)
train(
    model=model,
    full_loader=dataloader,
    optimizer=optimizer,
    num_epochs=10,
    device=device
)
