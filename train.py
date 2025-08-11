import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
from data_utils.dataset import PromptSegmentDataset
from segment_model.model_v7 import build_llm_seg
from torch.amp import autocast
from llava.mm_utils import get_model_name_from_path
from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model
from data_utils.dataset import create_dataloader
from loss import structure_loss, dice_score, BceDiceLoss
from tqdm import tqdm
import logging
import torch.nn.functional as F
from optimizers import Adam16

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

def evaluate(model, val_loader, device="cuda:2"): 
    dice_score_list = []
    print("Number of val sample", len(val_loader))
    for batch in tqdm(val_loader, desc="Evaluating"):
        model.eval()
        model.to(device)
        input_ids = batch['input_ids'].to(device)
        image_tensor = batch['image_tensor'].to(device)
        mask_tensor = batch['mask_tensor'].to(device)
        image_sam_tensor = batch['image_sam'].to(device)
        # answer_ids = batch['answers_ids'].to(device)
        # attention_mask = batch['attention_masks'].to(device)
        with torch.no_grad():
            outputs, _ = model(input_ids, image_tensor, image_sam_tensor)
            # print("outputs:", outputs)
            dice_score_value = dice_score(outputs, mask_tensor)
            # print("Dice score value:", dice_score_value)
            # print("loss:", loss.item())
            # print("dice_score_value:", dice_score_value)
            dice_score_list.append(dice_score_value.item())
        # break
    mean_dice = sum(dice_score_list) / len(dice_score_list)
    return mean_dice
        
def train(
    model,
    full_loader,
    optimizer,
    num_epochs=10,
    device="cuda:2"
):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=15,
        eta_min=1e-6
    )

    bce_dice_loss = BceDiceLoss()

    dataloader = full_loader["train"]
    val_dataloader = full_loader["val"]
    for epoch in range(num_epochs):
        model.train()
        model.to(device)
        ep_loss = 0
        total_llm_loss = 0
        total_segment_loss = 0
        total_cls_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            image_tensor = batch['image_tensor'].to(device)
            mask_tensor = batch['mask_tensor'].to(device)
            image_sam_tensor = batch['image_sam'].to(device)
            attention_mask = batch['attention_masks'].to(device)
            answers_ids = batch['answers_ids'].to(device)
            labels = batch['label'].to(device)
            torch.autograd.set_detect_anomaly(True)
            
            with autocast(dtype=torch.bfloat16, device_type=device):
                outputs_mask, output_cls, logit_loss = model(
                    input_ids = input_ids, 
                    image_tensor_for_vlm = image_tensor, 
                    image_tensor_for_image_enc = image_sam_tensor, 
                    attention_mask = attention_mask,
                    answers = answers_ids)
            # print("============/=========")
            # print("outputs:", outputs)
            outputs_mask = F.interpolate(outputs_mask, size=(1024, 1024), mode='bilinear', align_corners=False)
            cls_loss = nn.CrossEntropyLoss()(output_cls, labels)
            # print("cls_loss:", cls_loss.item())
            segment_loss = structure_loss(outputs_mask, mask_tensor)
            
            # if epoch < 2:
            # print(segment_loss, logit_loss, cls_loss)
            loss = segment_loss + 2 * cls_loss + 2 * logit_loss 

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()  

            ep_loss += loss.item()
            avg_loss = ep_loss / (progress_bar.n + 1)
            total_llm_loss += logit_loss.item()
            total_segment_loss += segment_loss.item()
            total_cls_loss += cls_loss.item()
            avg_llm_loss = total_llm_loss / (progress_bar.n + 1)
            avg_segment_loss = total_segment_loss / (progress_bar.n + 1)
            avg_cls_loss = total_cls_loss / (progress_bar.n + 1)
            if progress_bar.n % 1000 == 0:
                logging.info(f"Epoch [{epoch+1}/{num_epochs}], Step [{progress_bar.n}], Loss: {avg_loss}, LLM Loss: {avg_llm_loss}, Segment Loss: {avg_segment_loss}, Cls Loss: {avg_cls_loss}")
            progress_bar.set_postfix(loss=avg_loss, llm_loss=avg_llm_loss, segment_loss=avg_segment_loss, cls_loss=avg_cls_loss)
            # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
            # break
        scheduler.step()
        model.eval()
        ep_loss /= len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {ep_loss}")
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {ep_loss}")
        model.eval()
        mean_dice = evaluate(model, val_dataloader, device=device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Val mean Dice Score: {mean_dice}")
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Val mean Dice Score: {mean_dice}")
        model.save_model(f"/home/mamba/ML_project/Testing/Huy/llm_seg/training_results/train_sam_med_llava_med_new/llm_seg_{epoch+1}")
        # break

# torch.set_default_device("cuda")
device = "cuda:2"
model, tokenizer, image_processor, config = build_llm_seg(
    model_path="/home/mamba/ML_project/Testing/Huy/llm_seg/weight/llava-med-v1.5-mistral-7b",
    model_base=None,
    load_8bit=False,
    load_4bit=False,
    device=device
)

dataloader = create_dataloader(
    data_path="/home/mamba/ML_project/Testing/Huy/llm_seg/dataset/data",
    annotation_path="/home/mamba/ML_project/Testing/Huy/llm_seg/dataset/annotation_v3",
    data_config=config,
    image_processor=image_processor,
    tokenizer=tokenizer,
    batch_size=8,
    mode="train"
)

model.to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = 1e-4,
    weight_decay=1e-5,
    eps = 1e-6
)

# optimizer = torch.optim.Adam(
#     model.parameters(),
#     lr=1e-4,
#     betas=(0.9, 0.999),
#     eps=1e-8,
#     weight_decay=1e-5
# )
# optimizer = Adam16(
#     model.parameters(),
#     lr=1e-4,
#     betas=(0.9, 0.999),
#     eps=1e-8,
#     weight_decay=1e-5
# )

train_params = count_train_parameters(model)
print("Trainable parameters:", train_params)
train(
    model=model,
    full_loader=dataloader,
    optimizer=optimizer,
    num_epochs=20,
    device=device
)

# model.load_model("/home/mamba/ML_project/Testing/Huy/llm_seg/training_results/freeze_sam_med_llava_med/llm_seg_10")
