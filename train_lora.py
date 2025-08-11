import requests
from PIL import Image
from io import BytesIO
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import torch 
from tqdm import tqdm
from peft import LoraConfig, TaskType, get_peft_model
from data_utils.dataset import create_dataloader
import logging

logging.basicConfig(
    filename='logs/training_llava.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
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

def train(
    model,
    optimizer,
    scheduler,
    train_loader,
    criterion,
    num_epochs=20
):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )
    for epoch in range(num_epochs):
        model.train()
        model.to(torch.bfloat16)
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(model.device)
            image_tensor = batch['image_tensor'].to(model.device)
            mask_tensor = batch['mask_tensor'].to(model.device)
            image_sam_tensor = batch['image_sam'].to(model.device)
            answers_ids = batch['answers_ids'].to(model.device)
            image_size = image_tensor.shape[-2:]
            with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda:1"):
                optimizer.zero_grad()
                loss = model(
                    input_ids = answers_ids, 
                    images = image_tensor,
                    labels = answers_ids
                ).loss
                
                total_loss += loss.item()
                mean_loss = total_loss / (progress_bar.n + 1)
                progress_bar.set_postfix(loss=mean_loss)
                loss.backward()
                optimizer.step()
                scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
        model.save_pretrained(f"/home/mamba/ML_project/Testing/Huy/llm_seg/training_results/ablation/llava_lora_{epoch+1}")

if __name__ == "__main__":
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
    model.train()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01
    )
    full_loader = create_dataloader(
        data_path="/home/mamba/ML_project/Testing/Huy/llm_seg/dataset/data",
        annotation_path="/home/mamba/ML_project/Testing/Huy/llm_seg/dataset/annotation_v2",
        data_config=model.config,
        image_processor=image_processor,
        tokenizer=tokenizer,
        batch_size=8,
        mode="train"
    )
    train(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        train_loader=full_loader["train"],
        criterion=criterion,
        num_epochs=10
    )