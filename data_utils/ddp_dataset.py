from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
import random, numpy as np
from data_utils.utils import load_annotation, load_image, binary_loader
from llava.mm_utils import tokenizer_image_token
from llava.mm_utils import process_images
from torchvision import transforms 
import os 
import torch.nn as nn
def seed_worker(worker_id):
    """Ensure different randomness in each worker (important for augmentations)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

IGNORE_INDEX = 0
MAX_PROMPT_LENGTH = 512

class PromptSegmentDataset(Dataset):
    def __init__(
        self,
        data_path,
        annotation_path,
        data_config,
        image_processor,
        tokenizer,
        trainsize = 512,
        mode = "train"
    ):
        self.data_path = data_path
        self.annotation_path = annotation_path
        self.tokenizer = tokenizer
        self.annotation_df = None
        self.train_df, self.test_df = load_annotation(annotation_path)
        self.trainsize = trainsize
        self.annotation_df = self.train_df
        # if mode == "train":
        #     self.annotation_df = self.train_df
        # elif mode == "test":
        #     self.annotation_df = self.test_df

        self.IMAGE_TOKEN_INDEX = -200
        self.image_processor = image_processor
        self.data_config = data_config
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
        ])

        self.image_sam_transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.annotation_df)

    def answer_process(self, question, prompt, answer):
        # Process the answer to get the input_ids
        input_prompt = "<image>\n" + f"### User: {question} \n" + "### Assistant: \n" + answer
        # print("Input prompt:", input_prompt)
        answer_ids = tokenizer_image_token(
            input_prompt, 
            self.tokenizer, 
            self.IMAGE_TOKEN_INDEX, 
            return_tensors='pt'
        )
        # print(answer_ids)
        return answer_ids

    def prompt_process(self, prompt):
        # Process the prompt to get the input_ids and attention_mask
        # prompt_for_vlm = "<image> " + "### User: You are doing the segmentation for the tumour with the condition: " + prompt + " Where is the position of the tumour? \n"
        prompt_for_vlm = "<image> \n" + prompt 
        input_ids = tokenizer_image_token(
            prompt_for_vlm, 
            self.tokenizer, 
            self.IMAGE_TOKEN_INDEX, 
            return_tensors='pt'
        )
        # print("Input ids key full:", input_ids)
        return input_ids
    
    def process_image(self, image_path):
        # Process the image using the image processor
        image_pil = load_image(image_path)
        image_tensor = process_images(
            [image_pil], 
            self.image_processor, 
            self.data_config
        )
        # image_tensor = image_tensor.to(self.data_config.device, dtype=torch.float16)

        return image_tensor.squeeze(0).to(torch.float16)
    
    def process_sam_image(self, image_path):
        image_pil = load_image(image_path)
        image_sam_tensor = self.image_sam_transform(image_pil)
        return image_sam_tensor.to(torch.float32)

    def process_mask(self, mask_path):
        # Process the mask using the image processor
        mask_image = binary_loader(mask_path)
        mask_tensor = self.mask_transform(mask_image)
        return mask_tensor

    def __getitem__(self, idx):
        # Get the image path and prompt from the dataframe
        mask_path = os.path.join(self.data_path, self.annotation_df.iloc[idx]['image_path'])
        mask_path = mask_path.replace("\\", "/")
        image_path = mask_path.replace("train_masks", "train_images").replace("_Segmentation", "")
        if "ISIC" in image_path:
            image_path = image_path.replace(".png", ".jpg")
        if "ISIC" in image_path:
            label = 4
        elif "breast" in image_path:
            label = 1
        elif "brain" in image_path:
            label = 0
        # elif "dental" in image_path:
            # label = 2
        elif "lung_CT" in image_path:
            label = 2
        elif "lung_Xray" in image_path:
            label = 3
        else:
            label = 5
        
        prompt = self.annotation_df.iloc[idx]['position']
        question = self.annotation_df.iloc[idx]['question']
        answers = self.annotation_df.iloc[idx]['answer']
        mask_tensor = self.process_mask(mask_path)
        image_sam_tensor = self.process_sam_image(image_path)
        # Process the image and prompt
        image_tensor = self.process_image(image_path)
        input_ids = self.prompt_process(question)
        answers_ids = self.answer_process(question, prompt, answers)    
        return {
            'input_ids': input_ids,
            'image_tensor': image_tensor,
            'mask_tensor': mask_tensor,
            'answers_ids': answers_ids,
            "image_sam": image_sam_tensor,
            "label": label
        }
    
def collate_fn(batch):
    
    padded_input_ids = nn.utils.rnn.pad_sequence(
        [item['input_ids'].squeeze(0) for item in batch], 
        batch_first=True, 
        padding_value=IGNORE_INDEX
    )
    input_ids = padded_input_ids[:, :MAX_PROMPT_LENGTH]
    input_ids = input_ids.to(torch.int64)
    

    padded_answers_ids = nn.utils.rnn.pad_sequence(
        [item['answers_ids'] for item in batch], 
        batch_first=True, 
        padding_value=IGNORE_INDEX
    )

    answers_ids = padded_answers_ids[:, :MAX_PROMPT_LENGTH]
    answers_ids = answers_ids.to(torch.int64)

    attention_masks = torch.ones_like(answers_ids)
    attention_masks[answers_ids == IGNORE_INDEX] = 0
    attention_masks = attention_masks.to(torch.long)

    image_tensor = [item['image_tensor'] for item in batch]
    image_sam_tensor = [item['image_sam'] for item in batch]
    mask_tensor = [item['mask_tensor'] for item in batch]

    image_tensor = torch.stack(image_tensor, dim=0)
    mask_tensor = torch.stack(mask_tensor, dim=0)
    image_sam_tensor = torch.stack(image_sam_tensor, dim=0)
    return {
        'input_ids': input_ids,
        'image_tensor': image_tensor,
        'mask_tensor': mask_tensor,
        'answers_ids': answers_ids,
        'image_sam': image_sam_tensor,
        "attention_masks": attention_masks,
        "label": torch.tensor([item['label'] for item in batch])
    }

def create_dataloader(
    data_path,
    annotation_path,
    data_config,
    image_processor,
    tokenizer,
    batch_size=2,
    mode="train",
    num_workers=8
):
    dataset = PromptSegmentDataset(
        data_path=data_path,
        annotation_path=annotation_path,
        data_config=data_config,
        image_processor=image_processor,
        tokenizer=tokenizer,
        mode=mode
    )

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)]
    )

    # 🔑 Detect if distributed is initialized
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler   = DistributedSampler(val_dataset, shuffle=False)
        shuffle_train = False
        shuffle_val   = False
    else:
        train_sampler = None
        val_sampler   = None
        shuffle_train = True
        shuffle_val   = False

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=shuffle_train,         # ✅ single GPU fallback
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        persistent_workers=(num_workers > 0)
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=shuffle_val,           # ✅ single GPU fallback
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        persistent_workers=(num_workers > 0)
    )

    return {"train": train_dataloader, "val": val_dataloader}
