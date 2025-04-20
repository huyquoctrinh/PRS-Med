import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from data_utils.utils import load_annotation, load_image, binary_loader
from llava.mm_utils import tokenizer_image_token
from llava.mm_utils import process_images
from torchvision import transforms 
import os 

IGNORE_INDEX = -100
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
        if mode == "train":
            self.annotation_df = self.train_df
        elif mode == "test":
            self.annotation_df = self.test_df

        self.IMAGE_TOKEN_INDEX = -200
        self.image_processor = image_processor
        self.data_config = data_config
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
        ])

        self.image_sam_transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.annotation_df)

    def answer_process(self, answer):
        answer_ids = self.tokenizer.encode(
            answer, 
            add_special_tokens=False, 
            return_tensors='pt'
        ).squeeze(0)
        return answer_ids

    def prompt_process(self, prompt):
        # Process the prompt to get the input_ids and attention_mask
        prompt_for_vlm = "<image> " + prompt
        input_ids = tokenizer_image_token(        
            prompt_for_vlm, 
            self.tokenizer, 
            self.IMAGE_TOKEN_INDEX, 
            return_tensors='pt'
        ).unsqueeze(0)
        
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
        image_path = mask_path.replace("train_masks", "train_images")
        prompt = self.annotation_df.iloc[idx]['description']
        answers = self.annotation_df.iloc[idx]['position']
        mask_tensor = self.process_mask(mask_path)
        image_sam_tensor = self.process_sam_image(image_path)
        # Process the image and prompt
        image_tensor = self.process_image(image_path)
        input_ids = self.prompt_process(prompt)
        answers_ids = self.answer_process(answers)    
        return {
            'input_ids': input_ids,
            'image_tensor': image_tensor,
            'mask_tensor': mask_tensor,
            'answers_ids': answers_ids,
            "image_sam": image_sam_tensor
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
        'image_sam': image_sam_tensor
    }

def create_dataloader(
    data_path,
    annotation_path,
    data_config,
    image_processor,
    tokenizer,
    batch_size=2,
    mode="train"
):
    dataset = PromptSegmentDataset(
        data_path=data_path,
        annotation_path=annotation_path,
        data_config=data_config,
        image_processor=image_processor,
        tokenizer=tokenizer,
        mode=mode
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    return dataloader

# if __name__ == "__main__":

