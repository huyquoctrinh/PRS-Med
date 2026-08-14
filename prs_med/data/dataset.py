from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from llava.mm_utils import process_images, tokenizer_image_token

from prs_med.configs.base import DataConfig
from prs_med.data.utils import load_annotation, load_image, load_mask


IMAGE_TOKEN_INDEX = -200

LABEL_MAPPING: Dict[str, int] = {
    "brain": 0,
    "breast": 1,
    "lung_CT": 2,
    "lung_Xray": 3,
    "ISIC": 4,
}


def infer_label_from_path(image_path: str) -> int:
    for keyword, label in LABEL_MAPPING.items():
        if keyword.lower() in image_path.lower():
            return label
    return 5


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed_all(worker_seed)


class PromptSegmentDataset(Dataset):
    """Dataset producing multimodal prompts and segmentation targets."""

    def __init__(
        self,
        records,
        data_config: DataConfig,
        tokenizer,
        image_processor,
        processor_config,
        mode: str,
    ) -> None:
        self.records = records.reset_index(drop=True)
        self.data_config = data_config
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.processor_config = processor_config
        self.mode = mode
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
            ]
        )
        self.sam_transform = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def _prompt_for_vlm(self, question: str) -> str:
        return "<image> \n" + question

    def _prompt_for_answer(self, question: str, answer: str) -> str:
        return "<image>\n" + f"### User: {question} \n" + "### Assistant: \n" + answer

    def _tokenize(self, prompt: str) -> torch.Tensor:
        return tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).squeeze(0)

    def _process_image(self, image_path: str) -> torch.Tensor:
        image = load_image(image_path)
        processed = process_images([image], self.image_processor, self.processor_config)
        return processed.squeeze(0).to(torch.float16)

    def _process_sam_image(self, image_path: str) -> torch.Tensor:
        return self.sam_transform(load_image(image_path)).to(torch.float32)

    def _process_mask(self, mask_path: str) -> torch.Tensor:
        return self.mask_transform(load_mask(mask_path))

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records.iloc[index]

        mask_path = record["image_path"]
        image_path = mask_path.replace("train_masks", "train_images").replace("_Segmentation", "")
        if "ISIC" in image_path:
            image_path = image_path.replace(".png", ".jpg")

        full_mask_path = self.data_config.data_path.rstrip("/") + "/" + mask_path
        full_image_path = self.data_config.data_path.rstrip("/") + "/" + image_path

        label = infer_label_from_path(image_path)

        mask_tensor = self._process_mask(full_mask_path)
        sam_tensor = self._process_sam_image(full_image_path)
        vlm_tensor = self._process_image(full_image_path)

        prompt_question = record["question"]
        answer_text = record["answer"]

        input_ids = self._tokenize(self._prompt_for_vlm(prompt_question))
        answer_ids = self._tokenize(self._prompt_for_answer(prompt_question, answer_text))

        return {
            "input_ids": input_ids,
            "answers_ids": answer_ids,
            "image_tensor": vlm_tensor,
            "image_sam": sam_tensor,
            "mask_tensor": mask_tensor,
            "label": torch.tensor(label, dtype=torch.long),
        }


@dataclass
class PromptCollator:
    ignore_index: int
    max_prompt_length: int

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [item["input_ids"] for item in batch],
            batch_first=True,
            padding_value=self.ignore_index,
        )[:, : self.max_prompt_length].to(torch.int64)

        answers_ids = torch.nn.utils.rnn.pad_sequence(
            [item["answers_ids"] for item in batch],
            batch_first=True,
            padding_value=self.ignore_index,
        )[:, : self.max_prompt_length].to(torch.int64)

        attention_masks = torch.ones_like(answers_ids, dtype=torch.long)
        attention_masks[answers_ids == self.ignore_index] = 0

        image_tensor = torch.stack([item["image_tensor"] for item in batch], dim=0)
        image_sam = torch.stack([item["image_sam"] for item in batch], dim=0)
        mask_tensor = torch.stack([item["mask_tensor"] for item in batch], dim=0)
        labels = torch.stack([item["label"] for item in batch]).to(torch.long)

        return {
            "input_ids": input_ids,
            "answers_ids": answers_ids,
            "attention_masks": attention_masks,
            "image_tensor": image_tensor,
            "image_sam": image_sam,
            "mask_tensor": mask_tensor,
            "label": labels,
        }


def build_datasets(
    data_config: DataConfig,
    tokenizer,
    image_processor,
    processor_config,
) -> Tuple[PromptSegmentDataset, PromptSegmentDataset, PromptSegmentDataset]:
    train_df, test_df = load_annotation(data_config.annotation_path)
    train_size = int(len(train_df) * data_config.train_split)
    val_df = train_df.iloc[train_size:].reset_index(drop=True)
    train_df = train_df.iloc[:train_size].reset_index(drop=True)

    train_dataset = PromptSegmentDataset(
        train_df, data_config, tokenizer, image_processor, processor_config, mode="train"
    )
    val_dataset = PromptSegmentDataset(
        val_df, data_config, tokenizer, image_processor, processor_config, mode="val"
    )
    test_dataset = PromptSegmentDataset(
        test_df, data_config, tokenizer, image_processor, processor_config, mode="test"
    )
    return train_dataset, val_dataset, test_dataset


def build_dataloaders(
    data_config: DataConfig,
    tokenizer,
    image_processor,
    processor_config,
    distributed: bool,
) -> Dict[str, DataLoader]:
    train_dataset, val_dataset, _ = build_datasets(
        data_config, tokenizer, image_processor, processor_config
    )

    collator = PromptCollator(
        ignore_index=data_config.ignore_index,
        max_prompt_length=data_config.max_prompt_length,
    )

    train_sampler: Optional[DistributedSampler] = None
    val_sampler: Optional[DistributedSampler] = None

    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False)
        val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        persistent_workers=data_config.persistent_workers and data_config.num_workers > 0,
        collate_fn=collator,
        worker_init_fn=seed_worker if data_config.num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config.effective_val_batch_size(),
        sampler=val_sampler,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        persistent_workers=data_config.persistent_workers and data_config.num_workers > 0,
        collate_fn=collator,
        worker_init_fn=seed_worker if data_config.num_workers > 0 else None,
    )

    return {"train": train_loader, "val": val_loader}

