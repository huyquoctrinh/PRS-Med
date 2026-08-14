"""Multi-turn conversation dataset for [SEG]-gated segmentation (v1.5).

Builds input_ids/labels turn by turn using the mistral_instruct template.
Only assistant turns are supervised; user turns are IGNORE_INDEX.
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from prs_med.data.utils import binary_loader, load_image
from llava.mm_utils import process_images, tokenizer_image_token
from prs_med.models.seg_vocab import SEG_TOKENS

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
MAX_PROMPT_LENGTH = 1024

MODALITY_LABEL = {
    "brain_tumors_ct_scan": 0,
    "breast_tumors_ct_scan": 1,
    "lung_CT": 2,
    "lung_Xray": 3,
    "skin_rgbimage": 4,
    "polyp_endoscopy": 5,
    "total_segmentator": 6,
}


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    import random as _random
    _random.seed(worker_seed)


def label_from_path(image_path):
    p = str(image_path)
    if "ISIC" in p or "skin" in p:
        return 4
    if "breast" in p:
        return 1
    if "brain" in p:
        return 0
    if "lung_CT" in p:
        return 2
    if "lung_Xray" in p:
        return 3
    if "total_segmentator" in p:
        return 6
    return 5


class ConversationSegDataset(Dataset):
    def __init__(
        self,
        dialogues_path,
        data_path,
        image_processor,
        tokenizer,
        data_config,
        max_length=MAX_PROMPT_LENGTH,
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.data_config = data_config
        self.max_length = max_length

        self.records = []
        with open(dialogues_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

        self.seg_id_for_token = {
            t: tokenizer.convert_tokens_to_ids(t) for t in SEG_TOKENS
        }
        self.seg_ids = set(self.seg_id_for_token.values())

        self.mask_transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
        ])
        self.image_sam_transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.records)

    def build_conversation(self, conversations):
        tok = self.tokenizer
        ids, labels = [], []

        bos = tok.bos_token_id
        if bos is not None:
            ids.append(bos)
            labels.append(IGNORE_INDEX)

        turns = [
            (conversations[i]["value"], conversations[i + 1]["value"])
            for i in range(0, len(conversations) - 1, 2)
        ]

        for i, (user_msg, asst_msg) in enumerate(turns):
            img = DEFAULT_IMAGE_TOKEN + "\n" if i == 0 else ""
            user_part = f"[INST] {img}{user_msg} [/INST]"
            asst_part = f" {asst_msg}{tok.eos_token}"

            u = tokenizer_image_token(user_part, tok, IMAGE_TOKEN_INDEX)
            if bos is not None and len(u) and u[0] == bos:
                u = u[1:]
            a = tok(asst_part, add_special_tokens=False).input_ids

            ids.extend(u)
            labels.extend([IGNORE_INDEX] * len(u))
            ids.extend(a)
            labels.extend(a)

        if len(ids) > self.max_length:
            try:
                img_pos = ids.index(IMAGE_TOKEN_INDEX)
                head_ids = ids[: img_pos + 1]
                head_labels = labels[: img_pos + 1]
            except ValueError:
                head_ids, head_labels = [], []
            keep = self.max_length - len(head_ids)
            ids = head_ids + ids[-keep:]
            labels = head_labels + labels[-keep:]

        return (torch.tensor(ids, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long))

    def resolve_paths(self, rel_mask_path):
        mask_path = os.path.join(self.data_path, rel_mask_path).replace("\\", "/")
        image_path = mask_path
        for split in ("train", "val", "test"):
            image_path = image_path.replace(f"{split}_masks", f"{split}_images")
        image_path = image_path.replace("_Segmentation", "")
        if not os.path.isfile(image_path):
            for ext in (".jpg", ".png", ".jpeg"):
                cand = os.path.splitext(image_path)[0] + ext
                if os.path.isfile(cand):
                    image_path = cand
                    break
        return mask_path, image_path

    def _load_image(self, rel_image_path):
        image_path = os.path.join(self.data_path, rel_image_path).replace("\\", "/")
        image_pil = load_image(image_path)
        image_tensor = process_images(
            [image_pil], self.image_processor, self.data_config
        ).squeeze(0).to(torch.float16)
        image_sam = self.image_sam_transform(image_pil).to(torch.float32)
        return image_pil, image_tensor, image_sam

    def __getitem__(self, idx):
        rec = self.records[idx]
        input_ids, labels = self.build_conversation(rec["conversations"])

        if "targets" in rec:
            _, image_tensor, image_sam = self._load_image(rec["image_path"])
            id2mask = {}
            for t in rec["targets"]:
                tid = self.seg_id_for_token.get(t["seg_token"])
                if tid is None:
                    continue
                mpath = os.path.join(self.data_path, t["mask_path"]).replace("\\", "/")
                id2mask[tid] = self.mask_transform(binary_loader(mpath))
            masks, seg_gt_ids = [], []
            for tid in labels.tolist():
                if tid in id2mask:
                    masks.append(id2mask[tid])
                    seg_gt_ids.append(tid)
            return {
                "input_ids": input_ids,
                "labels": labels,
                "image_tensor": image_tensor,
                "image_sam": image_sam,
                "masks": masks,
                "seg_gt_ids": seg_gt_ids,
                "label": label_from_path(rec["image_path"]),
                "has_seg": len(masks) > 0,
            }

        mask_path, image_path = self.resolve_paths(rec["image_path"])
        image_pil = load_image(image_path)
        image_tensor = process_images(
            [image_pil], self.image_processor, self.data_config
        ).squeeze(0).to(torch.float16)
        image_sam = self.image_sam_transform(image_pil).to(torch.float32)
        mask_tensor = self.mask_transform(binary_loader(mask_path))
        has_seg = bool(rec.get("has_seg", False))
        seg_gt_ids = [tid for tid in labels.tolist() if tid in self.seg_ids]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "image_tensor": image_tensor,
            "image_sam": image_sam,
            "masks": [mask_tensor] if has_seg else [],
            "seg_gt_ids": seg_gt_ids,
            "label": label_from_path(rec["image_path"]),
            "has_seg": has_seg,
        }


def build_collate_fn(pad_token_id):
    def collate_fn(batch):
        input_ids = nn.utils.rnn.pad_sequence(
            [b["input_ids"] for b in batch],
            batch_first=True, padding_value=pad_token_id,
        )[:, :MAX_PROMPT_LENGTH].to(torch.int64)

        labels = nn.utils.rnn.pad_sequence(
            [b["labels"] for b in batch],
            batch_first=True, padding_value=IGNORE_INDEX,
        )[:, :MAX_PROMPT_LENGTH].to(torch.int64)

        attention_mask = (labels != IGNORE_INDEX) | (input_ids != pad_token_id)
        attention_mask = attention_mask.to(torch.long)

        flat_masks = [m for b in batch for m in b["masks"]]
        mask_tensor = torch.stack(flat_masks) if flat_masks else None
        seg_gt_ids = torch.tensor(
            [i for b in batch for i in b["seg_gt_ids"]], dtype=torch.long
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "image_tensor": torch.stack([b["image_tensor"] for b in batch]),
            "image_sam": torch.stack([b["image_sam"] for b in batch]),
            "mask_tensor": mask_tensor,
            "seg_gt_ids": seg_gt_ids,
            "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
            "has_seg": torch.tensor([b["has_seg"] for b in batch], dtype=torch.bool),
        }
    return collate_fn


def create_conv_dataloader(
    dialogues_dir,
    data_path,
    image_processor,
    tokenizer,
    data_config,
    batch_size=2,
    num_workers=8,
    prefix="dialogues",
):
    collate_fn = build_collate_fn(tokenizer.pad_token_id)
    loaders = {}
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()

    for split in ("train", "val"):
        path = os.path.join(dialogues_dir, f"{prefix}_{split}.jsonl")
        if not os.path.isfile(path):
            continue
        ds = ConversationSegDataset(
            dialogues_path=path,
            data_path=data_path,
            image_processor=image_processor,
            tokenizer=tokenizer,
            data_config=data_config,
        )
        sampler = (DistributedSampler(ds, shuffle=(split == "train"))
                   if distributed else None)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=(sampler is None and split == "train"),
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            persistent_workers=(num_workers > 0),
        )
    return loaders
