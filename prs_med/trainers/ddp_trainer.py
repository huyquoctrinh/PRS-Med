from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from prs_med.configs.base import AppConfig
from prs_med.trainers.metrics import dice_score, structure_loss
from prs_med.trainers.optimizer_factory import create_optimizer, create_scheduler
from prs_med.utils.distributed import (
    all_reduce_mean,
    all_reduce_sum,
    ddp_bar,
    is_main_process,
)


class DDPTrainer:
    def __init__(
        self,
        config: AppConfig,
        model: torch.nn.Module,
        dataloaders: Dict[str, DataLoader],
        device: torch.device,
    ) -> None:
        self.config = config
        self.model = model
        self.module = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        self.dataloaders = dataloaders
        self.device = device

        self.optimizer = create_optimizer(self.module, config.training.optimizer)

        total_steps = len(self.dataloaders["train"]) * config.training.epochs
        self.scheduler = create_scheduler(self.optimizer, config.training.scheduler, total_steps)

        self.cls_criterion = nn.CrossEntropyLoss()
        self.seg_criterion = structure_loss
        self.logger = logging.getLogger("trainer")

    def train(self) -> None:
        for epoch in range(self.config.training.epochs):
            self._set_epoch(epoch)
            train_stats = self._train_epoch(epoch)

            if is_main_process():
                self.logger.info(
                    "Epoch %d - loss: %.4f | llm: %.4f | seg: %.4f | cls: %.4f",
                    epoch + 1,
                    train_stats["loss"],
                    train_stats["llm_loss"],
                    train_stats["seg_loss"],
                    train_stats["cls_loss"],
                )

            if (epoch + 1) % self.config.training.eval_interval == 0:
                val_dice = self._evaluate()
                if is_main_process():
                    self.logger.info("Epoch %d - val dice: %.4f", epoch + 1, val_dice)

            if self.scheduler:
                self.scheduler.step()

            if (epoch + 1) % self.config.training.save_every == 0 and is_main_process():
                self._save_checkpoint(epoch + 1)

    def _set_epoch(self, epoch: int) -> None:
        train_loader = self.dataloaders["train"]
        val_loader = self.dataloaders["val"]
        for loader in (train_loader, val_loader):
            if isinstance(loader.sampler, DistributedSampler):
                loader.sampler.set_epoch(epoch)

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        self.module.train()

        total_loss = 0.0
        total_llm_loss = 0.0
        total_seg_loss = 0.0
        total_cls_loss = 0.0

        train_loader = self.dataloaders["train"]
        progress = ddp_bar(train_loader, desc=f"Epoch {epoch+1}/{self.config.training.epochs}")

        for step, batch in enumerate(progress, start=1):
            self.optimizer.zero_grad(set_to_none=True)

            inputs = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

            with autocast(device_type=self.device.type, dtype=self.module.target_dtype if hasattr(self.module, "target_dtype") else torch.float16):
                masks, cls_logits, llm_loss = self.model(
                    input_ids=inputs["input_ids"],
                    image_tensor_for_vlm=inputs["image_tensor"],
                    image_tensor_for_image_enc=inputs["image_sam"],
                    attention_mask=inputs["attention_masks"],
                    answers=inputs["answers_ids"],
                )

                seg_loss = self.seg_criterion(masks, inputs["mask_tensor"])
                cls_loss = self.cls_criterion(cls_logits, inputs["label"])

                if epoch < 5:
                    loss = seg_loss + 0.5 * cls_loss + llm_loss
                else:
                    loss = seg_loss + llm_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.module.parameters(), self.config.training.gradient_clip_norm)
            self.optimizer.step()

            total_loss += float(loss.detach())
            total_llm_loss += float(llm_loss.detach())
            total_seg_loss += float(seg_loss.detach())
            total_cls_loss += float(cls_loss.detach())

            if is_main_process():
                progress.set_postfix(
                    loss=f"{total_loss/step:.4f}",
                    llm=f"{total_llm_loss/step:.4f}",
                    seg=f"{total_seg_loss/step:.4f}",
                    cls=f"{total_cls_loss/step:.4f}",
                )

        epoch_loss = torch.tensor(total_loss / max(len(train_loader), 1), device=self.device)
        epoch_llm = torch.tensor(total_llm_loss / max(len(train_loader), 1), device=self.device)
        epoch_seg = torch.tensor(total_seg_loss / max(len(train_loader), 1), device=self.device)
        epoch_cls = torch.tensor(total_cls_loss / max(len(train_loader), 1), device=self.device)

        return {
            "loss": all_reduce_mean(epoch_loss).item(),
            "llm_loss": all_reduce_mean(epoch_llm).item(),
            "seg_loss": all_reduce_mean(epoch_seg).item(),
            "cls_loss": all_reduce_mean(epoch_cls).item(),
        }

    @torch.no_grad()
    def _evaluate(self) -> float:
        self.model.eval()
        self.module.eval()

        val_loader = self.dataloaders["val"]
        total_dice = torch.tensor(0.0, device=self.device)
        total_count = torch.tensor(0.0, device=self.device)

        for batch in ddp_bar(val_loader, desc="Evaluating"):
            inputs = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            masks, _ = self.model(
                input_ids=inputs["input_ids"],
                image_tensor_for_vlm=inputs["image_tensor"],
                image_tensor_for_image_enc=inputs["image_sam"],
                attention_mask=inputs["attention_masks"],
                answers=inputs["answers_ids"],
            )
            dice = dice_score(masks, inputs["mask_tensor"])
            total_dice += dice
            total_count += 1

        mean_dice = all_reduce_sum(total_dice) / torch.clamp_min(all_reduce_sum(total_count), 1.0)
        return mean_dice.item()

    def _save_checkpoint(self, epoch: int) -> None:
        output_dir = (
            self.config.checkpoint.output_dir
            if self.config.checkpoint
            else self.config.model.save_dir
        )
        checkpoint_path = Path(output_dir) / f"epoch_{epoch}"
        self.module.save_checkpoint(str(checkpoint_path))

