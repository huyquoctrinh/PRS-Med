"""DDP training for the v1.5 conversational [SEG]-gated segmentation model.

Uses separate input_ids/labels (instruction masking), segmentation loss only
over samples whose dialogue contains [SEG], gradient accumulation, linear
warmup, and NaN guards.
"""
import logging
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from prs_med.trainers.metrics import dice_score, structure_loss

IGNORE_INDEX = -100


def _is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def _bar(*args, **kwargs):
    return tqdm(*args, disable=not _is_main(), **kwargs)


def _all_reduce_sum(t):
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def _finite_or_zero(x, name, stats):
    if x is None:
        return None
    if not torch.isfinite(x):
        stats[name] = stats.get(name, 0) + 1
        return torch.zeros((), device=x.device, dtype=torch.float32)
    return x


def _mask_zero(ref):
    return torch.zeros((), device=ref.device, dtype=torch.float32)


def _run_batch(model, batch, device, stats=None):
    """Shared forward + loss for train and eval."""
    stats = {} if stats is None else stats
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    image_tensor = batch["image_tensor"].to(device, non_blocking=True)
    image_sam = batch["image_sam"].to(device, non_blocking=True)
    mask_tensor = batch["mask_tensor"]
    if mask_tensor is not None:
        mask_tensor = mask_tensor.to(device, non_blocking=True)
    seg_gt_ids = batch.get("seg_gt_ids")
    if seg_gt_ids is not None:
        seg_gt_ids = seg_gt_ids.to(device, non_blocking=True)
    cls_labels = batch["label"].to(device, non_blocking=True)

    outputs_mask, seg_batch_idx, output_cls, lm_loss = model(
        input_ids=input_ids,
        labels=labels,
        image_tensor_for_vlm=image_tensor,
        image_tensor_for_image_enc=image_sam,
        attention_mask=attention_mask,
    )

    inner = model.module if hasattr(model, "module") else model
    seg_ids = getattr(inner, "seg_token_ids", {"[SEG]": inner.seg_token_idx})
    n_seg_in_labels = 0
    for name, tid in seg_ids.items():
        c = int((labels == tid).sum())
        n_seg_in_labels += c
    stats["seg_in_labels"] = stats.get("seg_in_labels", 0) + n_seg_in_labels
    stats["seg_mask_fired"] = stats.get("seg_mask_fired", 0) + (
        0 if outputs_mask is None else outputs_mask.shape[0])
    stats["batches"] = stats.get("batches", 0) + 1

    cls_loss = nn.CrossEntropyLoss()(output_cls, cls_labels)

    if not bool((labels != IGNORE_INDEX).any()):
        lm_loss = torch.zeros((), device=device, dtype=torch.float32)

    kind = getattr(inner, "last_seg_kind", None)
    aligned = (
        outputs_mask is not None
        and mask_tensor is not None
        and seg_gt_ids is not None
        and outputs_mask.shape[0] == mask_tensor.shape[0]
        and kind is not None
        and kind.shape[0] == seg_gt_ids.shape[0]
        and torch.equal(kind.to(seg_gt_ids.device), seg_gt_ids)
    )

    if outputs_mask is not None and aligned:
        outputs_mask = F.interpolate(
            outputs_mask.float(), size=(1024, 1024),
            mode="bilinear", align_corners=False,
        )
        target = mask_tensor.float()
        segment_loss = structure_loss(outputs_mask, target)
        dice = dice_score(outputs_mask, target).detach()
    else:
        segment_loss = _mask_zero(output_cls)
        dice = None

    lm_loss = _finite_or_zero(lm_loss, "nan_lm", stats)
    segment_loss = _finite_or_zero(segment_loss, "nan_seg", stats)
    cls_loss = _finite_or_zero(cls_loss, "nan_cls", stats)
    if dice is not None and not torch.isfinite(dice):
        dice = None

    return segment_loss, cls_loss, lm_loss, dice


class DDPChatTrainer:
    """v1.5 DDP trainer for LLMSegChat."""

    NAN_ABORT_AFTER = 25

    def __init__(self, config, model, dataloaders, device):
        from prs_med.trainers.optimizer_factory import create_optimizer, create_scheduler
        self.config = config
        self.model = model
        self.device = device
        self.train_loader = dataloaders["train"]
        self.val_loader = dataloaders["val"]

        tc = config.training
        self.optimizer = create_optimizer(model, tc.optimizer)
        total_steps = tc.epochs * len(self.train_loader)
        self.scheduler = create_scheduler(self.optimizer, tc.scheduler, total_steps)

    def train(self):
        tc = self.config.training
        base_lrs = [g["lr"] for g in self.optimizer.param_groups]
        stats = {}
        consecutive_nan = 0
        global_step = 0

        for epoch in range(tc.epochs):
            for loader in (self.train_loader, self.val_loader):
                if isinstance(getattr(loader, "sampler", None), DistributedSampler):
                    loader.sampler.set_epoch(epoch)

            self.model.train()
            sums = {"loss": 0.0, "lm": 0.0, "seg": 0.0, "cls": 0.0}
            n_good = 0

            pbar = _bar(self.train_loader, desc=f"Epoch {epoch+1}/{tc.epochs}")
            self.optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(pbar):
                is_last = (step + 1 == len(self.train_loader))
                do_sync = ((step + 1) % tc.grad_accum_steps == 0) or is_last
                sync_ctx = nullcontext() if do_sync else self.model.no_sync()

                if global_step < tc.warmup_steps:
                    for g, base in zip(self.optimizer.param_groups, base_lrs):
                        g["lr"] = base * (global_step + 1) / tc.warmup_steps

                with sync_ctx:
                    with autocast(device_type="cuda", dtype=torch.bfloat16):
                        segment_loss, cls_loss, lm_loss, _ = _run_batch(
                            self.model, batch, self.device, stats)
                        loss = (tc.seg_weight * segment_loss
                                + tc.lm_weight * lm_loss
                                + tc.cls_weight * cls_loss)
                        loss = loss.mean() / tc.grad_accum_steps

                    if not torch.isfinite(loss):
                        stats["nan_loss"] = stats.get("nan_loss", 0) + 1
                        consecutive_nan += 1
                        self.optimizer.zero_grad(set_to_none=True)
                        if consecutive_nan >= self.NAN_ABORT_AFTER:
                            raise RuntimeError(
                                f"loss non-finite for {consecutive_nan} "
                                f"consecutive steps; aborting")
                        global_step += 1
                        continue
                    consecutive_nan = 0
                    loss.backward()

                sums["loss"] += float(loss.detach()) * tc.grad_accum_steps
                sums["lm"] += float(lm_loss.detach())
                sums["cls"] += float(cls_loss.detach())
                n_good += 1

                if do_sync:
                    gnorm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), tc.gradient_clip_norm)
                    if torch.isfinite(gnorm):
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                global_step += 1
                if _is_main():
                    d = max(n_good, 1)
                    pbar.set_postfix(loss=f"{sums['loss']/d:.4f}",
                                     lm=f"{sums['lm']/d:.4f}")

            if global_step >= tc.warmup_steps and self.scheduler is not None:
                self.scheduler.step()
                base_lrs = [g["lr"] for g in self.optimizer.param_groups]

            mean_dice = self._evaluate()

            if _is_main():
                d = max(n_good, 1)
                msg = (f"Epoch [{epoch+1}/{tc.epochs}] "
                       f"train loss {sums['loss']/d:.4f} | val Dice {mean_dice:.4f}")
                print(msg)
                logging.info(msg)

            if _is_main() and (epoch + 1) % tc.save_every == 0:
                self._save_checkpoint(epoch + 1)
            if dist.is_initialized():
                dist.barrier()

    @torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        local_sum = torch.tensor(0.0, device=self.device)
        local_cnt = torch.tensor(0.0, device=self.device)

        for batch in _bar(self.val_loader, desc="Evaluating"):
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                _, _, _, dice = _run_batch(self.model, batch, self.device)
            if dice is not None:
                local_sum += dice
                local_cnt += 1

        total = _all_reduce_sum(local_sum.clone())
        count = _all_reduce_sum(local_cnt.clone())
        return (total / torch.clamp_min(count, 1.0)).item()

    def _save_checkpoint(self, epoch):
        inner = self.model.module if hasattr(self.model, "module") else self.model
        save_dir = self.config.model.save_dir
        os.makedirs(save_dir, exist_ok=True)
        inner.save_checkpoint(os.path.join(save_dir, f"llm_seg_chat_{epoch}"))
