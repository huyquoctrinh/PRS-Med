from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class BceDiceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCELoss(reduction="mean")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_sigmoid = torch.sigmoid(pred)
        bce_loss = self.bce(pred_sigmoid, target.float())
        dice_loss = dice_loss_fn(pred_sigmoid, target)
        return bce_loss + dice_loss


def dice_loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    smooth = 1.0
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice_score = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice_score.mean()


def structure_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred_sigmoid = torch.sigmoid(pred)
    inter = ((pred_sigmoid * mask) * weit).sum(dim=(2, 3))
    union = ((pred_sigmoid + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def dice_score(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred_sigmoid = torch.sigmoid(pred)
    pred_resized = torch.nn.functional.interpolate(
        pred_sigmoid, size=(mask.shape[2], mask.shape[3]), mode="bilinear", align_corners=False
    )
    intersection = (pred_resized * mask).sum(dim=(2, 3))
    union = pred_resized.sum(dim=(2, 3)) + mask.sum(dim=(2, 3))
    union = torch.where(union == 0, intersection, union)
    dice = (2 * intersection + 1e-6) / (union + 1e-6)
    return dice.mean()


