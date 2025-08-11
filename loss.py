import torch.nn.functional as F
from torch import nn
import torch 

class BCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss(weight=weight, size_average=size_average, reduction='mean')

    def forward(self, pred, target):
        size = pred.size(0)
        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)
        pred_flat = pred_flat.float()
        target_flat = target_flat.float()
        loss = self.bceloss(pred_flat, target_flat)

        return loss
    
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1

        size = pred.size(0)

        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        intersection = pred_flat * target_flat
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_flat.sum(1) + target_flat.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss

class BceDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss(weight, size_average)
        self.dice = DiceLoss()

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = diceloss + bceloss

        return loss

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()

def iou_score(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3)) - inter
    iou = inter / (union + 1e-6)
    return iou.mean()

def dice_score(pred, mask):
    pred = torch.sigmoid(pred)
    # pred = torch.nn.functional.resize(pred, (mask.shape[0], 1, mask.shape[2], mask.shape[3]))
    pred = torch.nn.functional.interpolate(pred, size=(mask.shape[2], mask.shape[3]), mode='bilinear', align_corners=False)
    # mask = torch.nn.functional.resize(mask, (mask.shape[0], 1, mask.shape[2], mask.shape[3]))
    inter = (pred * mask).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + mask.sum(dim=(2, 3))
    union = torch.where(union == 0, inter, union)
    dice = (2 * inter + 1e-6) / (union + 1e-6)
    return dice.mean()
