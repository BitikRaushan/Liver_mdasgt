import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    """Binary mask loss on logits: BCE + soft Dice.
    (The paper lists 'sparse categorical entropy', which does not fit a 1-channel binary mask.)"""

    def __init__(self, bce_weight=0.5, eps=1e-6):
        super().__init__()
        self.w, self.eps = bce_weight, eps

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        p = torch.sigmoid(logits)
        inter = (p * target).sum((1, 2, 3))
        dice = 1 - (2 * inter + self.eps) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + self.eps)
        return self.w * bce + (1 - self.w) * dice.mean()
