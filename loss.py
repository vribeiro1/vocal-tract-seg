import torch
import torch.nn as nn


class MaskedBCEWithLogitsLoss(nn.Module):
    def __init__(self, weight=None, reduction="mean"):
        super(MaskedBCEWithLogitsLoss, self).__init__()
        self.bce_loss_fn = nn.BCEWithLogitsLoss(weight, reduction="none")

        if reduction == "none":
            self.reduction = lambda x: x
        else:
            self.reduction = getattr(torch, reduction)

    def forward(self, input_, target, class_masks):
        bce_loss = self.bce_loss_fn(input_, target)
        masked_loss = class_masks * bce_loss

        return self.reduction(masked_loss)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, input_, target, class_masks):
        input_prob = torch.sigmoid(input_)

        dims = (1, 2, 3)
        intersection = torch.sum(input_prob * target, dims)
        cardinality = torch.sum(input_prob + target, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)
        return torch.mean(1. - dice_score)
