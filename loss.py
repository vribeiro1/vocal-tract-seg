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
