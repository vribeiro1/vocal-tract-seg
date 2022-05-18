import math
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


def evaluate_jaccard(outputs, targets):
    eps = 1e-15
    jaccard_targets = (targets == 1).float()
    jaccard_outputs = torch.sigmoid(outputs)

    intersection = (jaccard_outputs * jaccard_targets).sum()
    union = jaccard_outputs.sum() + jaccard_targets.sum()

    jaccard = (intersection + eps) / (union - intersection + eps)

    return jaccard


def evaluate_dice(jaccard):
    return 2 * jaccard / (1 + jaccard)


class SoftJaccardBCEWithLogitsLoss:
    """
    Loss defined as BCE - log(soft_jaccard)
    Vladimir Iglovikov, Sergey Mushinskiy, Vladimir Osin,
    Satellite Imagery Feature Detection using Deep Convolutional Neural Network: A Kaggle Competition
    arXiv:1706.06169
    """
    eps = 10e-5

    def __init__(self, jaccard_weight=0.0):
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.jacc_weight = jaccard_weight

    def __call__(self, outputs, targets):
        bce_loss = self.bce_loss(outputs, targets)
        jaccard = evaluate_jaccard(outputs, targets)
        log_jaccard = math.log(jaccard + self.eps)
        loss = bce_loss - self.jacc_weight * log_jaccard

        return loss
