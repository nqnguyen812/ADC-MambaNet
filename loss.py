import torch
import torch.nn as nn
from metrics import dice_score

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        dice = dice_score(y_pred, y_true, smooth=1e-3)

        return 1 - dice

class bnce_loss(nn.Module):
    def __init__(self, alpha=0.9, beta=0.1):
        super(bnce_loss, self).__init__()
        self.alpha = alpha
        self.beta = beta


    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)

        y_true_pos = y_true.view(-1)
        y_pred_pos = y_pred.view(-1)
        true_pos = torch.sum(y_true_pos * y_pred_pos)
        false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
        w_bce = - (y_true_pos * torch.log(y_pred_pos + 1e-6) + (1 - y_true_pos) * torch.log(1 - y_pred_pos + 1e-6))
        loss = w_bce / (true_pos + self.alpha * false_neg + self.beta * false_pos + smooth)

        return loss.mean()

class dice_bnce_loss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super(dice_bnce_loss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.tspd_loss = bnce_loss()

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        tspd = self.tspd_loss(pred, target)
        loss = dice * (1 - self.bce_weight) + tspd * self.bce_weight
        return loss