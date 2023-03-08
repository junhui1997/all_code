import torch
import torch.nn as nn
class focal_loss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super().__init__()
        self.gamma = gamma
        if torch.cuda.is_available():
            self.loss = torch.nn.CrossEntropyLoss(weight=weight).cuda()
        else:
            self.loss = torch.nn.CrossEntropyLoss(weight=weight)

    def forward(self, pred, target):
        ce_loss = self.loss(pred, target)
        #ce_loss = torch.nn.functional.cross_entropy(pred, target, reduce=False)
        pt = torch.exp(-ce_loss)
        focal_loss = (1. - pt) ** self.gamma * ce_loss
        return torch.mean(focal_loss)