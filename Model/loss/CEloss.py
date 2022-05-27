"""
Cross Entropy.
By Kwanwoo Park, 2022.
"""
import torch.nn as nn

class CELoss_basic(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, pred, label):
        loss = self.loss(pred, label)
        return loss

def CEloss():
    return CELoss_basic(), "Cross Entropy Loss"