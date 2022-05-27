"""
Basic Identity Neck (Do Nothing).
By Kwanwoo Park, 2022.
"""
import torch.nn as nn

class Neck_basic(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feat_list):
        return feat_list

def Neck():
    return Neck_basic(), "Identity"