"""
Basic head for Classification.
By Kwanwoo Park, 2022.
"""
import torch.nn as nn

class Head_basic(nn.Module):
    def __init__(self, target_feat, Cins, class_num):
        super().__init__()
        self.TF = target_feat
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.FC = nn.Sequential(nn.Linear(in_features=Cins[target_feat], out_features=class_num, bias=True),
                                nn.ReLU(),
                                nn.Linear(in_features=class_num, out_features=class_num, bias=True))

    def forward(self, feat_list):
        x = self.pool(feat_list[self.TF])
        x = self.FC(self.flatten(x))
        return x

def Head(target_feat, Cins, class_num):
    return Head_basic(target_feat, Cins, class_num), "BasicHead"