"""
Builder for Model, Criterion and Dataloader.
By Kwanwoo Park, 2022.
"""
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler

from Model.body.resnet import *
from Model.neck.neck import *
from Model.head.head import *
from Model.loss.CEloss import *
from Datasets.ImageNet import Dataset


class Criterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss, name1 = CEloss()
        self.loss_name = name1

    def forward(self, pred, label):
        return [self.ce_loss(pred, label)]


class Model(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        self.body, body_name, out_channels = Resnet50()
        self.neck, neck_name = Neck()
        self.head, head_name = Head(-1, out_channels, class_num)
        self.model_name = body_name + " - " + neck_name + " - " + head_name

    def forward(self, input):
        feat_list = self.body(input)
        feat_list = self.neck(feat_list)
        output = self.head(feat_list)
        return output


class Dataloader(nn.Module):
    def __init__(self, root_folder, batch_size, sizes, worker, world_size, local_rank):
        super().__init__()
        self.root_folder = root_folder
        self.batch_size = batch_size
        self.sizes = sizes
        self.worker = worker
        self.world_size = world_size
        self.local_rank = local_rank
        self.name = ""

    def Train(self):
        TrainDataset = Dataset(self.root_folder, size=self.sizes, training=True)
        self.name = TrainDataset.name
        TrainSampler = DistributedSampler(TrainDataset, num_replicas=self.world_size, rank=self.local_rank, shuffle=True)
        TrainLoader = torch.utils.data.DataLoader(TrainDataset, batch_size=self.batch_size, shuffle=False,
                                                  num_workers=self.worker, pin_memory=True, drop_last=True, sampler=TrainSampler)
        return TrainLoader, TrainSampler

    def Val(self):
        ValDataset = Dataset(self.root_folder, size=self.sizes, training=False)
        self.name = ValDataset.name
        ValSampler = DistributedSampler(ValDataset, num_replicas=self.world_size, rank=self.local_rank, shuffle=False)
        ValLoader = torch.utils.data.DataLoader(ValDataset, batch_size=self.batch_size, shuffle=False,
                                                num_workers=self.worker, pin_memory=True, drop_last=True, sampler=ValSampler)
        return ValLoader, ValSampler
