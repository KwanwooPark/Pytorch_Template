"""
ImageNet Dataloader For Classification.

File structure
root(*ILSVRC/) -- Data        -- CLS-LOC -- train -- n**** -- *.JPEG
               |                         |- val   -- *.JPEG
               |                         |- test  -- *.JPEG
               |
               |- Annotations -- CLS-LOC -- train -- n**** -- *.xml
               |                         |- val   -- *.xml
               |
               |- ImageSets   -- CLS-LOC -- train_cls.txt
               |                         |- val.txt
               |                         |- text.txt
               |
               |- Labels      -- train.json (made)
                              |- val.josn   (made)

This is used in builder.py.
By Kwanwoo Park, 2022.
"""

import torch
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
import json

class Dataset(torch.utils.data.Dataset):
    def __init__(self, RootFolder, size, training):
        super(Dataset, self).__init__()
        self.name = "ImageNet"
        self.RootFolder = RootFolder
        self.training = training
        self.size = size
        self.class_num = 1000
        if training:
            InFile = open(self.RootFolder + "/Labels/train.json", "r")
            self.data_list = json.load(InFile)
            InFile.close()
            self.RootFolder += "/Data/CLS-LOC/train/"
        else:
            InFile = open(self.RootFolder + "/Labels/val.json", "r")
            self.data_list = json.load(InFile)
            InFile.close()
            self.RootFolder += "/Data/CLS-LOC/val/"

        self.h, self.w = size

        self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize,
        ])
        self.val_transforms = transforms.Compose([
            transforms.Resize((size[0] + 32, size[1] + 32)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            # normalize,
        ])


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image, label_one_hot = self.Load_data(idx)
        if self.training:
            image = self.train_transforms(image)
        else:
            image = self.val_transforms(image)
        return {"image": image, "label": label_one_hot, "index": idx}

    def Load_data(self, idx):
        img_path = self.RootFolder + self.data_list[idx][0]
        label = self.data_list[idx][1]
        image = Image.open(img_path).convert("RGB")
        label_one_hot = torch.zeros((self.class_num,), dtype=torch.float)
        label_one_hot[label] = 1
        return image, label_one_hot
