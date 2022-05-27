"""
Utils Code.
This contains (load_state_dict).
By Kwanwoo Park, 2022.
"""

import torch
import random
import numpy as np
import math
import shutil, os, glob


def Load_StateDict(load_state_dict, model):
    model_state_dict = model.state_dict()
    for key in load_state_dict:
        if key in model_state_dict:
            if load_state_dict[key].shape != model_state_dict[key].shape:
                print("[USER_PRINT] Skip Params %s (Unmatched Shape)." % key)
                load_state_dict[key] = model_state_dict[key]
        else:
            print("[USER_PRINT] Skip Params %s (Not in Model)." % key)

    for key in model_state_dict:
        if key not in load_state_dict:
            print("[USER_PRINT] There is no %s." % key)
            load_state_dict[key] = model_state_dict[key]
    model.load_state_dict(load_state_dict, strict=False)
    return model


def Fix_Randomness(RANDOM_SEED):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def Backup_code(save_path):
    save_path = save_path + "/code/"

    folder_list = [tmp[0] for tmp in os.walk("./")]

    skip_folders = [".idea", "pycache", "Save", "ILSVRC"]

    for folder in folder_list:
        breaked = False
        for skip_folder in skip_folders:
            if skip_folder in folder:
                breaked = True
                break
        if breaked:
            continue
        os.makedirs(save_path + folder, exist_ok=True)
        code_list = glob.glob(folder + "/*.py")
        for code_file in code_list:
            shutil.copy(code_file, save_path + folder)


class CustomScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_term, cosine_term, min_ratio, last_epoch=-1):
        phi = 3.14159265358979323846264338
        def lr_lambda(step):
            if step < warmup_term:
                return float(step) / float(max(1.0, warmup_term))
            elif step < (cosine_term + warmup_term):
                lr = (math.cos((step - warmup_term) / cosine_term * phi) + 1) / 2
                return lr * (1 - min_ratio) + min_ratio
            else:
                return min_ratio

        super(CustomScheduler, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)


class Evaluator():
    def __init__(self):
        self.t_num = 0
        self.loss = 0
        self.top1 = 0
        self.top5 = 0

    def update(self, pred, label, loss):
        e_num = pred.size(0)
        label = torch.tile(torch.argmax(label, dim=1, keepdim=True), (1, 5))
        _, pred = pred.topk(5, 1, True, True)

        self.loss = self.loss * (self.t_num / (self.t_num + e_num)) + torch.mean(loss) * (e_num / (self.t_num + e_num))
        self.top5 += torch.sum(pred == label)
        self.top1 += torch.sum(pred[:, 0] == label[:, 0])
        self.t_num += e_num

    def get_metric(self):
        metric = {}
        metric["top1"] = (self.top1 / self.t_num) * 100
        metric["top5"] = (self.top5 / self.t_num) * 100
        metric["loss"] = self.loss
        return metric