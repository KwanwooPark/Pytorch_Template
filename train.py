"""
Training Code.
By Kwanwoo Park, 2022.
"""
import time
from torch.multiprocessing import Process
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from Utils.utils import *
from builder import *
from config import cfg


class Trainer():
    def Setting_Memory(self, local_rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '2055'

        self.local_rank = local_rank
        self.world_size = world_size

        print("[USER_PRINT] Setting Memory of GPU [ %d / %d ]" % (self.local_rank, self.world_size))
        dist.init_process_group(backend='nccl', init_method='env://', world_size=self.world_size, rank=self.local_rank)
        self.device = torch.device('cuda:%d' % self.local_rank)
        torch.cuda.set_device(self.local_rank)
        torch.backends.cudnn.benchmark = True

        print("[USER_PRINT] Fix randomness (SEED = %d)" % cfg.RANDOM_SEED) if self.local_rank == cfg.R_GPU else False
        if cfg.RANDOM_SEED > 0:
            Fix_Randomness(cfg.RANDOM_SEED)
        dist.barrier()


    def Setting_Record(self):
        if self.local_rank == cfg.R_GPU:
            self.now_time = time.strftime("%Y%m%d%H%M%S")
            self.save_path = cfg.save_path + self.now_time + "/"
            print("[USER_PRINT] Save path: %s" % self.save_path)

            print("[USER_PRINT] Back up code and setting writer")
            Backup_code(self.save_path)
            os.makedirs(self.save_path + "/tb/", exist_ok=True)
            self.writer = SummaryWriter(self.save_path + "/tb/")



    def Setting_Building(self):
        self.Model = Model(cfg.class_num).to(self.device)
        print("[USER_PRINT] MODEL ARCHITECTURE: %s" % self.Model.model_name) if self.local_rank == cfg.R_GPU else False
        self.Model = nn.parallel.DistributedDataParallel(self.Model, device_ids=[self.local_rank], )
        for param in self.Model.parameters():
            dist.broadcast(param.data, src=cfg.R_GPU)

        self.Criterion = Criterion().to(self.device)
        print("[USER_PRINT] Criterion: %s" % self.Criterion.loss_name) if self.local_rank == cfg.R_GPU else False

        dataloader = Dataloader(root_folder=cfg.root_folder,
                                batch_size=cfg.batch_size,
                                sizes=(cfg.height, cfg.width),
                                worker=cfg.data_loader_worker,
                                world_size=self.world_size,
                                local_rank=self.local_rank)
        self.TrainDataLoader, self.TrainSampler = dataloader.Train()
        self.ValDataLoader, self.ValSampler = dataloader.Val()
        self.NumTrainIter = len(self.TrainDataLoader)
        self.NumValIter = len(self.ValDataLoader)
        print("[USER_PRINT] Dataset: %s" % dataloader.name) if self.local_rank == cfg.R_GPU else False
        dist.barrier()

    def Setting_Training(self):
        print("[USER_PRINT] Setting optimizer(%s), scheduler(%s), scaler" % (cfg.optimizer, cfg.scheduler)) if self.local_rank == cfg.R_GPU else False
        if cfg.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.Model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        else:
            self.optimizer = torch.optim.Adam(self.Model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

        if cfg.scheduler == "StepLR":
            self.lr_scheduler = StepLR(self.optimizer, step_size=cfg.step_size * self.NumTrainIter, gamma=cfg.lr_gamma)
        else:
            self.lr_scheduler = CustomScheduler(self.optimizer, warmup_term=cfg.warmup_size * self.NumTrainIter, cosine_term=cfg.cosine_size * self.NumTrainIter, min_ratio=cfg.min_lr)

        self.Scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_MixedPrecision)
        dist.barrier()


    def Load(self):
        if cfg.trained_path is None:
            print("[USER_PRINT] Not Load trained Model") if self.local_rank == cfg.R_GPU else False
            self.epoch, self.best_top1 = 0, 0
        else:
            print("[USER_PRINT] Load Trained Model: %s" % cfg.trained_path) if self.local_rank == cfg.R_GPU else False
            trained_path = cfg.trained_path + "/ckpt/best_top1.pth"
            load_dict = torch.load(trained_path, map_location=self.device)

            self.Model = Load_StateDict(load_dict["model"], self.Model)
            self.optimizer.load_state_dict(load_dict["optimizer"])
            self.lr_scheduler.load_state_dict(load_dict["lr_scheduler"])
            self.Scaler.load_state_dict(load_dict["scaler"])
            self.epoch = load_dict["epoch"] + 1
            self.best_top1 = load_dict["best_top1"]
            if self.local_rank == cfg.R_GPU:
                print("[USER_PRINT] Copy tensorboard files") if self.local_rank == cfg.R_GPU else False
                for tb_file in glob.glob(cfg.trained_path + "/tb/*"):
                    shutil.copy(tb_file, self.save_path + "/tb/")


    def Save(self, save_name):
        if self.local_rank == cfg.R_GPU:
            print("[USER_PRINT] Save Model in %s" % self.save_path)
            save_path = self.save_path + "/ckpt/"
            os.makedirs(save_path, exist_ok=True)
            save_dict = {
                "epoch": self.epoch,
                "model": self.Model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "scaler": self.Scaler.state_dict(),
                "best_top1": self.best_top1
            }
            torch.save(save_dict, save_path + save_name)

    def Train(self):
        print("[USER_PRINT] Start Train") if self.local_rank == cfg.R_GPU else False
        torch.cuda.empty_cache()
        self.Model.train()
        self.TrainSampler.set_epoch(self.epoch)
        dist.barrier()

        for num_iter, batch in enumerate(self.TrainDataLoader):
            for key in batch:
                batch[key] = batch[key].to(device=self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=cfg.use_MixedPrecision):
                preds = self.Model(batch["image"])
                loss_list = self.Criterion(preds, batch["label"])
                loss = loss_list[0]

            self.optimizer.zero_grad(set_to_none=True)
            self.Scaler.scale(loss).backward()
            self.Scaler.step(self.optimizer)
            self.Scaler.update()
            self.lr_scheduler.step()
            torch.cuda.synchronize()
            dist.barrier()

            loss_reduce = loss.clone()
            dist.all_reduce(loss_reduce)
            loss_reduce /= self.world_size
            if self.local_rank == cfg.R_GPU:
                lr = self.optimizer.param_groups[0]['lr']
                print("[USER_PRINT] [%d] [%5d / %5d] loss=%2.4f,  lr=%.6f" % (self.epoch, num_iter, self.NumTrainIter, loss_reduce, lr))
                self.writer.add_scalar("train/loss", loss_reduce, self.epoch * self.NumTrainIter + num_iter)
                self.writer.add_scalar("train/lr", lr, self.epoch * self.NumTrainIter + num_iter)

        self.epoch += 1
        dist.barrier()

    def Validate(self):
        print("[USER_PRINT] Start Evaluation") if self.local_rank == cfg.R_GPU else False
        evaluator = Evaluator()
        torch.cuda.empty_cache()
        self.Model.eval()

        pred_list = []
        label_list = []
        loss_list = []
        for _ in range(self.world_size):
            pred_list.append(torch.zeros((cfg.batch_size, cfg.class_num), device=self.device))
            label_list.append(torch.zeros((cfg.batch_size, cfg.class_num), device=self.device))
            loss_list.append(torch.zeros((1,), device=self.device))

        with torch.no_grad():
            for iteration, batch in enumerate(self.ValDataLoader):
                for key in batch:
                    batch[key] = batch[key].to(device=self.device, non_blocking=True)
                preds = self.Model(batch["image"])
                labels = batch["label"]
                loss = self.Criterion(preds, labels)[0]

                dist.barrier()
                dist.all_gather(pred_list, preds)
                dist.all_gather(label_list, labels)
                dist.all_gather(loss_list, loss)

                if self.local_rank == cfg.R_GPU:
                    print("[USER_PRINT] [%5d / %5d] Eval" % (iteration, self.NumValIter))
                    pred_gahter = torch.cat(pred_list, dim=0).detach().cpu()
                    label_gahter = torch.cat(label_list, dim=0).detach().cpu()
                    loss_gahter = torch.cat(loss_list, dim=0).detach().cpu()
                    evaluator.update(pred_gahter, label_gahter, loss_gahter)

        if self.local_rank == cfg.R_GPU:
            metric = evaluator.get_metric()
            for key, value in metric.items():
                self.writer.add_scalar("val/" + key, value, self.epoch)
            if self.best_top1 < metric["top1"]:
                self.best_top1 = metric["top1"]
                self.Save("best_top1.pth")
            print("[USER_PRINT] Evaluation finish, Top1=%2.2f, Top5=%2.2f, loss=%2.4f" % (metric["top1"], metric["top5"], metric["loss"]))

def main(local_rank, world_size):
    trainer = Trainer()
    trainer.Setting_Memory(local_rank, world_size)    # DDP, gpu, Randomness
    trainer.Setting_Record()                          # Record path, backup
    trainer.Setting_Building()                        # Model, criterion, dataloader
    trainer.Setting_Training()                        # Optimizer, schedular, scalar
    trainer.Load()                                    # Load dict

    # for i in range(cfg.max_epoch):
    while trainer.epoch <= cfg.max_epoch:
        trainer.Train()                               # Train
        trainer.Validate()                            # Val
        if trainer.local_rank == cfg.R_GPU:
            trainer.Save("%d.pth" % trainer.epoch)    # Save
    dist.barrier()
    dist.destroy_process_group()

if __name__ == '__main__':
    print("[USER_PRINT] number of proc per node  : %d" % cfg.process_per_node)

    if cfg.process_per_node > 1:
        processes = []
        for local_rank in range(cfg.process_per_node):

            process = Process(target=main, args=(local_rank, cfg.process_per_node))

            process.start()
            processes.append(process)

        for process in processes:
            process.join()

    else:
        main(0, cfg.process_per_node)



