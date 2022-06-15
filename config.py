"""
Config File.
By Kwanwoo Park, 2022.
"""


class Config():
    def __init__(self):
        """ Data """
        self.height = 224                 # image size
        self.width = 224                  # image size
        self.batch_size = 64              # batch size (per gpu)
        self.class_num = 1000             # number of class (dataset)

        """ Learning Rate & Scheduler """
        self.scheduler = "StepLR"         # select [StepLR, Custom]
        self.learning_rate = 0.1          # learning rate
        self.max_epoch = 90               # end epoch
        # StepLR
        self.step_size = 30               # lr step size (epoch)
        self.lr_gamma = 0.1               # lr gamma
        # Custom
        self.warmup_size = 1              # lr warmup size (epoch)
        self.cosine_size = 90             # lr cosine size (epoch)
        self.min_lr = 0.01                # minimum lr ratio

        """ optimizer """
        self.optimizer = "SGD"            # select [SGD, Adam]
        # SGD
        self.momentum = 0.9               # optimizer momentum
        self.weight_decay = 1e-4          # weight regularization
        # Adam
        self.weight_decay = 1e-4          # weight regularization

        """ Memory """
        self.process_per_node = 4         # number of process(=gpu) per machine (DDP)
        self.R_GPU = 0                    # Main GPU for Logging
        self.data_loader_worker = 2       # number of process per gpu for dataloader

        """ Skill """
        self.RANDOM_SEED = -1             # random seed (if -1, mean random)
        self.use_MixedPrecision = True    # Mixed precision scalar

        """ Path """
        self.trained_path = None          # path for load
        self.save_path = "./Save/"        # path for save
        self.root_folder = "./ILSVRC/"    # path for dataset


cfg = Config()

if __name__ == '__main__':
    print(cfg.__dict__)
