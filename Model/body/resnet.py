"""
ResNet For Body.
By Kwanwoo Park, 2022.
"""
import torch.nn as nn

class Stem(nn.Module):
    def __init__(self, Cout):
        super().__init__()
        self.conv1 = nn.Conv2d(3, Cout, kernel_size=(7, 7), padding=(3, 3), stride=(2, 2))
        self.bn1 = nn.BatchNorm2d(num_features=Cout)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(3, 3), stride= (2, 2), padding=(1, 1))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.pool(out)
        return out

class SmallBlock(nn.Module):
    def __init__(self, Cin, Cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(Cin, Cout, kernel_size=(3, 3), padding=(1, 1), stride=(stride, stride))
        self.bn1 = nn.BatchNorm2d(num_features=Cout)
        self.conv2 = nn.Conv2d(Cout, Cout, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(Cout)
        self.act = nn.ReLU()

        if stride > 1 or Cin != Cout:
            self.skip = nn.Sequential(nn.Conv2d(Cin, Cout, kernel_size=(1, 1), padding=(0, 0), stride=(stride, stride)),
                                      nn.BatchNorm2d(Cout))
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += self.skip(x)
        out = self.act(out)
        return out

class LargeBlock(nn.Module):
    def __init__(self, Cin, Cout, stride):
        super().__init__()
        Cmid = int(Cout / 4)
        self.conv1 = nn.Conv2d(Cin, Cmid, kernel_size=(1, 1), padding=(0, 0))
        self.bn1 = nn.BatchNorm2d(Cmid)

        self.conv2 = nn.Conv2d(Cmid, Cmid, kernel_size=(3, 3), padding=(1, 1), stride=(stride, stride))
        self.bn2 = nn.BatchNorm2d(Cmid)

        self.conv3 = nn.Conv2d(Cmid, Cout, kernel_size=(1, 1), padding=(0, 0))
        self.bn3 = nn.BatchNorm2d(Cout)
        self.act = nn.ReLU()

        if stride > 1 or Cin != Cout:
            self.skip = nn.Sequential(nn.Conv2d(Cin, Cout, kernel_size=(1, 1), padding=(0, 0), stride=(stride, stride)),
                                      nn.BatchNorm2d(Cout))
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out += self.skip(x)
        out = self.act(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, Num_blocks, expansion=1):
        super().__init__()
        self.stem = Stem(64)
        self.NC = 64
        self.Cins = Num_channel = [64, 128, 256, 512]
        self.expansion = expansion

        self.stage1 = self._make_stage(block, Num_channel[0], Num_blocks[0], 1)
        self.stage2 = self._make_stage(block, Num_channel[1], Num_blocks[1], 2)
        self.stage3 = self._make_stage(block, Num_channel[2], Num_blocks[2], 2)
        self.stage4 = self._make_stage(block, Num_channel[3], Num_blocks[3], 2)
        self.outputs = {}

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_stage(self, block, channel, iter, stride):
        stage = []
        stage.append(block(self.NC, channel * self.expansion, stride))
        self.NC = channel * self.expansion
        for _ in range(iter - 1):
            stage.append(block(self.NC, channel * self.expansion, 1))

        return nn.Sequential(*stage)

    def forward(self, x):
        feat4 = self.stem(x)
        feat4 = self.stage1(feat4)
        feat8 = self.stage2(feat4)
        feat16 = self.stage3(feat8)
        feat32 = self.stage4(feat16)
        return [feat4, feat8, feat16, feat32]


def Resnet18():
    return ResNet(SmallBlock, [2, 2, 2, 2], 1), "ResNet18", [64, 128, 256, 512]


def Resnet34():
    return ResNet(SmallBlock, [3, 4, 6, 3], 1), "ResNet34", [64, 128, 256, 512]


def Resnet50():
    return ResNet(LargeBlock, [3, 4, 6, 3], 4), "ResNet50", [256, 512, 1024, 2048]


def Resnet101():
    return ResNet(LargeBlock, [3, 4, 23, 3], 4), "ResNet101", [256, 512, 1024, 2048]
