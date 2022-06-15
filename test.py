"""
Test Code.
By Kwanwoo Park, 2022.
"""
from Model.body.resnet import *
from Model.neck.neck import *
from Model.head.head import *
import torch, json
from PIL import Image
import torchvision.transforms as transforms


def Load_StateDict(load_state_dict, model):
    for key in load_state_dict.copy():
        if key[0:7] == "module.":
            load_state_dict[key[7:]] = load_state_dict[key]
            del load_state_dict[key]

    model.load_state_dict(load_state_dict)
    return model


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


def make_onnx_file(model):
    x = torch.randn(1, 3, 224, 224, device="cuda")
    torch.onnx.export(model, x, "./model.onnx", input_names=["input"], output_names=["output"], opset_version=10)


def test_in_eval(model, json_file, img_root):
    InFile = open(json_file, "r")
    data_list = json.load(InFile)
    InFile.close()
    total = 0
    topk1 = 0
    topk5 = 0

    size = (224, 224)
    image_transforms = transforms.Compose([
        transforms.Resize((size[0] + 32, size[1] + 32)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])

    for data in data_list:
        img_path = img_root + data[0]
        image = Image.open(img_path).convert("RGB")
        image = image_transforms(image).unsqueeze(0).cuda()
        pred = model(image)
        label = data[1]

        _, pred = pred.topk(5, 1, True, True)
        pred = pred[0]
        total += 1
        if label in pred:
            topk5 += 1
            if pred[0] == label:
                topk1 += 1
        print("[USER_PRINT] [%5d/%5d] Top1 = %2.2f    Top5 = %2.2f" % (total, len(data_list), (topk1/total) * 100, (topk5/total) * 100))


if __name__ == '__main__':
    model = Model(1000).cuda().eval()
    load_dict = torch.load("./pretrained/resnet50_mp/ckpt/best_top1.pth")
    Load_StateDict(load_dict["model"], model)
    print("[USER_PRINT] Loaded epoch = %d, Top1 = %2.2f" % (load_dict["epoch"], load_dict["best_top1"]))
    make_onnx_file(model)
    test_in_eval(model, "./ILSVRC/Labels/val.json", "./ILSVRC/Data/CLS-LOC/val/")
