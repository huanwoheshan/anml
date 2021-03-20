from yacs.config import CfgNode as CN

MODEL_PATH = {
    'bninception': "~/.torch/models/bn_inception-52deb4733.pth",
    'resnet50': "~/.torch/models/resnet50-19c8e357.pth",
#    'resnet50': "/home/songkun/PycharmProjects/ms_loss_01/scripts/output/model_000200.pth",
    'resnet100': "~/.torch/models/resnet101-5d3b4d8f.pth"
}

MODEL_PATH = CN(MODEL_PATH)
