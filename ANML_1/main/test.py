

import argparse
import torch

from ret_benchmark.config import cfg
from data import build_data
from engine.trainer import do_train
from losses import build_loss
from modeling import build_model
from solver import build_lr_scheduler, build_optimizer
from utils.logger import setup_logger
from utils.checkpoint import Checkpointer

import os

from collections import OrderedDict

import torch
from torch.nn.modules import Sequential

from modeling.registry import HEADS

from modeling.registry import BACKBONES

from .bninception import BNInception
from .resnet import ResNet50



os.environ["CUDA_VISIBLE_DEVICES"] = "1"
logger = setup_logger(name='Train', level=cfg.LOGGER.LEVEL)
logger.info(cfg)

model = Sequential(OrderedDict([
    ('backbone', backbone),
    ('head', head)
]))

def build_model(cfg):
    backbone = build_backbone(cfg)
    head = build_head(cfg)

    model = Sequential(OrderedDict([
        ('backbone', backbone),
        ('head', head)
    ]))

    if cfg.MODEL.PRETRAIN == 'imagenet':
        print('Loading imagenet pretrianed model ...')
        pretrained_path = os.path.expanduser(cfg.MODEL.PRETRIANED_PATH[cfg.MODEL.BACKBONE.NAME])
        model.backbone.load_param(pretrained_path)
    elif os.path.exists(cfg.MODEL.PRETRAIN):
        ckp = torch.load(cfg.MODEL.PRETRAIN)
        model.load_state_dict(ckp['model'])
    return model


model = build_model(cfg)
device = torch.device(cfg.MODEL.DEVICE)
model.to(device)

model.eval()
logger.info('Validation')
labels = val_loader.dataset.label_list
labels = np.array([int(k) for k in labels])
feats = feat_extractor(model, val_loader, logger=logger)

ret_metric = RetMetric(feats=feats, labels=labels)
recall_curr = ret_metric.recall_k(1)