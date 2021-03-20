# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import argparse
import torch

from configs import cfg
from data import build_data
from engine.trainer import do_train
from losses import build_loss
from modeling import build_model
from solver import build_lr_scheduler, build_optimizer
from utils.logger import setup_logger
from utils.checkpoint import Checkpointer

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def train(cfg):
    logger = setup_logger(name='Train', level=cfg.LOGGER.LEVEL)
    logger.info(cfg)
    model = build_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    #model.to(cuda_device = 'cuda:9')

    criterion = build_loss(cfg)

    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    train_loader = build_data(cfg, is_train=True)
    val_loader = build_data(cfg, is_train=False)

    logger.info(train_loader.dataset)
    logger.info(val_loader.dataset)

    arguments = dict()
    arguments["iteration"] = 0

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    checkpointer = Checkpointer(model, optimizer, scheduler, cfg.SAVE_DIR)

    do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        logger
    )

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a retrieval network')
  parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='config file',
        default='/home/songkun/PycharmProjects/TPAMI/configs/example.yaml',
        type=str)
  return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg.merge_from_file(args.cfg_file)
    train(cfg)
