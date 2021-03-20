from .registry import LOSS
from .anml import AnmlLoss


def build_loss(cfg):
    loss_name = cfg.LOSSES.NAME
    assert loss_name in LOSS, \
        f'loss name {loss_name} is not registered in registry :{LOSS.keys()}'
    return LOSS[loss_name](cfg)