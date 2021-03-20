import torch
from torch import nn
import numpy as np
from losses.registry import LOSS

@LOSS.register('anml_loss')
class AnmlLoss(nn.Module):
    def __init__(self, cfg):
        super(AnmlLoss, self).__init__()
        self.margin = 0.09 #This parameter can speed up the algorithm, when it is set as 0.0. the algorithm can also obtain a competitive result.

        self.scale_pos = cfg.LOSSES.ANML_LOSS.SCALE_POS
        self.scale_neg = cfg.LOSSES.ANML_LOSS.SCALE_NEG

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)
        sim_mat = torch.matmul(feats, torch.t(feats))
        #        print(max(sim_mat[0,:]))

        epsilon = 1e-5
        loss = list()
        thresh1 = 0.501
        thresh2 = 0.531
        for i in range(batch_size):
            pos_pair_ = sim_mat[i][labels == labels[i]]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = sim_mat[i][labels != labels[i]]

            if len(neg_pair_) < 1 or len(pos_pair_) < 1:
                continue

            #neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]
            pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_)]

            neg_pair = neg_pair_
            #pos_pair = pos_pair_

            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            pos_loss = 1.0 / self.scale_pos * torch.log(
                ((torch.sum(torch.exp(-self.scale_pos * (pos_pair)))) + np.exp(-self.scale_pos * thresh1)) / (
                            len(pos_pair) + 1))
            neg_loss = 1.0 / self.scale_neg * torch.log(
                ((torch.sum(torch.exp(self.scale_neg * (neg_pair)))) + np.exp(self.scale_neg * thresh2)) / (
                            len(neg_pair) + 1))

            loss.append(torch.log(5.33 + torch.exp(pos_loss + neg_loss)))
                # loss.append(torch.clamp(pos_loss + neg_loss,1,2))

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)

        loss = sum(loss) / batch_size
        return loss