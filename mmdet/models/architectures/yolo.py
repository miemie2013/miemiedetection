#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date:
#   Description :
#
# ================================================================
import torch


class PPYOLO(torch.nn.Module):
    def __init__(self, backbone, head):
        super(PPYOLO, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x, im_size=None):
        body_feats = self.backbone(x)
        out = self.head.get_prediction(body_feats, im_size)
        return out

    def train_model(self, x, gt_box, targets):
        body_feats = self.backbone(x)
        out = self.head.get_loss(body_feats, gt_box, targets)
        return out

    def add_param_group(self, param_groups, base_lr, base_wd):
        self.backbone.add_param_group(param_groups, base_lr, base_wd)
        self.head.add_param_group(param_groups, base_lr, base_wd)



