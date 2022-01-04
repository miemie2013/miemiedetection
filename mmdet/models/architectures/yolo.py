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
    def __init__(self, backbone, fpn, head):
        super(PPYOLO, self).__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.head = head

    def forward(self, x, im_size=None):
        body_feats = self.backbone(x)
        fpn_feats = self.fpn(body_feats)
        out = self.head(fpn_feats, im_size)
        return out

    def train_model(self, x, gt_bbox, targets):
        body_feats = self.backbone(x)
        fpn_feats = self.fpn(body_feats)
        out = self.head.get_loss(fpn_feats, gt_bbox, targets)
        return out

    def add_param_group(self, param_groups, base_lr, base_wd, need_clip, clip_norm):
        self.backbone.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        self.fpn.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        self.head.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)



