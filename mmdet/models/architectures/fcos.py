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


class FCOS(torch.nn.Module):
    def __init__(self, backbone, fpn, head):
        super(FCOS, self).__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.head = head

    def forward(self, x, im_scale=None):
        body_feats = self.backbone(x)
        body_feats, spatial_scale = self.fpn(body_feats)
        out = self.head.get_prediction(body_feats, im_scale)
        return out

    def train_model(self, x, tag_labels, tag_bboxes, tag_centerness):
        body_feats = self.backbone(x)
        body_feats, spatial_scale = self.fpn(body_feats)
        out = self.head.get_loss(body_feats, tag_labels, tag_bboxes, tag_centerness)
        return out

    def add_param_group(self, param_groups, base_lr, base_wd):
        self.backbone.add_param_group(param_groups, base_lr, base_wd)
        self.fpn.add_param_group(param_groups, base_lr, base_wd)
        self.head.add_param_group(param_groups, base_lr, base_wd)



