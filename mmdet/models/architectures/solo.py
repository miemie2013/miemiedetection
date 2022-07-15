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


class SOLO(torch.nn.Module):
    def __init__(self, backbone, neck, solo_head, mask_head):
        super(SOLO, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.solov2_head = solo_head
        self.mask_head = mask_head

    def forward(self, x, im_shape=None, ori_shape=None, targets=None):
        '''
        获得损失（训练）、推理 都要放在forward()中进行，否则DDP会计算错误结果。
        '''
        body_feats = self.backbone(x)
        fpn_feats = self.neck(body_feats)
        seg_pred = self.mask_head(fpn_feats)
        out = self.solov2_head(fpn_feats, seg_pred, im_shape, ori_shape)
        return out

    def add_param_group(self, param_groups, base_lr, base_wd, need_clip, clip_norm):
        self.backbone.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        self.neck.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        self.mask_head.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        self.solov2_head.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)



