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


class DETR(torch.nn.Module):
    def __init__(self, backbone, transformer, detr_head, neck, post_process, with_mask=False, exclude_post_process=False):
        super(DETR, self).__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.detr_head = detr_head
        self.neck = neck
        self.post_process = post_process
        self.with_mask = with_mask
        self.exclude_post_process = exclude_post_process

    def forward(self, x, scale_factor=None, targets=None):
        '''
        获得损失（训练）、推理 都要放在forward()中进行，否则DDP会计算错误结果。
        '''
        body_feats = self.backbone(x)
        fpn_feats = self.neck(body_feats)
        out = self.head(fpn_feats, export_post_process=True)
        if self.training:
            out = self.head.get_loss(out, targets)
            return out
        else:
            out = self.head.post_process(out, scale_factor)
            return out

    def add_param_group(self, param_groups, base_lr, base_wd, need_clip, clip_norm):
        self.backbone.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        self.neck.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        self.head.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)



