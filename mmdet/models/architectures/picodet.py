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


class PicoDet(torch.nn.Module):
    def __init__(self, backbone, neck, head):
        super(PicoDet, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

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

    def export_ncnn(self, ncnn_data, bottom_names):
        body_feats_names = self.backbone.export_ncnn(ncnn_data, bottom_names)
        fpn_feats_names = self.neck.export_ncnn(ncnn_data, body_feats_names)
        outputs = self.head.export_ncnn(ncnn_data, fpn_feats_names)
        return outputs



