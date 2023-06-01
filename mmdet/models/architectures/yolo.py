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
    def __init__(self, backbone, neck, yolo_head):
        super(PPYOLO, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.yolo_head = yolo_head

    def forward(self, x, im_size=None, gt_bbox=None, targets=None):
        '''
        获得损失（训练）、推理 都要放在forward()中进行，否则DDP会计算错误结果。
        '''
        body_feats = self.backbone(x)
        fpn_feats = self.neck(body_feats)
        out = self.yolo_head(fpn_feats, im_size, gt_bbox, targets)
        return out

    def export_ncnn(self, ncnn_data, bottom_names):
        x = bottom_names[0]
        im_scale = bottom_names[1]
        body_feats_names = self.backbone.export_ncnn(ncnn_data, [x, ])
        fpn_feats_names = self.neck.export_ncnn(ncnn_data, body_feats_names)
        outputs = self.yolo_head.export_ncnn(ncnn_data, fpn_feats_names, [im_scale, ])
        return outputs



