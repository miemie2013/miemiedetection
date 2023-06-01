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


class PPYOLOE(torch.nn.Module):
    def __init__(self, backbone, neck, yolo_head, for_distill=False, feat_distill_place='neck_feats'):
        super(PPYOLOE, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.yolo_head = yolo_head
        self.for_distill = for_distill
        self.feat_distill_place = feat_distill_place
        self.is_teacher = False
        if for_distill:
            assert feat_distill_place in ['backbone_feats', 'neck_feats']

    def forward(self, x, scale_factor=None, targets=None):
        '''
        获得损失（训练）、推理 都要放在forward()中进行，否则DDP会计算错误结果。
        '''
        body_feats = self.backbone(x)
        fpn_feats = self.neck(body_feats)
        out = self.yolo_head(fpn_feats, targets)
        if self.training or self.is_teacher:
            if self.for_distill:
                if self.feat_distill_place == 'backbone_feats':
                    self.yolo_head.distill_pairs['backbone_feats'] = body_feats
                elif self.feat_distill_place == 'neck_feats':
                    self.yolo_head.distill_pairs['neck_feats'] = fpn_feats
                else:
                    raise ValueError
            return out
        else:
            out = self.yolo_head.post_process(out, scale_factor)
            return out

    def export_ncnn(self, ncnn_data, bottom_names):
        body_feats_names = self.backbone.export_ncnn(ncnn_data, bottom_names)
        fpn_feats_names = self.neck.export_ncnn(ncnn_data, body_feats_names)
        outputs = self.yolo_head.export_ncnn(ncnn_data, fpn_feats_names)
        return outputs



