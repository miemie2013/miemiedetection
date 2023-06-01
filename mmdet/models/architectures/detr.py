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

    def forward(self, inputs, scale_factor=None, targets=None):
        '''
        获得损失（训练）、推理 都要放在forward()中进行，否则DDP会计算错误结果。
        '''
        # Backbone
        body_feats = self.backbone(inputs['image'])

        # Neck
        if self.neck is not None:
            body_feats = self.neck(body_feats)

        # Transformer
        pad_mask = inputs.get('pad_mask', None)
        out_transformer = self.transformer(body_feats, pad_mask, inputs)

        # DETR Head
        if self.training:
            detr_losses = self.detr_head(out_transformer, body_feats, inputs)
            # detr_losses.update({
            #     'loss': paddle.add_n(
            #         [v for k, v in detr_losses.items() if 'log' not in k])
            # })
            return detr_losses
        else:
            preds = self.detr_head(out_transformer, body_feats)
            if self.exclude_post_process:
                bbox, bbox_num, mask = preds
            else:
                bbox, bbox_num, mask = self.post_process(
                    preds, inputs['im_shape'], inputs['scale_factor'],
                    inputs['image'].shape[2:])

            output = {'bbox': bbox, 'bbox_num': bbox_num}
            if self.with_mask:
                output['mask'] = mask
            return output



