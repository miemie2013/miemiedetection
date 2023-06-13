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
import torch.nn as nn
import torch.nn.functional as F


class BaseCls(torch.nn.Module):
    def __init__(self, backbone, in_channel, num_classes):
        super(BaseCls, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.fc = nn.Linear(in_channel, num_classes)
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, x, labels=None):
        '''
        获得损失（训练）、推理 都要放在forward()中进行，否则DDP会计算错误结果。
        '''
        x = self.backbone(x)
        if isinstance(x, list):
            x = x[-1]
        elif isinstance(x, dict):
            x = x['dark5']
        else:
            raise NotImplementedError
        avg_mean = x.mean([2, 3])
        logits = self.fc(avg_mean)
        if self.training:
            cls_targets = F.one_hot(labels.view(-1), self.num_classes)
            cls_targets = cls_targets.to(logits.dtype)
            loss_cls = self.bcewithlog_loss(logits, cls_targets)
            loss_cls = loss_cls.sum(1)
            loss_cls = loss_cls.mean()
            out_dict = {
                'total_loss': loss_cls,
            }
            return out_dict
        else:
            p = torch.sigmoid(logits)
            return p

