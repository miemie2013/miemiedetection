#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-10-15 14:50:03
#   Description : pytorch_ppyolo
#
# ================================================================
import torch
import torch.nn as nn
import torch as T
import torch.nn.functional as F
import numpy as np

from mmdet.models.bbox_utils import bbox_iou


class IouLoss(nn.Module):
    """
    iou loss, see https://arxiv.org/abs/1908.03851
    loss = 1.0 - iou * iou
    Args:
        loss_weight (float): iou loss weight, default is 2.5
        max_height (int): max height of input to support random shape input
        max_width (int): max width of input to support random shape input
        ciou_term (bool): whether to add ciou_term
        loss_square (bool): whether to square the iou term
    """

    def __init__(self,
                 loss_weight=2.5,
                 giou=False,
                 diou=False,
                 ciou=False,
                 loss_square=True):
        super(IouLoss, self).__init__()
        self.loss_weight = loss_weight
        self.giou = giou
        self.diou = diou
        self.ciou = ciou
        self.loss_square = loss_square

    def forward(self, pbox, gbox):
        iou = bbox_iou(
            pbox, gbox, giou=self.giou, diou=self.diou, ciou=self.ciou)
        if self.loss_square:
            loss_iou = 1 - iou * iou
        else:
            loss_iou = 1 - iou

        loss_iou = loss_iou * self.loss_weight
        return loss_iou


class IouAwareLoss(IouLoss):
    """
    iou aware loss, see https://arxiv.org/abs/1912.05992
    Args:
        loss_weight (float): iou aware loss weight, default is 1.0
        max_height (int): max height of input to support random shape input
        max_width (int): max width of input to support random shape input
    """

    def __init__(self, loss_weight=1.0, giou=False, diou=False, ciou=False):
        super(IouAwareLoss, self).__init__(
            loss_weight=loss_weight, giou=giou, diou=diou, ciou=ciou)

    def forward(self, ioup, pbox, gbox):
        iou = bbox_iou(
            pbox, gbox, giou=self.giou, diou=self.diou, ciou=self.ciou)
        iou.requires_grad = False
        loss_iou_aware = F.binary_cross_entropy_with_logits(
            ioup, iou, reduction='none')
        loss_iou_aware = loss_iou_aware * self.loss_weight
        return loss_iou_aware


class MyIOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(MyIOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        '''
        输入矩形的格式是cx cy w h
        '''
        assert pred.shape[0] == target.shape[0]

        boxes1 = pred
        boxes2 = target

        # 变成左上角坐标、右下角坐标
        boxes1_x0y0x1y1 = torch.cat([boxes1[:, :2] - boxes1[:, 2:] * 0.5,
                                     boxes1[:, :2] + boxes1[:, 2:] * 0.5], dim=-1)
        boxes2_x0y0x1y1 = torch.cat([boxes2[:, :2] - boxes2[:, 2:] * 0.5,
                                     boxes2[:, :2] + boxes2[:, 2:] * 0.5], dim=-1)

        # 两个矩形的面积
        boxes1_area = (boxes1_x0y0x1y1[:, 2] - boxes1_x0y0x1y1[:, 0]) * (boxes1_x0y0x1y1[:, 3] - boxes1_x0y0x1y1[:, 1])
        boxes2_area = (boxes2_x0y0x1y1[:, 2] - boxes2_x0y0x1y1[:, 0]) * (boxes2_x0y0x1y1[:, 3] - boxes2_x0y0x1y1[:, 1])

        # 相交矩形的左上角坐标、右下角坐标
        left_up = torch.maximum(boxes1_x0y0x1y1[:, :2], boxes2_x0y0x1y1[:, :2])
        right_down = torch.minimum(boxes1_x0y0x1y1[:, 2:], boxes2_x0y0x1y1[:, 2:])

        # 相交矩形的面积inter_area。iou
        inter_section = F.relu(right_down - left_up)
        inter_area = inter_section[:, 0] * inter_section[:, 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = inter_area / (union_area + 1e-16)


        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            # 包围矩形的左上角坐标、右下角坐标
            enclose_left_up = torch.minimum(boxes1_x0y0x1y1[:, :2], boxes2_x0y0x1y1[:, :2])
            enclose_right_down = torch.maximum(boxes1_x0y0x1y1[:, 2:], boxes2_x0y0x1y1[:, 2:])

            # 包围矩形的面积
            enclose_wh = enclose_right_down - enclose_left_up
            enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]

            giou = iou - (enclose_area - union_area) / enclose_area
            # giou限制在区间[-1.0, 1.0]内
            giou = torch.clamp(giou, -1.0, 1.0)
            loss = 1 - giou
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss





