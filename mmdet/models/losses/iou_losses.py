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
        # iou.requires_grad = False
        iou = iou.detach()
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

class GIoULoss(object):
    """
    Generalized Intersection over Union, see https://arxiv.org/abs/1902.09630
    Args:
        loss_weight (float): giou loss weight, default as 1
        eps (float): epsilon to avoid divide by zero, default as 1e-10
        reduction (string): Options are "none", "mean" and "sum". default as none
    """

    def __init__(self, loss_weight=1., eps=1e-10, reduction='none'):
        self.loss_weight = loss_weight
        self.eps = eps
        assert reduction in ('none', 'mean', 'sum')
        self.reduction = reduction

    def bbox_overlap(self, box1, box2, eps=1e-10):
        """calculate the iou of box1 and box2
        Args:
            box1 (Tensor): box1 with the shape (..., 4)
            box2 (Tensor): box1 with the shape (..., 4)
            eps (float): epsilon to avoid divide by zero
        Return:
            iou (Tensor): iou of box1 and box2
            overlap (Tensor): overlap of box1 and box2
            union (Tensor): union of box1 and box2
        """
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2

        xkis1 = torch.maximum(x1, x1g)
        ykis1 = torch.maximum(y1, y1g)
        xkis2 = torch.minimum(x2, x2g)
        ykis2 = torch.minimum(y2, y2g)
        w_inter = F.relu(xkis2 - xkis1)
        h_inter = F.relu(ykis2 - ykis1)
        overlap = w_inter * h_inter

        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2g - x1g) * (y2g - y1g)
        union = area1 + area2 - overlap + eps
        iou = overlap / union

        return iou, overlap, union

    def __call__(self, pbox, gbox, iou_weight=1., loc_reweight=None):
        # x1, y1, x2, y2 = paddle.split(pbox, num_or_sections=4, axis=-1)
        # x1g, y1g, x2g, y2g = paddle.split(gbox, num_or_sections=4, axis=-1)
        # torch的split和paddle有点不同，torch的第二个参数表示的是每一份的大小，paddle的第二个参数表示的是分成几份。
        x1, y1, x2, y2 = torch.split(pbox, split_size_or_sections=1, dim=-1)
        x1g, y1g, x2g, y2g = torch.split(gbox, split_size_or_sections=1, dim=-1)
        box1 = [x1, y1, x2, y2]
        box2 = [x1g, y1g, x2g, y2g]
        iou, overlap, union = self.bbox_overlap(box1, box2, self.eps)
        xc1 = torch.minimum(x1, x1g)
        yc1 = torch.minimum(y1, y1g)
        xc2 = torch.maximum(x2, x2g)
        yc2 = torch.maximum(y2, y2g)

        area_c = (xc2 - xc1) * (yc2 - yc1) + self.eps
        miou = iou - ((area_c - union) / area_c)
        if loc_reweight is not None:
            loc_reweight = torch.reshape(loc_reweight, shape=(-1, 1))
            loc_thresh = 0.9
            giou = 1 - (1 - loc_thresh
                        ) * miou - loc_thresh * miou * loc_reweight
        else:
            giou = 1 - miou
        if self.reduction == 'none':
            loss = giou
        elif self.reduction == 'sum':
            loss = torch.sum(giou * iou_weight)
        else:
            loss = torch.mean(giou * iou_weight)
        return loss * self.loss_weight




