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
import math
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
# import paddle.fluid as fluid
# from paddle import ParamAttr
# from paddle.regularizer import L2Decay
# from paddle.nn.initializer import Uniform
# from paddle.nn.initializer import Constant
# from paddle.vision.ops import DeformConv2D

def bbox_iou(box1, box2, giou=False, diou=False, ciou=False, eps=1e-9):
    """calculate the iou of box1 and box2

    Args:
        box1 (list): [x, y, w, h], all have the shape [b, na, h, w, 1]
        box2 (list): [x, y, w, h], all have the shape [b, na, h, w, 1]
        giou (bool): whether use giou or not, default False
        diou (bool): whether use diou or not, default False
        ciou (bool): whether use ciou or not, default False
        eps (float): epsilon to avoid divide by zero

    Return:
        iou (Tensor): iou of box1 and box1, with the shape [b, na, h, w, 1]
    """
    px1, py1, px2, py2 = box1
    gx1, gy1, gx2, gy2 = box2
    x1 = torch.max(px1, gx1)
    y1 = torch.max(py1, gy1)
    x2 = torch.min(px2, gx2)
    y2 = torch.min(py2, gy2)

    overlap = torch.clamp((x2 - x1), min=0) * torch.clamp((y2 - y1), min=0)

    area1 = (px2 - px1) * (py2 - py1)
    area1 = torch.clamp(area1, min=0)

    area2 = (gx2 - gx1) * (gy2 - gy1)
    area2 = torch.clamp(area2, min=0)

    union = area1 + area2 - overlap + eps
    iou = overlap / union

    if giou or ciou or diou:
        # convex w, h
        cw = torch.max(px2, gx2) - torch.min(px1, gx1)
        ch = torch.max(py2, gy2) - torch.min(py1, gy1)
        if giou:
            c_area = cw * ch + eps
            return iou - (c_area - union) / c_area
        else:
            # convex diagonal squared
            c2 = cw**2 + ch**2 + eps
            # center distance
            rho2 = ((px1 + px2 - gx1 - gx2)**2 + (py1 + py2 - gy1 - gy2)**2) / 4
            if diou:
                return iou - rho2 / c2
            else:
                w1, h1 = px2 - px1, py2 - py1 + eps
                w2, h2 = gx2 - gx1, gy2 - gy1 + eps
                delta = torch.atan(w1 / h1) - torch.atan(w2 / h2)
                v = (4 / math.pi**2) * torch.pow(delta, 2)
                alpha = v / (1 + eps - iou + v)
                alpha.requires_grad = False
                return iou - (rho2 / c2 + v * alpha)
    else:
        return iou



