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


def bbox2distance(points, bbox, max_dis=None, eps=0.1):
    """Decode bounding box based on distances.
    Args:
        points (Tensor): Shape (n, 2), [x, y].
        bbox (Tensor): Shape (n, 4), "xyxy" format
        max_dis (float): Upper bound of the distance.
        eps (float): a small value to ensure target < max_dis, instead <=
    Returns:
        Tensor: Decoded distances.
    """
    left = points[:, 0] - bbox[:, 0]
    top = points[:, 1] - bbox[:, 1]
    right = bbox[:, 2] - points[:, 0]
    bottom = bbox[:, 3] - points[:, 1]
    if max_dis is not None:
        left = torch.clamp(left, min=0, max=max_dis - eps)
        top = torch.clamp(top, min=0, max=max_dis - eps)
        right = torch.clamp(right, min=0, max=max_dis - eps)
        bottom = torch.clamp(bottom, min=0, max=max_dis - eps)
    return torch.stack([left, top, right, bottom], -1)


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.
        Args:
            points (Tensor): Shape (n, 2), [x, y].
            distance (Tensor): Distance from the given point to 4
                boundaries (left, top, right, bottom).
            max_shape (tuple): Shape of the image.
        Returns:
            Tensor: Decoded bboxes.
        """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = torch.clamp(x1, min=0, max=max_shape[1])
        y1 = torch.clamp(y1, min=0, max=max_shape[0])
        x2 = torch.clamp(x2, min=0, max=max_shape[1])
        y2 = torch.clamp(y2, min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], -1)


def bbox_center(boxes):
    # boxes              [A, 4]  矩形左上角坐标、右下角坐标；单位是像素。  返回矩形中心点坐标
    """Get bbox centers from boxes.
    Args:
        boxes (Tensor): boxes with shape (..., 4), "xmin, ymin, xmax, ymax" format.
    Returns:
        Tensor: boxes centers with shape (..., 2), "cx, cy" format.
    """
    boxes_cx = (boxes[..., 0] + boxes[..., 2]) / 2
    boxes_cy = (boxes[..., 1] + boxes[..., 3]) / 2
    return torch.stack([boxes_cx, boxes_cy], dim=-1)


def batch_distance2bbox(points, distance, max_shapes=None):
    """Decode distance prediction to bounding box for batch.
    像FCOS那样，将 ltrb 解码成 预测框左上角坐标、右下角坐标
    Args:
        points (Tensor): [B, ..., 2], "xy" format
        distance (Tensor): [B, ..., 4], "ltrb" format
        max_shapes (Tensor): [B, 2], "h,w" format, Shape of the image.
    Returns:
        Tensor: Decoded bboxes, "x1y1x2y2" format.
    """
    #       points.shape == [A, 2], value == [[0.5, 0.5], [1.5, 0.5], [2.5, 0.5], ..., [4.5, 6.5], [5.5, 6.5], [6.5, 6.5]]  是格子中心点坐标（单位是格子边长）
    # distance.shape == [N,  A, 4],   是预测的bbox，ltrb格式(均是正值且单位是格子边长)
    lt, rb = torch.split(distance, 2, -1)  # lt.shape == [N, A, 2],  rb.shape == [N, A, 2]
    # while tensor add parameters, parameters should be better placed on the second place
    x1y1 = -lt + points  # x1y1.shape == [N, A, 2],  预测的bbox左上角坐标
    x2y2 = rb + points   # x2y2.shape == [N, A, 2],  预测的bbox右下角坐标
    out_bbox = torch.cat([x1y1, x2y2], -1)  # out_bbox.shape == [N, A, 4],  预测的bbox左上角坐标、右下角坐标
    if max_shapes is not None:
        max_shapes = max_shapes.flip(-1).tile([1, 2])
        delta_dim = out_bbox.ndim - max_shapes.ndim
        for _ in range(delta_dim):
            max_shapes.unsqueeze_(1)
        out_bbox = torch.where(out_bbox < max_shapes, out_bbox, max_shapes)
        out_bbox = torch.where(out_bbox > 0, out_bbox, torch.zeros_like(out_bbox))
    return out_bbox


def iou_similarity(box1, box2, eps=1e-10):
    """Calculate iou of box1 and box2

    Args:
        box1 (Tensor): box with the shape [M1, 4]
        box2 (Tensor): box with the shape [M2, 4]

    Return:
        iou (Tensor): iou between box1 and box2 with the shape [M1, M2]
    """
    box1 = box1.unsqueeze(1)  # [M1, 4] -> [M1, 1, 4]
    box2 = box2.unsqueeze(0)  # [M2, 4] -> [1, M2, 4]
    px1y1, px2y2 = box1[:, :, 0:2], box1[:, :, 2:4]
    gx1y1, gx2y2 = box2[:, :, 0:2], box2[:, :, 2:4]
    x1y1 = torch.max(px1y1, gx1y1)
    x2y2 = torch.min(px2y2, gx2y2)
    overlap = torch.clamp((x2y2 - x1y1), min=0).prod(-1)
    area1 = torch.clamp((px2y2 - px1y1), min=0).prod(-1)
    area2 = torch.clamp((gx2y2 - gx1y1), min=0).prod(-1)
    union = area1 + area2 - overlap + eps
    return overlap / union

