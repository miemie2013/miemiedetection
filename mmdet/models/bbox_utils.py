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


def bbox2delta(src_boxes, tgt_boxes, weights=[1.0, 1.0, 1.0, 1.0]):
    """Encode bboxes to deltas.
    """
    src_w = src_boxes[:, 2] - src_boxes[:, 0]
    src_h = src_boxes[:, 3] - src_boxes[:, 1]
    src_ctr_x = src_boxes[:, 0] + 0.5 * src_w
    src_ctr_y = src_boxes[:, 1] + 0.5 * src_h

    tgt_w = tgt_boxes[:, 2] - tgt_boxes[:, 0]
    tgt_h = tgt_boxes[:, 3] - tgt_boxes[:, 1]
    tgt_ctr_x = tgt_boxes[:, 0] + 0.5 * tgt_w
    tgt_ctr_y = tgt_boxes[:, 1] + 0.5 * tgt_h

    wx, wy, ww, wh = weights
    dx = wx * (tgt_ctr_x - src_ctr_x) / src_w
    dy = wy * (tgt_ctr_y - src_ctr_y) / src_h
    dw = ww * torch.log(tgt_w / src_w)
    dh = wh * torch.log(tgt_h / src_h)

    deltas = torch.stack((dx, dy, dw, dh), dim=1)
    return deltas


def delta2bbox(deltas, boxes, weights=[1.0, 1.0, 1.0, 1.0], max_shape=None):
    """Decode deltas to boxes. Used in RCNNBox,CascadeHead,RCNNHead,RetinaHead.
    Note: return tensor shape [n,1,4]
        If you want to add a reshape, please add after the calling code instead of here.
    """
    clip_scale = math.log(1000.0 / 16)

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    wx, wy, ww, wh = weights
    dx = deltas[:, 0::4] / wx
    dy = deltas[:, 1::4] / wy
    dw = deltas[:, 2::4] / ww
    dh = deltas[:, 3::4] / wh
    # Prevent sending too large values into paddle.exp()
    dw = torch.clamp(dw, max=clip_scale)
    dh = torch.clamp(dh, max=clip_scale)

    pred_ctr_x = dx * widths.unsqueeze(1) + ctr_x.unsqueeze(1)
    pred_ctr_y = dy * heights.unsqueeze(1) + ctr_y.unsqueeze(1)
    pred_w = torch.exp(dw) * widths.unsqueeze(1)
    pred_h = torch.exp(dh) * heights.unsqueeze(1)

    pred_boxes = []
    pred_boxes.append(pred_ctr_x - 0.5 * pred_w)
    pred_boxes.append(pred_ctr_y - 0.5 * pred_h)
    pred_boxes.append(pred_ctr_x + 0.5 * pred_w)
    pred_boxes.append(pred_ctr_y + 0.5 * pred_h)
    pred_boxes = torch.stack(pred_boxes, dim=-1)

    if max_shape is not None:
        pred_boxes[..., 0::2] = torch.clamp(pred_boxes[..., 0::2], min=0., max=max_shape[1])
        pred_boxes[..., 1::2] = torch.clamp(pred_boxes[..., 1::2], min=0., max=max_shape[0])
    return pred_boxes



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


def batch_bbox2distance(points, bbox):
    """
    Args:
        points:     [A, 2]  格子中心点坐标
        bbox:    [N, A, 4]  预测框左上角坐标、右下角坐标
    Returns:
        lt:      [N, A, 2]  预测的lt
        rb:      [N, A, 2]  预测的rb
    """
    x1y1, x2y2 = torch.split(bbox, 2, -1)  # x1y1.shape == [N, A, 2],  x2y2.shape == [N, A, 2]
    lt = points - x1y1  # lt.shape == [N, A, 2],  预测的lt
    rb = x2y2 - points  # rb.shape == [N, A, 2],  预测的rb
    return lt, rb


def batch_anchor_is_in_gt(points, gt_bboxes):
    """
    Args:
        points:           [A, 2]  anchor中心点坐标
        gt_bboxes:   [N, 200, 4]  每个gt的左上角坐标、右下角坐标
    Returns:
        anchor_in_gt:      [N, 200, A]  anchor中心点 是否 落在某个gt内部, bool类型
    """
    points_ = points.unsqueeze(0).unsqueeze(0)   # [1,   1, A, 2]
    gt_bboxes_ = gt_bboxes.unsqueeze(2)          # [N, 200, 1, 4]
    x1y1, x2y2 = torch.split(gt_bboxes_, 2, -1)  # [N, 200, 1, 2], [N, 200, 1, 2]
    lt = points_ - x1y1  # [N, 200, A, 2]
    rb = x2y2 - points_  # [N, 200, A, 2]
    ltrb = torch.cat([lt, rb], -1)  # [N, 200, A, 4]
    anchor_in_gt = ltrb.min(dim=-1).values > 0.0  # [N, 200, A]
    anchor_in_gt_mask = anchor_in_gt.float()
    return anchor_in_gt, anchor_in_gt_mask


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

