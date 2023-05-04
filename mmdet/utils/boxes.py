#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import numpy as np

import torch
import torchvision
import torch.nn.functional as F

__all__ = [
    "filter_box",
    "postprocess",
    "my_multiclass_nms",
    "bboxes_iou",
    "matrix_iou",
    "adjust_box_anns",
    "xyxy2xywh",
    "xyxy2cxcywh",
    "cxcywh2xyxy",
    "bboxes_iou_batch",
]


def filter_box(output, scale_range):
    """
    output: (N, 5+class) shape
    """
    min_scale, max_scale = scale_range
    w = output[:, 2] - output[:, 0]
    h = output[:, 3] - output[:, 1]
    keep = (w * h > min_scale * min_scale) & (w * h < max_scale * max_scale)
    return output[keep]


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output

# 参考自mmdet/utils/boxes.py的postprocess()。为了保持和matrix_nms()一样的返回风格，重新写一下。
def my_multiclass_nms(bboxes, scores, score_threshold=0.7, nms_threshold=0.45, nms_top_k=1000, keep_top_k=100, class_agnostic=False):
    '''
    :param bboxes:   shape = [N, A,  4]   "左上角xy + 右下角xy"格式
    :param scores:   shape = [N, A, 80]
    :param score_threshold:
    :param nms_threshold:
    :param nms_top_k:
    :param keep_top_k:
    :param class_agnostic:
    :return:
    '''

    # 每张图片的预测结果
    output = [None for _ in range(len(bboxes))]
    # 每张图片分开遍历
    for i, (xyxy, score) in enumerate(zip(bboxes, scores)):
        '''
        :var xyxy:    shape = [A, 4]   "左上角xy + 右下角xy"格式
        :var score:   shape = [A, 80]
        '''

        # 每个预测框最高得分的分数和对应的类别id
        class_conf, class_pred = torch.max(score, 1, keepdim=True)

        # 分数超过阈值的预测框为True
        conf_mask = (class_conf.squeeze() >= score_threshold).squeeze()
        # 这样排序 (x1, y1, x2, y2, 得分, 类别id)
        detections = torch.cat((xyxy, class_conf, class_pred.float()), 1)
        # 只保留超过阈值的预测框
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        # 使用torchvision自带的nms、batched_nms
        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4],
                nms_threshold,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4],
                detections[:, 5],
                nms_threshold,
            )

        detections = detections[nms_out_index]

        # 保留得分最高的keep_top_k个
        sort_inds = torch.argsort(detections[:, 4], descending=True)
        if keep_top_k > 0 and len(sort_inds) > keep_top_k:
            sort_inds = sort_inds[:keep_top_k]
        detections = detections[sort_inds, :]

        # 为了保持和matrix_nms()一样的返回风格 cls、score、xyxy。
        detections = torch.cat((detections[:, 5:6], detections[:, 4:5], detections[:, :4]), 1)

        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i + 1e-12)


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox


def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes


def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes

def cxcywh2xyxy(bboxes):
    bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] * 0.5
    bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
    bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
    return bboxes

def bboxes_iou_batch(bboxes_a, bboxes_b, xyxy=True):
    """计算两组矩形两两之间的iou
    Args:
        bboxes_a: (tensor) bounding boxes, Shape: [N, A, 4].
        bboxes_b: (tensor) bounding boxes, Shape: [N, B, 4].
    Return:
      (tensor) iou, Shape: [N, A, B].
    """
    # 使用混合精度训练时，iou可能出现nan，所以转成torch.float32
    bboxes_a = bboxes_a.to(torch.float32)
    bboxes_b = bboxes_b.to(torch.float32)
    N = bboxes_a.shape[0]
    A = bboxes_a.shape[1]
    B = bboxes_b.shape[1]
    if xyxy:
        box_a = bboxes_a
        box_b = bboxes_b
    else:  # cxcywh格式
        box_a = torch.cat([bboxes_a[:, :, :2] - bboxes_a[:, :, 2:] * 0.5,
                           bboxes_a[:, :, :2] + bboxes_a[:, :, 2:] * 0.5], dim=-1)
        box_b = torch.cat([bboxes_b[:, :, :2] - bboxes_b[:, :, 2:] * 0.5,
                           bboxes_b[:, :, :2] + bboxes_b[:, :, 2:] * 0.5], dim=-1)

    box_a_rb = torch.reshape(box_a[:, :, 2:], (N, A, 1, 2))
    box_a_rb = torch.tile(box_a_rb, [1, 1, B, 1])
    box_b_rb = torch.reshape(box_b[:, :, 2:], (N, 1, B, 2))
    box_b_rb = torch.tile(box_b_rb, [1, A, 1, 1])
    max_xy = torch.minimum(box_a_rb, box_b_rb)

    box_a_lu = torch.reshape(box_a[:, :, :2], (N, A, 1, 2))
    box_a_lu = torch.tile(box_a_lu, [1, 1, B, 1])
    box_b_lu = torch.reshape(box_b[:, :, :2], (N, 1, B, 2))
    box_b_lu = torch.tile(box_b_lu, [1, A, 1, 1])
    min_xy = torch.maximum(box_a_lu, box_b_lu)

    inter = F.relu(max_xy - min_xy)
    inter = inter[:, :, :, 0] * inter[:, :, :, 1]

    box_a_w = box_a[:, :, 2]-box_a[:, :, 0]
    box_a_h = box_a[:, :, 3]-box_a[:, :, 1]
    area_a = box_a_h * box_a_w
    area_a = torch.reshape(area_a, (N, A, 1))
    area_a = torch.tile(area_a, [1, 1, B])  # [N, A, B]

    box_b_w = box_b[:, :, 2]-box_b[:, :, 0]
    box_b_h = box_b[:, :, 3]-box_b[:, :, 1]
    area_b = box_b_h * box_b_w
    area_b = torch.reshape(area_b, (N, 1, B))
    area_b = torch.tile(area_b, [1, A, 1])  # [N, A, B]

    union = area_a + area_b - inter + 1e-9
    return inter / union  # [N, A, B]

def iou_similarity(box1, box2):
    # 使用混合精度训练时，iou可能出现nan，所以转成torch.float32
    box1 = box1.to(torch.float32)
    box2 = box2.to(torch.float32)
    return bboxes_iou_batch(box1, box2, xyxy=True)
