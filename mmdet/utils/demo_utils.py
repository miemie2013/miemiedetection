#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import os

import numpy as np

__all__ = ["mkdir", "nms", "multiclass_nms", "demo_postprocess"]


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr, class_agnostic=True):
    """Multiclass NMS implemented in Numpy"""
    if class_agnostic:
        nms_method = multiclass_nms_class_agnostic
    else:
        nms_method = multiclass_nms_class_aware
    return nms_method(boxes, scores, nms_thr, score_thr)


def multiclass_nms_class_aware(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-aware version."""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-agnostic version."""
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)
    if keep:
        dets = np.concatenate(
            [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
        )
    return dets


def demo_postprocess(outputs, img_size, p6=False):

    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs


def numpy_jaccard(box_a, box_b):
    """计算两组矩形两两之间的iou
    Args:
        box_a: (tensor) bounding boxes, Shape: [A, 4].
        box_b: (tensor) bounding boxes, Shape: [B, 4].
    Return:
        ious: (tensor) Shape: [A, B]
    """
    A = box_a.shape[0]
    B = box_b.shape[0]
    box_a_x1y1 = np.reshape(box_a[:, 2:], (A, 1, 2))
    box_a_x1y1 = np.tile(box_a_x1y1, (1, B, 1))
    box_b_x1y1 = np.reshape(box_b[:, 2:], (1, B, 2))
    box_b_x1y1 = np.tile(box_b_x1y1, (A, 1, 1))
    box_a_x0y0 = np.reshape(box_a[:, :2], (A, 1, 2))
    box_a_x0y0 = np.tile(box_a_x0y0, (1, B, 1))
    box_b_x0y0 = np.reshape(box_b[:, :2], (1, B, 2))
    box_b_x0y0 = np.tile(box_b_x0y0, (A, 1, 1))

    max_xy = np.minimum(box_a_x1y1, box_b_x1y1)
    min_xy = np.maximum(box_a_x0y0, box_b_x0y0)

    inter = np.clip((max_xy - min_xy), 0.0, np.inf)
    inter = inter[:, :, 0] * inter[:, :, 1]

    area_a = ((box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1]))
    area_a = np.reshape(area_a, (A, 1))
    area_a = np.tile(area_a, (1, B))
    area_b = ((box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3]-box_b[:, 1]))
    area_b = np.reshape(area_b, (1, B))
    area_b = np.tile(area_b, (A, 1))

    union = area_a + area_b - inter
    return inter / union  # [A, B]


def _numpy_matrix_nms(bboxes, cate_labels, cate_scores, kernel='gaussian', sigma=2.0):
    """Matrix NMS for multi-class bboxes.
    Args:
        bboxes (Tensor): shape (n, 4)
        cate_labels (Tensor): shape (n), mask labels in descending order
        cate_scores (Tensor): shape (n), mask scores in descending order
        kernel (str):  'linear' or 'gaussian'
        sigma (float): std in gaussian method
    Returns:
        Tensor: cate_scores_update, tensors of shape (n)
    """
    n_samples = len(cate_labels)
    if n_samples == 0:
        return []

    # 计算一个n×n的IOU矩阵，两组矩形两两之间的IOU
    iou_matrix = numpy_jaccard(bboxes, bboxes)   # shape: [n_samples, n_samples]
    iou_matrix = np.triu(iou_matrix, 1)     # 只取上三角部分

    # label_specific matrix.
    cate_labels_x = np.reshape(cate_labels, (1, n_samples))
    cate_labels_x = np.tile(cate_labels_x, (n_samples, 1))   # shape: [n_samples, n_samples]
    # 第i行第j列表示的是第i个预测框和第j个预测框的类别id是否相同。我们抑制的是同类的预测框。
    label_matrix = (cate_labels_x == cate_labels_x.transpose(1, 0)).astype(np.float32)   # shape: [n_samples, n_samples]
    label_matrix = np.triu(label_matrix, 1)   # shape: [n_samples, n_samples]

    # IoU compensation
    # 非同类的iou置为0，同类的iou保留。逐列取最大iou
    compensate_iou = np.max((iou_matrix * label_matrix), axis=0)   # shape: [n_samples, ]
    compensate_iou = np.reshape(compensate_iou, (n_samples, 1))
    compensate_iou = np.tile(compensate_iou, (1, n_samples))   # shape: [n_samples, n_samples]

    # IoU decay
    # 非同类的iou置为0，同类的iou保留。
    decay_iou = iou_matrix * label_matrix   # shape: [n_samples, n_samples]

    # matrix nms
    if kernel == 'gaussian':
        decay_matrix = np.exp(-1 * sigma * (decay_iou ** 2))
        compensate_matrix = np.exp(-1 * sigma * (compensate_iou ** 2))
        decay_coefficient = np.min((decay_matrix / compensate_matrix), axis=0)
    elif kernel == 'linear':
        decay_matrix = (1-decay_iou)/(1-compensate_iou)
        decay_coefficient = np.min(decay_matrix, axis=0)
    else:
        raise NotImplementedError

    # 更新分数
    cate_scores_update = cate_scores * decay_coefficient
    return cate_scores_update


def numpy_matrix_nms(bboxes,
                     scores,
                     score_threshold,
                     post_threshold,
                     nms_top_k,
                     keep_top_k,
                     use_gaussian=False,
                     gaussian_sigma=2.):

    inds = np.where(scores >= score_threshold)
    cate_scores = scores[inds]
    if len(cate_scores) == 0:
        return np.zeros((1, 6), dtype=np.float32) - 1.0

    cate_labels = inds[1]
    bboxes = bboxes[inds[0]]

    # sort and keep top nms_top_k
    sort_inds = np.argsort(cate_scores)
    sort_inds = sort_inds[::-1]
    if nms_top_k > 0 and len(sort_inds) > nms_top_k:
        sort_inds = sort_inds[:nms_top_k]
    bboxes = bboxes[sort_inds, :]
    cate_scores = cate_scores[sort_inds]
    cate_labels = cate_labels[sort_inds]

    # Matrix NMS
    kernel = 'gaussian' if use_gaussian else 'linear'
    cate_scores = _numpy_matrix_nms(bboxes, cate_labels, cate_scores, kernel=kernel, sigma=gaussian_sigma)

    # filter.
    keep = np.where(cate_scores >= post_threshold)
    cate_scores = cate_scores[keep]
    if len(cate_scores) == 0:
        return np.zeros((1, 6), dtype=np.float32) - 1.0
    cate_labels = cate_labels[keep]
    bboxes = bboxes[keep]

    # sort and keep keep_top_k
    sort_inds = np.argsort(cate_scores)
    sort_inds = sort_inds[::-1]
    if len(sort_inds) > keep_top_k:
        sort_inds = sort_inds[:keep_top_k]
    bboxes = bboxes[sort_inds]
    cate_scores = cate_scores[sort_inds]
    cate_labels = cate_labels[sort_inds]

    cate_scores = np.reshape(cate_scores, (-1, 1))
    cate_labels = np.reshape(cate_labels, (-1, 1)).astype(np.float32)
    pred = np.concatenate([cate_labels, cate_scores, bboxes], 1)
    return pred
