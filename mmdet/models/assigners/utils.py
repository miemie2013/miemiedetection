# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F

__all__ = [
    'pad_gt', 'gather_topk_anchors', 'check_points_inside_bboxes',
    'compute_max_iou_anchor', 'compute_max_iou_gt',
    'generate_anchors_for_grid_cell'
]


def pad_gt(gt_labels, gt_bboxes, gt_scores=None):
    r""" Pad 0 in gt_labels and gt_bboxes.
    Args:
        gt_labels (Tensor|List[Tensor], int64): Label of gt_bboxes,
            shape is [B, n, 1] or [[n_1, 1], [n_2, 1], ...], here n = sum(n_i)
        gt_bboxes (Tensor|List[Tensor], float32): Ground truth bboxes,
            shape is [B, n, 4] or [[n_1, 4], [n_2, 4], ...], here n = sum(n_i)
        gt_scores (Tensor|List[Tensor]|None, float32): Score of gt_bboxes,
            shape is [B, n, 1] or [[n_1, 4], [n_2, 4], ...], here n = sum(n_i)
    Returns:
        pad_gt_labels (Tensor, int64): shape[B, n, 1]
        pad_gt_bboxes (Tensor, float32): shape[B, n, 4]
        pad_gt_scores (Tensor, float32): shape[B, n, 1]
        pad_gt_mask (Tensor, float32): shape[B, n, 1], 1 means bbox, 0 means no bbox
    """
    if isinstance(gt_labels, torch.Tensor) and isinstance(gt_bboxes, torch.Tensor):
        assert gt_labels.ndim == gt_bboxes.ndim and \
               gt_bboxes.ndim == 3
        pad_gt_mask = (
            gt_bboxes.sum(axis=-1, keepdim=True) > 0).to(gt_bboxes.dtype)
        if gt_scores is None:
            gt_scores = pad_gt_mask.clone()
        assert gt_labels.ndim == gt_scores.ndim

        return gt_labels, gt_bboxes, gt_scores, pad_gt_mask
    elif isinstance(gt_labels, list) and isinstance(gt_bboxes, list):
        assert len(gt_labels) == len(gt_bboxes), \
            'The number of `gt_labels` and `gt_bboxes` is not equal. '
        num_max_boxes = max([len(a) for a in gt_bboxes])
        batch_size = len(gt_bboxes)
        # pad label and bbox
        pad_gt_labels = paddle.zeros(
            [batch_size, num_max_boxes, 1], dtype=gt_labels[0].dtype)
        pad_gt_bboxes = paddle.zeros(
            [batch_size, num_max_boxes, 4], dtype=gt_bboxes[0].dtype)
        pad_gt_scores = paddle.zeros(
            [batch_size, num_max_boxes, 1], dtype=gt_bboxes[0].dtype)
        pad_gt_mask = paddle.zeros(
            [batch_size, num_max_boxes, 1], dtype=gt_bboxes[0].dtype)
        for i, (label, bbox) in enumerate(zip(gt_labels, gt_bboxes)):
            if len(label) > 0 and len(bbox) > 0:
                pad_gt_labels[i, :len(label)] = label
                pad_gt_bboxes[i, :len(bbox)] = bbox
                pad_gt_mask[i, :len(bbox)] = 1.
                if gt_scores is not None:
                    pad_gt_scores[i, :len(gt_scores[i])] = gt_scores[i]
        if gt_scores is None:
            pad_gt_scores = pad_gt_mask.clone()
        return pad_gt_labels, pad_gt_bboxes, pad_gt_scores, pad_gt_mask
    else:
        raise ValueError('The input `gt_labels` or `gt_bboxes` is invalid! ')


def gather_topk_anchors(metrics, topk, largest=True, topk_mask=None, eps=1e-9):
    r"""
    Args:
        metrics (Tensor, float32): shape[B, n, L], n: num_gts, L: num_anchors
        topk (int): The number of top elements to look for along the axis.
        largest (bool) : largest is a flag, if set to true,
            algorithm will sort by descending order, otherwise sort by
            ascending order. Default: True
        topk_mask (Tensor, bool|None): shape[B, n, topk], ignore bbox mask,
            Default: None
        eps (float): Default: 1e-9
    Returns:
        is_in_topk (Tensor, float32): shape[B, n, L], value=1. means selected
    """
    num_anchors = metrics.shape[-1]
    topk_metrics, topk_idxs = torch.topk(
        metrics, topk, dim=-1, largest=largest)
    if topk_mask is None:
        topk_mask = (topk_metrics.max(axis=-1, keepdim=True) > eps).tile(
            [1, 1, topk])
    topk_idxs = torch.where(topk_mask, topk_idxs, torch.zeros_like(topk_idxs))
    is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(axis=-2)
    is_in_topk = torch.where(is_in_topk > 1,
                              torch.zeros_like(is_in_topk), is_in_topk)
    return is_in_topk.to(metrics.dtype)


def check_points_inside_bboxes(points,
                               bboxes,
                               center_radius_tensor=None,
                               eps=1e-9,
                               sm_use=False):
    r"""
    Args:
        points (Tensor, float32): shape[L, 2], "xy" format, L: num_anchors
        bboxes (Tensor, float32): shape[B, n, 4], "xmin, ymin, xmax, ymax" format
        center_radius_tensor (Tensor, float32): shape [L, 1]. Default: None.
        eps (float): Default: 1e-9
    Returns:
        is_in_bboxes (Tensor, float32): shape[B, n, L], value=1. means selected
    """
    '''
    points       [A, 2]  点的坐标
    gt_bboxes    [N, 200, 4]  bbox的左上角坐标、右下角坐标
    Returns:
        is_in_bboxes (Tensor, float32): shape[N, 200, A], value=1. means selected
    '''
    points = points.unsqueeze(0).unsqueeze(1)  # [1, 1, A, 2]
    x, y = points.chunk(2, axis=-1)   # [1, 1, A, 1]   [1, 1, A, 1]
    xmin, ymin, xmax, ymax = bboxes.unsqueeze(2).chunk(4, axis=-1)  # [N, 200, 1, 1] [N, 200, 1, 1] [N, 200, 1, 1] [N, 200, 1, 1]
    # check whether `points` is in `bboxes`
    l = x - xmin  # [N, 200, A, 1]
    t = y - ymin  # [N, 200, A, 1]
    r = xmax - x  # [N, 200, A, 1]
    b = ymax - y  # [N, 200, A, 1]
    delta_ltrb = torch.cat([l, t, r, b], -1)  # [N, 200, A, 4]
    delta_ltrb_min, _ = delta_ltrb.min(-1)    # [N, 200, A]
    is_in_bboxes = (delta_ltrb_min > eps)     # [N, 200, A]
    if center_radius_tensor is not None:
        # check whether `points` is in `center_radius`
        center_radius_tensor = center_radius_tensor.unsqueeze(0).unsqueeze(1)
        cx = (xmin + xmax) * 0.5
        cy = (ymin + ymax) * 0.5
        l = x - (cx - center_radius_tensor)
        t = y - (cy - center_radius_tensor)
        r = (cx + center_radius_tensor) - x
        b = (cy + center_radius_tensor) - y
        delta_ltrb_c = torch.cat([l, t, r, b], -1)
        delta_ltrb_c_min = delta_ltrb_c.min(-1)
        is_in_center = (delta_ltrb_c_min > eps)
        if sm_use:
            return is_in_bboxes.to(bboxes.dtype), is_in_center.to(bboxes.dtype)
        else:
            return (torch.logical_and(is_in_bboxes, is_in_center),
                    torch.logical_or(is_in_bboxes, is_in_center))

    return is_in_bboxes.to(bboxes.dtype)


def compute_max_iou_anchor(ious):
    r"""
    返回： [N, 200, A]，对每个anchor，具有最高iou的gt处为1
    For each anchor, find the GT with the largest IOU.
    Args:
        ious (Tensor, float32): shape[B, n, L], n: num_gts, L: num_anchors
    Returns:
        is_max_iou (Tensor, float32): shape[B, n, L], value=1. means selected
    """
    num_max_boxes = ious.shape[-2]
    max_iou_index = ious.argmax(axis=-2)
    is_max_iou = F.one_hot(max_iou_index, num_max_boxes).permute((0, 2, 1))
    return is_max_iou.to(ious.dtype)


def compute_max_iou_gt(ious):
    r"""
    返回： [N, 200, A]，对每个gt，具有最高iou的anchor处为1
    For each GT, find the anchor with the largest IOU.
    Args:
        ious (Tensor, float32): shape[B, n, L], n: num_gts, L: num_anchors
    Returns:
        is_max_iou (Tensor, float32): shape[B, n, L], value=1. means selected
    """
    num_anchors = ious.shape[-1]
    max_iou_index = ious.argmax(axis=-1)
    is_max_iou = F.one_hot(max_iou_index, num_anchors)
    return is_max_iou.to(ious.dtype)


def generate_anchors_for_grid_cell(feats,
                                   fpn_strides,
                                   grid_cell_size=5.0,
                                   grid_cell_offset=0.5):
    r"""
    Like ATSS, generate anchors based on grid size.
    Args:
        feats (List[Tensor]): shape[s, (b, c, h, w)]
        fpn_strides (tuple|list): shape[s], stride for each scale feature
        grid_cell_size (float): anchor size
        grid_cell_offset (float): The range is between 0 and 1.
    Returns:
        anchors (Tensor): shape[l, 4], "xmin, ymin, xmax, ymax" format.
        anchor_points (Tensor): shape[l, 2], "x, y" format.
        num_anchors_list (List[int]): shape[s], contains [s_1, s_2, ...].
        stride_tensor (Tensor): shape[l, 1], contains the stride for each scale.
    """
    # feats里有4个张量，形状分别是[N, C, 52, 52], [N, C, 26, 26], [N, C, 13, 13], [N, C, 7, 7]
    # fpn_strides == [8, 16, 32, 64]
    # grid_cell_size==5.0  表示每个格子出1个先验框，先验框是5.0倍的格子边长
    assert len(feats) == len(fpn_strides)
    anchors = []
    anchor_points = []
    num_anchors_list = []
    stride_tensor = []
    for feat, stride in zip(feats, fpn_strides):
        _, _, h, w = feat.shape
        cell_half_size = grid_cell_size * stride * 0.5  # 先验框边长的一半
        shift_x = (torch.arange(end=w) + grid_cell_offset) * stride
        shift_y = (torch.arange(end=h) + grid_cell_offset) * stride
        # shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)   # shift_y.shape == [h, w]   shift_x.shape == [h, w]
        anchor = torch.stack(
            [
                shift_x - cell_half_size, shift_y - cell_half_size,
                shift_x + cell_half_size, shift_y + cell_half_size
            ],
            -1).to(feat.dtype)   # anchor.shape == [h, w, 4]  先验框左上角坐标、右下角坐标；单位是像素
        anchor_point = torch.stack([shift_x, shift_y], -1).to(feat.dtype)   # anchor_point.shape == [h, w, 2]  格子中心点坐标（单位是像素）

        anchors.append(anchor.reshape([-1, 4]))   # [h*w, 4]  先验框左上角坐标、右下角坐标；单位是像素
        anchor_points.append(anchor_point.reshape([-1, 2]))   # anchor_point.shape == [h*w, 2]  格子中心点坐标（单位是像素）
        num_anchors_list.append(len(anchors[-1]))   # h*w，格子数
        stride_tensor.append(
            torch.full(
                [num_anchors_list[-1], 1], stride, dtype=feat.dtype))   # [h*w, 1]  格子边长
    anchors = torch.cat(anchors)   # [A, 4]  先验框左上角坐标、右下角坐标；单位是像素
    anchors.requires_grad_(False)
    anchor_points = torch.cat(anchor_points)   # [A, 2]  格子中心点坐标（单位是像素）
    anchor_points.requires_grad_(False)
    stride_tensor = torch.cat(stride_tensor)   # [A, 1]  格子边长
    stride_tensor.requires_grad_(False)
    dvs = feats[0].device
    anchors = anchors.to(dvs)
    anchor_points = anchor_points.to(dvs)
    stride_tensor = stride_tensor.to(dvs)
    return anchors, anchor_points, num_anchors_list, stride_tensor
