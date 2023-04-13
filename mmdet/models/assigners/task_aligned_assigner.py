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
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.ops import gather_1d, gather_nd
from mmdet.utils.boxes import iou_similarity
from .utils import (gather_topk_anchors, check_points_inside_bboxes,
                    compute_max_iou_anchor)


def is_close_gt(anchor, gt, stride_lst, max_dist=2.0, alpha=2.):
    """Calculate distance ratio of box1 and box2 in batch for larger stride
        anchors dist/stride to promote the survive of large distance match
    Args:
        anchor (Tensor): box with the shape [L, 2]
        gt (Tensor): box with the shape [N, M2, 4]
    Return:
        dist (Tensor): dist ratio between box1 and box2 with the shape [N, M1, M2]
    """
    center1 = anchor.unsqueeze(0)
    center2 = (gt[..., :2] + gt[..., -2:]) / 2.
    center1 = center1.unsqueeze(1)  # [N, M1, 2] -> [N, 1, M1, 2]
    center2 = center2.unsqueeze(2)  # [N, M2, 2] -> [N, M2, 1, 2]

    stride = torch.cat([
        torch.full([x], 32 / pow(2, idx)) for idx, x in enumerate(stride_lst)
    ]).unsqueeze(0).unsqueeze(0)
    dist = torch.linalg.norm(center1 - center2, p=2, axis=-1) / stride
    dist_ratio = dist
    dist_ratio[dist < max_dist] = 1.
    dist_ratio[dist >= max_dist] = 0.
    return dist_ratio


class TaskAlignedAssigner(nn.Module):
    """TOOD: Task-aligned One-stage Object Detection
    """

    def __init__(self, topk=13, alpha=1.0, beta=6.0, eps=1e-9, is_close_gt=False):
        super(TaskAlignedAssigner, self).__init__()
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.is_close_gt = is_close_gt

    @torch.no_grad()
    def forward(self,
                pred_scores,
                pred_bboxes,
                anchor_points,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index,
                gt_scores=None):
        r"""This code is based on
            https://github.com/fcjian/TOOD/blob/master/mmdet/core/bbox/assigners/task_aligned_assigner.py

        The assignment is done in following steps
        1. compute alignment metric between all bbox (bbox of all pyramid levels) and gt
        2. select top-k bbox as candidates for each gt
        3. limit the positive sample's center in gt (because the anchor-free detector
           only can predict positive distance)
        4. if an anchor box is assigned to multiple gts, the one with the
           highest iou will be selected.
        Args:
            pred_scores (Tensor, float32): predicted class probability, shape(B, L, C)
            pred_bboxes (Tensor, float32): predicted bounding boxes, shape(B, L, 4)
            anchor_points (Tensor, float32): pre-defined anchors, shape(L, 2), "cxcy" format
            num_anchors_list (List): num of anchors in each level, shape(L)
            gt_labels (Tensor, int64|int32): Label of gt_bboxes, shape(B, n, 1)
            gt_bboxes (Tensor, float32): Ground truth bboxes, shape(B, n, 4)
            pad_gt_mask (Tensor, float32): 1 means bbox, 0 means no bbox, shape(B, n, 1)
            bg_index (int): background index
            gt_scores (Tensor|None, float32) Score of gt_bboxes, shape(B, n, 1)
        Returns:
            assigned_labels (Tensor): (B, L)
            assigned_bboxes (Tensor): (B, L, 4)
            assigned_scores (Tensor): (B, L, C)
        """
        assert pred_scores.ndim == pred_bboxes.ndim
        assert gt_labels.ndim == gt_bboxes.ndim and \
               gt_bboxes.ndim == 3

        batch_size, num_anchors, num_classes = pred_scores.shape
        _, num_max_boxes, _ = gt_bboxes.shape

        # negative batch
        if num_max_boxes == 0:
            assigned_labels = torch.full(
                [batch_size, num_anchors], bg_index, dtype=gt_labels.dtype)
            assigned_bboxes = torch.zeros([batch_size, num_anchors, 4])
            assigned_scores = torch.zeros(
                [batch_size, num_anchors, num_classes])
            mask_positive = torch.zeros([batch_size, 1, num_anchors])

            assigned_labels = assigned_labels.to(gt_bboxes.device)
            assigned_bboxes = assigned_bboxes.to(gt_bboxes.device)
            assigned_scores = assigned_scores.to(gt_bboxes.device)
            mask_positive = mask_positive.to(gt_bboxes.device)
            return assigned_labels, assigned_bboxes, assigned_scores, mask_positive

        # compute iou between gt and pred bbox, [B, n, L]
        ious = iou_similarity(gt_bboxes, pred_bboxes)
        # gather pred bboxes class score
        pred_scores = pred_scores.permute([0, 2, 1])
        batch_ind = torch.arange(
            end=batch_size, dtype=gt_labels.dtype).unsqueeze(-1)
        batch_ind = batch_ind.to(gt_labels.device)
        gt_labels_ind = torch.stack([batch_ind.repeat([1, num_max_boxes]), gt_labels.squeeze(-1)], -1)
        bbox_cls_scores = gather_nd(pred_scores, gt_labels_ind)
        # compute alignment metrics, [B, n, L]
        alignment_metrics = bbox_cls_scores.pow(self.alpha) * ious.pow(self.beta)

        # check the positive sample's center in gt, [B, n, L]
        if self.is_close_gt:
            is_in_gts = is_close_gt(anchor_points, gt_bboxes, num_anchors_list)
        else:
            is_in_gts = check_points_inside_bboxes(anchor_points, gt_bboxes)

        # select topk largest alignment metrics pred bbox as candidates
        # for each gt, [B, n, L]
        is_in_topk = gather_topk_anchors(
            alignment_metrics * is_in_gts,
            self.topk,
            topk_mask=pad_gt_mask.repeat([1, 1, self.topk]).to(torch.bool))

        # select positive sample, [B, n, L]
        mask_positive = is_in_topk * is_in_gts * pad_gt_mask

        # if an anchor box is assigned to multiple gts,
        # the one with the highest iou will be selected, [B, n, L]
        mask_positive_sum = mask_positive.sum(-2)
        if mask_positive_sum.max() > 1:
            mask_multiple_gts = (mask_positive_sum.unsqueeze(1) > 1).repeat(
                [1, num_max_boxes, 1])
            is_max_iou = compute_max_iou_anchor(ious)
            mask_positive = torch.where(mask_multiple_gts, is_max_iou,
                                         mask_positive)
            mask_positive_sum = mask_positive.sum(-2)
        assigned_gt_index = mask_positive.argmax(-2)

        # assigned target
        assigned_gt_index = assigned_gt_index + batch_ind * num_max_boxes
        assigned_gt_index = assigned_gt_index.to(torch.int64)
        assigned_labels = gather_1d(gt_labels.flatten(), assigned_gt_index.flatten())
        assigned_labels = assigned_labels.reshape([batch_size, num_anchors])
        assigned_labels = torch.where(
            mask_positive_sum > 0, assigned_labels,
            torch.full_like(assigned_labels, bg_index))

        assigned_bboxes = gather_1d(
            gt_bboxes.reshape([-1, 4]), assigned_gt_index.flatten())
        assigned_bboxes = assigned_bboxes.reshape([batch_size, num_anchors, 4])

        assigned_labels = assigned_labels.to(torch.int64)
        assigned_scores = F.one_hot(assigned_labels, num_classes + 1)
        ind = list(range(num_classes + 1))
        ind.remove(bg_index)
        assigned_scores = torch.index_select(
            assigned_scores, dim=-1, index=torch.Tensor(ind).to(torch.int32).to(assigned_scores.device))
        # rescale alignment metrics
        alignment_metrics *= mask_positive
        max_metrics_per_instance, _ = alignment_metrics.max(-1, keepdim=True)
        max_ious_per_instance, _ = (ious * mask_positive).max(-1, keepdim=True)
        alignment_metrics = alignment_metrics / (
            max_metrics_per_instance + self.eps) * max_ious_per_instance
        alignment_metrics, _ = alignment_metrics.max(-2)
        alignment_metrics = alignment_metrics.unsqueeze(-1)
        assigned_scores = assigned_scores * alignment_metrics

        return assigned_labels, assigned_bboxes, assigned_scores, mask_positive
