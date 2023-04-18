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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..bbox_utils import bbox_center
from .utils import (check_points_inside_bboxes, compute_max_iou_anchor,
                    compute_max_iou_gt)
from ..ops import index_sample_2d, gather_1d
from ...utils.boxes import iou_similarity
from ...utils.boxes import iou_similarity as batch_iou_similarity


class ATSSAssigner(nn.Module):
    """Bridging the Gap Between Anchor-based and Anchor-free Detection
     via Adaptive Training Sample Selection
    """

    def __init__(self,
                 topk=9,
                 num_classes=80,
                 force_gt_matching=False,
                 eps=1e-9,
                 sm_use=False):
        super(ATSSAssigner, self).__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.force_gt_matching = force_gt_matching
        self.eps = eps
        self.sm_use = sm_use

    def _gather_topk_pyramid(self, gt2anchor_distances, num_anchors_list,
                             pad_gt_mask):
        """"""
        '''
        gt2anchor_distances    [N, 200, A]   gt和先验框 两组矩形两两之间中心点的距离
        num_anchors_list       value = [52*52, 26*26, 13*13, 7*7]，每个特征图的格子数
        pad_gt_mask            [N, 200, 1]  是真gt还是填充的假gt
        '''
        pad_gt_mask = pad_gt_mask.repeat([1, 1, self.topk]).to(torch.bool)
        # gt2anchor_distances 最后一维，分成每个fpn特征图的格子
        gt2anchor_distances_list = torch.split(gt2anchor_distances, num_anchors_list, -1)
        # 若 num_anchors_list.value = [a0, a1, a2, a3]
        # 则 num_anchors_index.value = [a0, a0+a1, a0+a1+a2, a0+a1+a2+a3]
        num_anchors_index = np.cumsum(num_anchors_list).tolist()
        num_anchors_index = [0, ] + num_anchors_index[:-1]  # num_anchors_index.value = [0, a0, a0+a1, a0+a1+a2]  每个fpn特征图第一个格子在A维度的下标
        is_in_topk_list = []
        topk_idxs_list = []
        for distances, anchors_index in zip(gt2anchor_distances_list, num_anchors_index):
            # distances    [N, 200, 52*52]   gt和 这层fpn先验框 两组矩形两两之间中心点的距离
            # anchors_index.value = 0        每个fpn特征图第一个格子在A维度的下标
            num_anchors = distances.shape[-1]   # 这个fpn特征图的格子数
            # 对每个gt，取最近的topk个先验框。
            # topk_metrics    [N, 200, topk]   每个gt最近的topk个先验框 的距离
            # (把gt比作考试科目，把anchor比作学生，把每个fpn level比作不同的班级，topk_metrics即：当前班级每门科目得分的前topk名学生)
            # topk_idxs       [N, 200, topk]   每个gt最近的topk个先验框 在52*52维度的下标  (即：当前班级每门科目得分的前topk名学生的id)
            topk_metrics, topk_idxs = torch.topk(distances, self.topk, dim=-1, largest=False)
            topk_idxs_list.append(topk_idxs + anchors_index)   # 每个gt最近的topk个先验框 在A维度的下标
            # 假gt的topk_idxs置0
            topk_idxs = torch.where(pad_gt_mask, topk_idxs, torch.zeros_like(topk_idxs))
            # [N, 200, topk, num_anchors]  -> [N, 200, num_anchors]  (即：掩码。对于每门科目每个学生，该生是否拿到了该科目的前topk名)
            # 对于每个gt每个anchor，该anchor是否是该gt的前topk个最近的anchor
            is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(axis=-2)
            is_in_topk = torch.where(is_in_topk > 1, torch.zeros_like(is_in_topk), is_in_topk)  # 为了修复一些bug，可以不理解这句代码。
            is_in_topk_list.append(is_in_topk.to(gt2anchor_distances.dtype))
        is_in_topk_list = torch.cat(is_in_topk_list, -1)  # [N, 200, A]         对于每个gt每个anchor，该anchor是否是该gt在该fpn level上的前topk个最近的anchor
        topk_idxs_list = torch.cat(topk_idxs_list, -1)    # [N, 200, 4*topk]    每个fpn level, 每个gt最近的topk个先验框 在A维度的下标
        return is_in_topk_list, topk_idxs_list

    @torch.no_grad()
    def forward(self,
                anchor_bboxes,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index,
                gt_scores=None,
                pred_bboxes=None):
        r"""This code is based on
            https://github.com/fcjian/TOOD/blob/master/mmdet/core/bbox/assigners/atss_assigner.py

        The assignment is done in following steps
        1. compute iou between all bbox (bbox of all pyramid levels) and gt
        2. compute center distance between all bbox and gt
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center, so we total select k*l bbox as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as positive
        6. limit the positive sample's center in gt
        7. if an anchor box is assigned to multiple gts, the one with the
           highest iou will be selected.
        Args:
            anchor_bboxes (Tensor, float32): pre-defined anchors, shape(L, 4),
                    "xmin, xmax, ymin, ymax" format
            num_anchors_list (List): num of anchors in each level
            gt_labels (Tensor, int64|int32): Label of gt_bboxes, shape(B, n, 1)
            gt_bboxes (Tensor, float32): Ground truth bboxes, shape(B, n, 4)
            pad_gt_mask (Tensor, float32): 1 means bbox, 0 means no bbox, shape(B, n, 1)
            bg_index (int): background index
            gt_scores (Tensor|None, float32) Score of gt_bboxes,
                    shape(B, n, 1), if None, then it will initialize with one_hot label
            pred_bboxes (Tensor, float32, optional): predicted bounding boxes, shape(B, L, 4)
        Returns:
            assigned_labels (Tensor): (B, L)
            assigned_bboxes (Tensor): (B, L, 4)
            assigned_scores (Tensor): (B, L, C), if pred_bboxes is not None, then output ious
        """
        '''
        anchor_bboxes        [A, 4]  先验框左上角坐标、右下角坐标；单位是像素
        num_anchors_list     value = [52*52, 26*26, 13*13, 7*7]，每个特征图的格子数
        gt_labels            [N, 200, 1]  每个gt的label
        gt_bboxes            [N, 200, 4]  每个gt的左上角坐标、右下角坐标；单位是像素
        pad_gt_mask          [N, 200, 1]  是真gt还是填充的假gt
        bg_index==num_classes   背景的类别id
        gt_scores            None
        pred_bboxes          [N, A, 4],  预测框左上角坐标、右下角坐标；单位是像素
        
        作者提出了一种自适应的选取正样本的方法，具体方法如下：
        1.对于每个 pyramid level，先计算每个anchor的中心点和gt的中心点的L2距离，
          选取topk个anchor中心点离gt中心点最近的anchor为候选正样本（candidate positive samples）
        2.计算每个候选正样本和gt之间的IOU，计算这组IOU的均值和方差
          根据方差和均值，设置选取正样本的阈值：t=m+g ；m为均值，g为方差
        3.根据每一层的t从其候选正样本中选出真正需要加入训练的正样本
        '''
        assert gt_labels.ndim == gt_bboxes.ndim and \
               gt_bboxes.ndim == 3

        num_anchors, _ = anchor_bboxes.shape
        batch_size, num_max_boxes, _ = gt_bboxes.shape

        # negative batch   这一批全是没有gt的图片
        if num_max_boxes == 0:
            assigned_labels = torch.full(
                [batch_size, num_anchors], bg_index, dtype=gt_labels.dtype)
            assigned_bboxes = torch.zeros([batch_size, num_anchors, 4])
            assigned_scores = torch.zeros(
                [batch_size, num_anchors, self.num_classes])

            assigned_labels = assigned_labels.to(gt_bboxes.device)
            assigned_bboxes = assigned_bboxes.to(gt_bboxes.device)
            assigned_scores = assigned_scores.to(gt_bboxes.device)
            return assigned_labels, assigned_bboxes, assigned_scores
        # B=N
        # n=200
        # L=A
        # 1. [N, 200, A]  计算 gt和先验框 两组矩形两两之间的iou
        batch_anchor_bboxes = anchor_bboxes.unsqueeze(0).repeat([batch_size, 1, 1])  # [N, A, 4]  先验框左上角坐标、右下角坐标；单位是像素
        ious = iou_similarity(gt_bboxes, batch_anchor_bboxes)  # [N, 200, A]  两组矩形两两之间的iou

        # 2. [N, 200, A]  计算 gt和先验框 两组矩形两两之间中心点的距离
        gt_centers = bbox_center(gt_bboxes.reshape([-1, 4])).unsqueeze(1)  # [N*200, 1, 2]  每个gt的中心点坐标；单位是像素
        anchor_centers = bbox_center(anchor_bboxes)   # [A, 2]  先验框中心点坐标；单位是像素
        gt2anchor_distances = (gt_centers - anchor_centers.unsqueeze(0)) \
            .norm(2, dim=-1).reshape([batch_size, -1, num_anchors])   # [N, 200, A]  计算 gt和先验框 两组矩形两两之间中心点的距离

        # 3. 对每个 pyramid level, 每个gt取最近的topk个anchor
        # is_in_topk  [N, 200, A]         对于每个gt每个anchor，该anchor是否是该gt在该fpn level上的前topk个最近的anchor
        # topk_idxs   [N, 200, 4*topk]    每个fpn level, 每个gt最近的topk个先验框 在A维度的下标
        is_in_topk, topk_idxs = self._gather_topk_pyramid(
            gt2anchor_distances, num_anchors_list, pad_gt_mask)

        # 4. get corresponding iou for the these candidates, and compute the
        # mean and std, 5. set mean + std as the iou threshold
        iou_candidates = ious * is_in_topk   # [N, 200, A]  前topk个anchor与该gt的iou
        aaaaaa1 = iou_candidates.reshape((-1, iou_candidates.shape[-1]))   # [N*200, A]
        aaaaaa2 = topk_idxs.reshape((-1, topk_idxs.shape[-1]))             # [N*200, 4*topk]
        iou_threshold = index_sample_2d(aaaaaa1, aaaaaa2)   # [N*200, 4*topk]   每个fpn level, 前topk个anchor与该gt的iou
        iou_threshold = iou_threshold.reshape([batch_size, num_max_boxes, -1])   # [N, 200, 4*topk]
        # [N, 200, 1]  每个gt的均值和标准差相加，作为阈值
        iou_threshold = iou_threshold.mean(dim=-1, keepdim=True) + \
                        iou_threshold.std(dim=-1, keepdim=True)
        # [N, 200, A]  前topk个最近的anchor（候选正样本）做一次过滤，iou > 阈值的，保留下来。
        is_in_topk = torch.where(
            iou_candidates > iou_threshold.repeat([1, 1, num_anchors]),
            is_in_topk, torch.zeros_like(is_in_topk))

        # 6. [N, 200, A]   anchor中心点是否在gt里面
        if self.sm_use:
            is_in_gts = check_points_inside_bboxes(anchor_centers, gt_bboxes, sm_use=True)
        else:
            is_in_gts = check_points_inside_bboxes(anchor_centers, gt_bboxes)

        # [N, 200, A], 这才是最终正样本。 iou超过阈值的，且位于gt框内部的，且不是假gt的
        mask_positive = is_in_topk * is_in_gts * pad_gt_mask

        # 7. 如果一个anchor被分配给多个gt,
        # 它只分配给与其具有最高iou的gt.
        mask_positive_sum = mask_positive.sum(dim=-2)  # [N, A]  每个anchor被分配给了几个gt
        if mask_positive_sum.max() > 1:
            # [N, 1, A] -> [N, 200, A]   1个anchor对多个gt的anchor的掩码，重复num_max_boxes次
            mask_multiple_gts = (mask_positive_sum.unsqueeze(1) > 1).repeat([1, num_max_boxes, 1])
            if self.sm_use:
                is_max_iou = compute_max_iou_anchor(ious * mask_positive)
            else:
                # [N, 200, A]，对每个anchor，具有最高iou的gt处为1
                is_max_iou = compute_max_iou_anchor(ious)
            # [N, 200, A], 最终正样本。一对多的时候，选择具有最高iou的gt
            mask_positive = torch.where(mask_multiple_gts, is_max_iou, mask_positive.to(is_max_iou.dtype))
            # [N, A], 每个anchor学习的gt个数(最大是1最小是0)。也是anchor是否是正样本的掩码
            mask_positive_sum = mask_positive.sum(-2)
        # 8. 确保 每一个gt 被分配给anchor
        if self.force_gt_matching:
            # [N, 200, A]，对每个gt，具有最高iou的anchor处为1。 * pad_gt_mask表示假gt处都是0
            is_max_iou = compute_max_iou_gt(ious) * pad_gt_mask
            # [N, 1, A] -> [N, 200, A]    求和之后表示，每个anchor是几个gt最高iou的。 ==1表示只保留是1个gt的最高iou的
            mask_max_iou = (is_max_iou.sum(-2, keepdim=True) == 1).repeat([1, num_max_boxes, 1])
            # 反悔重写。 [N, 200, A], 最终正样本。
            mask_positive = torch.where(mask_max_iou, is_max_iou, mask_positive)
            # 反悔重写。 [N, A], 每个anchor学习的gt个数(最大是1最小是0)。也是anchor是否是正样本的掩码
            mask_positive_sum = mask_positive.sum(-2)
        # [N, A]   每个anchor被分配的gt在 200维度(num_max_boxes维度) 里的下标
        assigned_gt_index = mask_positive.argmax(-2)

        # assigned target
        # [N, 1]    value=[[0], [1], [2], ..., [N-1]]    图片的下标
        batch_ind = torch.arange(end=batch_size, dtype=gt_labels.dtype).unsqueeze(-1)
        batch_ind = batch_ind.to(assigned_gt_index.device)
        # [N, A]   每个anchor被分配的gt在 N*200维度(N*num_max_boxes维度) 里的下标
        assigned_gt_index = assigned_gt_index + batch_ind * num_max_boxes

        # gt_labels.shape = [N, 200, 1]
        # assigned_labels.shape = [N*A, ]    每个anchor学习的类别id
        assigned_labels = gather_1d(gt_labels.flatten(), index=assigned_gt_index.flatten().to(torch.int64))
        assigned_labels = assigned_labels.reshape([batch_size, num_anchors])  # assigned_labels.shape = [N, A]
        # [N, A]    正样本处保留原值，负样本处置为bg_index
        assigned_labels = torch.where(mask_positive_sum > 0, assigned_labels, torch.full_like(assigned_labels, bg_index))

        # assigned_bboxes.shape = [N*A, 4]    每个anchor学习的左上角坐标、右下角坐标；单位是像素
        assigned_bboxes = gather_1d(gt_bboxes.reshape([-1, 4]), index=assigned_gt_index.flatten().to(torch.int64))
        assigned_bboxes = assigned_bboxes.reshape([batch_size, num_anchors, 4])  # assigned_bboxes.shape = [N, A, 4]

        assigned_labels = assigned_labels.to(torch.int64)
        # [N, A, num_classes + 1]    每个anchor学习的one_hot向量
        assigned_scores = F.one_hot(assigned_labels, self.num_classes + 1)
        assigned_scores = assigned_scores.to(torch.float32)
        ind = list(range(self.num_classes + 1))  # ind.value = [0, 1, 2, ..., num_classes - 1, num_classes]
        ind.remove(bg_index)                     # ind.value = [0, 1, 2, ..., num_classes - 1]
        ind = torch.Tensor(ind).to(torch.int32).to(assigned_scores.device)   # ind变成张量
        # [N, A, num_classes]    每个anchor学习的one_hot向量
        # 相当于assigned_scores = assigned_scores[:, :, :-1]
        assigned_scores = torch.index_select(assigned_scores, dim=-1, index=ind)
        if pred_bboxes is not None:
            # assigned iou
            # [N, 200, A]  计算 gt和预测框 两组矩形两两之间的iou。只保留正样本的。
            ious = batch_iou_similarity(gt_bboxes, pred_bboxes) * mask_positive
            # [N, A]       每个预测框与所有gt的最高iou
            ious_max, _ = ious.max(-2)
            ious_max = ious_max.unsqueeze(-1)  # [N, A, 1]
            # [N, A, num_classes]    每个anchor学习的one_hot向量，目标类别处不是1而是正样本与所有gt的最高iou。用于 VarifocalLoss
            assigned_scores *= ious_max
        elif gt_scores is not None:
            gather_scores = gather_1d(
                gt_scores.flatten(), index=assigned_gt_index.flatten().to(torch.int64))
            gather_scores = gather_scores.reshape([batch_size, num_anchors])
            gather_scores = torch.where(mask_positive_sum > 0, gather_scores,
                                         torch.zeros_like(gather_scores))
            assigned_scores *= gather_scores.unsqueeze(-1)
        # if torch.isnan(assigned_scores).any():
        #     print()
        return assigned_labels, assigned_bboxes, assigned_scores, mask_positive
