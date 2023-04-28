
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..bbox_utils import bbox_center, batch_bbox2distance, batch_anchor_is_in_gt
from .utils import (check_points_inside_bboxes, compute_max_iou_anchor,
                    compute_max_iou_gt)
from ..ops import index_sample_2d, gather_1d
from ...utils.boxes import iou_similarity
from ...utils.boxes import iou_similarity as batch_iou_similarity


class PositionAssigner(nn.Module):
    def __init__(self,
                 num_fpn_stride=4,
                 topk=9,
                 num_classes=80,
                 force_gt_matching=False,
                 eps=1e-9,
                 score_threshold=0.5,
                 sm_use=False):
        super(PositionAssigner, self).__init__()
        self.num_fpn_stride = num_fpn_stride
        self.topk = topk
        self.num_classes = num_classes
        self.force_gt_matching = force_gt_matching
        self.eps = eps
        self.sm_use = sm_use
        self.avg_beta = 0.995
        self.step = 0
        self.score_threshold = score_threshold
        for stage_idx in range(num_fpn_stride):
            self.register_buffer('w_avg_%d' % stage_idx, torch.zeros([1, ]))
            self.register_buffer('h_avg_%d' % stage_idx, torch.zeros([1, ]))

    @torch.no_grad()
    def forward(self,
                centers,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index,
                gt_scores=None,
                pred_bboxes=None,
                pred_scores=None):
        r"""
        """
        '''
        centers              [A, 2]  先验框中心点坐标（单位是像素）
        num_anchors_list     value = [52*52, 26*26, 13*13, 7*7]，每个特征图的格子数
        gt_labels            [N, 200, 1]  每个gt的label
        gt_bboxes            [N, 200, 4]  每个gt的左上角坐标、右下角坐标；单位是像素
        pad_gt_mask          [N, 200, 1]  是真gt还是填充的假gt
        bg_index==num_classes   背景的类别id
        gt_scores            None
        pred_bboxes          [N, A, 4],  预测框左上角坐标、右下角坐标；单位是像素
        pred_scores          [N, A, num_classes],  预测分数，已经经过sigmoid激活

        新算法：
        算每个anchor的centerness，iou
        算gt两两之间的iou，作为分配样本的惩罚
        ema统计每个level平均的w h，作为 cost_fpn
        '''
        assert gt_labels.ndim == gt_bboxes.ndim and gt_bboxes.ndim == 3

        _, num_anchors, _ = pred_bboxes.shape
        batch_size, num_max_boxes, _ = gt_bboxes.shape
        print(pred_bboxes.shape)
        print(num_anchors_list)

        # 更新w h 统计值
        num_anchors_index = np.cumsum(num_anchors_list).tolist()
        num_anchors_index = [0, ] + num_anchors_index[:-1]  # num_anchors_index.value = [0, a0, a0+a1, a0+a1+a2]  每个fpn特征图第一个格子在A维度的下标
        avg_beta = min(self.avg_beta, (1 + self.step) / (10 + self.step))
        for stage_idx in range(self.num_fpn_stride):
            start_i = num_anchors_index[stage_idx]
            end_i = start_i + num_anchors_list[stage_idx]
            pred_bboxes_this_level = pred_bboxes[:, start_i:end_i, :]   # [N, 52*52, 4]
            pred_scores_this_level = pred_scores[:, start_i:end_i, :]   # [N, 52*52, num_classes]
            pred_scores_this_level_max, _ = pred_scores_this_level.max(-1)   # [N, 52*52]
            might_pos_bboxes_flag = pred_scores_this_level_max > self.score_threshold   # [N, 52*52]   bool类型，分数大于阈值处为True
            might_pos_bboxes_this_level = pred_bboxes_this_level[might_pos_bboxes_flag]   # [?, 4]  分数大于阈值的预测框
            if might_pos_bboxes_this_level.shape[0] > 0:
                bbox_w = might_pos_bboxes_this_level[:, 2] - might_pos_bboxes_this_level[:, 0]
                bbox_h = might_pos_bboxes_this_level[:, 3] - might_pos_bboxes_this_level[:, 1]
                w_avg = getattr(self, 'w_avg_%d' % stage_idx)
                h_avg = getattr(self, 'h_avg_%d' % stage_idx)
                w_avg.copy_(bbox_w.mean().lerp(w_avg, avg_beta))
                h_avg.copy_(bbox_h.mean().lerp(h_avg, avg_beta))
        self.step += 1

        # negative batch   这一批全是没有gt的图片
        if num_max_boxes == 0:
            assigned_labels = torch.full([batch_size, num_anchors], bg_index, dtype=gt_labels.dtype)
            assigned_bboxes = torch.zeros([batch_size, num_anchors, 4])
            assigned_scores = torch.zeros([batch_size, num_anchors, self.num_classes])
            mask_positive = torch.zeros([batch_size, 1, num_anchors])

            assigned_labels = assigned_labels.to(gt_bboxes.device)
            assigned_bboxes = assigned_bboxes.to(gt_bboxes.device)
            assigned_scores = assigned_scores.to(gt_bboxes.device)
            mask_positive = mask_positive.to(gt_bboxes.device)
            return assigned_labels, assigned_bboxes, assigned_scores, mask_positive

        # anchor_in_gt_mask   [N, 200, A]  anchor中心点 是否 落在某个gt内部, float类型
        _, anchor_in_gt_mask = batch_anchor_is_in_gt(centers, gt_bboxes)
        anchor_in_gt_mask *= pad_gt_mask

        # pred_lt   [N, A, 2],  预测的lt；单位是像素
        # pred_rb   [N, A, 2],  预测的rb；单位是像素
        pred_lt, pred_rb = batch_bbox2distance(centers, pred_bboxes)
        l = pred_lt[:, :, 0]
        t = pred_lt[:, :, 1]
        r = pred_rb[:, :, 0]
        b = pred_rb[:, :, 1]
        # centerness   [N, A],  每个预测框的centerness
        centerness = torch.min(l, r) * torch.min(t, b) / (torch.max(l, r) * torch.max(t, b))
        centerness = centerness.sqrt()
        # centerness   [N, 200, A],
        centerness = centerness.unsqueeze(1).repeat([1, num_max_boxes, 1])
        cost_centerness = (1. - centerness) * anchor_in_gt_mask + (1. - anchor_in_gt_mask) * 100000.0

        # 1. [N, 200, A]  计算 gt和预测框 两组矩形两两之间的iou
        ious = iou_similarity(gt_bboxes, pred_bboxes)  # [N, 200, A]  两组矩形两两之间的iou
        cost_iou = (1. - ious) * anchor_in_gt_mask + (1. - anchor_in_gt_mask) * 100000.0
        # print(ious)
        print('aaaaaaaaaaaaaaaaaa')
        print(ious.shape)

        centerness = 1


        assigned_labels = 1
        assigned_bboxes = 1
        assigned_scores = 1
        mask_positive = 1


        return assigned_labels, assigned_bboxes, assigned_scores, mask_positive
