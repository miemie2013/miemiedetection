
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

        # negative batch   这一批全是没有gt的图片
        if pad_gt_mask.sum() < 0.1:
            assigned_labels = torch.full([batch_size, num_anchors], bg_index, dtype=gt_labels.dtype)
            assigned_bboxes = torch.zeros([batch_size, num_anchors, 4])
            assigned_scores = torch.zeros([batch_size, num_anchors, self.num_classes])
            mask_positive = torch.zeros([batch_size, 1, num_anchors])

            assigned_labels = assigned_labels.to(gt_bboxes.device)
            assigned_bboxes = assigned_bboxes.to(gt_bboxes.device)
            assigned_scores = assigned_scores.to(gt_bboxes.device)
            mask_positive = mask_positive.to(gt_bboxes.device)
            return assigned_labels, assigned_bboxes, assigned_scores, mask_positive

        # 更新w h 统计值
        num_anchors_index = np.cumsum(num_anchors_list).tolist()
        num_anchors_index = [0, ] + num_anchors_index[:-1]  # num_anchors_index.value = [0, a0, a0+a1, a0+a1+a2]  每个fpn特征图第一个格子在A维度的下标
        avg_beta = min(self.avg_beta, (1 + self.step) / (10 + self.step))
        score_thr = min(self.score_threshold, self.step / (100 + self.step))
        w_avgs, h_avgs = [], []
        anchor_fpn_id = []
        for stage_idx in range(self.num_fpn_stride):
            start_i = num_anchors_index[stage_idx]
            end_i = start_i + num_anchors_list[stage_idx]
            anchor_fpn_id.append(torch.full([num_anchors_list[stage_idx], ], stage_idx, dtype=torch.int64))
            pred_bboxes_this_level = pred_bboxes[:, start_i:end_i, :]   # [N, 52*52, 4]
            pred_scores_this_level = pred_scores[:, start_i:end_i, :]   # [N, 52*52, num_classes]
            pred_scores_this_level_max, _ = pred_scores_this_level.max(-1)   # [N, 52*52]
            might_pos_bboxes_flag = pred_scores_this_level_max > score_thr   # [N, 52*52]   bool类型，分数大于阈值处为True
            might_pos_bboxes_this_level = pred_bboxes_this_level[might_pos_bboxes_flag]   # [?, 4]  分数大于阈值的预测框
            w_avg = getattr(self, 'w_avg_%d' % stage_idx)
            h_avg = getattr(self, 'h_avg_%d' % stage_idx)
            if might_pos_bboxes_this_level.shape[0] > 0:
                bbox_w = might_pos_bboxes_this_level[:, 2] - might_pos_bboxes_this_level[:, 0]
                bbox_h = might_pos_bboxes_this_level[:, 3] - might_pos_bboxes_this_level[:, 1]
                w_avg.copy_(bbox_w.mean().lerp(w_avg, avg_beta))
                h_avg.copy_(bbox_h.mean().lerp(h_avg, avg_beta))
            w_avgs.append(w_avg)
            h_avgs.append(h_avg)
        w_avgs = torch.stack(w_avgs, dim=0)
        h_avgs = torch.stack(h_avgs, dim=0)
        avg_wh = torch.cat([w_avgs, h_avgs], dim=1)   # [num_fpn_stride, 2]
        anchor_fpn_id = torch.cat(anchor_fpn_id, dim=0)   # [A, ]
        anchor_fpn_id = anchor_fpn_id.to(gt_bboxes.device)
        del w_avgs, h_avgs

        # 计算每个gt应该分配到哪个fpn level，1个gt只分配到1个fpn level
        gt_w = gt_bboxes[:, :, 2] - gt_bboxes[:, :, 0]
        gt_h = gt_bboxes[:, :, 3] - gt_bboxes[:, :, 1]
        gt_wh = torch.stack([gt_w, gt_h], dim=-1)   # [N, 200, 2]
        del gt_w, gt_h
        avg_wh = torch.reshape(avg_wh, [1, 1, self.num_fpn_stride, 2])     # [1, 1, num_fpn_stride, 2]
        gt_wh = torch.reshape(gt_wh, [batch_size, num_max_boxes, 1, 2])    # [N, 200,            1, 2]
        eps = 1e-5
        # [N, 200, num_fpn_stride]   每个gt和每个fpn level的匹配程度，cost越小越匹配。这里使用对勾函数y=x+1/x作为代价函数
        cost_fpn = avg_wh[..., 0] / (gt_wh[..., 0] + eps) + gt_wh[..., 0] / (avg_wh[..., 0] + eps) + \
                   avg_wh[..., 1] / (gt_wh[..., 1] + eps) + gt_wh[..., 1] / (avg_wh[..., 1] + eps)
        max_fpn_count = int(2.5 - self.step / (1000 + self.step))  # 最多是2，最少是1
        # max_fpn_count = 1
        # [N, 200, max_fpn_count]  每个gt被分配到的fpn_level的id
        _, gt_fpn_level = torch.topk(cost_fpn, max_fpn_count, dim=2, largest=False)
        anchor_fpn_id = torch.reshape(anchor_fpn_id, [1, 1, num_anchors])     # [1, 1, A]
        gt_select_fpn_mask = None  # [N, 200, A]  gt是否选择了这个anchor所在的fpn level
        if max_fpn_count == 1:
            gt_select_fpn_mask = ((gt_fpn_level - anchor_fpn_id) == 0).float()  # [N, 200, A]
        elif max_fpn_count == 2:
            gt_fpn_level0 = gt_fpn_level[:, :, 0:1]  # [N, 200, 1]
            gt_fpn_level1 = gt_fpn_level[:, :, 1:2]  # [N, 200, 1]
            gt_select_fpn_mask0 = ((gt_fpn_level0 - anchor_fpn_id) == 0).float()  # [N, 200, A]
            gt_select_fpn_mask1 = ((gt_fpn_level1 - anchor_fpn_id) == 0).float()  # [N, 200, A]
            gt_select_fpn_mask = gt_select_fpn_mask0 + gt_select_fpn_mask1  # [N, 200, A]
        elif max_fpn_count == 3:
            gt_fpn_level0 = gt_fpn_level[:, :, 0:1]  # [N, 200, 1]
            gt_fpn_level1 = gt_fpn_level[:, :, 1:2]  # [N, 200, 1]
            gt_fpn_level2 = gt_fpn_level[:, :, 2:3]  # [N, 200, 1]
            gt_select_fpn_mask0 = ((gt_fpn_level0 - anchor_fpn_id) == 0).float()  # [N, 200, A]
            gt_select_fpn_mask1 = ((gt_fpn_level1 - anchor_fpn_id) == 0).float()  # [N, 200, A]
            gt_select_fpn_mask2 = ((gt_fpn_level2 - anchor_fpn_id) == 0).float()  # [N, 200, A]
            gt_select_fpn_mask = gt_select_fpn_mask0 + gt_select_fpn_mask1 + gt_select_fpn_mask2  # [N, 200, A]
        elif max_fpn_count == 4:
            gt_fpn_level0 = gt_fpn_level[:, :, 0:1]  # [N, 200, 1]
            gt_fpn_level1 = gt_fpn_level[:, :, 1:2]  # [N, 200, 1]
            gt_fpn_level2 = gt_fpn_level[:, :, 2:3]  # [N, 200, 1]
            gt_fpn_level3 = gt_fpn_level[:, :, 3:4]  # [N, 200, 1]
            gt_select_fpn_mask0 = ((gt_fpn_level0 - anchor_fpn_id) == 0).float()  # [N, 200, A]
            gt_select_fpn_mask1 = ((gt_fpn_level1 - anchor_fpn_id) == 0).float()  # [N, 200, A]
            gt_select_fpn_mask2 = ((gt_fpn_level2 - anchor_fpn_id) == 0).float()  # [N, 200, A]
            gt_select_fpn_mask3 = ((gt_fpn_level3 - anchor_fpn_id) == 0).float()  # [N, 200, A]
            gt_select_fpn_mask = gt_select_fpn_mask0 + gt_select_fpn_mask1 + gt_select_fpn_mask2 + gt_select_fpn_mask3  # [N, 200, A]
        else:
            raise NotImplementedError("max_fpn_count == {} is not implemented.".format(max_fpn_count))
        gt_select_fpn_mask *= pad_gt_mask


        # anchor_in_gt_mask   [N, 200, A]  anchor中心点 是否 落在某个gt内部, float类型
        _, anchor_in_gt_mask = batch_anchor_is_in_gt(centers, gt_bboxes)
        # anchor_in_gt_mask *= pad_gt_mask    # 可以不写这句代码，  *= gt_select_fpn_mask 的时候已经等价于写了这句代码
        anchor_in_gt_mask *= gt_select_fpn_mask   # anchor中心点 只会落在 被分配到当前fpn level的gt内部

        # [N, 200, A]  计算 gt和预测框 两组矩形两两之间的iou
        ious = iou_similarity(gt_bboxes, pred_bboxes)  # [N, 200, A]  两组矩形两两之间的iou
        ious *= gt_select_fpn_mask  # [N, 200, A]  gt只和被分配到的fpn level的anchor计算iou，与其他fpn level的anchor的iou设置为0
        cost_iou = (1. - ious) * anchor_in_gt_mask + (1. - anchor_in_gt_mask) * 10000.0


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
        cost_centerness = (1. - centerness) * anchor_in_gt_mask + (1. - anchor_in_gt_mask) * 10000.0


        # [N, 200, A]  计算总的cost
        # cost = cost_iou + cost_centerness
        cost = cost_iou

        # [N, A]   对每个anchor，求与其最小cost的gt
        matched_gt_cost, matched_gt_index = cost.min(dim=1)
        neg_flag = matched_gt_cost > 0.5   # [N, A] 负样本处为True
        neg_mask = neg_flag.float().unsqueeze(-1)   # [N, A, 1] 负样本处为1

        mask_positive = F.one_hot(matched_gt_index, num_max_boxes)   # [N, A, 200]
        mask_positive = mask_positive.to(torch.float32)
        mask_positive *= (1. - neg_mask)
        mask_positive = mask_positive.permute([0, 2, 1])  # [N, 200, A]
        mask_positive *= pad_gt_mask
        pos_num_per_gt = mask_positive.sum(2)  # [N, 200]  每个gt的正样本数
        pppp = pos_num_per_gt.cpu().detach().numpy()
        print(pppp[:4, :4])
        print(pad_gt_mask[:4, :4, 0])

        # [N, 1]    value=[[0], [1], [2], ..., [N-1]]    图片的下标
        batch_ind = torch.arange(end=batch_size, dtype=gt_labels.dtype).unsqueeze(-1)
        batch_ind = batch_ind.to(gt_bboxes.device)
        matched_gt_index = matched_gt_index + batch_ind * num_max_boxes
        matched_gt_index_ = matched_gt_index.flatten().to(torch.int64)

        # 获取 assigned_labels
        assigned_labels = gather_1d(gt_labels.flatten(), index=matched_gt_index_)
        assigned_labels = assigned_labels.reshape([batch_size, num_anchors])  # assigned_labels.shape = [N, A]
        assigned_labels[neg_flag] = bg_index
        assigned_labels = assigned_labels.to(torch.int64)

        # 获取 assigned_bboxes
        assigned_bboxes = gather_1d(gt_bboxes.reshape([-1, 4]), index=matched_gt_index_)
        assigned_bboxes = assigned_bboxes.reshape([batch_size, num_anchors, 4])  # assigned_bboxes.shape = [N, A, 4]
        assigned_bboxes = 0. * neg_mask + (1. - neg_mask) * assigned_bboxes

        # 获取 assigned_scores
        # [N, A, num_classes + 1]    每个anchor学习的one_hot向量
        assigned_scores = F.one_hot(assigned_labels, self.num_classes + 1)
        assigned_scores = assigned_scores.to(torch.float32)
        ind = list(range(self.num_classes + 1))  # ind.value = [0, 1, 2, ..., num_classes - 1, num_classes]
        ind.remove(bg_index)                     # ind.value = [0, 1, 2, ..., num_classes - 1]
        ind = torch.Tensor(ind).to(torch.int32).to(assigned_scores.device)   # ind变成张量
        # [N, A, num_classes]    每个anchor学习的one_hot向量
        # 相当于assigned_scores = assigned_scores[:, :, :-1]
        assigned_scores = torch.index_select(assigned_scores, dim=-1, index=ind)
        # [N, A]       每个预测框与所有gt的最高iou (其实不是所有gt，是与所有被分配到这个fpn level的gt)
        ious_max, _ = ious.max(-2)
        ious_max = ious_max.unsqueeze(-1)  # [N, A, 1]
        # [N, A, num_classes]    每个anchor学习的one_hot向量，目标类别处不是1而是正样本与所有gt的最高iou。用于 VarifocalLoss
        assigned_scores *= ious_max


        # assigned_labels    [N, A]               每个anchor负责学习的gt的类别id，负样本处为bg_index==num_classes
        # assigned_bboxes    [N, A, 4]            每个anchor学习的左上角坐标、右下角坐标；单位是像素
        # assigned_scores    [N, A, num_classes]  每个anchor学习的one_hot向量，目标类别处不是1而是正样本与所有gt的最高iou。用于 VarifocalLoss
        mask_positive = 1
        self.step += 1
        return assigned_labels, assigned_bboxes, assigned_scores, mask_positive
