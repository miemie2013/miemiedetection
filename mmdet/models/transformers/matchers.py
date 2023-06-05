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
#
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ..losses.iou_losses import GIoULoss
from .utils import bbox_cxcywh_to_xyxy
from ..ops import gather_1d_dim1

__all__ = ['HungarianMatcher']



class HungarianMatcher(nn.Module):
    __shared__ = ['use_focal_loss', 'with_mask', 'num_sample_points']

    def __init__(self,
                 matcher_coeff={
                     'class': 1,
                     'bbox': 5,
                     'giou': 2,
                     'mask': 1,
                     'dice': 1
                 },
                 use_focal_loss=False,
                 with_mask=False,
                 num_sample_points=12544,
                 alpha=0.25,
                 gamma=2.0):
        r"""
        Args:
            matcher_coeff (dict): The coefficient of hungarian matcher cost.
        """
        super(HungarianMatcher, self).__init__()
        self.matcher_coeff = matcher_coeff
        self.use_focal_loss = use_focal_loss
        self.with_mask = with_mask
        self.num_sample_points = num_sample_points
        self.alpha = alpha
        self.gamma = gamma

        self.giou_loss = GIoULoss()


    @torch.no_grad()
    def forward(self,
                boxes,
                logits,
                gt_bbox,
                gt_class,
                pad_gt_mask,
                masks=None,
                gt_mask=None):
        r"""
        Args:
            boxes (Tensor): [b, query, 4]
            logits (Tensor): [b, query, num_classes]
            gt_bbox (List(Tensor)): list[[n, 4]]
            gt_class (List(Tensor)): list[[n, 1]]
            masks (Tensor|None): [b, query, h, w]
            gt_mask (List(Tensor)): list[[n, H, W]]

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = boxes.shape[:2]
        device = pad_gt_mask.device

        # num_gts = [len(a) for a in gt_class]
        num_gts = pad_gt_mask.sum([1, 2]).to(torch.int32).cpu().detach().numpy().tolist()
        if sum(num_gts) == 0:
            return [(paddle.to_tensor(
                [], dtype=paddle.int64), paddle.to_tensor(
                    [], dtype=paddle.int64)) for _ in range(bs)]

        # We flatten to compute the cost matrices in a batch
        # [batch_size * num_queries, num_classes]
        # logits = logits.detach()
        # paddle的softmax dim默认是-1，所以这里显式写上-1
        out_prob = None
        if self.use_focal_loss:
            out_prob = torch.sigmoid(logits.flatten(0, 1))
        else:
            out_prob = F.softmax(logits.flatten(0, 1), dim=-1)
        # [batch_size * num_queries, 4]
        # out_bbox = boxes.detach().flatten(0, 1)
        out_bbox = boxes.flatten(0, 1)

        # Also concat the target labels and boxes
        # tgt_ids = torch.cat(gt_class, dim=0).flatten()
        # tgt_bbox = torch.cat(gt_bbox, dim=0)
        real_gt = pad_gt_mask[:, :, 0] > 0.
        tgt_ids = gt_class[real_gt].flatten().to(torch.int64)
        tgt_bbox = gt_bbox[real_gt]

        # Compute the classification cost
        out_prob = gather_1d_dim1(out_prob, tgt_ids)
        if self.use_focal_loss:
            neg_cost_class = (1 - self.alpha) * (out_prob**self.gamma) * (-(
                1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * (
                (1 - out_prob)**self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -out_prob

        # Compute the L1 cost between boxes
        cost_bbox = (
            out_bbox.unsqueeze(1) - tgt_bbox.unsqueeze(0)).abs().sum(-1)

        # Compute the giou cost betwen boxes
        cost_giou = self.giou_loss(
            bbox_cxcywh_to_xyxy(out_bbox.unsqueeze(1)),
            bbox_cxcywh_to_xyxy(tgt_bbox.unsqueeze(0))).squeeze(-1)

        # Final cost matrix
        C = self.matcher_coeff['class'] * cost_class + \
            self.matcher_coeff['bbox'] * cost_bbox + \
            self.matcher_coeff['giou'] * cost_giou
        # Compute the mask cost and dice cost
        if self.with_mask:
            raise NotImplementedError

        C = C.reshape([bs, num_queries, -1])
        C = [a.squeeze(0) for a in C.chunk(bs)]  # tensor.chunk(N) paddle 和 pytorch 同义
        # sizes = [a.shape[0] for a in gt_bbox]
        sizes = num_gts
        indices = []
        # import paddle
        for i, c in enumerate(C):
            _input = c.split(sizes, -1)   # 这里和 paddle 的等价。对最后一维切分，size是每一份的大小。
            _input = _input[i]
            _input = _input.cpu().detach().numpy()
            _output = linear_sum_assignment(_input)
            indices.append(_output)
        outs = []
        for i, j in indices:
            v1 = torch.from_numpy(i).to(device).to(torch.int64)
            v2 = torch.from_numpy(j).to(device).to(torch.int64)
            outs.append((v1, v2))
        return outs
