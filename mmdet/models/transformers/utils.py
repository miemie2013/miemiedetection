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
# Modified from detrex (https://github.com/IDEA-Research/detrex)
# Copyright 2022 The IDEA Authors. All rights reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.ops import gather_1d, scatter_1d
from mmdet.models.initializer import xavier_uniform_, constant_

__all__ = [
    '_get_clones', 'bbox_cxcywh_to_xyxy',
    'bbox_xyxy_to_cxcywh', 'inverse_sigmoid',
    'deformable_attention_core_func'
]


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def bbox_cxcywh_to_xyxy(x):
    cxcy, wh = torch.split(x, 2, dim=-1)
    return torch.cat([cxcy - 0.5 * wh, cxcy + 0.5 * wh], dim=-1)


def bbox_xyxy_to_cxcywh(x):
    x1, y1, x2, y2 = torch.split(x, 1, -1)
    return torch.cat([(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)], dim=-1)


def inverse_sigmoid(x, eps=1e-5):
    x = torch.clamp(x, min=eps, max=1. - eps)
    return torch.log(x / (1. - x))


def deformable_attention_core_func(value, value_spatial_shapes,
                                   value_level_start_index, sampling_locations,
                                   attention_weights):
    """
    Args:
        value (Tensor): [bs, value_length, n_head, c]
        value_spatial_shapes (Tensor|List): [n_levels, 2]
        value_level_start_index (Tensor|List): [n_levels]
        sampling_locations (Tensor): [bs, query_length, n_head, n_levels, n_points, 2]
        attention_weights (Tensor): [bs, query_length, n_head, n_levels, n_points]

    Returns:
        output (Tensor): [bs, Length_{query}, C]
    """
    bs, _, n_head, c = value.shape
    _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape

    split_shape = [int(h * w) for h, w in value_spatial_shapes]
    value_list = torch.split(value, split_shape, dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        h = int(h)
        w = int(w)
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[level].flatten(2)
        value_l_ = value_l_.permute([0, 2, 1]).reshape([bs * n_head, c, h, w])
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].permute([0, 2, 1, 3, 4])
        sampling_grid_l_ = sampling_grid_l_.flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_*M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.permute([0, 2, 1, 3, 4]).reshape([bs * n_head, 1, Len_q, n_levels * n_points])
    output = torch.stack(sampling_value_list, dim=-2)
    output = output.flatten(-2)
    output = output * attention_weights
    output = output.sum(-1).reshape([bs, n_head * c, Len_q])
    return output.permute([0, 2, 1])


def get_contrastive_denoising_training_group(targets,
                                             num_classes,
                                             num_queries,
                                             class_embed,
                                             num_denoising=100,
                                             label_noise_ratio=0.5,
                                             box_noise_scale=1.0):
    if num_denoising <= 0:
        return None, None, None, None
    num_gts = targets["pad_gt_mask"].sum([1, 2]).to(torch.int32).cpu().detach().numpy().tolist()
    max_gt_num = max(num_gts)
    if max_gt_num == 0:
        return None, None, None, None

    num_group = num_denoising // max_gt_num
    num_group = 1 if num_group == 0 else num_group
    # 原版对每张图片的gt pad到 max_gt_num, 但是我已经在数据预处理阶段 pad了，所以不需要做
    # bs = len(targets["gt_class"])
    # input_query_class = paddle.full([bs, max_gt_num], num_classes, dtype='int32')
    # input_query_bbox = paddle.zeros([bs, max_gt_num, 4])
    # pad_gt_mask = paddle.zeros([bs, max_gt_num])
    # for i in range(bs):
    #     num_gt = num_gts[i]
    #     if num_gt > 0:
    #         input_query_class[i, :num_gt] = targets["gt_class"][i].squeeze(-1)
    #         input_query_bbox[i, :num_gt] = targets["gt_bbox"][i]
    #         pad_gt_mask[i, :num_gt] = 1
    input_query_class = targets["gt_class"].squeeze(-1).to(torch.int64)
    input_query_bbox = targets["gt_bbox"]
    pad_gt_mask = targets["pad_gt_mask"].squeeze(-1)
    bs = input_query_bbox.shape[0]
    device = input_query_bbox.device


    # each group has positive and negative queries.
    input_query_class = input_query_class.tile([1, 2 * num_group])
    input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])
    # positive and negative mask
    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
    positive_gt_mask = 1 - negative_gt_mask
    # contrastive denoising training positive index
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
    # nonzero() 返回[?, d], d是positive_gt_mask的维数，这里是2。 nonzero() 返回 positive_gt_mask 里非0值的坐标
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]   # 只取维度1的坐标
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])
    # total denoising queries
    num_denoising = int(max_gt_num * 2 * num_group)

    if label_noise_ratio > 0:
        input_query_class = input_query_class.flatten()
        pad_gt_mask = pad_gt_mask.flatten()
        # half of bbox prob
        mask = torch.rand(input_query_class.shape, device=device) < (label_noise_ratio * 0.5)
        chosen_idx = torch.nonzero(mask * pad_gt_mask).squeeze(-1)
        # randomly put a new one here
        new_label = torch.randint_like(chosen_idx, 0, num_classes, dtype=torch.int64, device=device)
        input_query_class = scatter_1d(input_query_class, chosen_idx, new_label)
        input_query_class = torch.reshape(input_query_class, [bs, num_denoising])
        pad_gt_mask = torch.reshape(pad_gt_mask, [bs, num_denoising])

    if box_noise_scale > 0:
        known_bbox = bbox_cxcywh_to_xyxy(input_query_bbox)

        diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale

        # rand_sign = paddle.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
        rand_sign = torch.randint_like(input_query_bbox, 0, 2, dtype=torch.int64, device=device) * 2 - 1
        rand_part = torch.rand(input_query_bbox.shape, device=device)
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (
            1 - negative_gt_mask)
        rand_part *= rand_sign
        known_bbox += rand_part * diff
        known_bbox = torch.clamp(known_bbox, min=0.0, max=1.0)
        input_query_bbox = bbox_xyxy_to_cxcywh(known_bbox)
        input_query_bbox = inverse_sigmoid(input_query_bbox)

    class_embed = torch.cat([class_embed, torch.zeros([1, class_embed.shape[-1]], device=device)])
    input_query_class = gather_1d(class_embed, input_query_class.flatten()).reshape([bs, num_denoising, -1])

    tgt_size = num_denoising + num_queries
    attn_mask = torch.ones([tgt_size, tgt_size], device=device) < 0
    # match query cannot see the reconstruction
    attn_mask[num_denoising:, :num_denoising] = True
    # reconstruct cannot see each other
    for i in range(num_group):
        if i == 0:
            attn_mask[max_gt_num * 2 * i:max_gt_num * 2 * (i + 1), max_gt_num *
                      2 * (i + 1):num_denoising] = True
        if i == num_group - 1:
            attn_mask[max_gt_num * 2 * i:max_gt_num * 2 * (i + 1), :max_gt_num *
                      i * 2] = True
        else:
            attn_mask[max_gt_num * 2 * i:max_gt_num * 2 * (i + 1), max_gt_num *
                      2 * (i + 1):num_denoising] = True
            attn_mask[max_gt_num * 2 * i:max_gt_num * 2 * (i + 1), :max_gt_num *
                      2 * i] = True
    attn_mask = ~attn_mask
    # attn_mask 会进入 MultiHeadAttention, 和 QK^T/sqrt(dk) 相加
    dn_meta = {
        "dn_positive_idx": dn_positive_idx,
        "dn_num_group": num_group,
        "dn_num_split": [num_denoising, num_queries]
    }

    return input_query_class, input_query_bbox, attn_mask, dn_meta


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.,
                 kdim=None,
                 vdim=None,
                 need_weights=False):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.aaaaaa = nn.MultiheadAttention
        self.need_weights = need_weights

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim:
            # 要注意，pytorch的fc层和paddle的fc层的权重weight需要转置一下才能等价！！！
            self.in_proj_weight = torch.nn.Parameter(torch.randn([3 * embed_dim, embed_dim]))
            self.in_proj_bias = torch.nn.Parameter(torch.full([3 * embed_dim], np.float32(0.)))
        else:
            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.k_proj = nn.Linear(self.kdim, embed_dim)
            self.v_proj = nn.Linear(self.vdim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self._type_list = ('q_proj', 'k_proj', 'v_proj')

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
            else:
                constant_(p)

    def compute_qkv(self, tensor, index):
        if self._qkv_same_embed_dim:
            # 要注意，pytorch的fc层和paddle的fc层的权重weight需要转置一下才能等价！！！
            weight = self.in_proj_weight[index * self.embed_dim:(index + 1) * self.embed_dim, :]
            bias = self.in_proj_bias[index * self.embed_dim:(index + 1) * self.embed_dim] if self.in_proj_bias is not None else None
            tensor = F.linear(tensor, weight=weight, bias=bias)
        else:
            tensor = getattr(self, self._type_list[index])(tensor)
        N, HW, _ = tensor.shape
        tensor = tensor.reshape([N, HW, self.num_heads, self.head_dim])
        tensor = tensor.permute([0, 2, 1, 3])   # [N, num_heads, HW, head_dim]
        return tensor

    def forward(self, query, key=None, value=None, attn_mask=None):
        key = query if key is None else key
        value = query if value is None else value
        # compute q ,k ,v
        q, k, v = (self.compute_qkv(t, i)
                   for i, t in enumerate([query, key, value]))

        # scale dot product attention
        product = q.matmul(k.permute([0, 1, 3, 2]))    # [N, num_heads, HW, HW]
        scaling = float(self.head_dim)**-0.5
        product = product * scaling

        if attn_mask is not None:
            # attn_mask 进入 MultiHeadAttention, 和 QK^T/sqrt(dk) 相加
            # attn_mask 原始值如果是 True, 则变成 0 , attn_mask 原始值如果是 False, 则变成 负无穷 .
            attn_mask = attn_mask.to(product.dtype)
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            product = product + attn_mask
        # paddle的softmax dim默认是-1，所以这里显式写上-1
        weights = F.softmax(product, dim=-1)
        if self.dropout:
            weights = F.dropout(
                weights,
                self.dropout,
                training=self.training)
        out = torch.matmul(weights, v)    # [N, num_heads, HW, head_dim]

        # combine heads
        out = out.permute([0, 2, 1, 3])    # [N, HW, num_heads, head_dim]
        N, HW, _, _ = out.shape
        out = torch.reshape(out, [N, HW, out.shape[2] * out.shape[3]])    # [N, HW, embed_dim]

        # project to output
        out = self.out_proj(out)    # [N, HW, embed_dim]

        outs = [out]
        if self.need_weights:
            outs.append(weights)
        return out if len(outs) == 1 else tuple(outs)


