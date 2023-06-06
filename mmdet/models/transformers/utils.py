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
    label_noise_ratio = -1.
    box_noise_scale = -1.
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
    input_query_class = targets["gt_class"].clone().squeeze(-1).to(torch.int64)
    # input_query_class = input_query_class2.clone() * 0 + num_classes
    # for i, num_gt in enumerate(num_gts):
    #     input_query_class[i, :num_gt] = input_query_class2[i, :num_gt]
    input_query_bbox = targets["gt_bbox"].clone()
    pad_gt_mask = targets["pad_gt_mask"].clone().squeeze(-1)
    bs = input_query_bbox.shape[0]
    device = input_query_bbox.device


    # each group has positive and negative queries.
    input_query_class_3 = input_query_class.repeat([1, 2 * num_group])
    input_query_bbox_3 = input_query_bbox.repeat([1, 2 * num_group, 1])
    pad_gt_mask = pad_gt_mask.repeat([1, 2 * num_group])
    # positive and negative mask
    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.repeat([1, num_group, 1])
    positive_gt_mask = 1 - negative_gt_mask
    # contrastive denoising training positive index
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
    # nonzero() 返回[?, d], d是positive_gt_mask的维数，这里是2。 nonzero() 返回 positive_gt_mask 里非0值的坐标
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]   # 只取维度1的坐标
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])
    # total denoising queries
    num_denoising = int(max_gt_num * 2 * num_group)

    # known_labels_expaned = input_query_class.clone()
    # known_bbox_expand = input_query_bbox.clone()

    if label_noise_ratio > 0:
        input_query_class = input_query_class.flatten()
        pad_gt_mask = pad_gt_mask.flatten()
        # half of bbox prob
        mask = torch.rand(input_query_class.shape, device=device) < (label_noise_ratio * 0.5)
        chosen_idx = torch.nonzero(mask * pad_gt_mask).squeeze(-1)
        # randomly put a new one here
        new_label = torch.randint_like(chosen_idx, 0, num_classes, dtype=torch.int64, device=device)
        # known_labels_expaned = scatter_1d(known_labels_expaned, chosen_idx, new_label)
        known_labels_expaned = torch.reshape(known_labels_expaned, [bs, num_denoising])
        pad_gt_mask = torch.reshape(pad_gt_mask, [bs, num_denoising])

    if box_noise_scale > 0:
        known_bbox = bbox_cxcywh_to_xyxy(input_query_bbox)

        diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale

        # rand_sign = paddle.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
        rand_sign = torch.randint_like(input_query_bbox, 0, 2, dtype=torch.int64, device=device) * 2 - 1
        rand_part = torch.rand(input_query_bbox.shape, device=device)
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (
            1 - negative_gt_mask)
        rand_part = rand_part * rand_sign
        known_bbox = known_bbox + rand_part * diff
        known_bbox = torch.clamp(known_bbox, min=0.0, max=1.0)
        input_query_bbox = bbox_xyxy_to_cxcywh(known_bbox)
        input_query_bbox = inverse_sigmoid(input_query_bbox)
    aaaaaaa = 11
    # input_query_class_3 = gather_1d(class_embed, input_query_class.flatten()) * pad_gt_mask.flatten()
    # input_query_class_4 = gather_1d(class_embed, input_query_class_3.flatten())
    input_query_class_4 = class_embed(input_query_class_3.flatten())
    input_query_class_4 = input_query_class_4.reshape([bs, num_denoising, -1])
    aa = 1

    # class_embed_ = torch.cat([class_embed, torch.zeros([1, class_embed.shape[-1]], device=device)])
    # input_query_class_3 = gather_1d(class_embed_, input_query_class.flatten()).reshape([bs, num_denoising, -1])

    tgt_size = num_denoising + num_queries
    attn_mask = torch.ones([tgt_size, tgt_size], device=device) < 0
    # match query cannot see the reconstruction
    # attn_mask[num_denoising:, :num_denoising] = True
    # reconstruct cannot see each other
    # for i in range(num_group):
    #     if i == 0:
    #         attn_mask[max_gt_num * 2 * i:max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1):num_denoising] = True
    #     if i == num_group - 1:
    #         attn_mask[max_gt_num * 2 * i:max_gt_num * 2 * (i + 1), :max_gt_num * i * 2] = True
    #     else:
    #         attn_mask[max_gt_num * 2 * i:max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1):num_denoising] = True
    #         attn_mask[max_gt_num * 2 * i:max_gt_num * 2 * (i + 1), :max_gt_num * 2 * i] = True
    attn_mask = ~attn_mask
    attn_mask.requires_grad_(False)
    # attn_mask 会进入 MultiHeadAttention, 和 QK^T/sqrt(dk) 相加
    dn_meta = {
        "dn_positive_idx": dn_positive_idx,
        "dn_num_group": num_group,
        "dn_num_split": [num_denoising, num_queries]
    }

    return input_query_class_4, input_query_bbox_3, attn_mask, dn_meta


def get_contrastive_denoising_training_group2(targets,
                                             num_classes,
                                             num_queries,
                                             class_embed,
                                             num_denoising=100,
                                             label_noise_ratio=0.5,
                                             box_noise_scale=1.0):
    """
        def prepare_for_cdn(dn_args, training, num_queries, num_classes, hidden_dim, label_enc):
        A major difference of DINO from DN-DETR is that the author process pattern embedding pattern embedding in its detector
        forward function and use learnable tgt embedding, so we change this function a little bit.
        :param dn_args: targets, dn_number, label_noise_ratio, box_noise_scale
        :param training: if it is training or inference
        :param num_queries: number of queires
        :param num_classes: number of classes
        :param hidden_dim: transformer hidden dim
        :param label_enc: encode labels in dn
        :return:
        """
    # targets, dn_number, label_noise_ratio, box_noise_scale = dn_args
    dn_number = num_denoising
    hidden_dim = 256
    # positive and negative dn queries
    dn_number = dn_number * 2
    known = [(torch.ones_like(t['labels'])).cuda() for t in targets]
    batch_size = len(known)
    known_num = [sum(k) for k in known]
    if int(max(known_num)) == 0:
        dn_number = 1
    else:
        if dn_number >= 100:
            dn_number = dn_number // (int(max(known_num) * 2))
        elif dn_number < 1:
            dn_number = 1
    if dn_number == 0:
        dn_number = 1
    unmask_bbox = unmask_label = torch.cat(known)
    labels = torch.cat([t['labels'] for t in targets])
    boxes = torch.cat([t['boxes'] for t in targets])
    batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])

    known_indice = torch.nonzero(unmask_label + unmask_bbox)
    known_indice = known_indice.view(-1)

    known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)
    known_labels = labels.repeat(2 * dn_number, 1).view(-1)
    known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
    known_bboxs = boxes.repeat(2 * dn_number, 1)
    known_labels_expaned = known_labels.clone()
    known_bbox_expand = known_bboxs.clone()

    if label_noise_ratio > 0:
        p = torch.rand_like(known_labels_expaned.float())
        chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(-1)  # half of bbox prob
        new_label = torch.randint_like(chosen_indice, 0, num_classes)  # randomly put a new one here
        known_labels_expaned.scatter_(0, chosen_indice, new_label)
    single_pad = int(max(known_num))

    pad_size = int(single_pad * 2 * dn_number)
    positive_idx = torch.tensor(range(len(boxes))).long().cuda().unsqueeze(0).repeat(dn_number, 1)
    positive_idx += (torch.tensor(range(dn_number)) * len(boxes) * 2).long().cuda().unsqueeze(1)
    positive_idx = positive_idx.flatten()
    negative_idx = positive_idx + len(boxes)
    if box_noise_scale > 0:
        known_bbox_ = torch.zeros_like(known_bboxs)
        known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
        known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2

        diff = torch.zeros_like(known_bboxs)
        diff[:, :2] = known_bboxs[:, 2:] / 2
        diff[:, 2:] = known_bboxs[:, 2:] / 2

        rand_sign = torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
        rand_part = torch.rand_like(known_bboxs)
        rand_part[negative_idx] += 1.0
        rand_part *= rand_sign
        known_bbox_ = known_bbox_ + torch.mul(rand_part,
                                              diff).cuda() * box_noise_scale
        known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
        known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
        known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]

    m = known_labels_expaned.long().to('cuda')
    input_label_embed = class_embed(m)
    input_bbox_embed = inverse_sigmoid(known_bbox_expand)

    padding_label = torch.zeros(pad_size, hidden_dim).cuda()
    padding_bbox = torch.zeros(pad_size, 4).cuda()

    input_query_label = padding_label.repeat(batch_size, 1, 1)
    input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

    map_known_indice = torch.tensor([]).to('cuda')
    if len(known_num):
        map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
        map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(2 * dn_number)]).long()
    if len(known_bid):
        input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
        input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

    tgt_size = pad_size + num_queries
    attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
    # match query cannot see the reconstruct
    attn_mask[pad_size:, :pad_size] = True
    # reconstruct cannot see each other
    for i in range(dn_number):
        if i == 0:
            attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
        if i == dn_number - 1:
            attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * i * 2] = True
        else:
            attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
            attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * 2 * i] = True

    dn_meta = {
        'pad_size': pad_size,
        'num_dn_group': dn_number,
    }

    return input_query_label, input_query_bbox, attn_mask, dn_meta



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


