# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
# Modified from Deformable-DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Modified from detrex (https://github.com/IDEA-Research/detrex)
# Copyright 2022 The IDEA Authors. All rights reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ops import gather_nd
from ..initializer import (linear_init_, constant_, xavier_uniform_, normal_, bias_init_with_prob)
from .utils import (_get_clones, get_contrastive_denoising_training_group, inverse_sigmoid)
from .utils import MultiHeadAttention

__all__ = ['RTDETRTransformer']




class MLP(nn.Module):
    """This code is based on
        https://github.com/facebookresearch/detr/blob/main/models/detr.py
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

        self._reset_parameters()

    @torch.no_grad()
    def _reset_parameters(self):
        for l in self.layers:
            linear_init_(l)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MSDeformableAttention(nn.Module):
    def __init__(self,
                 embed_dim=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 lr_mult=0.1):
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.sampling_offsets.weight.param_lr = float(lr_mult)
        self.sampling_offsets.bias.param_lr = float(lr_mult)

        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        from .utils import deformable_attention_core_func as ms_deformable_attn
        self.ms_deformable_attn_core = ms_deformable_attn

        self._reset_parameters()

    @torch.no_grad()
    def _reset_parameters(self):
        # sampling_offsets
        constant_(self.sampling_offsets.weight)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        aaaa, bbbbb = grid_init.abs().max(-1, keepdim=True)
        grid_init = grid_init / aaaa
        grid_init = grid_init.reshape([self.num_heads, 1, 1, 2]).tile(
            [1, self.num_levels, self.num_points, 1])
        scaling = torch.arange(
            1, self.num_points + 1,
            dtype=torch.float32).reshape([1, 1, -1, 1])
        grid_init *= scaling
        self.sampling_offsets.bias.copy_(grid_init.flatten())
        # attention_weights
        constant_(self.attention_weights.weight)
        constant_(self.attention_weights.bias)
        # proj
        xavier_uniform_(self.value_proj.weight)
        constant_(self.value_proj.bias)
        xavier_uniform_(self.output_proj.weight)
        constant_(self.output_proj.bias)

    def forward(self,
                query,
                reference_points,
                value,
                value_spatial_shapes,
                value_level_start_index,
                value_mask=None):
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]
        assert int(value_spatial_shapes.prod(1).sum()) == Len_v

        value = self.value_proj(value)
        if value_mask is not None:
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value *= value_mask
        value = value.reshape([bs, Len_v, self.num_heads, self.head_dim])

        sampling_offsets = self.sampling_offsets(query).reshape(
            [bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2])
        attention_weights = self.attention_weights(query).reshape(
            [bs, Len_q, self.num_heads, self.num_levels * self.num_points])
        attention_weights = F.softmax(attention_weights).reshape(
            [bs, Len_q, self.num_heads, self.num_levels, self.num_points])

        if reference_points.shape[-1] == 2:
            offset_normalizer = value_spatial_shapes.flip([1]).reshape(
                [1, 1, 1, self.num_levels, 1, 2])
            sampling_locations = reference_points.reshape([
                bs, Len_q, 1, self.num_levels, 1, 2
            ]) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2] + sampling_offsets /
                self.num_points * reference_points[:, :, None, :, None, 2:] *
                0.5)
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        output = self.ms_deformable_attn_core(
            value, value_spatial_shapes, value_level_start_index,
            sampling_locations, attention_weights)
        output = self.output_proj(output)

        return output



class PPMSDeformableAttention(MSDeformableAttention):
    def forward(self,
                query,
                reference_points,
                value,
                value_spatial_shapes,
                value_level_start_index,
                value_mask=None):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_level_start_index (List): [n_levels], [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value *= value_mask
        value = value.reshape([bs, Len_v, self.num_heads, self.head_dim])

        sampling_offsets = self.sampling_offsets(query).reshape(
            [bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2])
        attention_weights = self.attention_weights(query).reshape(
            [bs, Len_q, self.num_heads, self.num_levels * self.num_points])
        attention_weights = F.softmax(attention_weights, dim=-1).reshape(
            [bs, Len_q, self.num_heads, self.num_levels, self.num_points])

        if reference_points.shape[-1] == 2:
            raise NotImplementedError
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2] + sampling_offsets /
                self.num_points * reference_points[:, :, None, :, None, 2:] *
                0.5)
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        if not isinstance(query, torch.Tensor):
            raise NotImplementedError
        else:
            device = value.device
            value_spatial_shapes = torch.Tensor(value_spatial_shapes).to(device)
            value_level_start_index = torch.Tensor(value_level_start_index).to(device)
            output = self.ms_deformable_attn_core(
                value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights)
        output = self.output_proj(output)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 n_levels=4,
                 n_points=4,
                 weight_attr=None,
                 bias_attr=None):
        super(TransformerDecoderLayer, self).__init__()

        # self attention
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm1.weight.weight_decay = 0.0
        self.norm1.bias.weight_decay = 0.0

        # cross attention
        self.cross_attn = PPMSDeformableAttention(d_model, n_head, n_levels,
                                                  n_points, 1.0)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm2.weight.weight_decay = 0.0
        self.norm2.bias.weight_decay = 0.0

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias_attr)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias_attr)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm3.weight.weight_decay = 0.0
        self.norm3.bias.weight_decay = 0.0
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)
        xavier_uniform_(self.linear1.weight)
        xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                tgt,
                reference_points,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        if attn_mask is not None:
            # attn_mask 会进入 MultiHeadAttention, 和 QK^T/sqrt(dk) 相加。
            # attn_mask 如果是 True, 则变成 0 , attn_mask 如果是 False, 则变成 负无穷 .
            device = attn_mask.device
            attn_mask = torch.where(
                attn_mask.to(torch.bool),
                torch.zeros(attn_mask.shape, dtype=tgt.dtype, device=device),
                torch.ones(attn_mask.shape, dtype=tgt.dtype, device=device) * -100000.)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos_embed), reference_points, memory,
            memory_spatial_shapes, memory_level_start_index, memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.forward_ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(self,
                tgt,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                bbox_head,
                score_head,
                query_pos_head,
                attn_mask=None,
                memory_mask=None):
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = torch.sigmoid(ref_points_unact)
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)

            output = layer(output, ref_points_input, memory,
                           memory_spatial_shapes, memory_level_start_index,
                           attn_mask, memory_mask, query_pos_embed)

            inter_ref_bbox = torch.sigmoid(bbox_head[i](output) + inverse_sigmoid(
                ref_points_detach))

            if self.training:
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(
                        torch.sigmoid(bbox_head[i](output) + inverse_sigmoid(
                            ref_points)))
            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach(
            ) if self.training else inter_ref_bbox

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits)


class RTDETRTransformer(nn.Module):
    __shared__ = ['num_classes', 'hidden_dim', 'eval_size']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 backbone_feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 num_decoder_points=4,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learnt_init_query=True,
                 eval_size=None,
                 eval_idx=-1,
                 eps=1e-2):
        super(RTDETRTransformer, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(backbone_feat_channels) <= num_levels
        assert len(feat_strides) == len(backbone_feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_size = eval_size

        # backbone feature projection
        self._build_input_proj_layer(backbone_feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels,
            num_decoder_points, bias_attr=True)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer,
                                          num_decoder_layers, eval_idx)

        # denoising part
        self.denoising_class_embed = nn.Embedding(num_classes, hidden_dim)
        normal_(self.denoising_class_embed.weight)
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # encoder head
        layerNorm = nn.LayerNorm(hidden_dim)
        layerNorm.weight.weight_decay = 0.0
        layerNorm.bias.weight_decay = 0.0
        self.enc_output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), layerNorm)
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)
        ])

        self._reset_parameters()

    def _reset_parameters(self):
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01)
        linear_init_(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight)
        constant_(self.enc_bbox_head.layers[-1].bias)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight)
            constant_(reg_.layers[-1].bias)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for l in self.input_proj:
            xavier_uniform_(l[0].weight)

        # init encoder output anchors and valid_mask
        if self.eval_size:
            device = torch.zeros((1, )).cuda().device
            self.anchors, self.valid_mask = self._generate_anchors(device=device)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'backbone_feat_channels': [i.channels for i in input_shape]}

    def _build_input_proj_layer(self, backbone_feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in backbone_feat_channels:
            batchNorm2d = nn.BatchNorm2d(self.hidden_dim)
            batchNorm2d.weight.weight_decay = 0.0
            batchNorm2d.bias.weight_decay = 0.0
            self.input_proj.append(
                nn.Sequential(nn.Conv2d(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias=False), batchNorm2d))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_levels - len(backbone_feat_channels)):
            batchNorm2d = nn.BatchNorm2d(self.hidden_dim)
            batchNorm2d.weight.weight_decay = 0.0
            batchNorm2d.bias.weight_decay = 0.0
            self.input_proj.append(
                nn.Sequential(nn.Conv2d(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False), batchNorm2d))
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = torch.cat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)

    def forward(self, feats, pad_mask=None, gt_meta=None):
        # input projection and embedding
        (memory, spatial_shapes, level_start_index) = self._get_encoder_input(feats)

        # prepare denoising training
        if self.training:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(gt_meta,
                                            self.num_classes,
                                            self.num_queries,
                                            self.denoising_class_embed,
                                            self.num_denoising,
                                            self.label_noise_ratio,
                                            self.box_noise_scale)
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = \
            self._get_decoder_input(
            memory, spatial_shapes, denoising_class, denoising_bbox_unact)

        # decoder
        out_bboxes, out_logits = self.decoder(
            target,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask)
        return (out_bboxes, out_logits, enc_topk_bboxes, enc_topk_logits, dn_meta)

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          device=None,
                          dtype=torch.float32):
        if spatial_shapes is None:
            spatial_shapes = [
                [int(self.eval_size[0] / s), int(self.eval_size[1] / s)] for s in self.feat_strides
            ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(end=h, dtype=dtype, device=device), torch.arange(end=w, dtype=dtype, device=device))
            grid_xy = torch.stack([grid_x, grid_y], -1)

            valid_WH = torch.Tensor([h, w]).to(device).to(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**lvl)
            anchors.append(torch.cat([grid_xy, wh], -1).reshape([-1, h * w, 4]))

        anchors = torch.cat(anchors, 1)
        valid_mask = ((anchors > self.eps) & (anchors < 1 - self.eps)).all(-1, keepdim=True)
        # valid_mask = ((anchors > self.eps) & (anchors < 1 - self.eps)).sum(-1, keepdim=True) > 0.
        anchors = torch.log(anchors / (1 - anchors))
        # 这里和paddle不同，暂时设为较大的数
        # anchors = torch.where(valid_mask, anchors, paddle.to_tensor(float("inf")))
        anchors = torch.where(valid_mask, anchors, torch.ones_like(anchors) * 100000.)
        return anchors, valid_mask

    def _get_decoder_input(self,
                           memory,
                           spatial_shapes,
                           denoising_class=None,
                           denoising_bbox_unact=None):
        bs, _, _ = memory.shape
        # prepare input for decoder
        if self.training or self.eval_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
        memory_ = torch.where(valid_mask, memory, torch.zeros_like(memory))
        output_memory = self.enc_output(memory_)

        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors


        enc_outputs_class_max, _ = enc_outputs_class.max(-1)
        _, topk_ind = torch.topk(enc_outputs_class_max, self.num_queries, dim=1)
        # extract region proposal boxes
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype, device=memory.device)
        batch_ind = batch_ind.unsqueeze(-1).tile([1, self.num_queries])
        topk_ind = torch.stack([batch_ind, topk_ind], dim=-1)
        topk_ind.requires_grad_(False)

        reference_points_unact = gather_nd(enc_outputs_coord_unact, topk_ind)  # unsigmoided.
        enc_topk_bboxes = torch.sigmoid(reference_points_unact)
        if denoising_bbox_unact is not None:
            reference_points_unact = torch.cat([denoising_bbox_unact, reference_points_unact], 1)
        if self.training:
            reference_points_unact = reference_points_unact.detach()
        enc_topk_logits = gather_nd(enc_outputs_class, topk_ind)

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = gather_nd(output_memory, topk_ind)
            if self.training:
                target = target.detach()
        if denoising_class is not None:
            target = torch.cat([denoising_class, target], 1)

        return target, reference_points_unact, enc_topk_bboxes, enc_topk_logits
