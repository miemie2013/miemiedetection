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

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ops import get_act_fn
from ..network_blocks import BaseConv
from ..backbones.cspresnet import RepVggBlock
from .detr_transformer import TransformerEncoder
from ..initializer import xavier_uniform_, linear_init_, constant_
from ..layers import MultiHeadAttention

__all__ = ['HybridEncoder']



class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=False,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(
                hidden_channels, hidden_channels, act=act, act_name=act)
            for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = BaseConv(
                hidden_channels,
                out_channels,
                ksize=1,
                stride=1,
                bias=bias,
                act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


class TransformerLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False):
        super(TransformerLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        self.self_attn = MultiHeadAttention(d_model, nhead, attn_dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(act_dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = getattr(F, activation)
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None):
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class HybridEncoder(nn.Module):
    __shared__ = ['depth_mult', 'act', 'trt', 'eval_size']
    __inject__ = ['encoder_layer']

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 encoder_layer='TransformerLayer',
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 trt=False,
                 eval_size=None):
        super(HybridEncoder, self).__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_size = eval_size

        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            conv_layer = nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)
            bn_layer = nn.BatchNorm2d(hidden_dim)   # bn的weight和bias不使用L2正则
            self.input_proj.append(nn.Sequential(conv_layer, bn_layer))
        # encoder transformer,  num_encoder_layers==1
        self.encoder = nn.ModuleList([
            TransformerEncoder(encoder_layer, num_encoder_layers)
            for _ in range(len(use_encoder_idx))
        ])

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(BaseConv(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    act=act,
                    expansion=expansion))

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsample_convs.append(BaseConv(hidden_dim, hidden_dim, 3, stride=2, act=act))
            self.pan_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    act=act,
                    expansion=expansion))

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                device = torch.zeros((1, )).cuda().device
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_size[1] // stride, self.eval_size[0] // stride, device,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, device,
                                           embed_dim=256,
                                           temperature=10000.):
        grid_w = torch.arange(int(w), dtype=torch.float32, device=device)
        grid_h = torch.arange(int(h), dtype=torch.float32, device=device)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32, device=device) / pos_dim   # [pos_dim, ]  下标归一化到0到1之间
        omega = 1. / (temperature**omega)    # [pos_dim, ]  omega从1递减到接近 1/temperature
        omega = omega.unsqueeze(0)   # [1, pos_dim]

        grid_w = torch.reshape(grid_w, (h*w, 1))   # [h*w, 1]
        grid_h = torch.reshape(grid_h, (h*w, 1))   # [h*w, 1]

        out_w = grid_w @ omega   # 矩阵乘, [h*w, pos_dim]
        out_h = grid_h @ omega   # 矩阵乘, [h*w, pos_dim]

        out = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)  # [h*w, 4*pos_dim]
        out = out.unsqueeze(0)   # [1, h*w, hidden_dim]
        return out

    def forward(self, feats, for_mot=False):
        assert len(feats) == len(self.in_channels)
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute([0, 2, 1])
                device = src_flatten.device
                if self.training or self.eval_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(w, h, device, self.hidden_dim, self.pe_temperature)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None)
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute([0, 2, 1]).reshape([-1, self.hidden_dim, h, w])

        # top-down fpn
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh

            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode="nearest")
            feat_ = torch.cat([upsample_feat, feat_low], dim=1)
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](feat_)
            inner_outs.insert(0, inner_out)

        # bottom-up pan
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            feat_ = torch.cat([downsample_feat, feat_height], dim=1)
            out = self.pan_blocks[idx](feat_)
            outs.append(out)

        return outs
