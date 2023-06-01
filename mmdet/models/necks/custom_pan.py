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

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.backbones.cspresnet import ConvBNLayer, BasicBlock
from mmdet.models.custom_layers import DropBlock
from mmdet.models.ops import get_act_fn
from mmdet.models.custom_layers import ShapeSpec
import mmdet.models.ncnn_utils as ncnn_utils

__all__ = ['CustomCSPPAN']


class SPP(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 k,
                 pool_size,
                 act='swish',
                 act_name='swish',
                 data_format='NCHW'):
        super(SPP, self).__init__()
        self.pool = []
        self.data_format = data_format
        for i, size in enumerate(pool_size):
            name = 'pool{}'.format(i)
            pool = nn.MaxPool2d(
                    kernel_size=size,
                    stride=1,
                    padding=size // 2,
                    ceil_mode=False)
            self.add_module(name, pool)
            self.pool.append(pool)
        self.conv = ConvBNLayer(ch_in, ch_out, k, padding=k // 2, act=act, act_name=act_name)

    def forward(self, x):
        outs = [x]
        for pool in self.pool:
            outs.append(pool(x))
        if self.data_format == 'NCHW':
            y = torch.cat(outs, 1)
        else:
            y = torch.cat(outs, -1)

        y = self.conv(y)
        return y

    def export_ncnn(self, ncnn_data, bottom_names):
        concat_input = [bottom_names[0]]
        for pool in self.pool:
            pool_out = ncnn_utils.pooling(ncnn_data, bottom_names, op='MaxPool', pool=pool)
            concat_input.append(pool_out[0])

        # concat
        if self.data_format == 'NCHW':
            bottom_names = ncnn_utils.concat(ncnn_data, concat_input, dim=1)
        else:
            bottom_names = ncnn_utils.concat(ncnn_data, concat_input, dim=3)
        bottom_names = self.conv.export_ncnn(ncnn_data, bottom_names)
        return bottom_names


class CSPStage(nn.Module):
    def __init__(self, block_fn, ch_in, ch_out, n, act='swish', act_name='swish', spp=False):
        super(CSPStage, self).__init__()

        ch_mid = int(ch_out // 2)
        self.conv1 = ConvBNLayer(ch_in, ch_mid, 1, act=act, act_name=act_name)
        self.conv2 = ConvBNLayer(ch_in, ch_mid, 1, act=act, act_name=act_name)
        self.convs = nn.Sequential()
        next_ch_in = ch_mid
        for i in range(n):
            self.convs.add_module(
                str(i),
                eval(block_fn)(next_ch_in, ch_mid, act=act, act_name=act_name, shortcut=False))
            if i == (n - 1) // 2 and spp:
                self.convs.add_module(
                    'spp', SPP(ch_mid * 4, ch_mid, 1, [5, 9, 13], act=act, act_name=act_name))
            next_ch_in = ch_mid
        self.conv3 = ConvBNLayer(ch_mid * 2, ch_out, 1, act=act, act_name=act_name)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y2 = self.convs(y2)
        y = torch.cat([y1, y2], 1)
        y = self.conv3(y)
        return y

    def export_ncnn(self, ncnn_data, bottom_names):
        # 看conv1分支，是卷积操作
        y1 = self.conv1.export_ncnn(ncnn_data, bottom_names)

        # 看conv2分支，是卷积操作
        y2 = self.conv2.export_ncnn(ncnn_data, bottom_names)
        for layer in self.convs:
            y2 = layer.export_ncnn(ncnn_data, y2)

        # concat
        bottom_names = y1 + y2
        bottom_names = ncnn_utils.concat(ncnn_data, bottom_names, dim=1)

        bottom_names = self.conv3.export_ncnn(ncnn_data, bottom_names)
        return bottom_names


class CustomCSPPAN(nn.Module):
    __shared__ = ['norm_type', 'data_format', 'width_mult', 'depth_mult', 'trt']

    def __init__(self,
                 in_channels=[256, 512, 1024],
                 out_channels=[1024, 512, 256],
                 norm_type='bn',
                 act='leaky',
                 stage_fn='CSPStage',
                 block_fn='BasicBlock',
                 stage_num=1,
                 block_num=3,
                 drop_block=False,
                 block_size=3,
                 keep_prob=0.9,
                 spp=False,
                 data_format='NCHW',
                 width_mult=1.0,
                 depth_mult=1.0,
                 trt=False):

        super(CustomCSPPAN, self).__init__()
        out_channels = [max(round(c * width_mult), 1) for c in out_channels]
        block_num = max(round(block_num * depth_mult), 1)
        act_name = act
        act = get_act_fn(
            act, trt=trt) if act is None or isinstance(act,
                                                       (str, dict)) else act
        self.num_blocks = len(in_channels)
        self.data_format = data_format
        self._out_channels = out_channels
        in_channels = in_channels[::-1]
        fpn_stages = []
        fpn_routes = []
        for i, (ch_in, ch_out) in enumerate(zip(in_channels, out_channels)):
            if i > 0:
                ch_in += ch_pre // 2

            stage = nn.Sequential()
            for j in range(stage_num):
                stage.add_module(
                    str(j),
                    eval(stage_fn)(block_fn,
                                   ch_in if j == 0 else ch_out,
                                   ch_out,
                                   block_num,
                                   act=act,
                                   act_name=act_name,
                                   spp=(spp and i == 0)))

            if drop_block:
                stage.add_module('drop', DropBlock(block_size, keep_prob))

            fpn_stages.append(stage)

            if i < self.num_blocks - 1:
                fpn_routes.append(
                    ConvBNLayer(
                        ch_in=ch_out,
                        ch_out=ch_out // 2,
                        filter_size=1,
                        stride=1,
                        padding=0,
                        act=act,
                        act_name=act_name))

            ch_pre = ch_out

        self.fpn_stages = nn.ModuleList(fpn_stages)
        self.fpn_routes = nn.ModuleList(fpn_routes)

        pan_stages = []
        pan_routes = []
        for i in reversed(range(self.num_blocks - 1)):
            pan_routes.append(
                ConvBNLayer(
                    ch_in=out_channels[i + 1],
                    ch_out=out_channels[i + 1],
                    filter_size=3,
                    stride=2,
                    padding=1,
                    act=act,
                    act_name=act_name))

            ch_in = out_channels[i] + out_channels[i + 1]
            ch_out = out_channels[i]
            stage = nn.Sequential()
            for j in range(stage_num):
                stage.add_module(
                    str(j),
                    eval(stage_fn)(block_fn,
                                   ch_in if j == 0 else ch_out,
                                   ch_out,
                                   block_num,
                                   act=act,
                                   act_name=act_name,
                                   spp=False))
            if drop_block:
                stage.add_module('drop', DropBlock(block_size, keep_prob))

            pan_stages.append(stage)

        self.pan_stages = nn.ModuleList(pan_stages[::-1])
        self.pan_routes = nn.ModuleList(pan_routes[::-1])

    def forward(self, blocks, for_mot=False):
        blocks = blocks[::-1]
        fpn_feats = []

        for i, block in enumerate(blocks):
            if i > 0:
                block = torch.cat([route, block], 1)
            route = self.fpn_stages[i](block)
            fpn_feats.append(route)

            if i < self.num_blocks - 1:
                route = self.fpn_routes[i](route)
                route = F.interpolate(route, scale_factor=2.)

        pan_feats = [fpn_feats[-1], ]
        route = fpn_feats[-1]
        for i in reversed(range(self.num_blocks - 1)):
            block = fpn_feats[i]
            route = self.pan_routes[i](route)
            block = torch.cat([route, block], 1)
            route = self.pan_stages[i](block)
            pan_feats.append(route)

        return pan_feats[::-1]

    def export_ncnn(self, ncnn_data, bottom_names):
        blocks = bottom_names[::-1]
        fpn_feats = []

        for i, block in enumerate(blocks):
            if i > 0:
                bottom_names = route + [block, ]
                block = ncnn_utils.concat(ncnn_data, bottom_names, dim=1)
                block = block[0]
            route = [block, ]
            for layer in self.fpn_stages[i]:
                route = layer.export_ncnn(ncnn_data, route)
            fpn_feats.append(route[0])

            if i < self.num_blocks - 1:
                route = self.fpn_routes[i].export_ncnn(ncnn_data, route)
                route = ncnn_utils.interpolate(ncnn_data, route, scale_factor=2.)

        pan_feats = [fpn_feats[-1], ]
        route = fpn_feats[-1]
        for i in reversed(range(self.num_blocks - 1)):
            block = fpn_feats[i]
            route = self.pan_routes[i].export_ncnn(ncnn_data, [route, ])
            bottom_names = route + [block, ]
            block = ncnn_utils.concat(ncnn_data, bottom_names, dim=1)
            route = block
            for layer in self.pan_stages[i]:
                route = layer.export_ncnn(ncnn_data, route)
            route = route[0]
            pan_feats.append(route)

        return pan_feats[::-1]

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    @property
    def out_shape(self):
        return [ShapeSpec(channels=c) for c in self._out_channels]
