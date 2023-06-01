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

from mmdet.models.ops import get_act_fn
from mmdet.models.custom_layers import ShapeSpec
import mmdet.models.ncnn_utils as ncnn_utils


class ConvBNLayer(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size=3,
                 stride=1,
                 groups=1,
                 padding=0,
                 act=None,
                 act_name=None):
        super(ConvBNLayer, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False)

        self.bn = nn.BatchNorm2d(ch_out)
        self.act_name = act_name
        if act is None or isinstance(act, (str, dict)):
            self.act = get_act_fn(act)
        else:
            self.act = act

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x

    def export_ncnn(self, ncnn_data, bottom_names):
        if ncnn_utils.support_fused_activation(self.act_name):
            bottom_names = ncnn_utils.fuse_conv_bn(ncnn_data, bottom_names, self.conv, self.bn, self.act_name)
        else:
            bottom_names = ncnn_utils.fuse_conv_bn(ncnn_data, bottom_names, self.conv, self.bn)
            bottom_names = ncnn_utils.activation(ncnn_data, bottom_names, self.act_name)
        return bottom_names


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu', act_name='relu', alpha=False):
        super(RepVggBlock, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvBNLayer(
            ch_in, ch_out, 3, stride=1, padding=1, act=None)
        self.conv2 = ConvBNLayer(
            ch_in, ch_out, 1, stride=1, padding=0, act=None)
        self.act_name = act_name
        self.act = get_act_fn(act) if act is None or isinstance(act, (str, dict)) else act
        if alpha:
            self.alpha = torch.nn.Parameter(torch.randn([1, ]))
            torch.nn.init.constant_(self.alpha, 1.0)
        else:
            self.alpha = None

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            if self.alpha:
                y = self.conv1(x) + self.alpha * self.conv2(x)
            else:
                y = self.conv1(x) + self.conv2(x)
        y = self.act(y)
        return y

    def export_ncnn(self, ncnn_data, bottom_names):
        if hasattr(self, 'conv'):
            raise NotImplementedError("not implemented.")
        else:
            # 看conv1分支，是卷积操作
            add_0 = self.conv1.export_ncnn(ncnn_data, bottom_names)

            # 看conv2分支，是卷积操作
            add_1 = self.conv2.export_ncnn(ncnn_data, bottom_names)

            # 最后是逐元素相加
            bottom_names = add_0 + add_1
            bottom_names = ncnn_utils.binaryOp(ncnn_data, bottom_names, op='Add')

        # 最后是激活
        bottom_names = ncnn_utils.activation(ncnn_data, bottom_names, self.act_name)
        return bottom_names

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(
                in_channels=self.ch_in,
                out_channels=self.ch_out,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1)
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.copy_(kernel)
        self.conv.bias.copy_(bias)
        self.__delattr__('conv1')
        self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(
            kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.bn._mean
        running_var = branch.bn._variance
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn._epsilon
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))
        return kernel * t, beta - running_mean * gamma / std


class BasicBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu', act_name='relu', shortcut=True, use_alpha=False):
        super(BasicBlock, self).__init__()
        assert ch_in == ch_out
        self.conv1 = ConvBNLayer(ch_in, ch_out, 3, stride=1, padding=1, act=act, act_name=act_name)
        self.conv2 = RepVggBlock(ch_out, ch_out, act=act, act_name=act_name, alpha=use_alpha)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.shortcut:
            return x + y
        else:
            return y

    def export_ncnn(self, ncnn_data, bottom_names):
        if self.shortcut:
            add_0 = bottom_names

            # 看conv1层，是卷积操作
            y = self.conv1.export_ncnn(ncnn_data, bottom_names)
            # 看conv2层，是卷积操作
            y = self.conv2.export_ncnn(ncnn_data, y)

            # 最后是逐元素相加
            bottom_names = add_0 + y
            bottom_names = ncnn_utils.binaryOp(ncnn_data, bottom_names, op='Add')
        else:
            # 看conv1层，是卷积操作
            bottom_names = self.conv1.export_ncnn(ncnn_data, bottom_names)
            # 看conv2层，是卷积操作
            bottom_names = self.conv2.export_ncnn(ncnn_data, bottom_names)
        return bottom_names


class EffectiveSELayer(nn.Module):
    """ Effective Squeeze-Excitation
    From `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667
    """

    def __init__(self, channels, act='hardsigmoid', act_name='hardsigmoid'):
        super(EffectiveSELayer, self).__init__()
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.act_name = act_name
        self.act = get_act_fn(act) if act is None or isinstance(act, (
            str, dict)) else act

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.act(x_se)

    def export_ncnn(self, ncnn_data, bottom_names):
        # 看x_se分支，首先是mean操作，对应ncnn里的Reduction层
        x_se = ncnn_utils.reduction(ncnn_data, bottom_names, op='ReduceMean', input_dims=4, dims=(2, 3), keepdim=True)

        # 看x_se分支，然后是卷积操作
        x_se = ncnn_utils.conv2d(ncnn_data, x_se, self.fc)

        # 看x_se分支，然后是激活操作
        x_se = ncnn_utils.activation(ncnn_data, x_se, act_name=self.act_name)

        # 最后是逐元素相乘
        bottom_names = [bottom_names[0], x_se[0]]
        bottom_names = ncnn_utils.binaryOp(ncnn_data, bottom_names, op='Mul')
        return bottom_names


class CSPResStage(nn.Module):
    def __init__(self,
                 block_fn,
                 ch_in,
                 ch_out,
                 n,
                 stride,
                 act='relu',
                 act_name=None,
                 attn='eca',
                 use_alpha=False):
        super(CSPResStage, self).__init__()

        ch_mid = (ch_in + ch_out) // 2
        if stride == 2:
            self.conv_down = ConvBNLayer(
                ch_in, ch_mid, 3, stride=2, padding=1, act=act, act_name=act_name)
        else:
            self.conv_down = None
        self.conv1 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act, act_name=act_name)
        self.conv2 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act, act_name=act_name)
        self.blocks = nn.Sequential(*[
            block_fn(
                ch_mid // 2, ch_mid // 2, act=act, act_name=act_name, shortcut=True, use_alpha=use_alpha)
            for i in range(n)
        ])
        if attn:
            self.attn = EffectiveSELayer(ch_mid, act='hardsigmoid', act_name='hardsigmoid')
        else:
            self.attn = None

        self.conv3 = ConvBNLayer(ch_mid, ch_out, 1, act=act, act_name=act_name)

    def forward(self, x):
        if self.conv_down is not None:
            x = self.conv_down(x)
        y1 = self.conv1(x)
        y2 = self.blocks(self.conv2(x))
        y = torch.cat([y1, y2], 1)
        if self.attn is not None:
            y = self.attn(y)
        y = self.conv3(y)
        return y

    def export_ncnn(self, ncnn_data, bottom_names):
        if self.conv_down is not None:
            bottom_names = self.conv_down.export_ncnn(ncnn_data, bottom_names)
        # 看conv1层，是卷积操作
        y1 = self.conv1.export_ncnn(ncnn_data, bottom_names)

        # 看conv2层，是卷积操作
        temp = self.conv2.export_ncnn(ncnn_data, bottom_names)
        for layer in self.blocks:
            temp = layer.export_ncnn(ncnn_data, temp)
        y2 = temp

        # concat
        bottom_names = y1 + y2
        bottom_names = ncnn_utils.concat(ncnn_data, bottom_names, dim=1)

        if self.attn is not None:
            bottom_names = self.attn.export_ncnn(ncnn_data, bottom_names)
        bottom_names = self.conv3.export_ncnn(ncnn_data, bottom_names)
        return bottom_names


class CSPResNet(nn.Module):
    __shared__ = ['width_mult', 'depth_mult', 'trt']

    def __init__(self,
                 layers=[3, 6, 6, 3],
                 channels=[64, 128, 256, 512, 1024],
                 act='swish',
                 return_idx=[0, 1, 2, 3, 4],
                 depth_wise=False,
                 use_large_stem=False,
                 width_mult=1.0,
                 depth_mult=1.0,
                 freeze_at=-1,
                 trt=False,
                 use_alpha=False):
        super(CSPResNet, self).__init__()
        channels = [max(round(c * width_mult), 1) for c in channels]
        layers = [max(round(l * depth_mult), 1) for l in layers]
        act_name = act
        act = get_act_fn(
            act, trt=trt) if act is None or isinstance(act,
                                                       (str, dict)) else act

        if use_large_stem:
            self.stem = nn.Sequential()
            self.stem.add_module('conv1', ConvBNLayer(3, channels[0] // 2, 3, stride=2, padding=1, act=act, act_name=act_name))
            self.stem.add_module('conv2', ConvBNLayer(channels[0] // 2, channels[0] // 2, 3, stride=1, padding=1, act=act, act_name=act_name))
            self.stem.add_module('conv3', ConvBNLayer(channels[0] // 2, channels[0], 3, stride=1, padding=1, act=act, act_name=act_name))
        else:
            self.stem = nn.Sequential()
            self.stem.add_module('conv1', ConvBNLayer(3, channels[0] // 2, 3, stride=2, padding=1, act=act, act_name=act_name))
            self.stem.add_module('conv2', ConvBNLayer(channels[0] // 2, channels[0], 3, stride=1, padding=1, act=act, act_name=act_name))

        n = len(channels) - 1
        self.stages = nn.Sequential()
        for i in range(n):
            self.stages.add_module(str(i), CSPResStage(BasicBlock, channels[i], channels[i + 1], layers[i], 2, act=act, act_name=act_name, use_alpha=use_alpha))

        self._out_channels = channels[1:]
        self._out_strides = [4, 8, 16, 32]
        self.return_idx = return_idx

        if freeze_at >= 0:
            self._freeze_parameters(self.stem)
            for i in range(min(freeze_at + 1, n)):
                self._freeze_parameters(self.stages[i])

    def _freeze_parameters(self, m):
        for p in m.parameters():
            p.requires_grad_(False)

    def forward(self, inputs):
        x = self.stem(inputs)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)

        return outs

    def export_ncnn(self, ncnn_data, bottom_names):
        for layer in self.stem:
            bottom_names = layer.export_ncnn(ncnn_data, bottom_names)
        out_names = []
        for idx, stage in enumerate(self.stages):
            bottom_names = stage.export_ncnn(ncnn_data, bottom_names)
            if idx in self.return_idx:
                out_names.append(bottom_names[0])
        return out_names

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=self._out_channels[i], stride=self._out_strides[i])
            for i in self.return_idx
        ]
