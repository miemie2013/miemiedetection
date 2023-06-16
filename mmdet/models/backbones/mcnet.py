from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import Conv2d

from mmdet.models.initializer import kaiming_normal_



class ConvBNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups=1,
                 act='relu'):
        super().__init__()

        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            bias=False)
        kaiming_normal_(self.conv.weight)

        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03)
        if act == 'hardswish':
            self.act = nn.Hardswish()
        elif act == 'relu6':
            self.act = nn.ReLU6()
        elif act == 'relu':
            self.act = nn.ReLU()
        elif act is None:
            self.act = None
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act:
            x = self.act(x)
        return x


class RepBlock3x3(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super(RepBlock3x3, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvBNLayer(ch_in, ch_out, kernel_size=3, stride=1, act=None)
        self.conv2 = ConvBNLayer(ch_in, ch_out, kernel_size=1, stride=1, act=None)
        if act == 'hardswish':
            self.act = nn.Hardswish()
        elif act == 'relu6':
            self.act = nn.ReLU6()
        elif act == 'relu':
            self.act = nn.ReLU()
        elif act is None:
            self.act = None
        else:
            raise NotImplementedError

    def forward(self, x):
        y = self.conv1(x) + self.conv2(x)
        if self.act:
            y = self.act(y)
        return y


class ResBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 act='relu'):
        super().__init__()
        assert in_channels == out_channels
        self.conv1 = ConvBNLayer(in_channels, out_channels, kernel_size=1, stride=1, act=act)
        self.conv2 = RepBlock3x3(out_channels, out_channels, act=act)
        # self.conv2 = ConvBNLayer(out_channels, out_channels, kernel_size=3, stride=1, act=act)
        self.conv3 = ConvBNLayer(out_channels, out_channels, kernel_size=1, stride=1, act=act)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        return x + y



class MCNet(nn.Module):
    def __init__(self, scale=1.0, feature_maps=[3, 4, 5], act='relu'):
        super().__init__()
        self.scale = scale
        self.feature_maps = feature_maps
        base_channels = int(scale * 64)  # 64
        self.blocks1 = nn.Sequential()
        self.blocks1.add_module("blocks1_0", ConvBNLayer(3, base_channels, kernel_size=3, stride=2, act=act))
        self.blocks1.add_module("blocks1_1", ConvBNLayer(base_channels * 1, base_channels * 1, kernel_size=3, stride=1, act=act))
        self.blocks2 = nn.Sequential()
        self.blocks2.add_module("blocks2_0", ConvBNLayer(base_channels, base_channels * 2, kernel_size=3, stride=2, act=act))
        self.blocks2.add_module("blocks2_1", ConvBNLayer(base_channels * 2, base_channels * 2, kernel_size=3, stride=1, act=act))
        self.blocks2.add_module("blocks2_2", ConvBNLayer(base_channels * 2, base_channels * 2, kernel_size=3, stride=1, act=act))
        self.blocks3 = nn.Sequential()
        self.blocks3.add_module("blocks3_0", ConvBNLayer(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, act=act))
        self.blocks3.add_module("blocks3_1", ConvBNLayer(base_channels * 4, base_channels * 4, kernel_size=3, stride=1, act=act))
        self.blocks3.add_module("blocks3_2", ResBlock(base_channels * 4, base_channels * 4, act=act))
        self.blocks4 = nn.Sequential()
        self.blocks4.add_module("blocks4_0", ConvBNLayer(base_channels * 4, base_channels * 6, kernel_size=3, stride=2, act=act))
        self.blocks4.add_module("blocks4_1", ResBlock(base_channels * 6, base_channels * 6, act=act))
        self.blocks4.add_module("blocks4_2", ResBlock(base_channels * 6, base_channels * 6, act=act))
        self.blocks4.add_module("blocks4_3", ResBlock(base_channels * 6, base_channels * 6, act=act))
        self.blocks5 = nn.Sequential()
        self.blocks5.add_module("blocks5_0", ConvBNLayer(base_channels * 6, base_channels * 8, kernel_size=3, stride=2, act=act))
        self.blocks5.add_module("blocks5_1", ResBlock(base_channels * 8, base_channels * 8, act=act))
        self.blocks5.add_module("blocks5_2", ResBlock(base_channels * 8, base_channels * 8, act=act))

    def forward(self, inputs):
        outs = []
        x = self.blocks1(inputs)
        x = self.blocks2(x)
        x = self.blocks3(x)
        outs.append(x)
        x = self.blocks4(x)
        outs.append(x)
        x = self.blocks5(x)
        outs.append(x)
        return outs
