#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.backbones.darknet import Darknet
from mmdet.models.network_blocks import BaseConv

from mmdet.models.custom_layers import paddle_yolo_box, CoordConv, SPP, DropBlock, ConvBNLayer
import mmdet.models.ncnn_utils as ncnn_utils

class YOLOFPN(nn.Module):
    """
    YOLOFPN module. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=53,
        in_features=["dark3", "dark4", "dark5"],
    ):
        super().__init__()

        self.backbone = Darknet(depth)
        self.in_features = in_features

        # out 1
        self.out1_cbl = self._make_cbl(512, 256, 1)
        self.out1 = self._make_embedding([256, 512], 512 + 256)

        # out 2
        self.out2_cbl = self._make_cbl(256, 128, 1)
        self.out2 = self._make_embedding([128, 256], 256 + 128)

        # upsample
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def _make_cbl(self, _in, _out, ks):
        return BaseConv(_in, _out, ks, stride=1, act="lrelu")

    def _make_embedding(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                self._make_cbl(in_filters, filters_list[0], 1),
                self._make_cbl(filters_list[0], filters_list[1], 3),
                self._make_cbl(filters_list[1], filters_list[0], 1),
                self._make_cbl(filters_list[0], filters_list[1], 3),
                self._make_cbl(filters_list[1], filters_list[0], 1),
            ]
        )
        return m

    def load_pretrained_model(self, filename="./weights/darknet53.mix.pth"):
        with open(filename, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        print("loading pretrained weights...")
        self.backbone.load_state_dict(state_dict)

    def forward(self, inputs):
        """
        Args:
            inputs (Tensor): input image.

        Returns:
            Tuple[Tensor]: FPN output features..
        """
        #  backbone
        out_features = self.backbone(inputs)
        x2, x1, x0 = [out_features[f] for f in self.in_features]

        #  yolo branch 1
        x1_in = self.out1_cbl(x0)
        x1_in = self.upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out_dark4 = self.out1(x1_in)

        #  yolo branch 2
        x2_in = self.out2_cbl(out_dark4)
        x2_in = self.upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out_dark3 = self.out2(x2_in)

        outputs = (out_dark3, out_dark4, x0)
        return outputs


class PPYOLODetBlock(nn.Module):
    def __init__(self, cfg, name, data_format='NCHW'):
        """
        PPYOLODetBlock layer

        Args:
            cfg (list): layer configs for this block
            name (str): block name
            data_format (str): data format, NCHW or NHWC
        """
        super(PPYOLODetBlock, self).__init__()
        self.conv_module = nn.Sequential()
        for idx, (conv_name, layer, args, kwargs) in enumerate(cfg[:-1]):
            kwargs.update(name='{}.{}'.format(name, conv_name), data_format=data_format)
            self.conv_module.add_module(conv_name, layer(*args, **kwargs))

        # 要保存中间结果route，所以最后的self.tip层不放进self.conv_module里
        conv_name, layer, args, kwargs = cfg[-1]
        kwargs.update(name='{}.{}'.format(name, conv_name), data_format=data_format)
        self.tip = layer(*args, **kwargs)

    def forward(self, inputs):
        route = self.conv_module(inputs)
        tip = self.tip(route)
        return route, tip

    def export_ncnn(self, ncnn_data, bottom_names):
        for layer in self.conv_module:
            bottom_names = layer.export_ncnn(ncnn_data, bottom_names)
        tip = self.tip.export_ncnn(ncnn_data, bottom_names)
        return bottom_names, tip


class PPYOLOFPN(nn.Module):
    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 norm_type='bn',
                 freeze_norm=False,
                 data_format='NCHW',
                 coord_conv=False,
                 conv_block_num=2,
                 drop_block=False,
                 block_size=3,
                 keep_prob=0.9,
                 spp=False):
        """
        PPYOLOFPN layer

        Args:
            in_channels (list): input channels for fpn
            norm_type (str): batch norm type, default bn
            data_format (str): data format, NCHW or NHWC
            coord_conv (bool): whether use CoordConv or not
            conv_block_num (int): conv block num of each pan block
            drop_block (bool): whether use DropBlock or not
            block_size (int): block size of DropBlock
            keep_prob (float): keep probability of DropBlock
            spp (bool): whether use spp or not

        """
        super(PPYOLOFPN, self).__init__()
        assert len(in_channels) > 0, "in_channels length should > 0"
        self.in_channels = in_channels
        self.num_blocks = len(in_channels)
        # parse kwargs
        self.coord_conv = coord_conv
        self.drop_block = drop_block
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.spp = spp
        self.conv_block_num = conv_block_num
        self.data_format = data_format
        if self.coord_conv:
            ConvLayer = CoordConv
        else:
            ConvLayer = ConvBNLayer

        if self.drop_block:
            dropblock_cfg = [[
                'dropblock', DropBlock, [self.block_size, self.keep_prob],
                dict()
            ]]
        else:
            dropblock_cfg = []

        self._out_channels = []
        self.yolo_blocks = []
        self.routes = []
        for i, ch_in in enumerate(self.in_channels[::-1]):
            if i > 0:
                ch_in += 512 // (2**i)
            channel = 64 * (2**self.num_blocks) // (2**i)
            base_cfg = []
            c_in, c_out = ch_in, channel
            for j in range(self.conv_block_num):
                base_cfg += [
                    [
                        'conv{}'.format(2 * j), ConvLayer, [c_in, c_out, 1],
                        dict(
                            padding=0,
                            norm_type=norm_type,
                            freeze_norm=freeze_norm)
                    ],
                    [
                        'conv{}'.format(2 * j + 1), ConvBNLayer,
                        [c_out, c_out * 2, 3], dict(
                            padding=1,
                            norm_type=norm_type,
                            freeze_norm=freeze_norm)
                    ],
                ]
                c_in, c_out = c_out * 2, c_out

            base_cfg += [[
                'route', ConvLayer, [c_in, c_out, 1], dict(
                    padding=0, norm_type=norm_type, freeze_norm=freeze_norm)
            ], [
                'tip', ConvLayer, [c_out, c_out * 2, 3], dict(
                    padding=1, norm_type=norm_type, freeze_norm=freeze_norm)
            ]]

            if self.conv_block_num == 2:
                if i == 0:
                    if self.spp:
                        spp_cfg = [[
                            'spp', SPP, [channel * 4, channel, 1], dict(
                                pool_size=[5, 9, 13],
                                norm_type=norm_type,
                                freeze_norm=freeze_norm)
                        ]]
                    else:
                        spp_cfg = []
                    cfg = base_cfg[0:3] + spp_cfg + base_cfg[
                        3:4] + dropblock_cfg + base_cfg[4:6]
                else:
                    cfg = base_cfg[0:2] + dropblock_cfg + base_cfg[2:6]
            elif self.conv_block_num == 0:
                if self.spp and i == 0:
                    spp_cfg = [[
                        'spp', SPP, [c_in * 4, c_in, 1], dict(
                            pool_size=[5, 9, 13],
                            norm_type=norm_type,
                            freeze_norm=freeze_norm)
                    ]]
                else:
                    spp_cfg = []
                cfg = spp_cfg + dropblock_cfg + base_cfg
            name = 'yolo_block_{}'.format(i)
            yolo_block = PPYOLODetBlock(cfg, name)
            self.add_module(name, yolo_block)
            self.yolo_blocks.append(yolo_block)
            self._out_channels.append(channel * 2)
            if i < self.num_blocks - 1:
                name = 'yolo_transition_{}'.format(i)
                route = ConvBNLayer(
                        ch_in=channel,
                        ch_out=256 // (2**i),
                        filter_size=1,
                        stride=1,
                        padding=0,
                        norm_type=norm_type,
                        freeze_norm=freeze_norm,
                        data_format=data_format,
                        name=name)
                self.add_module(name, route)
                self.routes.append(route)

    def get_block(self, name):
        layer = getattr(self, name)
        return layer

    def forward(self, blocks, for_mot=False):
        assert len(blocks) == self.num_blocks
        blocks = blocks[::-1]
        yolo_feats = []

        # add embedding features output for multi-object tracking model
        if for_mot:
            emb_feats = []

        for i, block in enumerate(blocks):
            if i > 0:
                if self.data_format == 'NCHW':
                    block = torch.cat([route, block], 1)
                else:
                    block = torch.cat([route, block], -1)
            route, tip = self.yolo_blocks[i](block)
            yolo_feats.append(tip)

            if for_mot:
                # add embedding features output
                emb_feats.append(route)

            if i < self.num_blocks - 1:
                route = self.routes[i](route)
                route = F.interpolate(route, scale_factor=2., mode='nearest')

        if for_mot:
            return {'yolo_feats': yolo_feats, 'emb_feats': emb_feats}
        else:
            return yolo_feats

    def export_ncnn(self, ncnn_data, bottom_names):
        blocks = bottom_names[::-1]
        yolo_feats = []

        for i, block in enumerate(blocks):
            block = [block, ]
            if i > 0:
                if self.data_format == 'NCHW':
                    block = ncnn_utils.concat(ncnn_data, route + block, dim=1)
                else:
                    block = ncnn_utils.concat(ncnn_data, route + block, dim=3)
            route, tip = self.yolo_blocks[i].export_ncnn(ncnn_data, block)
            yolo_feats.append(tip[0])

            if i < self.num_blocks - 1:
                route = self.routes[i].export_ncnn(ncnn_data, route)
                route = ncnn_utils.interpolate(ncnn_data, route, scale_factor=2., mode='nearest')
        return yolo_feats


class PPYOLODetBlockCSP(nn.Module):
    def __init__(self,
                 cfg,
                 ch_in,
                 ch_out,
                 act,
                 norm_type,
                 name,
                 data_format='NCHW'):
        """
        PPYOLODetBlockCSP layer

        Args:
            cfg (list): layer configs for this block
            ch_in (int): input channel
            ch_out (int): output channel
            act (str): default mish
            name (str): block name
            data_format (str): data format, NCHW or NHWC
        """
        super(PPYOLODetBlockCSP, self).__init__()
        self.data_format = data_format
        self.conv1 = ConvBNLayer(
            ch_in,
            ch_out,
            1,
            padding=0,
            act=act,
            norm_type=norm_type,
            name=name + '.left',
            data_format=data_format)
        self.conv2 = ConvBNLayer(
            ch_in,
            ch_out,
            1,
            padding=0,
            act=act,
            norm_type=norm_type,
            name=name + '.right',
            data_format=data_format)
        self.conv3 = ConvBNLayer(
            ch_out * 2,
            ch_out * 2,
            1,
            padding=0,
            act=act,
            norm_type=norm_type,
            name=name,
            data_format=data_format)
        self.conv_module = nn.Sequential()
        for idx, (layer_name, layer, args, kwargs) in enumerate(cfg):
            kwargs.update(name=name + layer_name, data_format=data_format)
            layer_name = layer_name.replace('.', '_')
            self.conv_module.add_module(layer_name, layer(*args, **kwargs))

    def forward(self, inputs):
        conv_left = self.conv1(inputs)
        conv_right = self.conv2(inputs)
        conv_left = self.conv_module(conv_left)
        if self.data_format == 'NCHW':
            conv = torch.cat([conv_left, conv_right], 1)
        else:
            conv = torch.cat([conv_left, conv_right], -1)

        conv = self.conv3(conv)
        return conv, conv

    def export_ncnn(self, ncnn_data, bottom_names):
        conv_left = self.conv1.export_ncnn(ncnn_data, bottom_names)
        conv_right = self.conv2.export_ncnn(ncnn_data, bottom_names)
        for layer in self.conv_module:
            conv_left = layer.export_ncnn(ncnn_data, conv_left)
        if self.data_format == 'NCHW':
            conv = ncnn_utils.concat(ncnn_data, conv_left + conv_right, dim=1)
        else:
            conv = ncnn_utils.concat(ncnn_data, conv_left + conv_right, dim=3)

        conv = self.conv3.export_ncnn(ncnn_data, conv)
        conv2 = copy.deepcopy(conv)
        return conv, conv2


class PPYOLOPAN(nn.Module):
    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 norm_type='bn',
                 data_format='NCHW',
                 act='mish',
                 conv_block_num=3,
                 drop_block=False,
                 block_size=3,
                 keep_prob=0.9,
                 spp=False):
        """
        PPYOLOPAN layer with SPP, DropBlock and CSP connection.

        Args:
            in_channels (list): input channels for fpn
            norm_type (str): batch norm type, default bn
            data_format (str): data format, NCHW or NHWC
            act (str): activation function, default mish
            conv_block_num (int): conv block num of each pan block
            drop_block (bool): whether use DropBlock or not
            block_size (int): block size of DropBlock
            keep_prob (float): keep probability of DropBlock
            spp (bool): whether use spp or not

        """
        super(PPYOLOPAN, self).__init__()
        assert len(in_channels) > 0, "in_channels length should > 0"
        self.in_channels = in_channels
        self.num_blocks = len(in_channels)
        # parse kwargs
        self.drop_block = drop_block
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.spp = spp
        self.conv_block_num = conv_block_num
        self.data_format = data_format
        if self.drop_block:
            dropblock_cfg = [[
                'dropblock', DropBlock, [self.block_size, self.keep_prob],
                dict()
            ]]
        else:
            dropblock_cfg = []

        # fpn
        self.fpn_blocks = []
        self.fpn_routes = []
        fpn_channels = []
        for i, ch_in in enumerate(self.in_channels[::-1]):
            if i > 0:
                ch_in += 512 // (2**(i - 1))
            channel = 512 // (2**i)
            base_cfg = []
            for j in range(self.conv_block_num):
                base_cfg += [
                    # name, layer, args
                    [
                        '{}.0'.format(j), ConvBNLayer, [channel, channel, 1],
                        dict(
                            padding=0, act=act, norm_type=norm_type)
                    ],
                    [
                        '{}.1'.format(j), ConvBNLayer, [channel, channel, 3],
                        dict(
                            padding=1, act=act, norm_type=norm_type)
                    ]
                ]

            if i == 0 and self.spp:
                base_cfg[3] = [
                    'spp', SPP, [channel * 4, channel, 1], dict(
                        pool_size=[5, 9, 13], act=act, norm_type=norm_type)
                ]

            cfg = base_cfg[:4] + dropblock_cfg + base_cfg[4:]
            name = 'fpn_{}'.format(i)
            fpn_block = PPYOLODetBlockCSP(cfg, ch_in, channel, act, norm_type, name, data_format)
            self.add_module(name, fpn_block)
            self.fpn_blocks.append(fpn_block)
            fpn_channels.append(channel * 2)
            if i < self.num_blocks - 1:
                name = 'fpn_transition_{}'.format(i)
                route = ConvBNLayer(
                        ch_in=channel * 2,
                        ch_out=channel,
                        filter_size=1,
                        stride=1,
                        padding=0,
                        act=act,
                        norm_type=norm_type,
                        data_format=data_format,
                        name=name)
                self.add_module(name, route)
                self.fpn_routes.append(route)
        # pan
        self.pan_blocks = []
        self.pan_routes = []
        self._out_channels = [512 // (2**(self.num_blocks - 2)), ]
        for i in reversed(range(self.num_blocks - 1)):
            name = 'pan_transition_{}'.format(i)
            route = ConvBNLayer(
                    ch_in=fpn_channels[i + 1],
                    ch_out=fpn_channels[i + 1],
                    filter_size=3,
                    stride=2,
                    padding=1,
                    act=act,
                    norm_type=norm_type,
                    data_format=data_format,
                    name=name)
            self.add_module(name, route)
            self.pan_routes = [route, ] + self.pan_routes
            base_cfg = []
            ch_in = fpn_channels[i] + fpn_channels[i + 1]
            channel = 512 // (2**i)
            for j in range(self.conv_block_num):
                base_cfg += [
                    # name, layer, args
                    [
                        '{}.0'.format(j), ConvBNLayer, [channel, channel, 1],
                        dict(
                            padding=0, act=act, norm_type=norm_type)
                    ],
                    [
                        '{}.1'.format(j), ConvBNLayer, [channel, channel, 3],
                        dict(
                            padding=1, act=act, norm_type=norm_type)
                    ]
                ]

            cfg = base_cfg[:4] + dropblock_cfg + base_cfg[4:]
            name = 'pan_{}'.format(i)
            pan_block = PPYOLODetBlockCSP(cfg, ch_in, channel, act, norm_type, name, data_format)
            self.add_module(name, pan_block)
            self.pan_blocks = [pan_block, ] + self.pan_blocks
            self._out_channels.append(channel * 2)

        self._out_channels = self._out_channels[::-1]

    def forward(self, blocks, for_mot=False):
        assert len(blocks) == self.num_blocks
        blocks = blocks[::-1]
        fpn_feats = []

        # add embedding features output for multi-object tracking model
        if for_mot:
            emb_feats = []

        for i, block in enumerate(blocks):
            if i > 0:
                if self.data_format == 'NCHW':
                    block = torch.cat([route, block], 1)
                else:
                    block = torch.cat([route, block], -1)
            route, tip = self.fpn_blocks[i](block)
            fpn_feats.append(tip)

            if for_mot:
                # add embedding features output
                emb_feats.append(route)

            if i < self.num_blocks - 1:
                route = self.fpn_routes[i](route)
                route = F.interpolate(route, scale_factor=2., mode='nearest')

        pan_feats = [fpn_feats[-1], ]
        route = fpn_feats[self.num_blocks - 1]
        for i in reversed(range(self.num_blocks - 1)):
            block = fpn_feats[i]
            route = self.pan_routes[i](route)
            if self.data_format == 'NCHW':
                block = torch.cat([route, block], 1)
            else:
                block = torch.cat([route, block], -1)

            route, tip = self.pan_blocks[i](block)
            pan_feats.append(tip)

        # Debug. calc ddd with ncnn output.
        # import numpy as np
        # y2 = pan_feats[0].cpu().detach().numpy()
        # ncnn_output = 'D://GitHub/ncnn2/build/examples/output.txt'
        # with open(ncnn_output, 'r', encoding='utf-8') as f:
        #     for line in f:
        #         line = line.strip()
        # line = line[:-1]
        # ss = line.split(',')
        # y = []
        # for s in ss:
        #     y.append(float(s))
        # y = np.array(y).astype(np.float32)
        # y = np.reshape(y, y2.shape)
        # print(y2.shape)
        # ddd = np.sum((y - y2) ** 2)
        # print('ddd=%.9f' % ddd)
        # ddd2 = np.mean((y - y2) ** 2)
        # print('ddd=%.9f' % ddd2)
        if for_mot:
            return {'yolo_feats': pan_feats[::-1], 'emb_feats': emb_feats}
        else:
            return pan_feats[::-1]

    def export_ncnn(self, ncnn_data, bottom_names):
        blocks = bottom_names[::-1]
        fpn_feats = []

        for i, block in enumerate(blocks):
            block = [block, ]
            if i > 0:
                if self.data_format == 'NCHW':
                    block = ncnn_utils.concat(ncnn_data, route + block, dim=1)
                else:
                    block = ncnn_utils.concat(ncnn_data, route + block, dim=3)
            route, tip = self.fpn_blocks[i].export_ncnn(ncnn_data, block)
            fpn_feats.append(tip[0])

            if i < self.num_blocks - 1:
                route = self.fpn_routes[i].export_ncnn(ncnn_data, route)
                route = ncnn_utils.interpolate(ncnn_data, route, scale_factor=2., mode='nearest')

        pan_feats = [fpn_feats[-1], ]
        route = fpn_feats[self.num_blocks - 1]
        route = [route, ]
        for i in reversed(range(self.num_blocks - 1)):
            block = fpn_feats[i]
            block = [block, ]
            route = self.pan_routes[i].export_ncnn(ncnn_data, route)
            if self.data_format == 'NCHW':
                block = ncnn_utils.concat(ncnn_data, route + block, dim=1)
            else:
                block = ncnn_utils.concat(ncnn_data, route + block, dim=3)

            route, tip = self.pan_blocks[i].export_ncnn(ncnn_data, block)
            pan_feats.append(tip[0])
        return pan_feats[::-1]



