#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-08-21 19:33:37
#   Description : pytorch_fcos
#
# ================================================================
import torch
import copy

import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.custom_layers import ConvNormLayer


class FPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channel,
                 spatial_scales=[0.25, 0.125, 0.0625, 0.03125],
                 has_extra_convs=False,
                 extra_stage=1,
                 use_c5=True,
                 norm_type=None,
                 norm_decay=0.,
                 freeze_norm=False,
                 relu_before_extra_convs=True):
        super(FPN, self).__init__()
        self.out_channel = out_channel
        for s in range(extra_stage):
            spatial_scales = spatial_scales + [spatial_scales[-1] / 2.]
        self.spatial_scales = spatial_scales
        self.has_extra_convs = has_extra_convs
        self.extra_stage = extra_stage
        self.use_c5 = use_c5
        self.relu_before_extra_convs = relu_before_extra_convs
        self.norm_type = norm_type
        self.norm_decay = norm_decay
        self.freeze_norm = freeze_norm

        self.lateral_convs = []
        self.fpn_convs = []
        fan = out_channel * 3 * 3

        # stage index 0,1,2,3 stands for res2,res3,res4,res5 on ResNet Backbone
        # 0 <= st_stage < ed_stage <= 3
        st_stage = 4 - len(in_channels)
        ed_stage = st_stage + len(in_channels) - 1
        for i in range(st_stage, ed_stage + 1):
            if i == 3:
                lateral_name = 'fpn_inner_res5_sum'
            else:
                lateral_name = 'fpn_inner_res{}_sum_lateral'.format(i + 2)
            in_c = in_channels[i - st_stage]
            if self.norm_type is not None:
                lateral = ConvNormLayer(
                        ch_in=in_c,
                        ch_out=out_channel,
                        filter_size=1,
                        stride=1,
                        norm_type=self.norm_type,
                        norm_decay=self.norm_decay,
                        freeze_norm=self.freeze_norm,
                        initializer=XavierUniform(fan_out=in_c))
                self.add_module(lateral_name, lateral)
            else:
                lateral = nn.Conv2d(
                        in_channels=in_c,
                        out_channels=out_channel,
                        kernel_size=1,
                        weight_attr=ParamAttr(
                            initializer=XavierUniform(fan_out=in_c)))
                self.add_module(lateral_name, lateral)
            self.lateral_convs.append(lateral)

            fpn_name = 'fpn_res{}_sum'.format(i + 2)
            if self.norm_type is not None:
                fpn_conv = ConvNormLayer(
                        ch_in=out_channel,
                        ch_out=out_channel,
                        filter_size=3,
                        stride=1,
                        norm_type=self.norm_type,
                        norm_decay=self.norm_decay,
                        freeze_norm=self.freeze_norm,
                        initializer=XavierUniform(fan_out=fan))
                self.add_module(fpn_name, fpn_conv)
            else:
                fpn_conv = nn.Conv2d(
                        in_channels=out_channel,
                        out_channels=out_channel,
                        kernel_size=3,
                        padding=1,
                        weight_attr=ParamAttr(
                            initializer=XavierUniform(fan_out=fan)))
                self.add_module(fpn_name, fpn_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv levels for RetinaNet(use_c5)/FCOS(use_p5)
        if self.has_extra_convs:
            for i in range(self.extra_stage):
                lvl = ed_stage + 1 + i
                if i == 0 and self.use_c5:
                    in_c = in_channels[-1]
                else:
                    in_c = out_channel
                extra_fpn_name = 'fpn_{}'.format(lvl + 2)
                if self.norm_type is not None:
                    extra_fpn_conv = ConvNormLayer(
                            ch_in=in_c,
                            ch_out=out_channel,
                            filter_size=3,
                            stride=2,
                            norm_type=self.norm_type,
                            norm_decay=self.norm_decay,
                            freeze_norm=self.freeze_norm,
                            initializer=XavierUniform(fan_out=fan))
                    self.add_module(extra_fpn_name, extra_fpn_conv)
                else:
                    extra_fpn_conv = nn.Conv2d(
                            in_channels=in_c,
                            out_channels=out_channel,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            weight_attr=ParamAttr(
                                initializer=XavierUniform(fan_out=fan)))
                    self.add_module(extra_fpn_name, extra_fpn_conv)
                self.fpn_convs.append(extra_fpn_conv)

    def add_param_group(self, param_groups, base_lr, base_wd, need_clip, clip_norm):
        for i in range(0, self.num_backbone_stages):
            self.fpn_inner_convs[i].add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
            self.fpn_convs[i].add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        # 生成其它尺度的特征图时如果用的是卷积层
        highest_backbone_level = self.min_level + len(self.spatial_scale) - 1
        if self.has_extra_convs and self.max_level > highest_backbone_level:
            j = 0
            for i in range(highest_backbone_level + 1, self.max_level + 1):
                self.extra_convs[j].add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
                j += 1

    def forward(self, body_feats):
        '''
        一个示例
        :param body_feats:  [s8, s16, s32]
        :return:
                                     bs32
                                      |
                                     卷积
                                      |
                             bs16   [fs32]
                              |       |
                            卷积    上采样
                              |       |
                          lateral   topdown
                               \    /
                                相加
                                  |
                        bs8     [fs16]
                         |        |
                        卷积    上采样
                         |        |
                      lateral   topdown
                            \    /
                             相加
                               |
                             [fs8]

                fpn_inner_output = [fs32, fs16, fs8]
        然后  fs32, fs16, fs8  分别再接一个卷积得到 p5, p4, p3 ；
        p5 接一个卷积得到 p6， p6 接一个卷积得到 p7。
        '''
        laterals = []
        num_levels = len(body_feats)
        for i in range(num_levels):
            laterals.append(self.lateral_convs[i](body_feats[i]))

        for i in range(1, num_levels):
            lvl = num_levels - i
            upsample = F.interpolate(
                laterals[lvl],
                scale_factor=2.,
                mode='nearest', )
            laterals[lvl - 1] += upsample

        fpn_output = []
        for lvl in range(num_levels):
            fpn_output.append(self.fpn_convs[lvl](laterals[lvl]))

        if self.extra_stage > 0:
            # use max pool to get more levels on top of outputs (Faster R-CNN, Mask R-CNN)
            if not self.has_extra_convs:
                assert self.extra_stage == 1, 'extra_stage should be 1 if FPN has not extra convs'
                fpn_output.append(F.max_pool2d(fpn_output[-1], 1, stride=2))
            # add extra conv levels for RetinaNet(use_c5)/FCOS(use_p5)
            else:
                if self.use_c5:
                    extra_source = body_feats[-1]
                else:
                    extra_source = fpn_output[-1]
                fpn_output.append(self.fpn_convs[num_levels](extra_source))

                for i in range(1, self.extra_stage):
                    if self.relu_before_extra_convs:
                        fpn_output.append(self.fpn_convs[num_levels + i](F.relu(
                            fpn_output[-1])))
                    else:
                        fpn_output.append(self.fpn_convs[num_levels + i](
                            fpn_output[-1]))
        return fpn_output
