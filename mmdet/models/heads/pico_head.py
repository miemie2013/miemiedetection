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

import math
import copy
import numpy as np
import torch
from torch import distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ..initializer import normal_, constant_
from ..assigners.utils import generate_anchors_for_grid_cell
from ..bbox_utils import bbox_center, batch_distance2bbox, bbox2distance
from mmdet.models.custom_layers import ConvNormLayer
from .gfl_head import Integral, GFLHead
from mmdet.models.necks.csp_pan import DPModule
from mmdet.models.matrix_nms import matrix_nms
from mmdet.utils import my_multiclass_nms, get_world_size
from ..ops import gather_1d

eps = 1e-9


class PicoSE(nn.Module):
    def __init__(self, feat_channels):
        super(PicoSE, self).__init__()
        self.fc = nn.Conv2d(feat_channels, feat_channels, 1)
        self.conv = ConvNormLayer(feat_channels, feat_channels, 1, 1)

        self._init_weights()

    def _init_weights(self):
        normal_(self.fc.weight, std=0.001)

    def forward(self, feat, avg_feat):
        weight = torch.sigmoid(self.fc(avg_feat))
        out = self.conv(feat * weight)
        return out


class PicoFeat(nn.Module):
    """
    PicoFeat of PicoDet

    Args:
        feat_in (int): The channel number of input Tensor.
        feat_out (int): The channel number of output Tensor.
        num_convs (int): The convolution number of the LiteGFLFeat.
        norm_type (str): Normalization type, 'bn'/'sync_bn'/'gn'.
        share_cls_reg (bool): Whether to share the cls and reg output.
        act (str): The act of per layers.
        use_se (bool): Whether to use se module.
    """

    def __init__(self,
                 feat_in=256,
                 feat_out=96,
                 num_fpn_stride=3,
                 num_convs=2,
                 norm_type='bn',
                 share_cls_reg=False,
                 act='hard_swish',
                 use_se=False):
        super(PicoFeat, self).__init__()
        self.num_convs = num_convs
        self.norm_type = norm_type
        self.share_cls_reg = share_cls_reg
        self.act = act
        self.use_se = use_se
        self.cls_convs = []
        self.reg_convs = []
        if use_se:
            assert share_cls_reg == True, \
                'In the case of using se, share_cls_reg must be set to True'
            self.se = nn.ModuleList()
        for stage_idx in range(num_fpn_stride):
            cls_subnet_convs = []
            reg_subnet_convs = []
            for i in range(self.num_convs):
                in_c = feat_in if i == 0 else feat_out
                name = 'cls_conv_dw{}_{}'.format(stage_idx, i)
                cls_conv_dw = ConvNormLayer(
                        ch_in=in_c,
                        ch_out=feat_out,
                        filter_size=5,
                        stride=1,
                        groups=feat_out,
                        norm_type=norm_type,
                        bias_on=False,
                        lr_scale=2.)
                self.add_module(name, cls_conv_dw)
                cls_subnet_convs.append(cls_conv_dw)

                name = 'cls_conv_pw{}_{}'.format(stage_idx, i)
                cls_conv_pw = ConvNormLayer(
                        ch_in=in_c,
                        ch_out=feat_out,
                        filter_size=1,
                        stride=1,
                        norm_type=norm_type,
                        bias_on=False,
                        lr_scale=2.)
                self.add_module(name, cls_conv_pw)
                cls_subnet_convs.append(cls_conv_pw)

                if not self.share_cls_reg:
                    name = 'reg_conv_dw{}_{}'.format(stage_idx, i)
                    reg_conv_dw = ConvNormLayer(
                            ch_in=in_c,
                            ch_out=feat_out,
                            filter_size=5,
                            stride=1,
                            groups=feat_out,
                            norm_type=norm_type,
                            bias_on=False,
                            lr_scale=2.)
                    self.add_module(name, reg_conv_dw)
                    reg_subnet_convs.append(reg_conv_dw)

                    name = 'reg_conv_pw{}_{}'.format(stage_idx, i)
                    reg_conv_pw = ConvNormLayer(
                            ch_in=in_c,
                            ch_out=feat_out,
                            filter_size=1,
                            stride=1,
                            norm_type=norm_type,
                            bias_on=False,
                            lr_scale=2.)
                    self.add_module(name, reg_conv_pw)
                    reg_subnet_convs.append(reg_conv_pw)
            self.cls_convs.append(cls_subnet_convs)
            self.reg_convs.append(reg_subnet_convs)
            if use_se:
                self.se.append(PicoSE(feat_out))

    def act_func(self, x):
        if self.act == "leaky_relu":
            x = F.leaky_relu(x)
        elif self.act == "hard_swish":
            x = F.hardswish(x)
        elif self.act == "relu6":
            x = F.relu6(x)
        return x

    def forward(self, fpn_feat, stage_idx):
        assert stage_idx < len(self.cls_convs)
        cls_feat = fpn_feat
        reg_feat = fpn_feat
        for i in range(len(self.cls_convs[stage_idx])):
            cls_feat = self.act_func(self.cls_convs[stage_idx][i](cls_feat))
            reg_feat = cls_feat
            if not self.share_cls_reg:
                reg_feat = self.act_func(self.reg_convs[stage_idx][i](reg_feat))
        if self.use_se:
            avg_feat = F.adaptive_avg_pool2d(cls_feat, (1, 1))
            se_feat = self.act_func(self.se[stage_idx](cls_feat, avg_feat))
            return cls_feat, se_feat
        return cls_feat, reg_feat


class PicoHeadV2(GFLHead):
    """
    PicoHeadV2
    Args:
        conv_feat (object): Instance of 'PicoFeat'
        num_classes (int): Number of classes
        fpn_stride (list): The stride of each FPN Layer
        prior_prob (float): Used to set the bias init for the class prediction layer
        loss_class (object): Instance of VariFocalLoss.
        loss_dfl (object): Instance of DistributionFocalLoss.
        loss_bbox (object): Instance of bbox loss.
        assigner (object): Instance of label assigner.
        reg_max: Max value of integral set :math: `{0, ..., reg_max}`
                n QFL setting. Default: 7.
    """
    __inject__ = [
        'conv_feat', 'dgqp_module', 'loss_class', 'loss_dfl', 'loss_bbox',
        'static_assigner', 'assigner', 'nms'
    ]
    __shared__ = ['num_classes', 'eval_size']

    def __init__(self,
                 conv_feat='PicoFeatV2',
                 dgqp_module=None,
                 num_classes=80,
                 fpn_stride=[8, 16, 32],
                 prior_prob=0.01,
                 use_align_head=True,
                 loss_class='VariFocalLoss',
                 loss_dfl='DistributionFocalLoss',
                 loss_bbox='GIoULoss',
                 static_assigner_epoch=60,
                 static_assigner='ATSSAssigner',
                 assigner='TaskAlignedAssigner',
                 reg_max=16,
                 feat_in_chan=96,
                 nms=None,
                 nms_pre=1000,
                 cell_offset=0,
                 act='hard_swish',
                 grid_cell_scale=5.0,
                 eval_size=None,
                 nms_cfg=None):
        super(PicoHeadV2, self).__init__(
            conv_feat=conv_feat,
            dgqp_module=dgqp_module,
            num_classes=num_classes,
            fpn_stride=fpn_stride,
            prior_prob=prior_prob,
            loss_class=loss_class,
            loss_dfl=loss_dfl,
            loss_bbox=loss_bbox,
            reg_max=reg_max,
            feat_in_chan=feat_in_chan,
            nms=nms,
            nms_pre=nms_pre,
            cell_offset=cell_offset, )
        self.conv_feat = conv_feat
        self.num_classes = num_classes
        self.fpn_stride = fpn_stride
        self.prior_prob = prior_prob
        self.loss_vfl = loss_class
        self.loss_dfl = loss_dfl
        self.loss_bbox = loss_bbox

        self.static_assigner_epoch = static_assigner_epoch
        self.static_assigner = static_assigner
        self.assigner = assigner

        self.reg_max = reg_max
        self.feat_in_chan = feat_in_chan
        self.nms = nms
        self.nms_pre = nms_pre
        self.cell_offset = cell_offset
        self.act = act
        self.grid_cell_scale = grid_cell_scale
        self.use_align_head = use_align_head
        self.cls_out_channels = self.num_classes
        self.eval_size = eval_size
        self.nms_cfg = nms_cfg

        bias_init_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        # Clear the super class initialization
        self.gfl_head_cls = None
        self.gfl_head_reg = None
        self.scales_regs = None

        self.head_cls_list = nn.ModuleList()
        self.head_reg_list = nn.ModuleList()
        self.cls_align = nn.ModuleList()

        for i in range(len(fpn_stride)):
            name = "head_cls" + str(i)
            head_cls = nn.Conv2d(
                    in_channels=self.feat_in_chan,
                    out_channels=self.cls_out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0)
            normal_(head_cls.weight, mean=0., std=0.01)
            constant_(head_cls.bias, value=bias_init_value)
            self.add_module(name, head_cls)
            self.head_cls_list.append(head_cls)

            name = "head_reg" + str(i)
            head_reg = nn.Conv2d(
                    in_channels=self.feat_in_chan,
                    out_channels=4 * (self.reg_max + 1),
                    kernel_size=1,
                    stride=1,
                    padding=0)
            normal_(head_reg.weight, mean=0., std=0.01)
            constant_(head_reg.bias, value=0.)
            self.add_module(name, head_reg)
            self.head_reg_list.append(head_reg)
            if self.use_align_head:
                self.cls_align.append(
                    DPModule(
                        self.feat_in_chan,
                        1,
                        5,
                        act=self.act,
                        use_act_in_out=False))

        # initialize the anchor points
        if self.eval_size:
            self.anchor_points, self.stride_tensor = self._generate_anchors()

    def forward(self, fpn_feats, export_post_process=True):
        assert len(fpn_feats) == len(self.fpn_stride), "The size of fpn_feats is not equal to size of fpn_stride"

        if self.training:
            return self.forward_train(fpn_feats)
        else:
            return self.forward_eval(fpn_feats, export_post_process=export_post_process)

    def forward_train(self, fpn_feats):
        cls_score_list, reg_list, box_list = [], [], []
        # fpn_feats里有4个张量，形状分别是[N, C, 52, 52], [N, C, 26, 26], [N, C, 13, 13], [N, C, 7, 7]
        # self.fpn_stride == [8, 16, 32, 64]
        for i, (fpn_feat, stride) in enumerate(zip(fpn_feats, self.fpn_stride)):
            b, _, h, w = fpn_feat.shape
            # task decomposition
            # conv_cls_feat.shape == [N, C, 52, 52]
            # se_feat.shape ==       [N, C, 52, 52]
            conv_cls_feat, se_feat = self.conv_feat(fpn_feat, i)
            cls_logit = self.head_cls_list[i](se_feat)   # shape == [N,       num_classes, 52, 52]
            reg_pred = self.head_reg_list[i](se_feat)    # shape == [N, 4 * (reg_max + 1), 52, 52]

            # cls prediction and alignment
            if self.use_align_head:
                cls_prob = torch.sigmoid(self.cls_align[i](conv_cls_feat))
                cls_score = (torch.sigmoid(cls_logit) * cls_prob + eps).sqrt()   # [N, num_classes, 52, 52],  分数
            else:
                cls_score = torch.sigmoid(cls_logit)   # [N, num_classes, 52, 52],  分数

            bbox_pred = reg_pred.permute([0, 2, 3, 1])     # [N, 52, 52, 4 * (reg_max + 1)]
            b, _, cell_h, cell_w = cls_score.shape
            y, x = self.get_single_level_center_point(
                [cell_h, cell_w], stride, cell_offset=self.cell_offset)
            # center_points.shape == [52*52, 2], value == [[0.5*stride, 0.5*stride], [1.5*stride, 0.5*stride], [2.5*stride, 0.5*stride],
            # ..., [49.5*stride, 51.5*stride], [50.5*stride, 51.5*stride], [51.5*stride, 51.5*stride]]  是格子中心点坐标（单位是像素）
            center_points = torch.stack([x, y], dim=-1)
            bbox_pred = self.distribution_project(bbox_pred) * stride  # [N*52*52, 4],   是预测的bbox，ltrb格式(均是正值，乘了stride，单位是像素)
            bbox_pred = bbox_pred.reshape([b, cell_h * cell_w, 4])    # [N, 52*52, 4],   是预测的bbox，ltrb格式(均是正值，单位是像素)
            # center_points.shape == [52*52, 2], value == [[0.5*stride, 0.5*stride], [1.5*stride, 0.5*stride], [2.5*stride, 0.5*stride],
            # ..., [49.5*stride, 51.5*stride], [50.5*stride, 51.5*stride], [51.5*stride, 51.5*stride]]  是格子中心点坐标（单位是像素）
            center_points = center_points.to(bbox_pred.device)
            bbox_pred = batch_distance2bbox(center_points, bbox_pred)   # [N, 52*52, 4],  像FCOS那样，将 ltrb 解码成 预测框左上角坐标、右下角坐标；单位是像素
            cls_score_list.append(cls_score.flatten(2).permute([0, 2, 1]))   # [N, 52*52, num_classes],  分数
            reg_list.append(reg_pred.flatten(2).permute([0, 2, 1]))   # [N, 52*52, 4 * (reg_max + 1)],  回归
            box_list.append(bbox_pred / stride)   # [N, 52*52, 4],  预测框左上角坐标、右下角坐标；单位是格子边长

        cls_score_list = torch.cat(cls_score_list, 1)   # [N, A, num_classes],  分数
        box_list = torch.cat(box_list, 1)   # [N, A, 4],  预测框左上角坐标、右下角坐标；单位是格子边长
        reg_list = torch.cat(reg_list, 1)   # [N, A, 4 * (reg_max + 1)],  回归
        return cls_score_list, reg_list, box_list, fpn_feats

    def forward_eval(self, fpn_feats, export_post_process=True):
        if self.eval_size:
            anchor_points, stride_tensor = self.anchor_points, self.stride_tensor
        else:
            anchor_points, stride_tensor = self._generate_anchors(fpn_feats)
        anchor_points = anchor_points.to(fpn_feats[0].device)
        stride_tensor = stride_tensor.to(fpn_feats[0].device)
        cls_score_list, box_list = [], []
        # anchor_points.shape == [A, 2], value == [[0.5, 0.5], [1.5, 0.5], [2.5, 0.5], ..., [4.5, 6.5], [5.5, 6.5], [6.5, 6.5]]  是格子中心点坐标（单位是格子边长）
        # stride_tensor.shape == [A, 1], value == [[8.], [8.], [8.], ..., [64.], [64.], [64.]]   是格子边长（单位是像素）
        # fpn_feats里有4个张量，形状分别是[N, C, 52, 52], [N, C, 26, 26], [N, C, 13, 13], [N, C, 7, 7]
        # self.fpn_stride == [8, 16, 32, 64]
        for i, (fpn_feat, stride) in enumerate(zip(fpn_feats, self.fpn_stride)):
            _, _, h, w = fpn_feat.shape
            # task decomposition
            # conv_cls_feat.shape == [N, C, 52, 52]
            # se_feat.shape ==       [N, C, 52, 52]
            conv_cls_feat, se_feat = self.conv_feat(fpn_feat, i)
            cls_logit = self.head_cls_list[i](se_feat)   # shape == [N,       num_classes, 52, 52]
            reg_pred = self.head_reg_list[i](se_feat)    # shape == [N, 4 * (reg_max + 1), 52, 52]

            # cls prediction and alignment
            if self.use_align_head:
                cls_prob = torch.sigmoid(self.cls_align[i](conv_cls_feat))
                cls_score = (torch.sigmoid(cls_logit) * cls_prob + eps).sqrt()
            else:
                cls_score = torch.sigmoid(cls_logit)

            if not export_post_process:
                # Now only supports batch size = 1 in deploy
                cls_score_list.append(
                    cls_score.reshape([1, self.cls_out_channels, -1]).permute(
                        [0, 2, 1]))
                box_list.append(
                    reg_pred.reshape([1, (self.reg_max + 1) * 4, -1]).permute(
                        [0, 2, 1]))
            else:
                l = h * w
                cls_score_out = cls_score.reshape([-1, self.cls_out_channels, l])   # [N, num_classes, 52*52]
                bbox_pred = reg_pred.permute([0, 2, 3, 1])     # [N, 52, 52, 4 * (reg_max + 1)]
                bbox_pred = self.distribution_project(bbox_pred)  # [N*52*52*4, ],   是预测的bbox，ltrb格式(均是正值且单位是格子边长)
                bbox_pred = bbox_pred.reshape([-1, l, 4])         # [N, 52*52, 4],   是预测的bbox，ltrb格式(均是正值且单位是格子边长)
                cls_score_list.append(cls_score_out)
                box_list.append(bbox_pred)

        if export_post_process:
            cls_score_list = torch.cat(cls_score_list, -1)  # [N, 80, A]
            box_list = torch.cat(box_list, 1)               # [N,  A, 4],   是预测的bbox，ltrb格式(均是正值且单位是格子边长)
            box_list = batch_distance2bbox(anchor_points, box_list)   # [N,  A, 4],  像FCOS那样，将 ltrb 解码成 预测框左上角坐标、右下角坐标
            box_list *= stride_tensor        # [N,  A, 4]   预测框左上角坐标、右下角坐标；乘以格子边长，单位是像素

        return cls_score_list, box_list

    def get_loss(self, head_outs, gt_meta):
        '''
        pred_scores    [N, A, num_classes],  分数
        pred_regs      [N, A, 4 * (reg_max + 1)],  回归
        pred_bboxes    [N, A, 4],  预测框左上角坐标、右下角坐标；单位是格子边长
        fpn_feats      fpn输出的特征图
        '''
        pred_scores, pred_regs, pred_bboxes, fpn_feats = head_outs
        gt_labels = gt_meta['gt_class']   # [N, 200, 1]
        gt_bboxes = gt_meta['gt_bbox']    # [N, 200, 4]  每个gt的左上角坐标、右下角坐标；单位是像素
        gt_scores = gt_meta['gt_score'] if 'gt_score' in gt_meta else None
        num_imgs = gt_meta['gt_class'].shape[0]
        pad_gt_mask = gt_meta['pad_gt_mask']  # [N, 200, 1]

        # anchors              [A, 4]  先验框左上角坐标、右下角坐标；单位是像素
        # _                    [A, 2]  格子中心点坐标（单位是像素）
        # num_anchors_list     value = [52*52, 26*26, 13*13, 7*7]，每个特征图的格子数
        # stride_tensor_list   [A, 1]  格子边长
        anchors, _, num_anchors_list, stride_tensor_list = generate_anchors_for_grid_cell(
            fpn_feats, self.fpn_stride, self.grid_cell_scale, self.cell_offset)

        # centers              [A, 2]  先验框中心点坐标（单位是像素）
        centers = bbox_center(anchors)

        # label assignment
        if gt_meta['epoch_id'] < self.static_assigner_epoch:
            assigned_labels, assigned_bboxes, assigned_scores, _ = self.static_assigner(
                anchors,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index=self.num_classes,
                gt_scores=gt_scores,
                pred_bboxes=pred_bboxes.detach() * stride_tensor_list)

        else:
            assigned_labels, assigned_bboxes, assigned_scores, _ = self.assigner(
                pred_scores.detach(),
                pred_bboxes.detach() * stride_tensor_list,
                centers,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index=self.num_classes,
                gt_scores=gt_scores)

        assigned_bboxes /= stride_tensor_list

        centers_shape = centers.shape
        flatten_centers = centers.expand(
            [num_imgs, centers_shape[0], centers_shape[1]]).reshape([-1, 2])
        flatten_strides = stride_tensor_list.expand(
            [num_imgs, centers_shape[0], 1]).reshape([-1, 1])
        flatten_cls_preds = pred_scores.reshape([-1, self.num_classes])
        flatten_regs = pred_regs.reshape([-1, 4 * (self.reg_max + 1)])
        flatten_bboxes = pred_bboxes.reshape([-1, 4])
        flatten_bbox_targets = assigned_bboxes.reshape([-1, 4])
        flatten_labels = assigned_labels.reshape([-1])
        flatten_assigned_scores = assigned_scores.reshape([-1, self.num_classes])

        pos_inds = torch.nonzero(
            torch.logical_and((flatten_labels >= 0),
                               (flatten_labels < self.num_classes)),
            as_tuple=False).squeeze(1)

        num_total_pos = len(pos_inds)

        if num_total_pos > 0:
            pos_bbox_targets = gather_1d(flatten_bbox_targets, pos_inds)
            pos_decode_bbox_pred = gather_1d(flatten_bboxes, pos_inds)
            pos_reg = gather_1d(flatten_regs, pos_inds)
            pos_strides = gather_1d(flatten_strides, pos_inds)
            pos_centers = gather_1d(flatten_centers, pos_inds) / pos_strides

            weight_targets = flatten_assigned_scores.detach()
            weight_targets, _ = weight_targets.max(axis=1, keepdim=True)
            weight_targets = gather_1d(weight_targets, pos_inds)

            pred_corners = pos_reg.reshape([-1, self.reg_max + 1])
            target_corners = bbox2distance(pos_centers, pos_bbox_targets, self.reg_max).reshape([-1])
            # regression loss
            loss_bbox = torch.sum(self.loss_bbox(pos_decode_bbox_pred, pos_bbox_targets) * weight_targets)

            # dfl loss
            loss_dfl = self.loss_dfl(
                pred_corners,
                target_corners,
                weight=weight_targets.expand([-1, 4]).reshape([-1]),
                avg_factor=4.0)
        else:
            loss_bbox = torch.zeros([1])
            loss_dfl = torch.zeros([1])

        avg_factor = flatten_assigned_scores.sum()
        world_size = get_world_size()
        if world_size > 1:
            dist.all_reduce(avg_factor, op=dist.ReduceOp.SUM)
            avg_factor = avg_factor / world_size
        avg_factor = torch.clamp(avg_factor, min=1.)  # y = max(x, 1)
        loss_vfl = self.loss_vfl(flatten_cls_preds, flatten_assigned_scores, avg_factor=avg_factor)

        loss_bbox = loss_bbox / avg_factor
        loss_dfl = loss_dfl / avg_factor

        total_loss = loss_vfl + loss_bbox + loss_dfl
        loss_states = dict(total_loss=total_loss, loss_vfl=loss_vfl, loss_bbox=loss_bbox, loss_dfl=loss_dfl)

        return loss_states

    def _generate_anchors(self, feats=None):
        # just use in eval time
        anchor_points = []
        stride_tensor = []
        for i, stride in enumerate(self.fpn_stride):
            if feats is not None:
                _, _, h, w = feats[i].shape
            else:
                h = math.ceil(self.eval_size[0] / stride)
                w = math.ceil(self.eval_size[1] / stride)
            shift_x = torch.arange(end=w) + self.cell_offset
            shift_y = torch.arange(end=h) + self.cell_offset
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            anchor_point = torch.stack([shift_x, shift_y], dim=-1).float()
            anchor_points.append(anchor_point.reshape([-1, 2]))
            stride_tensor.append(torch.full([h * w, 1], stride, dtype=torch.float32))
        anchor_points = torch.cat(anchor_points)
        stride_tensor = torch.cat(stride_tensor)
        return anchor_points, stride_tensor

    def post_process(self, head_outs, scale_factor, export_nms=True):
        pred_scores, pred_bboxes = head_outs
        # pred_scores [N, 80, A]   0到1之间的值
        # pred_bboxes [N,  A, 4]   预测框左上角坐标、右下角坐标；单位是像素
        if not export_nms:
            return pred_bboxes, pred_scores
        else:
            # rescale: [h_scale, w_scale] -> [w_scale, h_scale, w_scale, h_scale]
            # torch的split和paddle有点不同，torch的第二个参数表示的是每一份的大小，paddle的第二个参数表示的是分成几份。
            scale_y, scale_x = torch.split(scale_factor, 1, dim=-1)
            scale_factor = torch.cat([scale_x, scale_y, scale_x, scale_y], dim=-1).reshape([-1, 1, 4])
            # scale bbox to origin image size.
            pred_bboxes /= scale_factor   # [N, A, 4]     pred_scores.shape = [N, 80, A]

            # nms
            preds = []
            nms_cfg = copy.deepcopy(self.nms_cfg)
            nms_type = nms_cfg.pop('nms_type')
            batch_size = pred_bboxes.shape[0]
            yolo_scores = pred_scores.permute((0, 2, 1))  #  [N, A, 80]
            if nms_type == 'matrix_nms':
                for i in range(batch_size):
                    pred = matrix_nms(pred_bboxes[i, :, :], yolo_scores[i, :, :], **nms_cfg)
                    preds.append(pred)
            elif nms_type == 'multiclass_nms':
                preds = my_multiclass_nms(pred_bboxes, yolo_scores, **nms_cfg)
            return preds
