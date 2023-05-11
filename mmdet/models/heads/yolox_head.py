#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import math
from loguru import logger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.losses.losses import IOUloss
from mmdet.models.network_blocks import BaseConv, DWConv
from mmdet.utils import bboxes_iou, cxcywh2xyxy, meshgrid, visualize_assign, yolox_batch_bboxes_iou
from mmdet.utils.boxes import bboxes_iou_batch


class YOLOXHead(nn.Module):
    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        self.use_l1 = False
        self.parallel_matching = True
        self.use_batch_assign = True
        self.use_batch_assign = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        grids = []
        strides = []

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid(output, k, stride_this_level, xin[0].type())
                grids.append(grid)
                strides.append(
                    torch.zeros(1, grid.shape[1]).fill_(stride_this_level).type_as(xin[0])
                )
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(batch_size, 1, 4, hsize, wsize)
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(batch_size, -1, 4)
                    origin_preds.append(reg_output.clone())
            else:
                output = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)

            outputs.append(output)

        if self.training:
            return self.get_losses(
                grids,
                strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
            )
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.permute(0, 2, 3, 1).reshape(batch_size, hsize * wsize, -1)
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs = torch.cat([
            (outputs[..., 0:2] + grids) * strides,
            torch.exp(outputs[..., 2:4]) * strides,
            outputs[..., 4:]
        ], dim=-1)
        return outputs

    def get_losses(
        self,
        grids,
        strides,
        labels,
        outputs,
        origin_preds,
    ):
        '''
        grids        格子左上角xy坐标，单位是格子边长
        strides      格子边长，单位是像素
        labels       [N, 120, 5]   gt的cid、cxcywh, 单位是像素
        outputs      [N, A, 5+n_cls]
        '''
        bbox_preds = outputs[:, :, :4]  # [N, A, 4]   预测的cxcywh, 单位是像素
        obj_preds = outputs[:, :, 4:5]  # [N, A, 1]       未经过sigmoid激活
        cls_preds = outputs[:, :, 5:]   # [N, A, n_cls]   未经过sigmoid激活

        # calculate targets
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # [N, ]  每张图片gt数

        A = outputs.shape[1]
        grids = torch.cat(grids, 1)      # [1, A, 2]  格子左上角xy坐标，单位是格子边长
        strides = torch.cat(strides, 1)  # [1, A]  格子边长，单位是像素
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        '''
        fg_masks      [N*A, ]          前景的mask, bool类型
        obj_targets   [N*A, 1]         前景的mask, float类型
        reg_targets   [num_fg, 4]      前景学习的gt的cxcywh, 单位是像素
        cls_targets   [num_fg, n_cls]  前景学习的one_hot向量，学习的类别处 不是1 而是 与匹配到的gt的iou
        l1_targets    [num_fg, 4]      前景学习的gt的cxcywh, 是未解码的cxcywh
        '''
        if self.use_batch_assign:
            num_gts, num_fg, fg_masks, obj_targets, reg_targets, cls_targets, l1_targets = \
                self.batch_assign(bbox_preds, obj_preds, cls_preds, labels, nlabel, grids, strides)
        else:
            num_gts, num_fg, fg_masks, obj_targets, reg_targets, cls_targets, l1_targets = \
                self.single_assign(bbox_preds, obj_preds, cls_preds, labels, nlabel, A, grids, strides)

        # num_gts, num_fg, fg_masks, obj_targets, reg_targets, cls_targets, l1_targets = \
        #     self.single_assign(bbox_preds, obj_preds, cls_preds, labels, nlabel, A, grids, strides)
        # num_gts2, num_fg2, fg_masks2, obj_targets2, reg_targets2, cls_targets2, l1_targets2 = \
        #     self.batch_assign(bbox_preds, obj_preds, cls_preds, labels, nlabel, grids, strides)
        # debug = False
        # debug = True
        # if debug:
        #     ddd = abs(num_gts - num_gts2)
        #     assert ddd < 0.0001
        #     ddd = abs(num_fg - num_fg2)
        #     assert ddd < 0.0001
        #     aaaaaaa1 = obj_targets.cpu().detach().numpy()
        #     aaaaaaa2 = obj_targets2.cpu().detach().numpy()
        #     ddd = np.sum((aaaaaaa1 - aaaaaaa2) ** 2)
        #     assert ddd < 0.0001
        #     aaaaaaa1 = reg_targets.cpu().detach().numpy()
        #     aaaaaaa2 = reg_targets2.cpu().detach().numpy()
        #     ddd = np.sum((aaaaaaa1 - aaaaaaa2) ** 2)
        #     assert ddd < 0.0001
        #     aaaaaaa1 = cls_targets.cpu().detach().numpy()
        #     aaaaaaa2 = cls_targets2.cpu().detach().numpy()
        #     ddd = np.sum((aaaaaaa1 - aaaaaaa2) ** 2)
        #     assert ddd < 0.0001
        #     if self.use_l1:
        #         aaaaaaa1 = l1_targets.cpu().detach().numpy()
        #         aaaaaaa2 = l1_targets2.cpu().detach().numpy()
        #         ddd = np.sum((aaaaaaa1 - aaaaaaa2) ** 2)
        #         assert ddd < 0.0001


        num_fg = max(num_fg, 1)
        loss_iou = (
            self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        ).sum() / num_fg
        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
        ).sum() / num_fg
        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            )
        ).sum() / num_fg
        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
            ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
        )

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    def single_assign(self, bbox_preds, obj_preds, cls_preds, labels, nlabel, A, grids, strides):
        # 这是即将返回的值
        num_gts = 0.0
        num_fg = 0.0
        fg_masks = []
        obj_targets = []
        reg_targets = []
        cls_targets = []
        l1_targets = []

        for batch_idx in range(bbox_preds.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                fg_mask = bbox_preds.new_zeros(A).bool()
                obj_target = bbox_preds.new_zeros((A, 1))
                reg_target = bbox_preds.new_zeros((0, 4))
                cls_target = bbox_preds.new_zeros((0, self.num_classes))
                l1_target = bbox_preds.new_zeros((0, 4))
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]   # [num_gt, 4]  gt的cxcywh, 单位是像素
                gt_classes = labels[batch_idx, :num_gt, 0]              # [num_gt, ]   gt的cid
                bbox_preds_ = bbox_preds[batch_idx]        # [A, 4]   预测的cxcywh, 单位是像素
                cls_preds_ = cls_preds[batch_idx]          # [A, n_cls]   未经过sigmoid激活
                obj_preds_ = obj_preds[batch_idx]          # [A, 1]       未经过sigmoid激活

                try:
                    (
                        fg_mask,
                        num_fg_img,
                        matched_gt_inds,
                        gt_matched_classes,
                        pred_ious_this_matching,
                    ) = self.get_assignments(  # noqa
                        num_gt,
                        gt_bboxes_per_image,
                        gt_classes,
                        bbox_preds_,
                        cls_preds_,
                        obj_preds_,
                        grids,
                        strides,
                    )
                except RuntimeError as e:
                    # TODO: the string might change, consider a better way
                    if "CUDA out of memory. " not in str(e):
                        raise  # RuntimeError might not caused by CUDA OOM

                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        fg_mask,
                        num_fg_img,
                        matched_gt_inds,
                        gt_matched_classes,
                        pred_ious_this_matching,
                    ) = self.get_assignments(  # noqa
                        num_gt,
                        gt_bboxes_per_image,
                        gt_classes,
                        bbox_preds_,
                        cls_preds_,
                        obj_preds_,
                        grids,
                        strides,
                        "cpu",
                    )
                '''
                fg_mask                     [A, ]       前景的mask
                num_fg_img                  anchor前景数
                matched_gt_inds             [num_fg_img, ]  前景匹配到的gt的下标
                gt_matched_classes          [num_fg_img, ]  前景匹配到的gt的cid
                pred_ious_this_matching     [num_fg_img, ]  前景匹配到的gt，与该gt的iou
                '''
                torch.cuda.empty_cache()
                num_fg += num_fg_img

                # [num_fg_img, n_cls]  前景学习的one_hot向量，学习的类别处 不是1 而是 与匹配到的gt的iou
                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)   # [A, 1]       前景的mask
                reg_target = gt_bboxes_per_image[matched_gt_inds]   # [num_fg_img, 4]  前景匹配到的gt的cxcywh, 单位是像素
                if self.use_l1:
                    x_shifts = grids[:, :, 0]  # [1, A]
                    y_shifts = grids[:, :, 1]  # [1, A]
                    # [num_fg_img, 4]  用来监督前景未解码的cxcywh
                    l1_target = self.get_l1_target(
                        bbox_preds.new_zeros((num_fg_img, 4)),
                        reg_target,
                        strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            fg_masks.append(fg_mask)
            obj_targets.append(obj_target.to(obj_preds.dtype))
            reg_targets.append(reg_target)
            cls_targets.append(cls_target)
            if self.use_l1:
                l1_targets.append(l1_target)

        fg_masks = torch.cat(fg_masks, 0)         # [N*A, ]          前景的mask, bool类型
        obj_targets = torch.cat(obj_targets, 0)   # [N*A, 1]         前景的mask, float类型
        reg_targets = torch.cat(reg_targets, 0)   # [num_fg, 4]      前景学习的gt的cxcywh, 单位是像素
        cls_targets = torch.cat(cls_targets, 0)   # [num_fg, n_cls]  前景学习的one_hot向量，学习的类别处 不是1 而是 与匹配到的gt的iou
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)   # [num_fg, 4]    前景学习的gt的cxcywh, 是未解码的cxcywh
        return num_gts, num_fg, fg_masks, obj_targets, reg_targets, cls_targets, l1_targets

    def batch_assign(self, bbox_preds, obj_preds, cls_preds, labels, nlabel, grids, strides):
        '''
        bbox_preds   [N, A, 4]       预测的cxcywh, 单位是像素
        cls_preds    [N, A, n_cls]   未经过sigmoid激活
        obj_preds    [N, A, 1]       未经过sigmoid激活
        labels       [N, G, 5]       gt的cid、cxcywh, 单位是像素
        '''
        # 这是即将返回的值
        l1_targets = []

        # 截掉填充的gt
        G = nlabel.max()
        labels = labels[:, :G, :]    # [N, G, 5]   gt的cid、cxcywh, 单位是像素
        masks = torch.ones((G+1, G+1), dtype=torch.uint8, device=obj_preds.device).tril(diagonal=-1)
        masks = masks[:, :-1]   # [G+1, G]
        pad_gt_mask = masks[nlabel, :].unsqueeze(-1)   # [N, G, 1]  是真gt还是填充的假gt
        pad_gt_mask = pad_gt_mask.float()
        num_gts = float(pad_gt_mask.sum())

        gt_bboxes = labels[:, :, 1:5]             # [N2, G, 4]  gt的cxcywh, 单位是像素
        gt_classes = labels[:, :, 0]              # [N2, G]     gt的cid
        try:
            (
                fg_mask,
                num_fg,
                matched_gt_inds,
                gt_matched_classes,
                pred_ious_this_matching,
            ) = self.get_batch_assignments(  # noqa
                gt_bboxes,
                gt_classes,
                bbox_preds,
                cls_preds,
                obj_preds,
                grids,
                strides,
                pad_gt_mask,
            )
        except RuntimeError as e:
            # TODO: the string might change, consider a better way
            if "CUDA out of memory. " not in str(e):
                raise  # RuntimeError might not caused by CUDA OOM

            logger.error(
                "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                   CPU mode is applied in this batch. If you want to avoid this issue, \
                   try to reduce the batch size or image size."
            )
            torch.cuda.empty_cache()
            (
                fg_mask,
                num_fg,
                matched_gt_inds,
                gt_matched_classes,
                pred_ious_this_matching,
            ) = self.get_batch_assignments(  # noqa
                gt_bboxes,
                gt_classes,
                bbox_preds,
                cls_preds,
                obj_preds,
                grids,
                strides,
                pad_gt_mask,
                "cpu",
            )
        '''
        fg_mask                     [N, A]      前景mask, bool类型
        num_fg                      anchor前景数
        matched_gt_inds             [num_fg, ]  前景匹配到的gt的下标。已经加上了图片偏移
        gt_matched_classes          [num_fg, ]  前景匹配到的gt的cid
        pred_ious_this_matching     [num_fg, ]  前景匹配到的gt，与该gt的iou
        '''
        torch.cuda.empty_cache()
        fg_masks = fg_mask.flatten()         # [N*A, ]          前景的mask, bool类型
        fg_masks.requires_grad_(False)

        # [num_fg, n_cls]  前景学习的one_hot向量，学习的类别处 不是1 而是 与匹配到的gt的iou
        cls_targets = F.one_hot(
            gt_matched_classes.to(torch.int64), self.num_classes
        ) * pred_ious_this_matching.unsqueeze(-1)
        reg_targets = gt_bboxes.reshape([-1, 4])[matched_gt_inds]   # [num_fg, 4]  前景匹配到的gt的cxcywh, 单位是像素
        if self.use_l1:
            x_shifts = grids[:, :, 0]  # [1, A]
            y_shifts = grids[:, :, 1]  # [1, A]
            N = obj_preds.shape[0]
            x_shifts = x_shifts.repeat([N, 1]).flatten()  # [N*A, ]
            y_shifts = y_shifts.repeat([N, 1]).flatten()  # [N*A, ]
            strides_ = strides.repeat([N, 1]).flatten()   # [N*A, ]
            # [num_fg, 4]  用来监督前景未解码的cxcywh
            l1_targets = self.get_l1_target(
                bbox_preds.new_zeros((num_fg, 4)),
                reg_targets,
                strides_[fg_masks],
                x_shifts=x_shifts[fg_masks],
                y_shifts=y_shifts[fg_masks],
            )

        obj_targets = fg_masks.unsqueeze(-1).to(obj_preds.dtype)   # [N*A, 1]         前景的mask, float类型
        num_fg = float(num_fg)
        obj_targets.requires_grad_(False)
        reg_targets.requires_grad_(False)
        cls_targets.requires_grad_(False)
        if self.use_l1:
            l1_targets.requires_grad_(False)
        return num_gts, num_fg, fg_masks, obj_targets, reg_targets, cls_targets, l1_targets

    @torch.no_grad()
    def get_assignments(
        self,
        num_gt,
        gt_bboxes_per_image,
        gt_classes,
        bbox_preds,
        cls_preds,
        obj_preds,
        grids,
        strides,
        mode="gpu",
    ):
        '''
        num_gt     当前图片gt数量
        gt_bboxes_per_image      [num_gt, 4]  gt的cxcywh, 单位是像素
        gt_classes               [num_gt, ]   gt的cid
        bbox_preds               [A, 4]   预测的cxcywh, 单位是像素
        cls_preds                [A, n_cls]   未经过sigmoid激活
        obj_preds                [A, 1]       未经过sigmoid激活
        grids                    [1, A, 2]  格子左上角xy坐标，单位是格子边长
        strides                  [1, A]  格子边长，单位是像素
        '''
        if mode == "cpu":
            print("-----------Using CPU for the Current Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            bbox_preds = bbox_preds.cpu().float()
            cls_preds = cls_preds.cpu().float()
            obj_preds = obj_preds.cpu().float()
            grids = grids.cpu()
            strides = strides.cpu().float()

        # fg_mask             [A, ]         anchor至少落在1个"范围框"内时, 为True
        # geometry_relation   [num_gt, M]   只保留那些至少落在1个"范围框"内的M个anchor
        fg_mask, geometry_relation = self.get_geometry_constraint(
            gt_bboxes_per_image,
            strides,
            grids,
        )

        # 抽出候选正样本
        bbox_preds = bbox_preds[fg_mask]  # [M, 4]  候选正样本预测的cxcywh, 单位是像素
        cls_preds_ = cls_preds[fg_mask]   # [M, n_cls]   候选正样本cls, 未经过sigmoid激活
        obj_preds_ = obj_preds[fg_mask]   # [M, 1]       候选正样本obj, 未经过sigmoid激活
        num_in_boxes_anchor = bbox_preds.shape[0]   # M, 在"范围框"内的anchor数量(候选正样本数量)

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bbox_preds = bbox_preds.cpu()

        # [num_gt, M]  gt和候选正样本两两之间的iou
        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bbox_preds, False)

        # [num_gt, n_cls]   gt的one_hot向量
        gt_cls_per_image = F.one_hot(gt_classes.to(torch.int64), self.num_classes).float()
        # [num_gt, M]  iou的cost，iou越大cost越小
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):   # 不使用fp16，因为 F.binary_cross_entropy 对fp16不安全
            # [M, n_cls]   M个候选正样本预测的各类别分数, 开了根号
            cls_preds_ = (
                cls_preds_.float().sigmoid_() * obj_preds_.float().sigmoid_()
            ).sqrt()
            # 二值交叉熵。input形状是 [num_gt, M, n_cls]，M个候选正样本预测的各类别分数，重复num_gt次
            # target形状是          [num_gt, M, n_cls]，num_gt个gt的one_hot向量，重复M次
            # 计算的是gt和候选正样本两两之间的二值交叉熵cost。pair_wise_cls_loss形状为 [num_gt, M]
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.unsqueeze(0).repeat(num_gt, 1, 1),
                gt_cls_per_image.unsqueeze(1).repeat(1, num_in_boxes_anchor, 1),
                reduction="none"
            ).sum(-1)
        del cls_preds_

        # [num_gt, M]  总的cost，iou cost的权重是3。
        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + float(1e6) * (~geometry_relation)
        )

        '''
        num_fg                      anchor前景数
        matched_gt_inds             [num_fg, ]  前景匹配到的gt的下标
        gt_matched_classes          [num_fg, ]  前景匹配到的gt的cid
        pred_ious_this_matching     [num_fg, ]  前景匹配到的gt，与该gt的iou
        '''
        (
            num_fg,
            matched_gt_inds,
            gt_matched_classes,
            pred_ious_this_matching,
        ) = self.simota_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            fg_mask,
            num_fg,
            matched_gt_inds,
            gt_matched_classes,
            pred_ious_this_matching,
        )

    @torch.no_grad()
    def get_batch_assignments(
        self,
        gt_bboxes,
        gt_classes,
        bbox_preds,
        cls_preds,
        obj_preds,
        grids,
        strides,
        pad_gt_mask,
        mode="gpu",
    ):
        '''
        gt_bboxes      [N, G, 4]  gt的cxcywh, 单位是像素
        gt_classes     [N, G]     gt的cid
        bbox_preds     [N, A, 4]       预测的cxcywh, 单位是像素
        cls_preds      [N, A, n_cls]   未经过sigmoid激活
        obj_preds      [N, A, 1]       未经过sigmoid激活
        grids          [1, A, 2]  格子左上角xy坐标，单位是格子边长
        strides        [1, A]  格子边长，单位是像素
        pad_gt_mask    [N, G, 1]  是真gt还是填充的假gt, float类型
        '''
        G = gt_bboxes.shape[1]
        A = bbox_preds.shape[1]
        if mode == "cpu":
            print("-----------Using CPU for the Current Batch-------------")
            gt_bboxes = gt_bboxes.cpu().float()
            gt_classes = gt_classes.cpu().float()
            bbox_preds = bbox_preds.cpu().float()
            cls_preds = cls_preds.cpu().float()
            obj_preds = obj_preds.cpu().float()
            grids = grids.cpu()
            strides = strides.cpu().float()
            pad_gt_mask = pad_gt_mask.cpu()

        # is_in_centers    [N, G, A]  若某个格子中心点落在某个"范围框"内, 值为True
        is_in_centers = self.get_batch_geometry_constraint(
            gt_bboxes,
            strides,
            grids,
        )
        if mode == "cpu":
            gt_bboxes = gt_bboxes.cpu()
            bbox_preds = bbox_preds.cpu()
        # [N, G, A]  计算 gt和预测框 两组矩形两两之间的iou
        pair_wise_ious = yolox_batch_bboxes_iou(gt_bboxes, bbox_preds, xyxy=False)  # [N, G, A]  两组矩形两两之间的iou
        candidate_pos = is_in_centers.sum(1) > 0                      # [N, A]  候选正样本。若anchor落在任意gt中心内部则为True
        candidate_pos = candidate_pos.unsqueeze(1).repeat([1, G, 1])  # [N, G, A]  候选正样本。若anchor落在任意gt中心内部则为True, 重复G次
        # 非候选正样本 与 所有gt 的iou置为0
        pair_wise_ious *= candidate_pos.float()
        # 假gt 和 所有anchor 的iou置为0
        pair_wise_ious *= pad_gt_mask
        # [N, G, A]  iou的cost，iou越大cost越小
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)
        pair_wise_ious_loss += float(1e6) * (~candidate_pos)   # 非候选正样本+float(1e6)

        # [N, G, n_cls]   gt的one_hot向量
        gt_cls = F.one_hot(gt_classes.to(torch.int64), self.num_classes).float()
        if mode == "cpu":
            cls_preds, obj_preds = cls_preds.cpu(), obj_preds.cpu()

        with torch.cuda.amp.autocast(enabled=False):   # 不使用fp16，因为 F.binary_cross_entropy 对fp16不安全
            # [N, A, n_cls]   anchor预测的各类别分数, 开了根号
            # 和 get_assignments()不同！！tensor.sigmoid_()是inplace的！！！会改变原来的cls_preds和obj_preds, 所以这两个张量先clone()新建副本
            cls_preds_ = (
                cls_preds.clone().float().sigmoid_() * obj_preds.clone().float().sigmoid_()
            ).sqrt()
            # 二值交叉熵。input形状是 [N, G, A, n_cls]，anchor预测的各类别分数，重复G次
            # target形状是          [N, G, A, n_cls]，gt的one_hot向量，重复A次
            # 计算的是gt和anchor两两之间的二值交叉熵cost。pair_wise_cls_loss形状为 [N, G, A]
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.unsqueeze(1).repeat(1, G, 1, 1),
                gt_cls.unsqueeze(2).repeat(1, 1, A, 1),
                reduction="none"
            ).sum(-1)
            pair_wise_cls_loss += float(1e6) * (~candidate_pos)   # 非候选正样本+float(1e6)
        del cls_preds_

        # [N, G, A]  总的cost，iou cost的权重是3。
        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + float(1e6) * (~is_in_centers)
            + float(1e6) * (~candidate_pos)
        )
        # 假gt 和 所有anchor 的cost都是 float(1e6)
        cost = cost * pad_gt_mask + (1. - pad_gt_mask) * float(1e6)

        '''
        fg_mask                     [N, A]      前景mask, bool类型
        num_fg                      anchor前景数
        matched_gt_inds             [num_fg, ]  前景匹配到的gt的下标。已经加上了图片偏移
        gt_matched_classes          [num_fg, ]  前景匹配到的gt的cid
        pred_ious_this_matching     [num_fg, ]  前景匹配到的gt，与该gt的iou
        '''
        (
            fg_mask,
            num_fg,
            matched_gt_inds,
            gt_matched_classes,
            pred_ious_this_matching,
        ) = self.batch_simota_matching(cost, pair_wise_ious, gt_classes, is_in_centers, pad_gt_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            fg_mask,
            num_fg,
            matched_gt_inds,
            gt_matched_classes,
            pred_ious_this_matching,
        )

    def get_geometry_constraint(
        self, gt_bboxes_per_image, strides, grids,
    ):
        """
        Calculate whether the center of an object is located in a fixed range of
        an anchor. This is used to avert inappropriate matching. It can also reduce
        the number of candidate anchors so that the GPU memory is saved.
        """
        '''
        gt_bboxes_per_image      [num_gt, 4]  gt的cxcywh, 单位是像素
        strides                  [1, A]  格子边长，单位是像素
        grids                    [1, A, 2]  格子左上角xy坐标，单位是格子边长
        '''
        # in fixed center
        center_radius = 1.5
        center_dist = strides * center_radius   # [1, A]  格子边长*1.5倍，单位是像素。每个gt中心点形成一个"范围框"，里面框住的格子中心点作为候选正样本。
        gt_bboxes_ = gt_bboxes_per_image.unsqueeze(1)    # [num_gt, 1, 4]
        center_dist = center_dist.unsqueeze(2)           # [     1, A, 1]
        center_x1y1 = gt_bboxes_[:, :, :2] - center_dist     # [num_gt, A, 2]   "范围框"的x1y1, 单位是像素
        center_x2y2 = gt_bboxes_[:, :, :2] + center_dist     # [num_gt, A, 2]   "范围框"的x2y2, 单位是像素

        points_ = (grids + 0.5) * strides.unsqueeze(2)   # [1, A, 2]  格子中心点xy坐标，单位是像素
        lt = points_ - center_x1y1  # [num_gt, A, 2]
        rb = center_x2y2 - points_  # [num_gt, A, 2]
        ltrb = torch.cat([lt, rb], -1)  # [num_gt, A, 4]
        is_in_centers = ltrb.min(dim=-1).values > 0.0  # [num_gt, A]  若某个格子中心点落在某个"范围框"内, 值为True
        anchor_filter = is_in_centers.sum(dim=0) > 0   # [A, ]        anchor至少落在1个"范围框"内时, 为True
        geometry_relation = is_in_centers[:, anchor_filter]   # [num_gt, M]   只保留那些至少落在1个"范围框"内的M个anchor
        return anchor_filter, geometry_relation

    def get_batch_geometry_constraint(
        self, gt_bboxes, strides, grids,
    ):
        """
        Calculate whether the center of an object is located in a fixed range of
        an anchor. This is used to avert inappropriate matching. It can also reduce
        the number of candidate anchors so that the GPU memory is saved.
        """
        '''
        gt_bboxes                [N, G, 4]  gt的cxcywh, 单位是像素
        strides                  [1, A]  格子边长，单位是像素
        grids                    [1, A, 2]  格子左上角xy坐标，单位是格子边长
        '''
        # in fixed center
        center_radius = 1.5
        center_dist = strides * center_radius   # [1, A]  格子边长*1.5倍，单位是像素。每个gt中心点形成一个"范围框"，里面框住的格子中心点作为候选正样本。
        gt_bboxes_ = gt_bboxes.unsqueeze(2)                    # [N, G, 1, 4]
        center_dist = center_dist.unsqueeze(0).unsqueeze(-1)   # [1, 1, A, 1]
        center_x1y1 = gt_bboxes_[:, :, :, :2] - center_dist     # [N, G, A, 2]   "范围框"的x1y1, 单位是像素
        center_x2y2 = gt_bboxes_[:, :, :, :2] + center_dist     # [N, G, A, 2]   "范围框"的x2y2, 单位是像素

        points_ = (grids + 0.5) * strides.unsqueeze(2)   # [1, A, 2]  格子中心点xy坐标，单位是像素
        points_ = points_.unsqueeze(0)    # [1, 1, A, 2]
        lt = points_ - center_x1y1  # [N, G, A, 2]
        rb = center_x2y2 - points_  # [N, G, A, 2]
        ltrb = torch.cat([lt, rb], -1)  # [N, G, A, 4]
        is_in_centers = ltrb.min(dim=-1).values > 0.0  # [N, G, A]  若某个格子中心点落在某个"范围框"内, 值为True
        # anchor_filter = is_in_centers.sum(dim=1) > 0   # [N, A]        anchor至少落在1个"范围框"内时, 为True
        # geometry_relation = is_in_centers[:, anchor_filter]   # [num_gt, M]   只保留那些至少落在1个"范围框"内的M个anchor
        return is_in_centers

    def simota_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        '''
        为每个gt动态分配不同的正样本数。
        cost             [num_gt, M]  总的cost
        pair_wise_ious   [num_gt, M]  gt和候选正样本两两之间的iou
        gt_classes       [num_gt, ]   gt的cid
        num_gt           当前图片gt数目
        fg_mask          [A, ]     anchor至少落在1个"范围框"内时, 为True
        '''
        # [num_gt, M]  全是0，类型是uint8省内存
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        n_candidate_k = min(10, pair_wise_ious.size(1))   # 选10个候选正样本，如果M < 10，选M个。下面假设M>10，n_candidate_k==10
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)   # [num_gt, 10]  每个gt取10个最大iou的候选正样本。
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)           # [num_gt, ]  iou求和作为正样本数。

        # 对每个gt，取cost最小的k个候选正样本去学习。
        if self.parallel_matching:
            max_k = dynamic_ks.max()
            masks = torch.ones((max_k, max_k), dtype=torch.uint8, device=cost.device).tril(diagonal=0)   # [max_k, max_k]
            fill_value = masks[(dynamic_ks - 1).long(), :]   # [num_gt, max_k]   每个gt要填入 matching_matrix[num_gt, M]  的值
            _, pos_idx = torch.topk(cost, k=max_k, largest=False)   # [num_gt, max_k]   每个gt前max_k个cost最小的下标
            M = cost.shape[1]
            offset = torch.arange(start=0, end=M*num_gt, step=M, dtype=torch.int64, device=cost.device).unsqueeze(-1)  # [num_gt, 1]
            pos_idx_1d = (pos_idx + offset).flatten()   # [num_gt*max_k, ]
            matching_matrix = matching_matrix.flatten()
            matching_matrix[pos_idx_1d] = fill_value.flatten()
            matching_matrix = matching_matrix.reshape(cost.shape)
            del topk_ious, dynamic_ks, max_k, masks, fill_value, pos_idx, offset, pos_idx_1d
        else:
            # naive implementation.
            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx], largest=False)
                matching_matrix[gt_idx][pos_idx] = 1
            del topk_ious, dynamic_ks, pos_idx

        # [M, ]  M个候选正样本匹配的gt数
        anchor_matching_gt = matching_matrix.sum(0)
        # deal with the case that one anchor matches multiple ground-truths
        if anchor_matching_gt.max() > 1:
            multiple_match_mask = anchor_matching_gt > 1  # [M, ]  M个候选正样本 一对多 处为1
            matching_matrix[:, multiple_match_mask] *= 0  # 一对多的候选正样本，不匹配任何gt
            _, cost_argmin = torch.min(cost[:, multiple_match_mask], dim=0)
            matching_matrix[cost_argmin, multiple_match_mask] = 1  # 一对多的候选正样本，匹配cost最小的gt
        fg_mask_inboxes = anchor_matching_gt > 0    # 可能有的候选正样本匹配了0个gt。将匹配多于0个gt的候选正样本作为最终正样本。
        num_fg = fg_mask_inboxes.sum().item()       # anchor前景数

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)  # [num_fg, ]  前景匹配到的gt的下标
        gt_matched_classes = gt_classes[matched_gt_inds]                 # [num_fg, ]  前景匹配到的gt的cid

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]  # [num_fg, ]  前景匹配到的gt，与该gt的iou
        return num_fg, matched_gt_inds, gt_matched_classes, pred_ious_this_matching

    def batch_simota_matching(self, cost, pair_wise_ious, gt_classes, is_in_centers, pad_gt_mask):
        '''
        为每个gt动态分配不同的正样本数。
        cost             [N, G, A]  总的cost
        pair_wise_ious   [N, G, A]  gt和 所有anchor 两两之间的iou
        gt_classes       [N, G]     gt的cid
        is_in_centers    [N, G, A]  若某个格子中心点落在某个"范围框"内, 值为True
        pad_gt_mask      [N, G, 1]  是真gt还是填充的假gt, float类型
        '''
        N, G, A = cost.shape
        # [N, G, A]  全是0，类型是uint8省内存
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        n_candidate_k = 10    # 选10个候选样本
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=2)   # [N, G, 10]  每个gt取10个最大iou的anchor。
        dynamic_ks = torch.clamp(topk_ious.sum(2).int(), min=1)           # [N, G]  iou求和作为正样本数。
        dynamic_ks *= pad_gt_mask[:, :, 0].int()    # [N, G]  假gt正样本数为0。

        # 对每个gt，取cost最小的k个anchor去学习。
        if self.parallel_matching:
            max_k = dynamic_ks.max()
            masks = torch.ones((max_k + 1, max_k + 1), dtype=torch.uint8, device=cost.device).tril(diagonal=-1)
            masks = masks[:, :-1]  # [max_k+1, max_k]
            fill_value = masks[dynamic_ks.long(), :]  # [N, G, max_k]
            _, pos_idx = torch.topk(cost, k=max_k, largest=False)  # [N, G, max_k]
            offset = torch.arange(start=0, end=N * G * A, step=A, dtype=torch.int64, device=cost.device).unsqueeze(-1)  # [N*G, 1]
            pos_idx = pos_idx.reshape([N * G, -1])  # [N*G, max_k]
            pos_idx_1d = (pos_idx + offset).flatten()  # [N*G*max_k, ]
            matching_matrix = matching_matrix.flatten()
            matching_matrix[pos_idx_1d] = fill_value.flatten()
            matching_matrix = matching_matrix.reshape(cost.shape)
            del topk_ious, dynamic_ks, max_k, masks, fill_value, pos_idx, offset, pos_idx_1d
        else:
            # naive implementation.
            for b_idx in range(N):
                for gt_idx in range(G):
                    if dynamic_ks[b_idx, gt_idx] == 0:   # 假gt跳过
                        continue
                    _, pos_idx = torch.topk(cost[b_idx, gt_idx], k=dynamic_ks[b_idx, gt_idx], largest=False)
                    matching_matrix[b_idx, gt_idx][pos_idx] = 1
            del topk_ious, dynamic_ks, pos_idx

        # [N, A]  anchor匹配的gt数
        anchor_matching_gt = matching_matrix.sum(1)
        matching_matrix = matching_matrix.permute((0, 2, 1))   # [N, A, G]
        # deal with the case that one anchor matches multiple ground-truths
        if anchor_matching_gt.max() > 1:
            multiple_match_mask = anchor_matching_gt > 1  # [N, A]  anchor 一对多 处为1
            matching_matrix[multiple_match_mask] *= 0  # 一对多的anchor，不匹配任何gt
            cost = cost.permute((0, 2, 1))   # [N, A, G]
            multiple_match_cost = cost[multiple_match_mask]   # [?, G]   一对多的anchor对所有gt的cost
            _, cost_argmin = torch.min(multiple_match_cost, dim=1)   # [?, ]   一对多的anchor最小cost的gt下标
            matching_matrix[multiple_match_mask, cost_argmin] = 1  # 一对多的anchor，匹配cost最小的gt
        fg_mask = anchor_matching_gt > 0    # [N, A]  可能有的anchor匹配了0个gt。将匹配多于0个gt的anchor作为最终正样本。
        num_fg = fg_mask.sum().item()       # anchor前景数

        # [N, A]    图片的下标
        image_ind = torch.arange(end=N, dtype=torch.int64, device=gt_classes.device).unsqueeze(-1).repeat([1, A])
        matched_image_inds = image_ind[fg_mask]   # [num_fg, ]  前景匹配到的gt所在图片的下标
        del image_ind

        matched_gt_inds = matching_matrix[fg_mask].argmax(1)           # [num_fg, ]  前景匹配到的gt的下标
        matched_gt_inds += matched_image_inds * G                      # [num_fg, ]  前景匹配到的gt的下标。加上图片偏移
        gt_matched_classes = gt_classes.flatten()[matched_gt_inds]     # [num_fg, ]  前景匹配到的gt的cid

        pair_wise_ious = pair_wise_ious.permute((0, 2, 1))   # [N, A, G]
        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(2)[fg_mask]  # [num_fg, ]  前景匹配到的gt，与该gt的iou
        return fg_mask, num_fg, matched_gt_inds, gt_matched_classes, pred_ious_this_matching
