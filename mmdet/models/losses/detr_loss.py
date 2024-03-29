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
from torch import distributed as dist

from .iou_losses import GIoULoss
from mmdet.models.transformers.utils import bbox_cxcywh_to_xyxy
from mmdet.utils import get_world_size
from ..bbox_utils import bbox_iou
from ..ops import gather_1d, scatter_1d

__all__ = ['DETRLoss', 'DINOLoss']



class DETRLoss(nn.Module):
    __shared__ = ['num_classes', 'use_focal_loss']
    __inject__ = ['matcher']

    def __init__(self,
                 num_classes=80,
                 matcher='HungarianMatcher',
                 loss_coeff={
                     'class': 1,
                     'bbox': 5,
                     'giou': 2,
                     'no_object': 0.1,
                     'mask': 1,
                     'dice': 1
                 },
                 aux_loss=True,
                 use_focal_loss=False,
                 use_vfl=False,
                 use_uni_match=False,
                 uni_match_ind=0):
        r"""
        Args:
            num_classes (int): The number of classes.
            matcher (HungarianMatcher): It computes an assignment between the targets
                and the predictions of the network.
            loss_coeff (dict): The coefficient of loss.
            aux_loss (bool): If 'aux_loss = True', loss at each decoder layer are to be used.
            use_focal_loss (bool): Use focal loss or not.
        """
        super(DETRLoss, self).__init__()

        self.num_classes = num_classes
        self.matcher = matcher
        self.loss_coeff = loss_coeff
        self.aux_loss = aux_loss
        self.use_focal_loss = use_focal_loss
        self.use_vfl = use_vfl
        self.use_uni_match = use_uni_match
        self.uni_match_ind = uni_match_ind

        # if not self.use_focal_loss:
        #     self.loss_coeff['class'] = torch.ones([num_classes + 1, ], dtype=torch.float32) * loss_coeff['class']
        #     self.loss_coeff['class'][-1] = loss_coeff['no_object']
        self.giou_loss = GIoULoss()

    def _get_loss_class(self,
                        logits,
                        gt_class,
                        pad_gt_mask,
                        match_indices,
                        bg_index,
                        num_gts,
                        postfix="",
                        iou_score=None):
        # logits: [b, query, num_classes], gt_class: list[[n, 1]]
        name_class = "loss_class" + postfix
        device = pad_gt_mask.device

        target_label = torch.full(logits.shape[:2], bg_index, dtype=torch.int64, device=device)
        bs, num_query_objects = target_label.shape
        # num_gt = sum(len(a) for a in gt_class)
        num_gt = int(pad_gt_mask.sum())
        if num_gt > 0:
            index, updates = self._get_index_updates(num_query_objects, gt_class, match_indices)
            target_label = scatter_1d(target_label.reshape([-1, 1]), index, updates.to(torch.int64))
            target_label = target_label.reshape([bs, num_query_objects])
        if self.use_focal_loss:
            raise NotImplementedError
        else:
            # loss_ = F.cross_entropy(logits, target_label, weight=self.loss_coeff['class'])
            assigned_scores = F.one_hot(target_label, self.num_classes + 1)
            assigned_scores = assigned_scores.to(device).to(torch.float32)
            assigned_scores = assigned_scores[:, :, :-1]
            eps = 1e-9
            p = F.softmax(logits, dim=-1)
            loss_ = assigned_scores * (0 - torch.log(p + eps))
            loss_ = loss_.sum(-1) * self.loss_coeff['class']
            # loss_ = loss_.mean()   # 先不用mean()  原版是 reduction='mean'
            loss_ = loss_.sum()
        return {name_class: loss_}

    def _get_loss_bbox(self, boxes, gt_bbox, pad_gt_mask, match_indices, num_gts, postfix=""):
        # boxes: [b, query, 4], gt_bbox: list[[n, 4]]
        name_bbox = "loss_bbox" + postfix
        name_giou = "loss_giou" + postfix

        loss = dict()
        device = boxes.device
        if pad_gt_mask.sum() < 1:
            # 由于我pad了，num_gt的计算要重新写一下
            # raise NotImplementedError
            loss[name_bbox] = torch.zeros([1, ], device=device)
            loss[name_giou] = torch.zeros([1, ], device=device)
            return loss

        src_bbox, target_bbox = self._get_src_target_assign(boxes, gt_bbox, match_indices)
        target_bbox.requires_grad = False
        loss[name_bbox] = self.loss_coeff['bbox'] * F.l1_loss(src_bbox, target_bbox, reduction='sum') / num_gts
        loss[name_giou] = self.giou_loss(bbox_cxcywh_to_xyxy(src_bbox), bbox_cxcywh_to_xyxy(target_bbox))
        loss[name_giou] = loss[name_giou].sum() / num_gts
        loss[name_giou] = self.loss_coeff['giou'] * loss[name_giou]
        return loss

    def _get_loss_mask(self, masks, gt_mask, pad_gt_mask, match_indices, num_gts, postfix=""):
        # masks: [b, query, h, w], gt_mask: list[[n, H, W]]
        name_mask = "loss_mask" + postfix
        name_dice = "loss_dice" + postfix

        loss = dict()
        device = masks.device
        if pad_gt_mask.sum() < 1:
            # 由于我pad了，num_gt的计算要重新写一下
            # raise NotImplementedError
            loss[name_mask] = torch.zeros([1, ], device=device)
            loss[name_dice] = torch.zeros([1, ], device=device)
            return loss

        src_masks, target_masks = self._get_src_target_assign(masks, gt_mask,
                                                              match_indices)
        src_masks = F.interpolate(
            src_masks.unsqueeze(0),
            size=target_masks.shape[-2:],
            mode="bilinear")[0]
        loss[name_mask] = self.loss_coeff['mask'] * F.sigmoid_focal_loss(
            src_masks,
            target_masks,
            paddle.to_tensor(
                [num_gts], dtype='float32'))
        loss[name_dice] = self.loss_coeff['dice'] * self._dice_loss(
            src_masks, target_masks, num_gts)
        return loss

    def _dice_loss(self, inputs, targets, num_gts):
        inputs = F.sigmoid(inputs)
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / num_gts

    def _get_loss_aux(self,
                      boxes,
                      logits,
                      gt_bbox,
                      gt_class,
                      pad_gt_mask,
                      bg_index,
                      num_gts,
                      dn_match_indices=None,
                      postfix="",
                      masks=None,
                      gt_mask=None):
        loss_class = 0.
        loss_bbox, loss_giou = 0., 0.
        loss_mask, loss_dice = 0., 0.
        if dn_match_indices is not None:
            match_indices = dn_match_indices
        elif self.use_uni_match:
            match_indices = self.matcher(
                boxes[self.uni_match_ind],
                logits[self.uni_match_ind],
                gt_bbox,
                gt_class,
                pad_gt_mask,
                masks=masks[self.uni_match_ind] if masks is not None else None,
                gt_mask=gt_mask)
        for i, (aux_boxes, aux_logits) in enumerate(zip(boxes, logits)):
            aux_masks = masks[i] if masks is not None else None
            if not self.use_uni_match and dn_match_indices is None:
                match_indices = self.matcher(
                    aux_boxes,
                    aux_logits,
                    gt_bbox,
                    gt_class,
                    pad_gt_mask,
                    masks=aux_masks,
                    gt_mask=gt_mask)
            if self.use_vfl:
                if pad_gt_mask.sum() > 0:
                    src_bbox, target_bbox = self._get_src_target_assign(
                        aux_boxes.detach(), gt_bbox, match_indices)
                    iou_score = bbox_iou(
                        bbox_cxcywh_to_xyxy(src_bbox).split(1, -1),
                        bbox_cxcywh_to_xyxy(target_bbox).split(1, -1))
                else:
                    iou_score = None
            else:
                iou_score = None
            loss_class = loss_class + \
                self._get_loss_class(aux_logits, gt_class, pad_gt_mask, match_indices,
                                     bg_index, num_gts, postfix, iou_score)[
                                         'loss_class' + postfix]
            loss_ = self._get_loss_bbox(aux_boxes, gt_bbox, pad_gt_mask, match_indices, num_gts, postfix)
            loss_bbox = loss_bbox + loss_['loss_bbox' + postfix]
            loss_giou = loss_giou + loss_['loss_giou' + postfix]
            if masks is not None and gt_mask is not None:
                loss_ = self._get_loss_mask(aux_masks, gt_mask, pad_gt_mask, match_indices,
                                            num_gts, postfix)
                loss_mask = loss_mask + loss_['loss_mask' + postfix]
                loss_dice = loss_dice + loss_['loss_dice' + postfix]
        loss = {
            "loss_class_aux" + postfix: loss_class,
            "loss_bbox_aux" + postfix: loss_bbox,
            "loss_giou_aux" + postfix: loss_giou
        }
        if masks is not None and gt_mask is not None:
            loss["loss_mask_aux" + postfix] = loss_mask
            loss["loss_dice_aux" + postfix] = loss_dice
        return loss

    def _get_index_updates(self, num_query_objects, target, match_indices):
        batch_idx = torch.cat([
            torch.full_like(src, i) for i, (src, _) in enumerate(match_indices)
        ], dim=0)
        src_idx = torch.cat([src for (src, _) in match_indices], dim=0)
        src_idx += (batch_idx * num_query_objects)
        target_assign = torch.cat([
            gather_1d(t, dst) for t, (_, dst) in zip(target, match_indices)
        ], dim=0)
        return src_idx, target_assign

    def _get_src_target_assign(self, src, target, match_indices):
        device = src.device
        src_assign = []
        for t, (I, _) in zip(src, match_indices):
            v = None
            if len(I) > 0:
                v = gather_1d(t, I)
            else:
                v = torch.zeros([0, t.shape[-1]], device=device)
            src_assign.append(v)
        src_assign = torch.cat(src_assign, dim=0)

        target_assign = []
        for t, (_, J) in zip(target, match_indices):
            v = None
            if len(J) > 0:
                v = gather_1d(t, J)
            else:
                v = torch.zeros([0, t.shape[-1]], device=device)
            target_assign.append(v)
        target_assign = torch.cat(target_assign, dim=0)

        return src_assign, target_assign

    def _get_num_gts(self, targets, dtype="float32"):
        # num_gts = sum(len(a) for a in targets)
        # num_gts = paddle.to_tensor([num_gts], dtype=dtype)
        # if paddle.distributed.get_world_size() > 1:
        #     paddle.distributed.all_reduce(num_gts)
        #     num_gts /= paddle.distributed.get_world_size()
        # num_gts = paddle.clip(num_gts, min=1.)
        # return num_gts
        raise NotImplementedError

    def _get_num_gts_by_pad_gt_mask(self, pad_gt_mask):
        num_gts = pad_gt_mask.sum()
        world_size = get_world_size()
        if world_size > 1:
            dist.all_reduce(num_gts, op=dist.ReduceOp.SUM)
            num_gts = num_gts / world_size
        num_gts = torch.clamp(num_gts, min=1.)  # y = max(x, 1)
        return num_gts

    def _get_prediction_loss(self,
                             boxes,
                             logits,
                             gt_bbox,
                             gt_class,
                             pad_gt_mask,
                             masks=None,
                             gt_mask=None,
                             postfix="",
                             dn_match_indices=None,
                             num_gts=1):
        if dn_match_indices is None:
            match_indices = self.matcher(
                boxes, logits, gt_bbox, gt_class, pad_gt_mask, masks=masks, gt_mask=gt_mask)
        else:
            match_indices = dn_match_indices

        if self.use_vfl:
            if pad_gt_mask.sum() > 0:
                src_bbox, target_bbox = self._get_src_target_assign(boxes.detach(), gt_bbox, match_indices)

                bbox1 = bbox_cxcywh_to_xyxy(src_bbox)
                bbox2 = bbox_cxcywh_to_xyxy(target_bbox)
                bbox1 = torch.split(bbox1, 1, -1)
                bbox2 = torch.split(bbox2, 1, -1)
                iou_score = bbox_iou(bbox1, bbox2)
            else:
                iou_score = None
        else:
            iou_score = None

        loss = dict()
        loss.update(
            self._get_loss_class(logits, gt_class, pad_gt_mask, match_indices,
                                 self.num_classes, num_gts, postfix, iou_score))
        loss.update(
            self._get_loss_bbox(boxes, gt_bbox, pad_gt_mask, match_indices, num_gts, postfix))
        if masks is not None and gt_mask is not None:
            loss.update(
                self._get_loss_mask(masks, gt_mask, pad_gt_mask, match_indices, num_gts, postfix))
        return loss

    def forward(self,
                boxes,
                logits,
                gt_bbox,
                gt_class,
                pad_gt_mask,
                masks=None,
                gt_mask=None,
                postfix="",
                **kwargs):
        r"""
        Args:
            boxes (Tensor): [l, b, query, 4]
            logits (Tensor): [l, b, query, num_classes]
            gt_bbox (List(Tensor)): list[[n, 4]]
            gt_class (List(Tensor)): list[[n, 1]]
            masks (Tensor, optional): [l, b, query, h, w]
            gt_mask (List(Tensor), optional): list[[n, H, W]]
            postfix (str): postfix of loss name
        """

        dn_match_indices = kwargs.get("dn_match_indices", None)
        num_gts = kwargs.get("num_gts", None)
        if num_gts is None:
            num_gts = self._get_num_gts(gt_class)

        total_loss = self._get_prediction_loss(
            boxes[-1],
            logits[-1],
            gt_bbox,
            gt_class,
            pad_gt_mask,
            masks=masks[-1] if masks is not None else None,
            gt_mask=gt_mask,
            postfix=postfix,
            dn_match_indices=dn_match_indices,
            num_gts=num_gts)

        if self.aux_loss:
            total_loss.update(
                self._get_loss_aux(
                    boxes[:-1],
                    logits[:-1],
                    gt_bbox,
                    gt_class,
                    pad_gt_mask,
                    self.num_classes,
                    num_gts,
                    dn_match_indices,
                    postfix,
                    masks=masks[:-1] if masks is not None else None,
                    gt_mask=gt_mask))

        return total_loss


class DINOLoss(DETRLoss):
    def forward(self,
                boxes,
                logits,
                gt_bbox,
                gt_class,
                pad_gt_mask,
                masks=None,
                gt_mask=None,
                postfix="",
                dn_out_bboxes=None,
                dn_out_logits=None,
                dn_meta=None,
                **kwargs):
        num_gts = self._get_num_gts_by_pad_gt_mask(pad_gt_mask)
        total_loss = super(DINOLoss, self).forward(
            boxes, logits, gt_bbox, gt_class, pad_gt_mask, num_gts=num_gts)

        if dn_meta is not None:
            dn_positive_idx, dn_num_group = \
                dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
            assert len(gt_class) == len(dn_positive_idx)

            # denoising match indices
            dn_match_indices = self.get_dn_match_indices(
                gt_class, dn_positive_idx, dn_num_group, pad_gt_mask)

            # compute denoising training loss
            num_gts *= dn_num_group
            dn_loss = super(DINOLoss, self).forward(
                dn_out_bboxes,
                dn_out_logits,
                gt_bbox,
                gt_class,
                pad_gt_mask,
                postfix="_dn",
                dn_match_indices=dn_match_indices,
                num_gts=num_gts)
            total_loss.update(dn_loss)
        else:
            pass
            # raise NotImplementedError
        return total_loss

    @staticmethod
    def get_dn_match_indices(labels, dn_positive_idx, dn_num_group, pad_gt_mask):
        dn_match_indices = []
        device = labels.device
        num_gts = pad_gt_mask.sum([1, 2]).to(torch.int32).cpu().detach().numpy().tolist()
        for i in range(len(labels)):
            num_gt = num_gts[i]
            if num_gt > 0:
                gt_idx = torch.arange(end=num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile([dn_num_group])
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(
                    [0], dtype=torch.int64, device=device), torch.zeros(
                        [0], dtype=torch.int64, device=device)))
        return dn_match_indices

