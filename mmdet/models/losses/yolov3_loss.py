#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-10-15 14:50:03
#   Description : pytorch_ppyolo
#
# ================================================================
import torch
import torch.nn as nn
import torch as T
import numpy as np

from mmdet.models.custom_layers import paddle_yolo_box
from mmdet.models.matrix_nms import jaccard

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence




class YOLOv3Loss(nn.Module):
    """
    Combined loss for YOLOv3 network

    Args:
        batch_size (int): training batch size
        ignore_thresh (float): threshold to ignore confidence loss
        label_smooth (bool): whether to use label smoothing
        use_fine_grained_loss (bool): whether use fine grained YOLOv3 loss
                                      instead of fluid.layers.yolov3_loss
    """

    def __init__(self,
                 ignore_thresh=0.7,
                 label_smooth=True,
                 use_fine_grained_loss=False,
                 iou_loss=None,
                 iou_aware_loss=None,
                 downsample=[32, 16, 8],
                 scale_x_y=1.,
                 match_score=False):
        super(YOLOv3Loss, self).__init__()
        self._ignore_thresh = ignore_thresh
        self._label_smooth = label_smooth
        self._use_fine_grained_loss = use_fine_grained_loss
        self._iou_loss = iou_loss
        self._iou_aware_loss = iou_aware_loss
        self.downsample = downsample
        self.scale_x_y = scale_x_y
        self.match_score = match_score

    def forward(self, outputs, gt_box, gt_label, gt_score, targets, anchors,
                anchor_masks, mask_anchors, num_classes):
        return self._get_fine_grained_loss(
            outputs, targets, gt_box, num_classes,
            mask_anchors, self._ignore_thresh)

    def _get_fine_grained_loss(self,
                               outputs,
                               targets,
                               gt_box,
                               num_classes,
                               mask_anchors,
                               ignore_thresh,
                               eps=1.e-10):
        """
        Calculate fine grained YOLOv3 loss

        Args:
            outputs ([Variables]): List of Variables, output of backbone stages
            targets ([Variables]): List of Variables, The targets for yolo
                                   loss calculatation.
            gt_box (Variable): The ground-truth boudding boxes.
            batch_size (int): The training batch size
            num_classes (int): class num of dataset
            mask_anchors ([[float]]): list of anchors in each output layer
            ignore_thresh (float): prediction bbox overlap any gt_box greater
                                   than ignore_thresh, objectness loss will
                                   be ignored.

        Returns:
            Type: dict
                xy_loss (Variable): YOLOv3 (x, y) coordinates loss
                wh_loss (Variable): YOLOv3 (w, h) coordinates loss
                obj_loss (Variable): YOLOv3 objectness score loss
                cls_loss (Variable): YOLOv3 classification loss

        """

        assert len(outputs) == len(targets), \
            "YOLOv3 output layer number not equal target number"

        batch_size = gt_box.shape[0]
        loss_xys, loss_whs, loss_objs, loss_clss = 0.0, 0.0, 0.0, 0.0
        if self._iou_loss is not None:
            loss_ious = 0.0
        if self._iou_aware_loss is not None:
            loss_iou_awares = 0.0
        for i, (output, target,
                anchors) in enumerate(zip(outputs, targets, mask_anchors)):
            downsample = self.downsample[i]
            an_num = len(anchors) // 2
            if self._iou_aware_loss is not None:
                ioup, output = self._split_ioup(output, an_num, num_classes)
            x, y, w, h, obj, cls = self._split_output(output, an_num,
                                                      num_classes)
            tx, ty, tw, th, tscale, tobj, tcls = self._split_target(target)

            tscale_tobj = tscale * tobj

            scale_x_y = self.scale_x_y if not isinstance(
                self.scale_x_y, Sequence) else self.scale_x_y[i]

            if (abs(scale_x_y - 1.0) < eps):
                sigmoid_x = torch.sigmoid(x)
                loss_x = tx * (0 - torch.log(sigmoid_x + 1e-9)) + (1 - tx) * (0 - torch.log(1 - sigmoid_x + 1e-9))
                loss_x *= tscale_tobj
                loss_x = loss_x.sum((1, 2, 3))
                sigmoid_y = torch.sigmoid(y)
                loss_y = ty * (0 - torch.log(sigmoid_y + 1e-9)) + (1 - ty) * (0 - torch.log(1 - sigmoid_y + 1e-9))
                loss_y *= tscale_tobj
                loss_y = loss_y.sum((1, 2, 3))
            else:
                # Grid Sensitive
                dx = scale_x_y * torch.sigmoid(x) - 0.5 * (scale_x_y - 1.0)
                dy = scale_x_y * torch.sigmoid(y) - 0.5 * (scale_x_y - 1.0)
                loss_x = torch.abs(dx - tx) * tscale_tobj
                loss_x = loss_x.sum((1, 2, 3))
                loss_y = torch.abs(dy - ty) * tscale_tobj
                loss_y = loss_y.sum((1, 2, 3))

            # NOTE: we refined loss function of (w, h) as L1Loss
            loss_w = torch.abs(w - tw) * tscale_tobj
            loss_w = loss_w.sum((1, 2, 3))
            loss_h = torch.abs(h - th) * tscale_tobj
            loss_h = loss_h.sum((1, 2, 3))
            if self._iou_loss is not None:
                loss_iou = self._iou_loss(x, y, w, h, tx, ty, tw, th, anchors,
                                          downsample, batch_size,
                                          scale_x_y)
                loss_iou = loss_iou * tscale_tobj
                loss_iou = loss_iou.sum((1, 2, 3))
                loss_ious += loss_iou.mean()

            if self._iou_aware_loss is not None:
                loss_iou_aware = self._iou_aware_loss(
                    ioup, x, y, w, h, tx, ty, tw, th, anchors, downsample,
                    batch_size, scale_x_y)
                loss_iou_aware = loss_iou_aware * tobj
                loss_iou_aware = loss_iou_aware.sum((1, 2, 3))
                loss_iou_awares += loss_iou_aware.mean()

            loss_obj_pos, loss_obj_neg = self._calc_obj_loss(
                output, obj, tobj, gt_box, batch_size, anchors,
                num_classes, downsample, self._ignore_thresh, scale_x_y)

            sigmoid_cls = torch.sigmoid(cls)
            loss_cls = tcls * (0 - torch.log(sigmoid_cls + 1e-9)) + (1 - tcls) * (0 - torch.log(1 - sigmoid_cls + 1e-9))
            loss_cls = loss_cls.sum(4)
            loss_cls *= tobj
            loss_cls = loss_cls.sum((1, 2, 3))

            loss_xys += (loss_x + loss_y).mean()
            loss_whs += (loss_w + loss_h).mean()
            loss_objs += (loss_obj_pos + loss_obj_neg).mean()
            loss_clss += loss_cls.mean()

        total_loss = loss_xys + loss_whs + loss_objs + loss_clss
        losses_all = {
            "loss_xy": loss_xys,
            "loss_wh": loss_whs,
            "loss_obj": loss_objs,
            "loss_cls": loss_clss,
        }
        if self._iou_loss is not None:
            losses_all["loss_iou"] = loss_ious
            total_loss += loss_ious
        if self._iou_aware_loss is not None:
            losses_all["loss_iou_aware"] = loss_iou_awares
            total_loss += loss_iou_awares
        losses_all["total_loss"] = total_loss
        return losses_all

    def _split_ioup(self, output, an_num, num_classes):
        """
        Split output feature map to output, predicted iou
        along channel dimension
        """
        ioup = output[:, :an_num, :, :]
        ioup = torch.sigmoid(ioup)

        oriout = output[:, an_num:, :, :]
        return (ioup, oriout)

    def _split_output(self, output, an_num, num_classes):
        """
        Split output feature map to x, y, w, h, objectness, classification
        along channel dimension
        """
        batch_size = output.shape[0]
        output_size = output.shape[2]
        output = output.reshape((batch_size, an_num, 5 + num_classes, output_size, output_size))
        x = output[:, :, 0, :, :]
        y = output[:, :, 1, :, :]
        w = output[:, :, 2, :, :]
        h = output[:, :, 3, :, :]
        obj = output[:, :, 4, :, :]
        cls = output[:, :, 5:, :, :]
        cls = cls.permute(0, 1, 3, 4, 2)
        return (x, y, w, h, obj, cls)

    def _split_target(self, target):
        """
        split target to x, y, w, h, objectness, classification
        along dimension 2

        target is in shape [N, an_num, 6 + class_num, H, W]
        """
        tx = target[:, :, 0, :, :]
        ty = target[:, :, 1, :, :]
        tw = target[:, :, 2, :, :]
        th = target[:, :, 3, :, :]

        tscale = target[:, :, 4, :, :]
        tobj = target[:, :, 5, :, :]

        tcls = target[:, :, 6:, :, :]
        tcls = tcls.permute(0, 1, 3, 4, 2)
        tcls.requires_grad = False

        return (tx, ty, tw, th, tscale, tobj, tcls)

    def _calc_obj_loss(self, output, obj, tobj, gt_box, batch_size, anchors,
                       num_classes, downsample, ignore_thresh, scale_x_y):
        # A prediction bbox overlap any gt_bbox over ignore_thresh,
        # objectness loss will be ignored, process as follows:

        _anchors = np.array(anchors)
        _anchors = np.reshape(_anchors, (-1, 2)).astype(np.float32)

        im_size = torch.ones((batch_size, 2), dtype=torch.float32, device=output.device)
        im_size.requires_grad = False
        bbox, prob = paddle_yolo_box(output, _anchors, downsample,
                                     num_classes, scale_x_y, im_size, clip_bbox=False,
                                     conf_thresh=0.0)

        # 2. split pred bbox and gt bbox by sample, calculate IoU between pred bbox
        #    and gt bbox in each sample
        ious = []
        for pred, gt in zip(bbox, gt_box):

            def box_xywh2xyxy(box):
                x = box[:, 0:1]
                y = box[:, 1:2]
                w = box[:, 2:3]
                h = box[:, 3:4]
                return torch.cat(
                    [
                        x - w / 2.,
                        y - h / 2.,
                        x + w / 2.,
                        y + h / 2.,
                    ], dim=1)

            gt = box_xywh2xyxy(gt)   # [50, 4]
            ious.append(jaccard(pred, gt).unsqueeze(0))   # [1, 3*13*13, 50]

        iou = torch.cat(ious, dim=0)   # [bz, 3*13*13, 50]   每张图片的这个输出层的所有预测框（比如3*13*13个）与所有gt（50个）两两之间的iou
        # 3. Get iou_mask by IoU between gt bbox and prediction bbox,
        #    Get obj_mask by tobj(holds gt_score), calculate objectness loss
        max_iou, _ = iou.max(-1)   # [bz, 3*13*13]   预测框与所有gt最高的iou
        iou_mask = (max_iou <= ignore_thresh).float()   # [bz, 3*13*13]   候选负样本处为1
        if self.match_score:
            max_prob, _ = prob.max(-1)   # [bz, 3*13*13]   预测框所有类别最高分数
            iou_mask = iou_mask * (max_prob <= 0.25).float()   # 最高分数低于0.25的预测框，被视作负样本或者忽略样本，虽然在训练初期该分数不可信。
        output_shape = output.shape
        an_num = len(anchors) // 2
        iou_mask = iou_mask.reshape((output_shape[0], an_num, output_shape[2], output_shape[3]))   # [bz, 3, 13, 13]   候选负样本处为1
        iou_mask.requires_grad = False

        # NOTE: tobj holds gt_score, obj_mask holds object existence mask
        obj_mask = (tobj > 0.).float()   # [bz, 3, 13, 13]  正样本处为1
        obj_mask.requires_grad = False

        # 候选负样本 中的 非正样本 才是负样本。所有样本中，正样本和负样本之外的样本是忽略样本。
        noobj_mask = (1.0 - obj_mask) * iou_mask   # [N, 3, n_grid, n_grid]  负样本处为1
        noobj_mask.requires_grad = False

        # For positive objectness grids, objectness loss should be calculated
        # For negative objectness grids, objectness loss is calculated only iou_mask == 1.0
        sigmoid_obj = torch.sigmoid(obj)
        loss_obj_pos = tobj * (0 - torch.log(sigmoid_obj + 1e-9))   # 由于有mixup增强，tobj正样本处不一定为1.0
        loss_obj_neg = noobj_mask * (0 - torch.log(1 - sigmoid_obj + 1e-9))   # 负样本的损失
        loss_obj_pos = loss_obj_pos.sum((1, 2, 3))
        loss_obj_neg = loss_obj_neg.sum((1, 2, 3))

        return loss_obj_pos, loss_obj_neg




