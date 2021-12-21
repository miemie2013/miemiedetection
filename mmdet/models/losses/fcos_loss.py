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




class FCOSLoss(nn.Module):
    """
    FCOSLoss
    Args:
        loss_alpha (float): alpha in focal loss
        loss_gamma (float): gamma in focal loss
        iou_loss_type(str): location loss type, IoU/GIoU/LINEAR_IoU
        reg_weights(float): weight for location loss
    """

    def __init__(self,
                 loss_alpha=0.25,
                 loss_gamma=2.0,
                 iou_loss_type="IoU",
                 reg_weights=1.0):
        super(FCOSLoss, self).__init__()
        self.loss_alpha = loss_alpha
        self.loss_gamma = loss_gamma
        self.iou_loss_type = iou_loss_type
        self.reg_weights = reg_weights

    def __flatten_tensor(self, input, channel_first=False):
        """
        Flatten a Tensor
        Args:
            input   (Variables): Input Tensor
            channel_first(bool): if true the dimension order of
                Tensor is [N, C, H, W], otherwise is [N, H, W, C]
        Return:
            input_channel_last (Variables): The flattened Tensor in channel_last style
        """
        if channel_first:
            input_channel_last = fluid.layers.transpose(
                input, perm=[0, 2, 3, 1])
        else:
            input_channel_last = input
        input_channel_last = fluid.layers.flatten(input_channel_last, axis=3)
        return input_channel_last

    def __iou_loss(self, pred, targets, positive_mask, weights=None):
        """
        Calculate the loss for location prediction
        Args:
            pred          (Variables): bounding boxes prediction
            targets       (Variables): targets for positive samples
            positive_mask (Variables): mask of positive samples
            weights       (Variables): weights for each positive samples
        Return:
            loss (Varialbes): location loss
        """
        positive_mask = fluid.layers.reshape(positive_mask, (-1, ))   # [批大小*所有格子数, ]
        plw = pred[:, 0] * positive_mask   # [批大小*所有格子数, ]， 预测的l
        pth = pred[:, 1] * positive_mask   # [批大小*所有格子数, ]， 预测的t
        prw = pred[:, 2] * positive_mask   # [批大小*所有格子数, ]， 预测的r
        pbh = pred[:, 3] * positive_mask   # [批大小*所有格子数, ]， 预测的b
        tlw = targets[:, 0] * positive_mask   # [批大小*所有格子数, ]， 真实的l
        tth = targets[:, 1] * positive_mask   # [批大小*所有格子数, ]， 真实的t
        trw = targets[:, 2] * positive_mask   # [批大小*所有格子数, ]， 真实的r
        tbh = targets[:, 3] * positive_mask   # [批大小*所有格子数, ]， 真实的b
        tlw.stop_gradient = True
        trw.stop_gradient = True
        tth.stop_gradient = True
        tbh.stop_gradient = True
        area_target = (tlw + trw) * (tth + tbh)      # [批大小*所有格子数, ]， 真实的面积
        area_predict = (plw + prw) * (pth + pbh)     # [批大小*所有格子数, ]， 预测的面积
        ilw = fluid.layers.elementwise_min(plw, tlw)   # [批大小*所有格子数, ]， 相交矩形的l
        irw = fluid.layers.elementwise_min(prw, trw)   # [批大小*所有格子数, ]， 相交矩形的r
        ith = fluid.layers.elementwise_min(pth, tth)   # [批大小*所有格子数, ]， 相交矩形的t
        ibh = fluid.layers.elementwise_min(pbh, tbh)   # [批大小*所有格子数, ]， 相交矩形的b
        clw = fluid.layers.elementwise_max(plw, tlw)   # [批大小*所有格子数, ]， 包围矩形的l
        crw = fluid.layers.elementwise_max(prw, trw)   # [批大小*所有格子数, ]， 包围矩形的r
        cth = fluid.layers.elementwise_max(pth, tth)   # [批大小*所有格子数, ]， 包围矩形的t
        cbh = fluid.layers.elementwise_max(pbh, tbh)   # [批大小*所有格子数, ]， 包围矩形的b
        area_inter = (ilw + irw) * (ith + ibh)   # [批大小*所有格子数, ]， 相交矩形的面积
        ious = (area_inter + 1.0) / (
            area_predict + area_target - area_inter + 1.0)
        ious = ious * positive_mask
        if self.iou_loss_type.lower() == "linear_iou":
            loss = 1.0 - ious
        elif self.iou_loss_type.lower() == "giou":
            area_uniou = area_predict + area_target - area_inter
            area_circum = (clw + crw) * (cth + cbh) + 1e-7
            giou = ious - (area_circum - area_uniou) / area_circum
            loss = 1.0 - giou
        elif self.iou_loss_type.lower() == "iou":
            loss = 0.0 - fluid.layers.log(ious)
        elif self.iou_loss_type.lower() == "ciou":
            # 预测的矩形。cx_cy_w_h格式，以格子中心点为坐标原点。
            pred_cx = (prw - plw) * 0.5
            pred_cy = (pbh - pth) * 0.5
            pred_w = (plw + prw)
            pred_h = (pth + pbh)
            pred_cx = L.reshape(pred_cx, (-1, 1))
            pred_cy = L.reshape(pred_cy, (-1, 1))
            pred_w = L.reshape(pred_w, (-1, 1))
            pred_h = L.reshape(pred_h, (-1, 1))
            pred_cx_cy_w_h = L.concat([pred_cx, pred_cy, pred_w, pred_h], -1)   # [批大小*所有格子数, 4]

            # 真实的矩形。cx_cy_w_h格式，以格子中心点为坐标原点。
            true_cx = (trw - tlw) * 0.5
            true_cy = (tbh - tth) * 0.5
            true_w = (tlw + trw)
            true_h = (tth + tbh)
            true_cx = L.reshape(true_cx, (-1, 1))
            true_cy = L.reshape(true_cy, (-1, 1))
            true_w = L.reshape(true_w, (-1, 1))
            true_h = L.reshape(true_h, (-1, 1))
            true_cx_cy_w_h = L.concat([true_cx, true_cy, true_w, true_h], -1)   # [批大小*所有格子数, 4]

            # 预测的矩形。x0y0x1y1格式，以格子中心点为坐标原点。
            boxes1_x0y0x1y1 = L.concat([pred_cx_cy_w_h[:, :2] - pred_cx_cy_w_h[:, 2:] * 0.5,
                                        pred_cx_cy_w_h[:, :2] + pred_cx_cy_w_h[:, 2:] * 0.5], axis=-1)

            # 真实的矩形。x0y0x1y1格式，以格子中心点为坐标原点。
            boxes2_x0y0x1y1 = L.concat([true_cx_cy_w_h[:, :2] - true_cx_cy_w_h[:, 2:] * 0.5,
                                        true_cx_cy_w_h[:, :2] + true_cx_cy_w_h[:, 2:] * 0.5], axis=-1)

            # 包围矩形的左上角坐标、右下角坐标，shape 都是 (批大小*所有格子数, 2)
            enclose_left_up = L.elementwise_min(boxes1_x0y0x1y1[:, :2], boxes2_x0y0x1y1[:, :2])
            enclose_right_down = L.elementwise_max(boxes1_x0y0x1y1[:, 2:], boxes2_x0y0x1y1[:, 2:])

            # 包围矩形的对角线的平方
            enclose_wh = enclose_right_down - enclose_left_up
            enclose_c2 = L.pow(enclose_wh[:, 0], 2) + L.pow(enclose_wh[:, 1], 2)

            # 两矩形中心点距离的平方
            p2 = L.pow(pred_cx_cy_w_h[:, 0] - true_cx_cy_w_h[:, 0], 2) \
                 + L.pow(pred_cx_cy_w_h[:, 1] - true_cx_cy_w_h[:, 1], 2)

            # 增加av。加上除0保护防止nan。
            atan1 = L.atan(pred_cx_cy_w_h[:, 2] / (pred_cx_cy_w_h[:, 3] + 1e-9))
            atan2 = L.atan(true_cx_cy_w_h[:, 2] / (true_cx_cy_w_h[:, 3] + 1e-9))
            v = 4.0 * L.pow(atan1 - atan2, 2) / (math.pi ** 2)
            a = v / (1 - ious + v)
            ciou = ious - 1.0 * p2 / (enclose_c2 + 1e-9) - 1.0 * a * v
            loss = 1.0 - ciou
        else:
            raise KeyError
        loss = fluid.layers.reshape(loss, (-1, 1))   # [批大小*所有格子数, 1]
        if weights is not None:
            loss = loss * weights
        return loss

    def __iou(self, pred, targets, positive_mask, weights=None):
        """
        Calculate the loss for location prediction
        Args:
            pred          (Variables): bounding boxes prediction
            targets       (Variables): targets for positive samples
            positive_mask (Variables): mask of positive samples
            weights       (Variables): weights for each positive samples
        Return:
            loss (Varialbes): location loss
        """
        positive_mask = fluid.layers.reshape(positive_mask, (-1, ))   # [批大小*所有格子数, ]
        plw = pred[:, 0] * positive_mask   # [批大小*所有格子数, ]， 预测的l
        pth = pred[:, 1] * positive_mask   # [批大小*所有格子数, ]， 预测的t
        prw = pred[:, 2] * positive_mask   # [批大小*所有格子数, ]， 预测的r
        pbh = pred[:, 3] * positive_mask   # [批大小*所有格子数, ]， 预测的b
        tlw = targets[:, 0] * positive_mask   # [批大小*所有格子数, ]， 真实的l
        tth = targets[:, 1] * positive_mask   # [批大小*所有格子数, ]， 真实的t
        trw = targets[:, 2] * positive_mask   # [批大小*所有格子数, ]， 真实的r
        tbh = targets[:, 3] * positive_mask   # [批大小*所有格子数, ]， 真实的b
        tlw.stop_gradient = True
        trw.stop_gradient = True
        tth.stop_gradient = True
        tbh.stop_gradient = True
        area_target = (tlw + trw) * (tth + tbh)      # [批大小*所有格子数, ]， 真实的面积
        area_predict = (plw + prw) * (pth + pbh)     # [批大小*所有格子数, ]， 预测的面积
        ilw = fluid.layers.elementwise_min(plw, tlw)   # [批大小*所有格子数, ]， 相交矩形的l
        irw = fluid.layers.elementwise_min(prw, trw)   # [批大小*所有格子数, ]， 相交矩形的r
        ith = fluid.layers.elementwise_min(pth, tth)   # [批大小*所有格子数, ]， 相交矩形的t
        ibh = fluid.layers.elementwise_min(pbh, tbh)   # [批大小*所有格子数, ]， 相交矩形的b
        area_inter = (ilw + irw) * (ith + ibh)   # [批大小*所有格子数, ]， 相交矩形的面积
        ious = (area_inter + 1.0) / (
            area_predict + area_target - area_inter + 1.0)
        ious = ious * positive_mask
        ious = fluid.layers.reshape(ious, (-1, 1))   # [批大小*所有格子数, 1]
        return ious

    def __call__(self, cls_logits, bboxes_reg, centerness, iou_awares, tag_labels,
                 tag_bboxes, tag_center):
        """
        Calculate the loss for classification, location and centerness
        Args:
            cls_logits (list): 预测结果list。里面每个元素是[N, 80, 格子行数, 格子列数]     从 大感受野 到 小感受野
            bboxes_reg (list): 预测结果list。里面每个元素是[N,  4, 格子行数, 格子列数]     从 大感受野 到 小感受野
            centerness (list): 预测结果list。里面每个元素是[N,  1, 格子行数, 格子列数]     从 大感受野 到 小感受野
            tag_labels (list): 真实标签list。里面每个元素是[N, 格子行数, 格子列数,  1]     从 小感受野 到 大感受野
            tag_bboxes (list): 真实标签list。里面每个元素是[N, 格子行数, 格子列数,  4]     从 小感受野 到 大感受野
            tag_center (list): 真实标签list。里面每个元素是[N, 格子行数, 格子列数,  1]     从 小感受野 到 大感受野
        Return:
            loss (dict): loss composed by classification loss, bounding box
        """
        cls_logits_flatten_list = []
        bboxes_reg_flatten_list = []
        centerness_flatten_list = []
        iou_awares_flatten_list = []
        self.use_iou_aware = iou_awares[0] is not None
        tag_labels_flatten_list = []
        tag_bboxes_flatten_list = []
        tag_center_flatten_list = []
        num_lvl = len(cls_logits)
        for lvl in range(num_lvl):
            cls_logits_flatten_list.append(
                self.__flatten_tensor(cls_logits[num_lvl - 1 - lvl], True))   # 从 小感受野 到 大感受野 遍历cls_logits
            bboxes_reg_flatten_list.append(
                self.__flatten_tensor(bboxes_reg[num_lvl - 1 - lvl], True))
            centerness_flatten_list.append(
                self.__flatten_tensor(centerness[num_lvl - 1 - lvl], True))
            if self.use_iou_aware:
                iou_awares_flatten_list.append(
                    self.__flatten_tensor(iou_awares[num_lvl - 1 - lvl], True))
            tag_labels_flatten_list.append(
                self.__flatten_tensor(tag_labels[lvl], False))   # 从 小感受野 到 大感受野 遍历tag_labels
            tag_bboxes_flatten_list.append(
                self.__flatten_tensor(tag_bboxes[lvl], False))
            tag_center_flatten_list.append(
                self.__flatten_tensor(tag_center[lvl], False))

        # 顺序都是从 小感受野 到 大感受野
        cls_logits_flatten = fluid.layers.concat(   # [批大小*所有格子数, 80]， 预测的类别
            cls_logits_flatten_list, axis=0)
        bboxes_reg_flatten = fluid.layers.concat(   # [批大小*所有格子数,  4]， 预测的lrtb
            bboxes_reg_flatten_list, axis=0)
        centerness_flatten = fluid.layers.concat(   # [批大小*所有格子数,  1]， 预测的centerness
            centerness_flatten_list, axis=0)
        iou_awares_flatten = None
        if self.use_iou_aware:
            iou_awares_flatten = fluid.layers.concat(   # [批大小*所有格子数,  1]， 预测的iou_aware
                iou_awares_flatten_list, axis=0)
        tag_labels_flatten = fluid.layers.concat(   # [批大小*所有格子数,  1]， 真实的类别id
            tag_labels_flatten_list, axis=0)
        tag_bboxes_flatten = fluid.layers.concat(   # [批大小*所有格子数,  4]， 真实的lrtb
            tag_bboxes_flatten_list, axis=0)
        tag_center_flatten = fluid.layers.concat(   # [批大小*所有格子数,  1]， 真实的centerness
            tag_center_flatten_list, axis=0)

        mask_positive = tag_labels_flatten > 0        # [批大小*所有格子数,  1]， 正样本处为True
        mask_positive.stop_gradient = True
        mask_positive_float = fluid.layers.cast(mask_positive, dtype="float32")   # [批大小*所有格子数,  1]， 正样本处为1
        mask_positive_float.stop_gradient = True
        num_positive_fp32 = fluid.layers.reduce_sum(mask_positive_float)   # 这一批的正样本数
        num_positive_int32 = fluid.layers.cast(num_positive_fp32, dtype="int32")
        num_positive_int32 = num_positive_int32 * 0 + 1
        num_positive_fp32.stop_gradient = True
        num_positive_int32.stop_gradient = True
        normalize_sum = fluid.layers.sum(tag_center_flatten)
        normalize_sum.stop_gradient = True
        normalize_sum = fluid.layers.reduce_sum(mask_positive_float * normalize_sum)   # 正样本的centerness求和
        normalize_sum.stop_gradient = True

        cls_loss = fluid.layers.sigmoid_focal_loss(
            cls_logits_flatten, tag_labels_flatten,
            num_positive_int32) / fluid.layers.elementwise_max(fluid.layers.ones((1, ), dtype='float32'), num_positive_fp32)   # 当没有gt时，即num_positive_fp32==0时，focal_loss什么都不除。
        reg_loss = self.__iou_loss(
            bboxes_reg_flatten, tag_bboxes_flatten, mask_positive_float,
            tag_center_flatten) * mask_positive_float / (normalize_sum + 1e-9)
        ctn_loss = fluid.layers.sigmoid_cross_entropy_with_logits(
            x=centerness_flatten,
            label=tag_center_flatten) * mask_positive_float / (num_positive_fp32 + 1e-9)

        loss_all = {
            "loss_centerness": fluid.layers.reduce_sum(ctn_loss),
            "loss_cls": fluid.layers.reduce_sum(cls_loss),
            "loss_box": fluid.layers.reduce_sum(reg_loss)
        }

        iouk = self.__iou(bboxes_reg_flatten, tag_bboxes_flatten, mask_positive_float,
                          tag_center_flatten)
        if self.use_iou_aware:
            ioup = L.sigmoid(iou_awares_flatten)
            iouk.stop_gradient = True
            loss_iou_aware = fluid.layers.cross_entropy(ioup, iouk, soft_label=True)
            loss_iou_aware = loss_iou_aware * mask_positive_float
            loss_all["loss_iou_aware"] = fluid.layers.reduce_sum(loss_iou_aware)
        return loss_all




