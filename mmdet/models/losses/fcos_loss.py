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
import torch.nn.functional as F


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
            input_channel_last = input.permute(0, 2, 3, 1)  # [N, H, W, C]
        else:
            input_channel_last = input
        input_channel_last = torch.reshape(input_channel_last, (-1, input_channel_last.shape[3]))  # [N*H*W, C]
        return input_channel_last

    def sigmoid_focal_loss(self, logits, labels, alpha=0.25, gamma=2.0, eps=1e-9):
        p = torch.sigmoid(logits)
        pos_loss = labels * (0 - torch.log(p + eps)) * torch.pow(1 - p, gamma) * alpha
        neg_loss = (1.0 - labels) * (0 - torch.log(1 - p + eps)) * torch.pow(p, gamma) * (1 - alpha)
        focal_loss = pos_loss + neg_loss
        focal_loss = focal_loss.sum()
        focal_loss = focal_loss.reshape((1, ))
        return focal_loss

    def sigmoid_cross_entropy_with_logits(self, logits, labels, eps=1e-9):
        p = torch.sigmoid(logits)
        pos_loss = labels * (0 - torch.log(p + eps))
        neg_loss = (1.0 - labels) * (0 - torch.log(1 - p + eps))
        bce_loss = pos_loss + neg_loss
        return bce_loss

    def __iou_loss(self, pred, targets, positive_mask, weights=None):
        """
        Calculate the loss for location prediction
        Args:
            pred (Tensor): bounding boxes prediction
            targets (Tensor): targets for positive samples
            positive_mask (Tensor): mask of positive samples
            weights (Tensor): weights for each positive samples
        Return:
            loss (Tensor): location loss
        """
        plw = pred[:, 0] * positive_mask
        pth = pred[:, 1] * positive_mask
        prw = pred[:, 2] * positive_mask
        pbh = pred[:, 3] * positive_mask

        tlw = targets[:, 0] * positive_mask
        tth = targets[:, 1] * positive_mask
        trw = targets[:, 2] * positive_mask
        tbh = targets[:, 3] * positive_mask
        tlw.requires_grad = False
        trw.requires_grad = False
        tth.requires_grad = False
        tbh.requires_grad = False

        ilw = torch.minimum(plw, tlw)
        irw = torch.minimum(prw, trw)
        ith = torch.minimum(pth, tth)
        ibh = torch.minimum(pbh, tbh)

        clw = torch.maximum(plw, tlw)
        crw = torch.maximum(prw, trw)
        cth = torch.maximum(pth, tth)
        cbh = torch.maximum(pbh, tbh)

        area_predict = (plw + prw) * (pth + pbh)
        area_target = (tlw + trw) * (tth + tbh)
        area_inter = (ilw + irw) * (ith + ibh)
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
            loss = 0.0 - torch.log(ious)
        else:
            raise KeyError
        if weights is not None:
            loss = loss * weights
        return loss

    def forward(self, cls_logits, bboxes_reg, centerness, tag_labels, tag_bboxes, tag_center):
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
        tag_labels_flatten_list = []
        tag_bboxes_flatten_list = []
        tag_center_flatten_list = []
        num_lvl = len(cls_logits)
        for lvl in range(num_lvl):
            cls_logits_flatten_list.append(self.__flatten_tensor(cls_logits[num_lvl - 1 - lvl], True))   # 从 小感受野 到 大感受野 遍历cls_logits
            bboxes_reg_flatten_list.append(self.__flatten_tensor(bboxes_reg[num_lvl - 1 - lvl], True))
            centerness_flatten_list.append(self.__flatten_tensor(centerness[num_lvl - 1 - lvl], True))
            tag_labels_flatten_list.append(self.__flatten_tensor(tag_labels[lvl], False))   # 从 小感受野 到 大感受野 遍历tag_labels
            tag_bboxes_flatten_list.append(self.__flatten_tensor(tag_bboxes[lvl], False))
            tag_center_flatten_list.append(self.__flatten_tensor(tag_center[lvl], False))

        # 顺序都是从 小感受野 到 大感受野
        cls_logits_flatten = torch.cat(cls_logits_flatten_list, 0)   # [批大小*所有格子数, 80]， 预测的类别
        bboxes_reg_flatten = torch.cat(bboxes_reg_flatten_list, 0)   # [批大小*所有格子数,  4]， 预测的lrtb
        centerness_flatten = torch.cat(centerness_flatten_list, 0)   # [批大小*所有格子数,  1]， 预测的centerness

        tag_labels_flatten = torch.cat(tag_labels_flatten_list, 0)   # [批大小*所有格子数,  1]， 真实的类别id
        tag_bboxes_flatten = torch.cat(tag_bboxes_flatten_list, 0)   # [批大小*所有格子数,  4]， 真实的lrtb
        tag_center_flatten = torch.cat(tag_center_flatten_list, 0)   # [批大小*所有格子数,  1]， 真实的centerness
        tag_labels_flatten.requires_grad = False
        tag_bboxes_flatten.requires_grad = False
        tag_center_flatten.requires_grad = False

        mask_positive_bool = tag_labels_flatten > 0
        mask_positive_bool.requires_grad = False
        mask_positive_float = mask_positive_bool.float()
        mask_positive_float.requires_grad = False

        num_positive_fp32 = mask_positive_float.sum()
        num_positive_fp32.requires_grad = False
        num_positive_int32 = num_positive_fp32.int()
        num_positive_int32 = num_positive_int32 * 0 + 1
        num_positive_int32.requires_grad = False

        normalize_sum = (tag_center_flatten * mask_positive_float).sum()
        normalize_sum.requires_grad = False

        # 1. cls_logits: sigmoid_focal_loss
        # expand onehot labels
        num_classes = cls_logits_flatten.shape[-1]
        tag_labels_flatten = tag_labels_flatten.squeeze(-1)
        tag_labels_flatten = tag_labels_flatten.long()
        tag_labels_flatten_bin = F.one_hot(
            tag_labels_flatten, num_classes=1 + num_classes)
        tag_labels_flatten_bin = tag_labels_flatten_bin[:, 1:]
        # sigmoid_focal_loss
        cls_loss = self.sigmoid_focal_loss(cls_logits_flatten, tag_labels_flatten_bin, self.loss_alpha, self.loss_gamma) / num_positive_fp32

        # 2. bboxes_reg: giou_loss
        mask_positive_float = mask_positive_float.squeeze(-1)
        tag_center_flatten = tag_center_flatten.squeeze(-1)
        reg_loss = self.__iou_loss(
            bboxes_reg_flatten,
            tag_bboxes_flatten,
            mask_positive_float,
            weights=tag_center_flatten)
        reg_loss = reg_loss * mask_positive_float / normalize_sum

        # 3. centerness: sigmoid_cross_entropy_with_logits_loss
        centerness_flatten = centerness_flatten.squeeze(-1)
        ctn_loss = self.sigmoid_cross_entropy_with_logits(centerness_flatten, tag_center_flatten)
        ctn_loss = ctn_loss * mask_positive_float / num_positive_fp32

        ctn_loss = ctn_loss.sum()
        cls_loss = cls_loss.sum()
        reg_loss = reg_loss.sum()
        total_loss = ctn_loss + cls_loss + reg_loss
        loss_all = {
            "total_loss": total_loss,
            "loss_centerness": ctn_loss,
            "loss_cls": cls_loss,
            "loss_box": reg_loss
        }
        return loss_all




