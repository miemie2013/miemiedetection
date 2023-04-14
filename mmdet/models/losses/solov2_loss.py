#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date:
#   Description :
#
# ================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F


class SOLOv2Loss(object):
    """
    SOLOv2Loss
    Args:
        ins_loss_weight (float): Weight of instance loss.
        focal_loss_gamma (float): Gamma parameter for focal loss.
        focal_loss_alpha (float): Alpha parameter for focal loss.
    """

    def __init__(self,
                 ins_loss_weight=3.0,
                 focal_loss_gamma=2.0,
                 focal_loss_alpha=0.25):
        self.ins_loss_weight = ins_loss_weight
        self.focal_loss_gamma = focal_loss_gamma
        self.focal_loss_alpha = focal_loss_alpha

    def _dice_loss(self, input, target):
        input = torch.reshape(input, shape=(input.shape[0], -1))
        target = torch.reshape(target, shape=(target.shape[0], -1))
        a = torch.sum(input * target, axis=1)
        b = torch.sum(input * input, axis=1) + 0.001
        c = torch.sum(target * target, axis=1) + 0.001
        d = (2 * a) / (b + c)
        return 1 - d

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

    def __call__(self, ins_pred_list, ins_label_list, cate_preds, cate_labels,
                 num_ins):
        """
        Get loss of network of SOLOv2.
        Args:
            ins_pred_list (list): Variable list of instance branch output.
            ins_label_list (list): List of instance labels pre batch.
            cate_preds (list): Concat Variable list of categroy branch output.
            cate_labels (list): Concat list of categroy labels pre batch.
            num_ins (int): Number of positive samples in a mini-batch.
        Returns:
            loss_ins (Variable): The instance loss Variable of SOLOv2 network.
            loss_cate (Variable): The category loss Variable of SOLOv2 network.
        """

        #1. Ues dice_loss to calculate instance loss
        loss_ins = []
        total_weights = torch.zeros([1], dtype=torch.float32, device=num_ins.device)
        for input, target in zip(ins_pred_list, ins_label_list):
            if input is None:
                continue
            target = target.to(torch.float32)
            target = torch.reshape(target, shape=[-1, input.shape[-2], input.shape[-1]])
            weights = (torch.sum(target, axis=[1, 2]) > 0).to(torch.float32)
            input = torch.sigmoid(input)
            dice_loss = self._dice_loss(input, target)
            dice_out = dice_loss * weights
            total_weights += torch.sum(weights)
            loss_ins.append(dice_out)

        if len(loss_ins) == 0:
            loss_ins = torch.zeros([]).to(num_ins.device)
        else:
            loss_ins = torch.sum(torch.cat(loss_ins)) / total_weights
            loss_ins = loss_ins * self.ins_loss_weight

        #2. Ues sigmoid_focal_loss to calculate category loss
        # expand onehot labels
        num_classes = cate_preds.shape[-1]
        cate_labels = cate_labels.to(torch.int64)
        cate_labels_bin = F.one_hot(cate_labels, num_classes + 1)
        cate_labels_bin = cate_labels_bin[:, 1:]

        loss_cate = self.sigmoid_focal_loss(
            cate_preds,
            labels=cate_labels_bin,
            gamma=self.focal_loss_gamma,
            alpha=self.focal_loss_alpha)
        loss_cate = loss_cate / (num_ins + 1.)

        return loss_ins, loss_cate



