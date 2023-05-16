#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date:
#   Description :
#
# ================================================================


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.initializer import kaiming_uniform_, constant_
from mmdet.models.losses.iou_losses import GIoULoss



def feature_norm(feat):
    # Normalize the feature maps to have zero mean and unit variances.
    assert len(feat.shape) == 4
    N, C, H, W = feat.shape
    feat = feat.permute([1, 0, 2, 3]).reshape([C, -1])
    mean = feat.mean(-1, keepdim=True)
    std = feat.std(-1, keepdim=True)
    feat = (feat - mean) / (std + 1e-6)
    return feat.reshape([C, N, H, W]).permute([1, 0, 2, 3])



class KnowledgeDistillationKLDivLoss(nn.Module):
    """Loss function for knowledge distilling using KL divergence.

    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
        T (int): Temperature for distillation.
    """

    def __init__(self, reduction='mean', loss_weight=1.0, T=10):
        super(KnowledgeDistillationKLDivLoss, self).__init__()
        assert reduction in ('none', 'mean', 'sum')
        assert T >= 1
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.T = T

    def knowledge_distillation_kl_div_loss(self,
                                           pred,
                                           soft_label,
                                           T,
                                           detach_target=True):
        r"""Loss function for knowledge distilling using KL divergence.

        Args:
            pred (Tensor): Predicted logits with shape (N, n + 1).
            soft_label (Tensor): Target logits with shape (N, N + 1).
            T (int): Temperature for distillation.
            detach_target (bool): Remove soft_label from automatic differentiation
        """
        assert pred.shape == soft_label.shape
        target = F.softmax(soft_label / T, dim=1)
        if detach_target:
            target = target.detach()

        kd_loss = F.kl_div(
            F.log_softmax(
                pred / T, dim=1), target, reduction='none').mean(1) * (T * T)

        return kd_loss

    def forward(self,
                pred,
                soft_label,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (Tensor): Predicted logits with shape (N, n + 1).
            soft_label (Tensor): Target logits with shape (N, N + 1).
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')

        reduction = (reduction_override
                     if reduction_override else self.reduction)

        loss_kd_out = self.knowledge_distillation_kl_div_loss(
            pred, soft_label, T=self.T)

        if weight is not None:
            loss_kd_out = weight * loss_kd_out

        if avg_factor is None:
            if reduction == 'none':
                loss = loss_kd_out
            elif reduction == 'mean':
                loss = loss_kd_out.mean()
            elif reduction == 'sum':
                loss = loss_kd_out.sum()
        else:
            # if reduction is mean, then average the loss by avg_factor
            if reduction == 'mean':
                loss = loss_kd_out.sum() / avg_factor
            # if reduction is 'none', then do nothing, otherwise raise an error
            elif reduction != 'none':
                raise ValueError(
                    'avg_factor can not be used with reduction="sum"')

        loss_kd = self.loss_weight * loss
        return loss_kd


class DistillPPYOLOELoss(nn.Module):
    def __init__(
            self,
            loss_weight={'logits': 4.0, 'feat': 1.0},
            logits_distill=True,
            logits_loss_weight={'class': 1.0, 'iou': 2.5, 'dfl': 0.5},
            logits_ld_distill=False,
            logits_ld_params={'weight': 20000, 'T': 10},
            feat_distill=True,
            feat_distiller='fgd',
            feat_distill_place='neck_feats',
            teacher_width_mult=1.0,  # L
            student_width_mult=0.75,  # M
            feat_out_channels=[768, 384, 192]):
        super(DistillPPYOLOELoss, self).__init__()
        self.loss_weight_logits = loss_weight['logits']
        self.loss_weight_feat = loss_weight['feat']
        self.logits_distill = logits_distill
        self.logits_ld_distill = logits_ld_distill
        self.feat_distill = feat_distill

        if logits_distill and self.loss_weight_logits > 0:
            self.bbox_loss_weight = logits_loss_weight['iou']
            self.dfl_loss_weight = logits_loss_weight['dfl']
            self.qfl_loss_weight = logits_loss_weight['class']
            self.loss_bbox = GIoULoss()

        if logits_ld_distill:
            self.loss_kd = KnowledgeDistillationKLDivLoss(
                loss_weight=logits_ld_params['weight'], T=logits_ld_params['T'])

        if feat_distill and self.loss_weight_feat > 0:
            assert feat_distiller in ['cwd', 'fgd', 'pkd', 'mgd', 'mimic']
            assert feat_distill_place in ['backbone_feats', 'neck_feats']
            self.feat_distill_place = feat_distill_place
            self.t_channel_list = [
                int(c * teacher_width_mult) for c in feat_out_channels
            ]
            self.s_channel_list = [
                int(c * student_width_mult) for c in feat_out_channels
            ]
            self.distill_feat_loss_modules = nn.ModuleList()
            for i in range(len(feat_out_channels)):
                if feat_distiller == 'cwd':
                    raise ValueError
                elif feat_distiller == 'fgd':
                    feat_loss_module = FGDFeatureLoss(
                        student_channels=self.s_channel_list[i],
                        teacher_channels=self.t_channel_list[i],
                        normalize=True,
                        alpha_fgd=0.00001,
                        beta_fgd=0.000005,
                        gamma_fgd=0.00001,
                        lambda_fgd=0.00000005)
                elif feat_distiller == 'pkd':
                    raise ValueError
                elif feat_distiller == 'mgd':
                    raise ValueError
                elif feat_distiller == 'mimic':
                    raise ValueError
                else:
                    raise ValueError
                self.distill_feat_loss_modules.append(feat_loss_module)

    def quality_focal_loss(self,
                           pred_logits,
                           soft_target_logits,
                           beta=2.0,
                           use_sigmoid=False,
                           num_total_pos=None):
        if use_sigmoid:
            func = F.binary_cross_entropy_with_logits
            soft_target = F.sigmoid(soft_target_logits)
            pred_sigmoid = F.sigmoid(pred_logits)
            preds = pred_logits
        else:
            func = F.binary_cross_entropy
            soft_target = soft_target_logits
            pred_sigmoid = pred_logits
            preds = pred_sigmoid

        scale_factor = pred_sigmoid - soft_target
        loss = func(
            preds, soft_target, reduction='none') * scale_factor.abs().pow(beta)
        loss = loss.sum(1)

        if num_total_pos is not None:
            loss = loss.sum() / num_total_pos
        else:
            loss = loss.mean()
        return loss

    def bbox_loss(self, s_bbox, t_bbox, weight_targets=None):
        # [x,y,w,h]
        if weight_targets is not None:
            loss = torch.sum(self.loss_bbox(s_bbox, t_bbox) * weight_targets)
            avg_factor = weight_targets.sum()
            loss = loss / avg_factor
        else:
            loss = torch.mean(self.loss_bbox(s_bbox, t_bbox))
        return loss

    def distribution_focal_loss(self,
                                pred_corners,
                                target_corners,
                                weight_targets=None):
        target_corners_label = F.softmax(target_corners, dim=-1)
        loss_dfl = F.cross_entropy(
            pred_corners,
            target_corners_label,
            reduction='none')

        if weight_targets is not None:
            loss_dfl = loss_dfl * (weight_targets.expand([-1, 4]).reshape([-1]))
            loss_dfl = loss_dfl.sum(-1) / weight_targets.sum()
        else:
            loss_dfl = loss_dfl.mean(-1)
        return loss_dfl / 4.0  # 4 direction

    def main_kd(self, mask_positive, pred_scores, soft_cls, num_classes):
        num_pos = mask_positive.sum()
        if num_pos > 0:
            cls_mask = mask_positive.unsqueeze(-1).tile([1, 1, num_classes])
            pred_scores_pos = pred_scores[cls_mask].reshape([-1, num_classes])
            soft_cls_pos = soft_cls[cls_mask].reshape([-1, num_classes])
            loss_kd = self.loss_kd(pred_scores_pos, soft_cls_pos, avg_factor=num_pos)
        else:
            loss_kd = torch.zeros([1])
        return loss_kd

    def forward(self, teacher_model, student_model, im_shape, gt_bbox, pad_gt_mask):
        teacher_distill_pairs = teacher_model.yolo_head.distill_pairs
        student_distill_pairs = student_model.yolo_head.distill_pairs
        if self.logits_distill and self.loss_weight_logits > 0:
            distill_bbox_loss, distill_dfl_loss, distill_cls_loss = 0., 0., 0.

            distill_cls_loss += self.quality_focal_loss(
                    student_distill_pairs['pred_cls_scores'].reshape(
                        (-1, student_distill_pairs['pred_cls_scores'].shape[-1]
                         )),
                    teacher_distill_pairs['pred_cls_scores'].detach().reshape(
                        (-1, teacher_distill_pairs['pred_cls_scores'].shape[-1]
                         )),
                    num_total_pos=student_distill_pairs['pos_num'],
                    use_sigmoid=False)
            if 'pred_bboxes_pos' in student_distill_pairs and \
                    'pred_bboxes_pos' in teacher_distill_pairs and \
                    'bbox_weight' in student_distill_pairs:
                distill_bbox_loss += self.bbox_loss(student_distill_pairs['pred_bboxes_pos'],
                               teacher_distill_pairs['pred_bboxes_pos'].detach(),
                               weight_targets=student_distill_pairs['bbox_weight']
                               )
            else:
                distill_bbox_loss += torch.zeros([1])

            if 'pred_dist_pos' in student_distill_pairs and \
                     'pred_dist_pos' in teacher_distill_pairs and \
                     'bbox_weight' in student_distill_pairs:
                distill_dfl_loss += self.distribution_focal_loss(
                    student_distill_pairs['pred_dist_pos'].reshape(
                        (-1, student_distill_pairs['pred_dist_pos'].shape[-1])),
                    teacher_distill_pairs['pred_dist_pos'].detach().reshape(
                        (-1, teacher_distill_pairs['pred_dist_pos'].shape[-1])), \
                    weight_targets=student_distill_pairs['bbox_weight']
                )
            else:
                distill_dfl_loss += torch.zeros([1])

            logits_loss = distill_bbox_loss * self.bbox_loss_weight + distill_cls_loss * self.qfl_loss_weight + distill_dfl_loss * self.dfl_loss_weight

            if self.logits_ld_distill:
                loss_kd = self.main_kd(
                    student_distill_pairs['mask_positive_select'],
                    student_distill_pairs['pred_cls_scores'],
                    teacher_distill_pairs['pred_cls_scores'],
                    student_model.yolo_head.num_classes, )
                logits_loss += loss_kd
        else:
            logits_loss = torch.zeros([1])

        if self.feat_distill and self.loss_weight_feat > 0:
            feat_loss = 0.0
            assert self.feat_distill_place in student_distill_pairs
            assert self.feat_distill_place in teacher_distill_pairs
            stu_feats = student_distill_pairs[self.feat_distill_place]
            tea_feats = teacher_distill_pairs[self.feat_distill_place]
            for i, loss_module in enumerate(self.distill_feat_loss_modules):
                feat_loss += loss_module(stu_feats[i], tea_feats[i], im_shape, gt_bbox, pad_gt_mask)
        else:
            feat_loss = torch.zeros([1])

        student_model.yolo_head.distill_pairs.clear()
        teacher_model.yolo_head.distill_pairs.clear()
        return logits_loss * self.loss_weight_logits, feat_loss * self.loss_weight_feat


class FGDFeatureLoss(nn.Module):
    """
    Focal and Global Knowledge Distillation for Detectors
    The code is reference from https://github.com/yzd-v/FGD/blob/master/mmdet/distillation/losses/fgd.py

    Args:
        student_channels (int): The number of channels in the student's FPN feature map. Default to 256.
        teacher_channels (int): The number of channels in the teacher's FPN feature map. Default to 256.
        normalize (bool): Whether to normalize the feature maps.
        temp (float, optional): The temperature coefficient. Defaults to 0.5.
        alpha_fgd (float, optional): The weight of fg_loss. Defaults to 0.001
        beta_fgd (float, optional): The weight of bg_loss. Defaults to 0.0005
        gamma_fgd (float, optional): The weight of mask_loss. Defaults to 0.001
        lambda_fgd (float, optional): The weight of relation_loss. Defaults to 0.000005
    """

    def __init__(self,
                 student_channels,
                 teacher_channels,
                 normalize=False,
                 loss_weight=1.0,
                 temp=0.5,
                 alpha_fgd=0.001,
                 beta_fgd=0.0005,
                 gamma_fgd=0.001,
                 lambda_fgd=0.000005):
        super(FGDFeatureLoss, self).__init__()
        self.normalize = normalize
        self.loss_weight = loss_weight
        self.temp = temp
        self.alpha_fgd = alpha_fgd
        self.beta_fgd = beta_fgd
        self.gamma_fgd = gamma_fgd
        self.lambda_fgd = lambda_fgd

        if student_channels != teacher_channels:
            self.align = nn.Conv2d(
                student_channels,
                teacher_channels,
                kernel_size=1,
                stride=1,
                padding=0)
            kaiming_uniform_(self.align.weight)
            student_channels = teacher_channels
        else:
            self.align = None

        self.conv_mask_s = nn.Conv2d(student_channels, 1, kernel_size=1)
        self.conv_mask_t = nn.Conv2d(teacher_channels, 1, kernel_size=1)
        kaiming_uniform_(self.conv_mask_s.weight)
        kaiming_uniform_(self.conv_mask_t.weight)

        self.stu_conv_block = nn.Sequential(
            nn.Conv2d(
                student_channels,
                student_channels // 2,
                kernel_size=1),
            nn.LayerNorm([student_channels // 2, 1, 1]),
            nn.ReLU(),
            nn.Conv2d(
                student_channels // 2,
                student_channels,
                kernel_size=1))
        constant_(self.stu_conv_block[0].weight, 0.0)
        constant_(self.stu_conv_block[-1].weight, 0.0)
        self.tea_conv_block = nn.Sequential(
            nn.Conv2d(
                teacher_channels,
                teacher_channels // 2,
                kernel_size=1),
            nn.LayerNorm([teacher_channels // 2, 1, 1]),
            nn.ReLU(),
            nn.Conv2d(
                teacher_channels // 2,
                teacher_channels,
                kernel_size=1))
        constant_(self.tea_conv_block[0].weight, 0.0)
        constant_(self.tea_conv_block[-1].weight, 0.0)

    def spatial_channel_attention(self, x, t=0.5):
        N, C, H, W = x.shape
        _f = torch.abs(x)
        spatial_map = torch.reshape(torch.mean(_f, dim=1, keepdim=True) / t, [N, -1])
        spatial_map = F.softmax(spatial_map, dim=1, dtype=torch.float32) * H * W
        spatial_att = torch.reshape(spatial_map, [N, H, W])

        channel_map = torch.mean(torch.mean(_f, dim=2, keepdim=False), dim=2, keepdim=False)
        channel_att = F.softmax(channel_map / t, dim=1, dtype=torch.float32) * C
        return [spatial_att, channel_att]

    def spatial_pool(self, x, mode="teacher"):
        batch, channel, width, height = x.shape
        x_copy = x
        x_copy = torch.reshape(x_copy, [batch, channel, height * width])
        x_copy = x_copy.unsqueeze(1)
        if mode.lower() == "student":
            context_mask = self.conv_mask_s(x)
        else:
            context_mask = self.conv_mask_t(x)

        context_mask = torch.reshape(context_mask, [batch, 1, height * width])
        context_mask = F.softmax(context_mask, dim=2)
        context_mask = context_mask.unsqueeze(-1)
        context = torch.matmul(x_copy, context_mask)
        context = torch.reshape(context, [batch, channel, 1, 1])
        return context

    def mask_loss(self, stu_channel_att, tea_channel_att, stu_spatial_att,
                  tea_spatial_att):
        def _func(a, b):
            return torch.sum(torch.abs(a - b)) / len(a)

        mask_loss = _func(stu_channel_att, tea_channel_att) + _func(
            stu_spatial_att, tea_spatial_att)
        return mask_loss

    def feature_loss(self, stu_feature, tea_feature, mask_fg, mask_bg,
                     tea_channel_att, tea_spatial_att):
        mask_fg = mask_fg.unsqueeze(1)
        mask_bg = mask_bg.unsqueeze(1)
        tea_channel_att = tea_channel_att.unsqueeze(-1).unsqueeze(-1)
        tea_spatial_att = tea_spatial_att.unsqueeze(1)

        fea_t = torch.multiply(tea_feature, tea_spatial_att.sqrt())
        fea_t = torch.multiply(fea_t, tea_channel_att.sqrt())
        fg_fea_t = torch.multiply(fea_t, mask_fg.sqrt())
        bg_fea_t = torch.multiply(fea_t, mask_bg.sqrt())

        fea_s = torch.multiply(stu_feature, tea_spatial_att.sqrt())
        fea_s = torch.multiply(fea_s, tea_channel_att.sqrt())
        fg_fea_s = torch.multiply(fea_s, mask_fg.sqrt())
        bg_fea_s = torch.multiply(fea_s, mask_bg.sqrt())

        fg_loss = F.mse_loss(fg_fea_s, fg_fea_t, reduction="sum") / len(mask_fg)
        bg_loss = F.mse_loss(bg_fea_s, bg_fea_t, reduction="sum") / len(mask_bg)
        return fg_loss, bg_loss

    def relation_loss(self, stu_feature, tea_feature):
        context_s = self.spatial_pool(stu_feature, "student")
        context_t = self.spatial_pool(tea_feature, "teacher")
        out_s = stu_feature + self.stu_conv_block(context_s)
        out_t = tea_feature + self.tea_conv_block(context_t)
        rela_loss = F.mse_loss(out_s, out_t, reduction="sum") / len(out_s)
        return rela_loss

    def mask_value(self, mask, xl, xr, yl, yr, value):
        mask[xl:xr, yl:yr] = torch.maximum(mask[xl:xr, yl:yr], value)
        return mask

    def forward(self, stu_feature, tea_feature, im_shape, gt_bboxes, pad_gt_mask):
        assert stu_feature.shape[-2:] == stu_feature.shape[-2:]
        # only distill feature with labeled GTbox  只对有gt的图片蒸馏
        num_boxes = pad_gt_mask.sum([1, 2])
        has_gt = num_boxes > 0.
        stu_feature = stu_feature[has_gt]
        tea_feature = tea_feature[has_gt]
        gt_bboxes = gt_bboxes[has_gt]
        # num_max_boxes = num_boxes.max().to(torch.int32)
        # pad_gt_mask = pad_gt_mask[:, :num_max_boxes, :]

        if self.align is not None:
            stu_feature = self.align(stu_feature)

        if self.normalize:
            stu_feature = feature_norm(stu_feature)
            tea_feature = feature_norm(tea_feature)

        tea_spatial_att, tea_channel_att = self.spatial_channel_attention(tea_feature, self.temp)
        stu_spatial_att, stu_channel_att = self.spatial_channel_attention(stu_feature, self.temp)

        mask_fg = torch.zeros(tea_spatial_att.shape)
        mask_bg = torch.ones_like(tea_spatial_att)
        one_tmp = torch.ones([*tea_spatial_att.shape[1:]])
        zero_tmp = torch.zeros([*tea_spatial_att.shape[1:]])
        mask_fg.requires_grad = False
        mask_bg.requires_grad = False
        one_tmp.requires_grad = False
        zero_tmp.requires_grad = False
        mask_fg = mask_fg.to(tea_spatial_att.device)
        one_tmp = one_tmp.to(tea_spatial_att.device)
        zero_tmp = zero_tmp.to(tea_spatial_att.device)

        wmin, wmax, hmin, hmax = [], [], [], []

        if len(gt_bboxes) == 0:
            loss = self.relation_loss(stu_feature, tea_feature)
            return self.lambda_fgd * loss

        N, _, H, W = stu_feature.shape
        for i in range(N):
            tmp_box = torch.ones_like(gt_bboxes[i])
            tmp_box.requires_grad = False
            tmp_box[:, 0] = gt_bboxes[i][:, 0] / im_shape[1] * W
            tmp_box[:, 2] = gt_bboxes[i][:, 2] / im_shape[1] * W
            tmp_box[:, 1] = gt_bboxes[i][:, 1] / im_shape[0] * H
            tmp_box[:, 3] = gt_bboxes[i][:, 3] / im_shape[0] * H

            zero = torch.zeros_like(tmp_box[:, 0], dtype=torch.int32)
            ones = torch.ones_like(tmp_box[:, 2], dtype=torch.int32)
            zero.requires_grad = False
            ones.requires_grad = False
            wmin.append(torch.maximum(zero, torch.floor(tmp_box[:, 0]).to(torch.int32)))
            wmax.append(torch.ceil(tmp_box[:, 2]).to(torch.int32))
            hmin.append(torch.maximum(zero, torch.floor(tmp_box[:, 1]).to(torch.int32)))
            hmax.append(torch.ceil(tmp_box[:, 3]).to(torch.int32))

            area_recip = 1.0 / (
                    hmax[i].reshape([1, -1]) + 1 - hmin[i].reshape([1, -1])) / (
                                 wmax[i].reshape([1, -1]) + 1 - wmin[i].reshape([1, -1]))

            for j in range(len(gt_bboxes[i])):
                if gt_bboxes[i][j].sum() > 0:
                    mask_fg[i] = self.mask_value(
                        mask_fg[i], hmin[i][j], hmax[i][j] + 1, wmin[i][j],
                                                wmax[i][j] + 1, area_recip[0][j])

            mask_bg[i] = torch.where(mask_fg[i] > zero_tmp, zero_tmp, one_tmp)

            if torch.sum(mask_bg[i]):
                mask_bg[i] /= torch.sum(mask_bg[i])

        fg_loss, bg_loss = self.feature_loss(stu_feature, tea_feature, mask_fg,
                                             mask_bg, tea_channel_att,
                                             tea_spatial_att)
        mask_loss = self.mask_loss(stu_channel_att, tea_channel_att,
                                   stu_spatial_att, tea_spatial_att)
        rela_loss = self.relation_loss(stu_feature, tea_feature)
        loss = self.alpha_fgd * fg_loss + self.beta_fgd * bg_loss \
               + self.gamma_fgd * mask_loss + self.lambda_fgd * rela_loss
        return loss * self.loss_weight


class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([
            math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand([channel, 1, window_size, window_size])
        return window

    def _ssim(self, img1, img2, window, window_size, channel,
              size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(
            img1 * img1, window, padding=window_size // 2,
            groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(
            img2 * img2, window, padding=window_size // 2,
            groups=channel) - mu2_sq
        sigma12 = F.conv2d(
            img1 * img2, window, padding=window_size // 2,
            groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
                1e-12 + (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean([1, 2, 3])

    def forward(self, img1, img2):
        channel = img1.shape[1]
        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            self.window = window
            self.channel = channel

        return self._ssim(img1, img2, window, self.window_size, channel,
                          self.size_average)
