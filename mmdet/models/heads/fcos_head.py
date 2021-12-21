#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date:
#   Description :
#
# ================================================================
import numpy as np
import math
import copy
import torch
import torch as T

from mmdet.models.custom_layers import Conv2dUnit
from mmdet.models.matrix_nms import matrix_nms


class FCOSHead(torch.nn.Module):
    def __init__(self,
                 in_channel,
                 num_classes,
                 fpn_stride=[8, 16, 32, 64, 128],
                 thresh_with_ctr=True,
                 prior_prob=0.01,
                 num_convs=4,
                 norm_type="gn",
                 fcos_loss=None,
                 norm_reg_targets=True,
                 centerness_on_reg=True,
                 use_dcn_in_tower=False,
                 use_dcn_bias=False,
                 nms_cfg=None
                 ):
        super(FCOSHead, self).__init__()
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.fpn_stride = fpn_stride[::-1]
        self.thresh_with_ctr = thresh_with_ctr
        self.prior_prob = prior_prob
        self.num_convs = num_convs
        self.norm_reg_targets = norm_reg_targets
        self.centerness_on_reg = centerness_on_reg
        self.norm_type = norm_type
        self.fcos_loss = fcos_loss
        self.nms_cfg = nms_cfg


        self.scales_on_reg = torch.nn.ParameterList()       # 回归分支（预测框坐标）的系数
        self.cls_convs = torch.nn.ModuleList()   # 每个fpn输出特征图  共享的  再进行卷积的卷积层，用于预测类别
        self.reg_convs = torch.nn.ModuleList()   # 每个fpn输出特征图  共享的  再进行卷积的卷积层，用于预测坐标
        # 用于预测centerness
        self.ctn_pred = Conv2dUnit(in_channel, 1, 3, stride=1, bias_attr=True, act=None, name="fcos_head_centerness")

        # 每个fpn输出特征图  共享的  卷积层。
        for lvl in range(0, self.num_convs):
            bias_attr = True

            in_ch = self.in_channel
            cls_conv_layer = Conv2dUnit(in_ch, self.in_channel, 3, stride=1, bias_attr=bias_attr, norm_type=norm_type, norm_groups=32, bias_lr=2.0,
                                        act='relu', name='fcos_head_cls_tower_conv_{}'.format(lvl))
            self.cls_convs.append(cls_conv_layer)


            in_ch = self.in_channel
            reg_conv_layer = Conv2dUnit(in_ch, self.in_channel, 3, stride=1, bias_attr=bias_attr, norm_type=norm_type, norm_groups=32, bias_lr=2.0,
                                        act='relu', name='fcos_head_reg_tower_conv_{}'.format(lvl))
            self.reg_convs.append(reg_conv_layer)

        # 类别分支最后的卷积。设置偏移的初始值使得各类别预测概率初始值为self.prior_prob (根据激活函数是sigmoid()时推导出，和RetinaNet中一样)
        bias_init_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        self.cls_pred = Conv2dUnit(in_channel, self.num_classes, 3, stride=1, bias_attr=True, act=None, name="fcos_head_cls")
        torch.nn.init.constant_(self.cls_pred.conv.bias, bias_init_value)
        # 坐标分支最后的卷积
        self.reg_pred = Conv2dUnit(in_channel, 4, 3, stride=1, bias_attr=True, act=None, name="fcos_head_reg")


        self.n_layers = len(self.fpn_stride)      # 有n个输出层
        for i in range(self.n_layers):     # 遍历每个输出层
            scale = torch.nn.Parameter(torch.randn(1, ))
            torch.nn.init.constant_(scale, 1.0)
            self.scales_on_reg.append(scale)

        self.relu = torch.nn.ReLU(inplace=True)

    def _fcos_head(self, fpn_feat, fpn_stride, i, is_training=False):
        """
        Args:
            fpn_feat (Variables): feature map from FPN
            fpn_stride     (int): the stride of current feature map
            is_training   (bool): whether is train or test mode
        """
        fpn_scale = self.scales_on_reg[i]
        subnet_blob_cls = fpn_feat
        subnet_blob_reg = fpn_feat
        for lvl in range(0, self.num_convs):
            subnet_blob_cls = self.cls_convs[lvl](subnet_blob_cls)
            subnet_blob_reg = self.reg_convs[lvl](subnet_blob_reg)

        cls_logits = self.cls_pred(subnet_blob_cls)   # 通道数变成类别数
        bbox_reg = self.reg_pred(subnet_blob_reg)     # 通道数变成4
        bbox_reg = bbox_reg * fpn_scale     # 预测坐标的特征图整体乘上fpn_scale，是一个可学习参数
        # 如果 归一化坐标分支，bbox_reg进行relu激活
        if self.norm_reg_targets:
            bbox_reg = self.relu(bbox_reg)
            if not is_training:   # 验证状态的话，bbox_reg再乘以下采样倍率
                bbox_reg = bbox_reg * fpn_stride
        else:
            bbox_reg = T.exp(bbox_reg)

        # ============= centerness分支，默认是用坐标分支接4个卷积层之后的结果subnet_blob_reg =============
        if self.centerness_on_reg:
            centerness = self.ctn_pred(subnet_blob_reg)
        else:
            centerness = self.ctn_pred(subnet_blob_cls)

        return cls_logits, bbox_reg, centerness

    def _get_output(self, fpn_feats, is_training=False):
        """
        Args:
            fpn_feats (list): the list of fpn feature maps。[p7, p6, p5, p4, p3]
            is_training (bool): whether is train or test mode
        Return:
            cls_logits (Variables): prediction for classification
            bboxes_reg (Variables): prediction for bounding box
            centerness (Variables): prediction for ceterness
        """
        cls_logits = []
        bboxes_reg = []
        centerness = []
        assert len(fpn_feats) == len(self.fpn_stride), \
            "The size of fpn_feats is not equal to size of fpn_stride"

        # 所有的fpn特征图共用这些卷积层。但是独占一个缩放因子self.scales_on_reg[i]。
        i = 0
        for fpn_feat, fpn_stride in zip(fpn_feats, self.fpn_stride):
            cls_pred, bbox_pred, ctn_pred = self._fcos_head(fpn_feat, fpn_stride, i, is_training=is_training)
            cls_logits.append(cls_pred)
            bboxes_reg.append(bbox_pred)
            centerness.append(ctn_pred)
            i += 1
        return cls_logits, bboxes_reg, centerness

    def _compute_locations(self, features):
        """
        Args:
            features (list): List of Variables for FPN feature maps. [p7, p6, p5, p4, p3]
        Return:
            Anchor points for each feature map pixel
        """
        locations = []
        for lvl, feature in enumerate(features):
            shape_fm = feature.shape
            h = shape_fm[2]
            w = shape_fm[3]
            fpn_stride = self.fpn_stride[lvl]
            shift_x = torch.arange(0, w, dtype=T.float32, device=feature.device) * fpn_stride
            shift_y = torch.arange(0, h, dtype=T.float32, device=feature.device) * fpn_stride
            shift_x = shift_x.unsqueeze(0)
            shift_y = shift_y.unsqueeze(1)
            shift_x = torch.tile(shift_x, [h, 1])
            shift_y = torch.tile(shift_y, [1, w])
            shift_x = torch.reshape(shift_x, (-1, ))
            shift_y = torch.reshape(shift_y, (-1, ))
            location = torch.stack([shift_x, shift_y], dim=-1) + fpn_stride // 2
            location.requires_grad = False
            locations.append(location)
        return locations

    def _postprocessing_by_level(self, locations, box_cls, box_reg, box_ctn, im_scale):
        """
        Args:
            locations (Variables): anchor points for current layer
            box_cls   (Variables): categories prediction
            box_reg   (Variables): bounding box prediction
            box_ctn   (Variables): centerness prediction
            im_scale  (Variables): [scale, ] for input images
        Return:
            box_cls_ch_last  (Variables): score for each category, in [N, C, M]
                C is the number of classes and M is the number of anchor points
            box_reg_decoding (Variables): decoded bounding box, in [N, M, 4]
                last dimension is [x1, y1, x2, y2]
        """
        batch_size = box_cls.shape[0]
        num_classes = self.num_classes

        # =========== 类别概率，[N, 80, H*W] ===========
        box_cls_ch_last = T.reshape(box_cls, (batch_size, num_classes, -1))  # [N, 80, H*W]
        box_cls_ch_last = T.sigmoid(box_cls_ch_last)  # 类别概率用sigmoid()激活，[N, 80, H*W]

        # =========== 坐标(4个偏移)，[N, H*W, 4] ===========
        box_reg_ch_last = box_reg.permute(0, 2, 3, 1)  # [N, H, W, 4]
        box_reg_ch_last = T.reshape(box_reg_ch_last, (batch_size, -1, 4))  # [N, H*W, 4]，坐标不用再接激活层，直接预测。

        # =========== centerness，[N, 1, H*W] ===========
        box_ctn_ch_last = T.reshape(box_ctn, (batch_size, 1, -1))  # [N, 1, H*W]
        box_ctn_ch_last = T.sigmoid(box_ctn_ch_last)  # centerness用sigmoid()激活，[N, 1, H*W]

        box_reg_decoding = T.cat(  # [N, H*W, 4]
            [
                locations[:, 0:1] - box_reg_ch_last[:, :, 0:1],  # 左上角x坐标
                locations[:, 1:2] - box_reg_ch_last[:, :, 1:2],  # 左上角y坐标
                locations[:, 0:1] + box_reg_ch_last[:, :, 2:3],  # 右下角x坐标
                locations[:, 1:2] + box_reg_ch_last[:, :, 3:4]  # 右下角y坐标
            ],
            dim=-1)
        # 除以图片缩放
        im_scale = T.reshape(im_scale, (batch_size, 1, 1))  # [N, 1, 1]
        box_reg_decoding /= im_scale  # [N, H*W, 4]，最终坐标=坐标/图片缩放
        if self.thresh_with_ctr:
            box_cls_ch_last = box_cls_ch_last * box_ctn_ch_last  # [N, 80, H*W]，最终分数=类别概率*centerness
        return box_cls_ch_last, box_reg_decoding

    def get_loss(self, input, tag_labels, tag_bboxes, tag_centerness):
        """
        Calculate the loss for FCOS
        Args:
            input           (list): List of Variables for feature maps from FPN layers
            tag_labels     (Variables): category targets for each anchor point
            tag_bboxes     (Variables): bounding boxes  targets for positive samples
            tag_centerness (Variables): centerness targets for positive samples
        Return:
            loss (dict): loss composed by classification loss, bounding box
                regression loss and centerness regression loss
        """
        # cls_logits里面每个元素是[N, 80, 格子行数, 格子列数]
        # bboxes_reg里面每个元素是[N,  4, 格子行数, 格子列数]
        # centerness里面每个元素是[N,  1, 格子行数, 格子列数]
        # is_training=True表示训练状态。
        # 训练状态的话，bbox_reg不会乘以下采样倍率，这样得到的坐标单位1表示当前层的1个格子边长。
        # 因为在Gt2FCOSTarget中设置了norm_reg_targets=True对回归的lrtb进行了归一化，归一化方式是除以格子边长（即下采样倍率），
        # 所以网络预测的lrtb的单位1实际上代表了当前层的1个格子边长。
        cls_logits, bboxes_reg, centerness, iou_awares = self._get_output(
            input, is_training=True)
        loss = self.fcos_loss(cls_logits, bboxes_reg, centerness, iou_awares, tag_labels,
                              tag_bboxes, tag_centerness)
        return loss

    def get_prediction(self, input, im_scale):
        """
        Decode the prediction
        Args:
            input: [p7, p6, p5, p4, p3]
            im_scale(Variables): [scale, ] for input images
        Return:
            the bounding box prediction
        """
        # cls_logits里面每个元素是[N, 80, 格子行数, 格子列数]
        # bboxes_reg里面每个元素是[N,  4, 格子行数, 格子列数]
        # centerness里面每个元素是[N,  1, 格子行数, 格子列数]
        # is_training=False表示验证状态。
        # 验证状态的话，bbox_reg再乘以下采样倍率，这样得到的坐标是相对于输入图片宽高的坐标。
        cls_logits, bboxes_reg, centerness = self._get_output(input, is_training=False)

        # locations里面每个元素是[格子行数*格子列数, 2]。即格子中心点相对于输入图片宽高的xy坐标。
        locations = self._compute_locations(input)

        pred_boxes = []
        pred_scores = []
        for _, (
                pts, cls, box, ctn
        ) in enumerate(zip(locations, cls_logits, bboxes_reg, centerness)):
            pred_scores_lvl, pred_boxes_lvl = self._postprocessing_by_level(
                pts, cls, box, ctn, im_scale)
            pred_boxes.append(pred_boxes_lvl)     # [N, H*W, 4]，最终坐标
            pred_scores.append(pred_scores_lvl)   # [N, 80, H*W]，最终分数
        pred_boxes = T.cat(pred_boxes, dim=1)    # [N, A, 4]，最终坐标
        pred_scores = T.cat(pred_scores, dim=2)  # [N, 80, A]，最终分数
        pred_scores = pred_scores.permute(0, 2, 1)  # [N, A, 80]，最终分数

        # nms
        preds = []
        nms_cfg = copy.deepcopy(self.nms_cfg)
        nms_type = nms_cfg.pop('nms_type')
        batch_size = pred_boxes.shape[0]
        if nms_type == 'matrix_nms':
            for i in range(batch_size):
                pred = matrix_nms(pred_boxes[i, :, :], pred_scores[i, :, :], **nms_cfg)
                preds.append(pred)
        # elif nms_type == 'multiclass_nms':
        #     for i in range(batch_size):
        #         pred = fluid.layers.multiclass_nms(pred_boxes[i:i+1, :, :], pred_scores[i:i+1, :, :], background_label=-1, **nms_cfg)
        #         preds.append(pred)
        return preds
