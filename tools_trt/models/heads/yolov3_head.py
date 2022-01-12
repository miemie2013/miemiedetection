#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-10-23 09:13:23
#   Description : paddle2.0_ppyolo
#
# ================================================================
import numpy as np
import torch
import torch as T
import torch.nn.functional as F
import copy
import math

from tools_trt.models.custom_layers import paddle_yolo_box, CoordConv, Conv2dUnit, SPP, DropBlock


# sigmoid()函数的反函数。先取倒数再减一，取对数再取相反数。
def _de_sigmoid(x, eps=1e-7):
    # x限制在区间[eps, 1 / eps]内
    x = torch.clamp(x, eps, 1 / eps)

    # 先取倒数再减一
    x = 1.0 / x - 1.0

    # e^(-x)限制在区间[eps, 1 / eps]内
    x = torch.clamp(x, eps, 1 / eps)

    # 取对数再取相反数
    x = -torch.log(x)
    return x


class YOLOv3Head(object):
    def __init__(self,
                 in_channels=[1024, 512, 256],
                 anchors=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                          [59, 119], [116, 90], [156, 198], [373, 326]],
                 anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 downsample=[32, 16, 8],
                 scale_x_y=1.05,
                 clip_bbox=True,
                 num_classes=80,
                 loss='YOLOv3Loss',
                 iou_aware=False,
                 iou_aware_factor=0.4,
                 nms_cfg=None,
                 data_format='NCHW'):
        """
        Head for YOLOv3 network

        Args:
            num_classes (int): number of foreground classes
            anchors (list): anchors
            anchor_masks (list): anchor masks
            loss (object): YOLOv3Loss instance
            iou_aware (bool): whether to use iou_aware
            iou_aware_factor (float): iou aware factor
            data_format (str): data format, NCHW or NHWC
        """
        super(YOLOv3Head, self).__init__()
        assert len(in_channels) > 0, "in_channels length should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.loss = loss

        self.iou_aware = iou_aware
        self.iou_aware_factor = iou_aware_factor

        self.parse_anchor(anchors, anchor_masks)
        self.num_outputs = len(self.anchors)
        self.data_format = data_format
        self.nms_cfg = nms_cfg
        self.export_onnx = False

        self.anchor_masks = anchor_masks
        self.downsample = downsample
        self.scale_x_y = scale_x_y
        self.clip_bbox = clip_bbox
        _anchors = copy.deepcopy(anchors)
        _anchors = np.array(_anchors)
        _anchors = _anchors.astype(np.float32)
        self._anchors = _anchors   # [9, 2]

        self.yolo_outputs = []
        for i in range(len(self.anchors)):

            if self.iou_aware:
                num_filters = len(self.anchors[i]) * (self.num_classes + 6)
            else:
                num_filters = len(self.anchors[i]) * (self.num_classes + 5)
            name = 'yolo_output.{}'.format(i)
            bias_init = None
            # if self.focalloss_on_obj:
            #     # 设置偏移的初始值使得obj预测概率初始值为self.prior_prob (根据激活函数是sigmoid()时推导出)
            #     bias_init_value = -math.log((1 - self.prior_prob) / self.prior_prob)
            #     bias_init_array = np.zeros((num_filters, ), np.float32)
            #     an_num = len(self.anchor_masks[i])
            #     start = 0
            #     stride = (self.num_classes + 5)
            #     if self.iou_aware:
            #         start = an_num
            #     # 只设置置信位
            #     for o in range(an_num):
            #         bias_init_array[start + o * stride + 4] = bias_init_value
            #     bias_init = fluid.initializer.NumpyArrayInitializer(bias_init_array)
            conv = Conv2dUnit(self.in_channels[i], num_filters, 1, stride=1, bias_attr=True, act=None,
                              bias_init=bias_init, name=name)
            self.yolo_outputs.append(conv)

    def parse_anchor(self, anchors, anchor_masks):
        self.anchors = [[anchors[i] for i in mask] for mask in anchor_masks]
        self.mask_anchors = []
        anchor_num = len(anchors)
        for masks in anchor_masks:
            self.mask_anchors.append([])
            for mask in masks:
                assert mask < anchor_num, "anchor mask index overflow"
                self.mask_anchors[-1].extend(anchors[mask])

    def add_param_group(self, param_groups, base_lr, base_wd, need_clip, clip_norm):
        for layer in self.yolo_outputs:
            if isinstance(layer, Conv2dUnit):
                layer.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)

    def get_loss(self, feats, gt_bbox, targets):
        assert len(feats) == len(self.anchors)
        yolo_outputs = []
        for i, feat in enumerate(feats):
            yolo_output = self.yolo_outputs[i](feat)
            if self.data_format == 'NHWC':
                yolo_output = yolo_output.permute(0, 3, 1, 2)
            yolo_outputs.append(yolo_output)
        return self.loss(yolo_outputs, gt_bbox, targets, self.anchors)

    def __call__(self, feats, im_size, network, state_dict):
        assert len(feats) == len(self.anchors)
        yolo_outputs = []
        for i, feat in enumerate(feats):
            conv_name = 'head.yolo_outputs.%d' % i
            w = state_dict[conv_name + '.conv.weight']
            b = state_dict[conv_name + '.conv.bias']
            weights = [w, b]
            yolo_output = self.yolo_outputs[i](feat, network, weights, only_conv=True)
            if self.data_format == 'NHWC':
                yolo_output = yolo_output.permute(0, 3, 1, 2)
            yolo_outputs.append(yolo_output)
        # outputs里为大中小感受野的输出
        outputs = yolo_outputs

        boxes = []
        scores = []
        for i, out in enumerate(outputs):
            if self.iou_aware:
                na = len(self.anchors[i])
                ioup, x = out[:, 0:na, :, :], out[:, na:, :, :]
                b, c, h, w = x.shape
                no = c // na
                x = x.reshape((b, na, no, h * w))
                ioup = ioup.reshape((b, na, 1, h * w))
                obj = x[:, :, 4:5, :]
                ioup = torch.sigmoid(ioup)
                obj = torch.sigmoid(obj)
                obj_t = (obj**(1 - self.iou_aware_factor)) * (
                    ioup**self.iou_aware_factor)
                obj_t = _de_sigmoid(obj_t)
                loc_t = x[:, :, :4, :]
                cls_t = x[:, :, 5:, :]
                y_t = torch.cat([loc_t, obj_t, cls_t], 2)
                out = y_t.reshape((b, c, h, w))
            box, score = paddle_yolo_box(out, self._anchors[self.anchor_masks[i]], self.downsample[i],
                                         self.num_classes, self.scale_x_y, im_size, self.clip_bbox,
                                         conf_thresh=self.nms_cfg['score_threshold'])
            boxes.append(box)
            scores.append(score)
        yolo_boxes = torch.cat(boxes, 1)  # [N, A,  4]
        yolo_scores = torch.cat(scores, 1)  # [N, A, 80]
        if self.export_onnx:
            decode_output = torch.cat([yolo_boxes, yolo_scores], 2)  # [N, A, 4+80]
            return decode_output

        # nms
        preds = []
        nms_cfg = copy.deepcopy(self.nms_cfg)
        nms_type = nms_cfg.pop('nms_type')
        batch_size = yolo_boxes.shape[0]
        if nms_type == 'matrix_nms':
            for i in range(batch_size):
                pred = matrix_nms(yolo_boxes[i, :, :], yolo_scores[i, :, :], **nms_cfg)
                preds.append(pred)
        return preds




