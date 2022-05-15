# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

# from ppdet.modeling.layers import MultiClassNMS
from mmdet.models.assigners.utils import generate_anchors_for_grid_cell
from mmdet.models.backbones.cspresnet import ConvBNLayer
from mmdet.models.bbox_utils import batch_distance2bbox
from mmdet.models.matrix_nms import matrix_nms
from mmdet.models.ops import get_static_shape, paddle_distributed_is_initialized, get_act_fn
from mmdet.models.initializer import bias_init_with_prob, constant_, normal_
from mmdet.models.losses.iou_losses import GIoULoss
from mmdet.utils import my_multiclass_nms


def print_diff(dic, key, tensor):
    if tensor is not None:  # 有的梯度张量可能是None
        aaaaaa1 = dic[key]
        aaaaaa2 = tensor.cpu().detach().numpy()
        ddd = np.sum((aaaaaa1 - aaaaaa2) ** 2)
        print('diff=%.6f (%s)' % (ddd, key))


class ESEAttn(nn.Module):
    def __init__(self, feat_channels, act='swish'):
        super(ESEAttn, self).__init__()
        self.fc = nn.Conv2d(feat_channels, feat_channels, 1)
        self.conv = ConvBNLayer(feat_channels, feat_channels, 1, act=act)

        self._init_weights()

    def _init_weights(self):
        normal_(self.fc.weight, std=0.001)

    def forward(self, feat, avg_feat):
        weight = torch.sigmoid(self.fc(avg_feat))
        return self.conv(feat * weight)

    def add_param_group(self, param_groups, base_lr, base_wd, need_clip, clip_norm):
        self.conv.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        if self.fc.weight.requires_grad:
            param_group_conv_weight = {'params': [self.fc.weight]}
            param_group_conv_weight['lr'] = base_lr * 1.0
            param_group_conv_weight['base_lr'] = base_lr * 1.0
            param_group_conv_weight['weight_decay'] = base_wd
            param_group_conv_weight['need_clip'] = need_clip
            param_group_conv_weight['clip_norm'] = clip_norm
            param_groups.append(param_group_conv_weight)
        if self.fc.bias.requires_grad:
            param_group_conv_bias = {'params': [self.fc.bias]}
            param_group_conv_bias['lr'] = base_lr * 1.0
            param_group_conv_bias['base_lr'] = base_lr * 1.0
            param_group_conv_bias['weight_decay'] = base_wd
            param_group_conv_bias['need_clip'] = need_clip
            param_group_conv_bias['clip_norm'] = clip_norm
            param_groups.append(param_group_conv_bias)


class PPYOLOEHead(nn.Module):
    __shared__ = ['num_classes', 'eval_size', 'trt', 'exclude_nms']
    __inject__ = ['static_assigner', 'assigner', 'nms']

    def __init__(self,
                 in_channels=[1024, 512, 256],
                 num_classes=80,
                 act='swish',
                 fpn_strides=(32, 16, 8),
                 grid_cell_scale=5.0,
                 grid_cell_offset=0.5,
                 reg_max=16,
                 static_assigner_epoch=4,
                 use_varifocal_loss=True,
                 static_assigner='ATSSAssigner',
                 assigner='TaskAlignedAssigner',
                 nms='MultiClassNMS',
                 eval_size=None,
                 loss_weight={
                     'class': 1.0,
                     'iou': 2.5,
                     'dfl': 0.5,
                 },
                 trt=False,
                 nms_cfg=None,
                 exclude_nms=False):
        super(PPYOLOEHead, self).__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max
        self.iou_loss = GIoULoss()
        self.loss_weight = loss_weight
        self.use_varifocal_loss = use_varifocal_loss
        self.eval_size = eval_size

        self.static_assigner_epoch = static_assigner_epoch
        self.static_assigner = static_assigner
        self.assigner = assigner
        self.nms = nms
        # if isinstance(self.nms, MultiClassNMS) and trt:
        #     self.nms.trt = trt
        self.exclude_nms = exclude_nms
        self.nms_cfg = nms_cfg
        # stem
        self.stem_cls = nn.ModuleList()
        self.stem_reg = nn.ModuleList()
        act = get_act_fn(
            act, trt=trt) if act is None or isinstance(act,
                                                       (str, dict)) else act
        for in_c in self.in_channels:
            self.stem_cls.append(ESEAttn(in_c, act=act))
            self.stem_reg.append(ESEAttn(in_c, act=act))
        # pred head
        self.pred_cls = nn.ModuleList()
        self.pred_reg = nn.ModuleList()
        for in_c in self.in_channels:
            self.pred_cls.append(
                nn.Conv2d(
                    in_c, self.num_classes, 3, padding=1))
            self.pred_reg.append(
                nn.Conv2d(
                    in_c, 4 * (self.reg_max + 1), 3, padding=1))
        # projection conv
        self.proj_conv = nn.Conv2d(self.reg_max + 1, 1, 1, bias=False)
        self._init_weights()

    def add_param_group(self, param_groups, base_lr, base_wd, need_clip, clip_norm):
        for i in range(len(self.in_channels)):
            self.stem_cls[i].add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
            self.stem_reg[i].add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
            if self.pred_cls[i].weight.requires_grad:
                param_group_conv_weight = {'params': [self.pred_cls[i].weight]}
                param_group_conv_weight['lr'] = base_lr * 1.0
                param_group_conv_weight['base_lr'] = base_lr * 1.0
                param_group_conv_weight['weight_decay'] = base_wd
                param_group_conv_weight['need_clip'] = need_clip
                param_group_conv_weight['clip_norm'] = clip_norm
                param_groups.append(param_group_conv_weight)
            if self.pred_cls[i].bias.requires_grad:
                param_group_conv_bias = {'params': [self.pred_cls[i].bias]}
                param_group_conv_bias['lr'] = base_lr * 1.0
                param_group_conv_bias['base_lr'] = base_lr * 1.0
                param_group_conv_bias['weight_decay'] = base_wd
                param_group_conv_bias['need_clip'] = need_clip
                param_group_conv_bias['clip_norm'] = clip_norm
                param_groups.append(param_group_conv_bias)
            if self.pred_reg[i].weight.requires_grad:
                param_group_conv_weight2 = {'params': [self.pred_reg[i].weight]}
                param_group_conv_weight2['lr'] = base_lr * 1.0
                param_group_conv_weight2['base_lr'] = base_lr * 1.0
                param_group_conv_weight2['weight_decay'] = base_wd
                param_group_conv_weight2['need_clip'] = need_clip
                param_group_conv_weight2['clip_norm'] = clip_norm
                param_groups.append(param_group_conv_weight2)
            if self.pred_reg[i].bias.requires_grad:
                param_group_conv_bias2 = {'params': [self.pred_reg[i].bias]}
                param_group_conv_bias2['lr'] = base_lr * 1.0
                param_group_conv_bias2['base_lr'] = base_lr * 1.0
                param_group_conv_bias2['weight_decay'] = base_wd
                param_group_conv_bias2['need_clip'] = need_clip
                param_group_conv_bias2['clip_norm'] = clip_norm
                param_groups.append(param_group_conv_bias2)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def _init_weights(self):
        bias_cls = bias_init_with_prob(0.01)
        for cls_, reg_ in zip(self.pred_cls, self.pred_reg):
            constant_(cls_.weight)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.weight)
            constant_(reg_.bias, 1.0)

        self.proj = torch.linspace(0, self.reg_max, self.reg_max + 1)
        self.proj.requires_grad = False
        self.proj_conv.weight.requires_grad_(False)
        self.proj_conv.weight.copy_(
            self.proj.reshape([1, self.reg_max + 1, 1, 1]))

        if self.eval_size:
            anchor_points, stride_tensor = self._generate_anchors()
            self.register_buffer('anchor_points', anchor_points)
            self.register_buffer('stride_tensor', stride_tensor)

    def forward_train(self, feats, targets):
        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
                feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset)

        cls_score_list, reg_distri_list = [], []
        for i, feat in enumerate(feats):
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                         feat)
            reg_distri = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            # cls and reg
            cls_score = torch.sigmoid(cls_logit)
            cls_score_list.append(cls_score.flatten(2).permute((0, 2, 1)))
            reg_distri_list.append(reg_distri.flatten(2).permute((0, 2, 1)))
        cls_score_list = torch.cat(cls_score_list, 1)
        reg_distri_list = torch.cat(reg_distri_list, 1)

        # import numpy as np
        # dic = np.load('../aaa.npz')
        # cls_score_list = torch.Tensor(dic['cls_score_list'])
        # reg_distri_list = torch.Tensor(dic['reg_distri_list'])
        # anchors = torch.Tensor(dic['anchors'])
        # anchor_points = torch.Tensor(dic['anchor_points'])
        # stride_tensor = torch.Tensor(dic['stride_tensor'])
        # gt_class = torch.Tensor(dic['gt_class'])
        # gt_bbox = torch.Tensor(dic['gt_bbox'])
        # pad_gt_mask = torch.Tensor(dic['pad_gt_mask'])
        # targets['gt_class'] = gt_class
        # targets['gt_bbox'] = gt_bbox
        # targets['pad_gt_mask'] = pad_gt_mask
        #
        # loss = torch.Tensor(dic['loss'])
        # loss_cls = torch.Tensor(dic['loss_cls'])
        # loss_iou = torch.Tensor(dic['loss_iou'])
        # loss_dfl = torch.Tensor(dic['loss_dfl'])
        # loss_l1 = torch.Tensor(dic['loss_l1'])

        losses = self.get_loss([
            cls_score_list, reg_distri_list, anchors, anchor_points,
            num_anchors_list, stride_tensor
        ], targets)
        return losses

    def _generate_anchors(self, feats=None):
        # just use in eval time
        anchor_points = []
        stride_tensor = []
        for i, stride in enumerate(self.fpn_strides):
            if feats is not None:
                _, _, h, w = feats[i].shape
            else:
                h = int(self.eval_size[0] / stride)
                w = int(self.eval_size[1] / stride)
            shift_x = torch.arange(end=w) + self.grid_cell_offset
            shift_y = torch.arange(end=h) + self.grid_cell_offset
            # shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            anchor_point = torch.stack([shift_x, shift_y], -1).to(torch.float32)
            anchor_points.append(anchor_point.reshape([-1, 2]))
            stride_tensor.append(
                torch.full(
                    [h * w, 1], stride, dtype=torch.float32))
        anchor_points = torch.cat(anchor_points)
        stride_tensor = torch.cat(stride_tensor)
        return anchor_points, stride_tensor

    def forward_eval(self, feats):
        if self.eval_size:
            anchor_points, stride_tensor = self.anchor_points, self.stride_tensor
        else:
            anchor_points, stride_tensor = self._generate_anchors(feats)
        cls_score_list, reg_dist_list = [], []
        for i, feat in enumerate(feats):
            b, _, h, w = feat.shape
            l = h * w
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                         feat)
            reg_dist = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            reg_dist = reg_dist.reshape([-1, 4, self.reg_max + 1, l])
            reg_dist = reg_dist.permute((0, 2, 1, 3))
            reg_dist = self.proj_conv(F.softmax(reg_dist, dim=1))
            # cls and reg
            cls_score = torch.sigmoid(cls_logit)
            cls_score_list.append(cls_score.reshape([b, self.num_classes, l]))
            reg_dist_list.append(reg_dist.reshape([b, 4, l]))

        cls_score_list = torch.cat(cls_score_list, -1)  # [N, 80, A]
        reg_dist_list = torch.cat(reg_dist_list, -1)    # [N,  4, A]

        return cls_score_list, reg_dist_list, anchor_points, stride_tensor

    def forward(self, feats, targets=None):
        assert len(feats) == len(self.fpn_strides), \
            "The size of feats is not equal to size of fpn_strides"

        if self.training:
            return self.forward_train(feats, targets)
        else:
            return self.forward_eval(feats)

    @staticmethod
    def _focal_loss(score, label, alpha=0.25, gamma=2.0):
        weight = (score - label).pow(gamma)
        if alpha > 0:
            alpha_t = alpha * label + (1 - alpha) * (1 - label)
            weight *= alpha_t

        # loss = F.binary_cross_entropy(
        #     score, label, weight=weight, reduction='sum')

        score = score.to(torch.float32)
        eps = 1e-9
        loss = label * (0 - torch.log(score + eps)) + \
               (1.0 - label) * (0 - torch.log(1.0 - score + eps))
        loss *= weight
        loss = loss.sum()
        return loss

    @staticmethod
    def _varifocal_loss(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label

        # loss = F.binary_cross_entropy(
        #     pred_score, gt_score, weight=weight, reduction='sum')

        # pytorch的F.binary_cross_entropy()的weight不能向前传播梯度，但是
        # paddle的F.binary_cross_entropy()的weight可以向前传播梯度（给pred_score），
        # 所以这里手动实现F.binary_cross_entropy()
        # 使用混合精度训练时，pred_score类型是torch.float16，需要转成torch.float32避免log(0)=nan
        pred_score = pred_score.to(torch.float32)
        eps = 1e-9
        loss = gt_score * (0 - torch.log(pred_score + eps)) + \
               (1.0 - gt_score) * (0 - torch.log(1.0 - pred_score + eps))
        loss *= weight
        loss = loss.sum()
        return loss

    def _bbox_decode(self, anchor_points, pred_dist):
        b, l, _ = get_static_shape(pred_dist)
        device = pred_dist.device
        pred_dist = pred_dist.reshape([b, l, 4, self.reg_max + 1])
        pred_dist = F.softmax(pred_dist, dim=-1)
        pred_dist = pred_dist.matmul(self.proj.to(device))
        return batch_distance2bbox(anchor_points, pred_dist)

    def _bbox2distance(self, points, bbox):
        x1y1, x2y2 = torch.split(bbox, 2, -1)
        lt = points - x1y1
        rb = x2y2 - points
        return torch.cat([lt, rb], -1).clamp(0, self.reg_max - 0.01)

    def _df_loss(self, pred_dist, target):
        target_left = target.to(torch.int64)
        target_right = target_left + 1
        weight_left = target_right.to(torch.float32) - target
        weight_right = 1 - weight_left

        eps = 1e-9
        # 使用混合精度训练时，pred_dist类型是torch.float16，pred_dist_act类型是torch.float32
        pred_dist_act = F.softmax(pred_dist, dim=-1)
        target_left_onehot = F.one_hot(target_left, pred_dist_act.shape[-1])
        target_right_onehot = F.one_hot(target_right, pred_dist_act.shape[-1])
        loss_left = target_left_onehot * (0 - torch.log(pred_dist_act + eps))
        loss_right = target_right_onehot * (0 - torch.log(pred_dist_act + eps))
        loss_left = loss_left.sum(-1) * weight_left
        loss_right = loss_right.sum(-1) * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)

    def _bbox_loss(self, pred_dist, pred_bboxes, anchor_points, assigned_labels,
                   assigned_bboxes, assigned_scores, assigned_scores_sum):
        # select positive samples mask
        mask_positive = (assigned_labels != self.num_classes)
        num_pos = mask_positive.sum()
        # pos/neg loss
        if num_pos > 0:
            # l1 + iou
            bbox_mask = mask_positive.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(pred_bboxes,
                                                   bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes, bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(
                assigned_scores.sum(-1), mask_positive).unsqueeze(-1)

            loss_l1 = F.l1_loss(pred_bboxes_pos, assigned_bboxes_pos)

            loss_iou = self.iou_loss(pred_bboxes_pos,
                                     assigned_bboxes_pos) * bbox_weight
            loss_iou = loss_iou.sum() / assigned_scores_sum

            dist_mask = mask_positive.unsqueeze(-1).repeat(
                [1, 1, (self.reg_max + 1) * 4])
            pred_dist_pos = torch.masked_select(
                pred_dist, dist_mask).reshape([-1, 4, self.reg_max + 1])
            assigned_ltrb = self._bbox2distance(anchor_points, assigned_bboxes)
            assigned_ltrb_pos = torch.masked_select(
                assigned_ltrb, bbox_mask).reshape([-1, 4])
            loss_dfl = self._df_loss(pred_dist_pos,
                                     assigned_ltrb_pos) * bbox_weight
            loss_dfl = loss_dfl.sum() / assigned_scores_sum
        else:
            loss_l1 = torch.zeros([]).to(pred_dist.device)
            loss_iou = torch.zeros([]).to(pred_dist.device)
            loss_dfl = pred_dist.sum() * 0.
            # loss_l1 = None
            # loss_iou = None
            # loss_dfl = None
        return loss_l1, loss_iou, loss_dfl

    def get_loss(self, head_outs, gt_meta):
        pred_scores, pred_distri, anchors,\
        anchor_points, num_anchors_list, stride_tensor = head_outs
        device = pred_scores.device
        anchors = anchors.to(device)
        anchor_points = anchor_points.to(device)
        stride_tensor = stride_tensor.to(device)

        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self._bbox_decode(anchor_points_s, pred_distri)

        gt_labels = gt_meta['gt_class']
        gt_labels = gt_labels.to(torch.int64)
        gt_bboxes = gt_meta['gt_bbox']
        pad_gt_mask = gt_meta['pad_gt_mask']

        # miemie2013: 剪掉填充的gt
        num_boxes = pad_gt_mask.sum([1, 2])
        num_max_boxes = num_boxes.max().to(torch.int32)
        pad_gt_mask = pad_gt_mask[:, :num_max_boxes, :]
        gt_labels = gt_labels[:, :num_max_boxes, :]
        gt_bboxes = gt_bboxes[:, :num_max_boxes, :]

        # label assignment
        if gt_meta['epoch_id'] < self.static_assigner_epoch:
            assigned_labels, assigned_bboxes, assigned_scores = \
                self.static_assigner(
                    anchors,
                    num_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    pad_gt_mask,
                    bg_index=self.num_classes,
                    pred_bboxes=pred_bboxes.detach() * stride_tensor)
            alpha_l = 0.25
            # import numpy as np
            # dic = np.load('../aa2.npz')
            # print_diff(dic, 'assigned_labels', assigned_labels)
            # print_diff(dic, 'assigned_bboxes', assigned_bboxes)
            # print_diff(dic, 'assigned_scores', assigned_scores)
            # assigned_labels = torch.Tensor(dic['assigned_labels']).to(torch.int64)
            # assigned_bboxes = torch.Tensor(dic['assigned_bboxes']).to(torch.float32)
            # assigned_scores = torch.Tensor(dic['assigned_scores']).to(torch.float32)
            # print()
        else:
            assigned_labels, assigned_bboxes, assigned_scores = \
                self.assigner(
                pred_scores.detach(),
                pred_bboxes.detach() * stride_tensor,
                anchor_points,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index=self.num_classes)
            alpha_l = -1
        # rescale bbox
        assigned_bboxes /= stride_tensor
        # cls loss
        if self.use_varifocal_loss:
            one_hot_label = F.one_hot(assigned_labels,
                                      self.num_classes + 1)[..., :-1]
            loss_cls = self._varifocal_loss(pred_scores, assigned_scores,
                                            one_hot_label)
        else:
            loss_cls = self._focal_loss(pred_scores, assigned_scores, alpha_l)

        # assigned_scores_sum2 = assigned_scores.sum()
        assigned_scores_sum = F.relu(assigned_scores.sum() - 1.) + 1.   # y = max(x, 1)
        # 每张卡上的assigned_scores_sum求平均，而且max(x, 1)
        # if paddle_distributed_is_initialized():
        #     paddle.distributed.all_reduce(assigned_scores_sum)
        #     assigned_scores_sum = (assigned_scores_sum / paddle.distributed.get_world_size()).clamp(min=1)
        loss_cls /= assigned_scores_sum

        loss_l1, loss_iou, loss_dfl = \
            self._bbox_loss(pred_distri, pred_bboxes, anchor_points_s,
                            assigned_labels, assigned_bboxes, assigned_scores,
                            assigned_scores_sum)

        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou + \
               self.loss_weight['dfl'] * loss_dfl
        out_dict = {
            'total_loss': loss,
            'loss_cls': loss_cls,
            'loss_iou': loss_iou,
            'loss_dfl': loss_dfl,
            'loss_l1': loss_l1,
        }
        return out_dict

    def post_process(self, head_outs, scale_factor):
        pred_scores, pred_dist, anchor_points, stride_tensor = head_outs
        pred_bboxes = batch_distance2bbox(anchor_points, pred_dist.permute((0, 2, 1)))
        pred_bboxes *= stride_tensor
        # scale bbox to origin
        # torch的split和paddle有点不同，torch的第二个参数表示的是每一份的大小，paddle的第二个参数表示的是分成几份。
        scale_y, scale_x = torch.split(scale_factor, 1, -1)
        scale_factor = torch.cat(
            [scale_x, scale_y, scale_x, scale_y], -1).reshape([-1, 1, 4])
        pred_bboxes /= scale_factor   # [N, A, 4]     pred_scores.shape = [N, 80, A]
        if self.exclude_nms:
            # `exclude_nms=True` just use in benchmark
            return pred_bboxes.sum(), pred_scores.sum()
        else:
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
            # bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
            # return bbox_pred, bbox_num
