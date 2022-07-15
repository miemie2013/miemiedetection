# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from mmdet.models.custom_layers import DropBlock, ConvNormLayer
from mmdet.models.initializer import Normal, XavierNormal, XavierUniform

from six.moves import zip
import numpy as np



class SOLOv2MaskHead(nn.Module):
    """
    MaskHead of SOLOv2.
    The code of this function is based on:
        https://github.com/WXinlong/SOLO/blob/master/mmdet/models/mask_heads/mask_feat_head.py

    Args:
        in_channels (int): The channel number of input Tensor.
        out_channels (int): The channel number of output Tensor.
        start_level (int): The position where the input starts.
        end_level (int): The position where the input ends.
        use_dcn_in_tower (bool): Whether to use dcn in tower or not.
    """
    __shared__ = ['norm_type']

    def __init__(self,
                 in_channels=256,
                 mid_channels=128,
                 out_channels=256,
                 start_level=0,
                 end_level=3,
                 use_dcn_in_tower=False,
                 norm_type='gn'):
        super(SOLOv2MaskHead, self).__init__()
        assert start_level >= 0 and end_level >= start_level
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.use_dcn_in_tower = use_dcn_in_tower
        self.range_level = end_level - start_level + 1
        self.use_dcn = True if self.use_dcn_in_tower else False
        self.convs_all_levels = []
        self.norm_type = norm_type
        for i in range(start_level, end_level + 1):
            conv_feat_name = 'mask_feat_head_convs_all_levels_{}'.format(i)
            conv_pre_feat = nn.Sequential()
            if i == start_level:
                conv_pre_feat.add_module(
                    conv_feat_name + '_conv' + str(i),
                    ConvNormLayer(
                        ch_in=self.in_channels,
                        ch_out=self.mid_channels,
                        filter_size=3,
                        stride=1,
                        use_dcn=self.use_dcn,
                        norm_type=self.norm_type))
                self.add_module('conv_pre_feat' + str(i), conv_pre_feat)
                self.convs_all_levels.append(conv_pre_feat)
            else:
                for j in range(i):
                    ch_in = 0
                    if j == 0:
                        ch_in = self.in_channels + 2 if i == end_level else self.in_channels
                    else:
                        ch_in = self.mid_channels
                    conv_pre_feat.add_module(
                        conv_feat_name + '_conv' + str(j),
                        ConvNormLayer(
                            ch_in=ch_in,
                            ch_out=self.mid_channels,
                            filter_size=3,
                            stride=1,
                            use_dcn=self.use_dcn,
                            norm_type=self.norm_type))
                    conv_pre_feat.add_module(
                        conv_feat_name + '_conv' + str(j) + 'act', nn.ReLU())
                    conv_pre_feat.add_module(
                        'upsample' + str(i) + str(j),
                        nn.Upsample(
                            scale_factor=2, mode='bilinear'))
                self.add_module('conv_pre_feat' + str(i), conv_pre_feat)
                self.convs_all_levels.append(conv_pre_feat)

        conv_pred_name = 'mask_feat_head_conv_pred_0'
        self.conv_pred = ConvNormLayer(
                ch_in=self.mid_channels,
                ch_out=self.out_channels,
                filter_size=1,
                stride=1,
                use_dcn=self.use_dcn,
                norm_type=self.norm_type)
        self.add_module(conv_pred_name, self.conv_pred)

    def forward(self, inputs):
        """
        Get SOLOv2MaskHead output.

        Args:
            inputs(list[Tensor]): feature map from each necks with shape of [N, C, H, W]
        Returns:
            ins_pred(Tensor): Output of SOLOv2MaskHead head
        """
        feat_all_level = F.relu(self.convs_all_levels[0](inputs[0]))
        for i in range(1, self.range_level):
            input_p = inputs[i]
            if i == (self.range_level - 1):
                input_feat = input_p
                x_range = torch.linspace(-1, 1, input_feat.shape[-1], dtype=torch.float32, device=input_feat.device)
                y_range = torch.linspace(-1, 1, input_feat.shape[-2], dtype=torch.float32, device=input_feat.device)
                y, x = torch.meshgrid([y_range, x_range])
                x = x.unsqueeze(0).unsqueeze(0)
                y = y.unsqueeze(0).unsqueeze(0)
                y = torch.tile(y, [input_feat.shape[0], 1, 1, 1])
                x = torch.tile(x, [input_feat.shape[0], 1, 1, 1])
                coord_feat = torch.cat([x, y], 1)
                input_p = torch.cat([input_p, coord_feat], 1)
            feat_all_level = feat_all_level.add_(self.convs_all_levels[i](input_p))
        ins_pred = F.relu(self.conv_pred(feat_all_level))

        return ins_pred


def matrix_nms(seg_masks, cate_labels, cate_scores, kernel='gaussian', sigma=2.0, sum_masks=None):
    """Matrix NMS for multi-class masks.
    Args:
        seg_masks (Tensor): shape (n, h, w)
        cate_labels (Tensor): shape (n), mask labels in descending order
        cate_scores (Tensor): shape (n), mask scores in descending order
        kernel (str):  'linear' or 'gauss'
        sigma (float): std in gaussian method
        sum_masks (Tensor): The sum of seg_masks
    Returns:
        Tensor: cate_scores_update, tensors of shape (n)
    """
    n_samples = len(cate_labels)
    if n_samples == 0:
        return []
    if sum_masks is None:
        sum_masks = seg_masks.sum((1, 2)).float()
    seg_masks = seg_masks.reshape(n_samples, -1).float()
    # inter.
    inter_matrix = torch.mm(seg_masks, seg_masks.transpose(1, 0))
    # union.
    sum_masks_x = sum_masks.expand(n_samples, n_samples)
    # iou.
    iou_matrix = (inter_matrix / (sum_masks_x + sum_masks_x.transpose(1, 0) - inter_matrix)).triu(diagonal=1)
    # label_specific matrix.
    cate_labels_x = cate_labels.expand(n_samples, n_samples)
    label_matrix = (cate_labels_x == cate_labels_x.transpose(1, 0)).float().triu(diagonal=1)

    # IoU compensation
    compensate_iou, _ = (iou_matrix * label_matrix).max(0)
    compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)

    # IoU decay
    decay_iou = iou_matrix * label_matrix

    # matrix nms
    if kernel == 'gaussian':
        decay_matrix = torch.exp(-1 * sigma * (decay_iou ** 2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou ** 2))
        decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
    elif kernel == 'linear':
        decay_matrix = (1-decay_iou)/(1-compensate_iou)
        decay_coefficient, _ = decay_matrix.min(0)
    else:
        raise NotImplementedError

    # update the score.
    cate_scores_update = cate_scores * decay_coefficient
    return cate_scores_update


class SOLOv2Head(nn.Module):
    """
    Head block for SOLOv2 network

    Args:
        num_classes (int): Number of output classes.
        in_channels (int): Number of input channels.
        seg_feat_channels (int): Num_filters of kernel & categroy branch convolution operation.
        stacked_convs (int): Times of convolution operation.
        num_grids (list[int]): List of feature map grids size.
        kernel_out_channels (int): Number of output channels in kernel branch.
        dcn_v2_stages (list): Which stage use dcn v2 in tower. It is between [0, stacked_convs).
        segm_strides (list[int]): List of segmentation area stride.
        solov2_loss (object): SOLOv2Loss instance.
        score_threshold (float): Threshold of categroy score.
        mask_nms (object): MaskMatrixNMS instance.
    """
    __inject__ = ['solov2_loss', 'mask_nms']
    __shared__ = ['norm_type', 'num_classes']

    def __init__(self,
                 num_classes=80,
                 in_channels=256,
                 seg_feat_channels=256,
                 stacked_convs=4,
                 num_grids=[40, 36, 24, 16, 12],
                 kernel_out_channels=256,
                 dcn_v2_stages=[],
                 segm_strides=[8, 8, 16, 32, 32],
                 solov2_loss=None,
                 score_threshold=0.1,
                 mask_threshold=0.5,
                 mask_nms=None,
                 norm_type='gn',
                 nms_cfg=None,
                 drop_block=False):
        super(SOLOv2Head, self).__init__()
        self.nms_cfg = nms_cfg
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.seg_num_grids = num_grids
        self.cate_out_channels = self.num_classes
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.kernel_out_channels = kernel_out_channels
        self.dcn_v2_stages = dcn_v2_stages
        self.segm_strides = segm_strides
        self.solov2_loss = solov2_loss
        self.mask_nms = mask_nms
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        self.norm_type = norm_type
        self.drop_block = drop_block

        self.kernel_pred_convs = []
        self.cate_pred_convs = []
        for i in range(self.stacked_convs):
            use_dcn = True if i in self.dcn_v2_stages else False
            ch_in = self.in_channels + 2 if i == 0 else self.seg_feat_channels
            kernel_conv = ConvNormLayer(
                    ch_in=ch_in,
                    ch_out=self.seg_feat_channels,
                    filter_size=3,
                    stride=1,
                    use_dcn=use_dcn,
                    norm_type=self.norm_type)
            self.add_module('bbox_head_kernel_convs_' + str(i), kernel_conv)
            self.kernel_pred_convs.append(kernel_conv)
            ch_in = self.in_channels if i == 0 else self.seg_feat_channels
            cate_conv = ConvNormLayer(
                    ch_in=ch_in,
                    ch_out=self.seg_feat_channels,
                    filter_size=3,
                    stride=1,
                    use_dcn=use_dcn,
                    norm_type=self.norm_type)
            self.add_module('bbox_head_cate_convs_' + str(i), cate_conv)
            self.cate_pred_convs.append(cate_conv)

        self.solo_kernel = nn.Conv2d(
                self.seg_feat_channels,
                self.kernel_out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True)
        solo_kernel_initializer = Normal(mean=0., std=0.01)
        solo_kernel_initializer.init(self.solo_kernel.weight)
        torch.nn.init.constant_(self.solo_kernel.bias, 0.0)
        self.add_module('bbox_head_solo_kernel', self.solo_kernel)

        self.solo_cate = nn.Conv2d(
                self.seg_feat_channels,
                self.cate_out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True)
        solo_cate_initializer = Normal(mean=0., std=0.01)
        solo_cate_initializer.init(self.solo_cate.weight)
        torch.nn.init.constant_(self.solo_cate.bias, float(-np.log((1 - 0.01) / 0.01)))
        self.add_module('bbox_head_solo_cate', self.solo_cate)

        if self.drop_block and self.training:
            self.drop_block_fun = DropBlock(
                block_size=3, keep_prob=0.9, name='solo_cate.dropblock')

    def _points_nms(self, heat, kernel_size=2):
        hmax = F.max_pool2d(heat, kernel_size=kernel_size, stride=1, padding=1)
        keep = (hmax[:, :, :-1, :-1] == heat)
        keep = keep.to(torch.float32)
        return heat * keep

    def _split_feats(self, feats):
        # [p2, p3, p4, p5, p6]
        # 有5个张量，5个张量的strides=[8, 8, 16, 32, 32]，所以先对首尾张量进行插值。
        # 一定要设置align_corners=False, align_mode=0才能和原版SOLO输出一致。
        x0 = F.interpolate(feats[0], scale_factor=0.5, align_corners=False, mode='bilinear')
        x4 = F.interpolate(feats[4], size=feats[3].shape[-2:], align_corners=False, mode='bilinear')
        return (x0, feats[1], feats[2], feats[3], x4)

    def forward(self, input, seg_pred, im_shape, ori_shape):
        """
        Get SOLOv2 head output

        Args:
            input (list): List of Tensors, output of backbone or neck stages
        Returns:
            cate_pred_list (list): Tensors of each category branch layer
            kernel_pred_list (list): Tensors of each kernel branch layer
        """
        feats = self._split_feats(input)
        # 有5个张量，5个张量的strides=[8, 8, 16, 32, 32]
        cate_pred_list = []
        kernel_pred_list = []
        for idx in range(len(self.seg_num_grids)):
            seg_num_grid = self.seg_num_grids[idx]   # 格子数。特征图会被插值成 格子数*格子数 的分辨率。
            cate_pred, kernel_pred = self._get_output_single(feats[idx], seg_num_grid)
            cate_pred_list.append(cate_pred)
            kernel_pred_list.append(kernel_pred)

        # return cate_pred_list, kernel_pred_list
        if self.training:
            pass
            # return self.forward_train(feats, targets)
        else:
            seg_masks, cate_labels, cate_scores, bbox_num = self.forward_eval(cate_pred_list, kernel_pred_list,
                                                                              seg_pred, im_shape, ori_shape)
            outs = {
                "segm": seg_masks,
                "bbox_num": bbox_num,
                'cate_label': cate_labels,
                'cate_score': cate_scores
            }
            return outs

    def _get_output_single(self, input, seg_num_grid):
        ins_kernel_feat = input
        # CoordConv
        x_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-1], dtype=torch.float32, device=ins_kernel_feat.device)
        y_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-2], dtype=torch.float32, device=ins_kernel_feat.device)
        y, x = torch.meshgrid([y_range, x_range])
        x = x.unsqueeze(0).unsqueeze(0)
        y = y.unsqueeze(0).unsqueeze(0)
        y = torch.tile(y, [ins_kernel_feat.shape[0], 1, 1, 1])
        x = torch.tile(x, [ins_kernel_feat.shape[0], 1, 1, 1])
        coord_feat = torch.cat([x, y], 1)
        ins_kernel_feat = torch.cat([ins_kernel_feat, coord_feat], 1)

        # kernel branch
        kernel_feat = ins_kernel_feat
        kernel_feat = F.interpolate(
            kernel_feat,
            size=[seg_num_grid, seg_num_grid],
            mode='bilinear',
            align_corners=False)
        cate_feat = kernel_feat[:, :-2, :, :]

        for kernel_layer in self.kernel_pred_convs:
            kernel_feat = F.relu(kernel_layer(kernel_feat))
        if self.drop_block and self.training:
            kernel_feat = self.drop_block_fun(kernel_feat)
        kernel_pred = self.solo_kernel(kernel_feat)
        # cate branch
        for cate_layer in self.cate_pred_convs:
            cate_feat = F.relu(cate_layer(cate_feat))
        if self.drop_block and self.training:
            cate_feat = self.drop_block_fun(cate_feat)
        cate_pred = self.solo_cate(cate_feat)

        if not self.training:
            cate_pred = self._points_nms(torch.sigmoid(cate_pred), kernel_size=2)
            cate_pred = cate_pred.permute((0, 2, 3, 1))
        return cate_pred, kernel_pred

    def get_loss(self, cate_preds, kernel_preds, ins_pred, ins_labels,
                 cate_labels, grid_order_list, fg_num):
        """
        Get loss of network of SOLOv2.

        Args:
            cate_preds (list): Tensor list of categroy branch output.
            kernel_preds (list): Tensor list of kernel branch output.
            ins_pred (list): Tensor list of instance branch output.
            ins_labels (list): List of instance labels pre batch.
            cate_labels (list): List of categroy labels pre batch.
            grid_order_list (list): List of index in pre grid.
            fg_num (int): Number of positive samples in a mini-batch.
        Returns:
            loss_ins (Tensor): The instance loss Tensor of SOLOv2 network.
            loss_cate (Tensor): The category loss Tensor of SOLOv2 network.
        """
        batch_size = paddle.shape(grid_order_list[0])[0]
        ins_pred_list = []
        for kernel_preds_level, grid_orders_level in zip(kernel_preds,
                                                         grid_order_list):
            if grid_orders_level.shape[1] == 0:
                ins_pred_list.append(None)
                continue
            grid_orders_level = paddle.reshape(grid_orders_level, [-1])
            reshape_pred = paddle.reshape(
                kernel_preds_level,
                shape=(paddle.shape(kernel_preds_level)[0],
                       paddle.shape(kernel_preds_level)[1], -1))
            reshape_pred = paddle.transpose(reshape_pred, [0, 2, 1])
            reshape_pred = paddle.reshape(
                reshape_pred, shape=(-1, paddle.shape(reshape_pred)[2]))
            gathered_pred = paddle.gather(reshape_pred, index=grid_orders_level)
            gathered_pred = paddle.reshape(
                gathered_pred,
                shape=[batch_size, -1, paddle.shape(gathered_pred)[1]])
            cur_ins_pred = ins_pred
            cur_ins_pred = paddle.reshape(
                cur_ins_pred,
                shape=(paddle.shape(cur_ins_pred)[0],
                       paddle.shape(cur_ins_pred)[1], -1))
            ins_pred_conv = paddle.matmul(gathered_pred, cur_ins_pred)
            cur_ins_pred = paddle.reshape(
                ins_pred_conv,
                shape=(-1, paddle.shape(ins_pred)[-2],
                       paddle.shape(ins_pred)[-1]))
            ins_pred_list.append(cur_ins_pred)

        num_ins = paddle.sum(fg_num)
        cate_preds = [
            paddle.reshape(
                paddle.transpose(cate_pred, [0, 2, 3, 1]),
                shape=(-1, self.cate_out_channels)) for cate_pred in cate_preds
        ]
        flatten_cate_preds = torch.cat(cate_preds)
        new_cate_labels = []
        for cate_label in cate_labels:
            new_cate_labels.append(paddle.reshape(cate_label, shape=[-1]))
        cate_labels = torch.cat(new_cate_labels)

        loss_ins, loss_cate = self.solov2_loss(
            ins_pred_list, ins_labels, flatten_cate_preds, cate_labels, num_ins)

        return {'loss_ins': loss_ins, 'loss_cate': loss_cate}

    def forward_eval(self, cate_preds, kernel_preds, seg_pred, im_shape, ori_shape):
        """
        Get prediction result of SOLOv2 network

        Args:
            cate_preds (list): List of Variables, output of categroy branch.
            kernel_preds (list): List of Variables, output of kernel branch.
            seg_pred (list): List of Variables, output of mask head stages.
            im_shape (Variables): [h, w] for input images.
            scale_factor (Variables): [scale, scale] for input images.
        Returns:
            seg_masks (Tensor): The prediction segmentation.
            cate_labels (Tensor): The prediction categroy label of each segmentation.
            seg_masks (Tensor): The prediction score of each segmentation.
        """
        num_levels = len(cate_preds)
        featmap_size = seg_pred.shape[-2:]
        seg_masks_list = []
        cate_labels_list = []
        cate_scores_list = []
        bbox_num_list = []
        cate_preds = [cate_pred * 1.0 for cate_pred in cate_preds]
        kernel_preds = [kernel_pred * 1.0 for kernel_pred in kernel_preds]

        batch_size = ori_shape.shape[0]
        # Currently only supports batch size == 1
        for idx in range(batch_size):
            cate_pred_list = [
                torch.reshape(cate_preds[i][idx], shape=(-1, self.cate_out_channels))
                for i in range(num_levels)
            ]
            seg_pred_list = seg_pred[idx:idx+1, :, :, :]
            kernel_pred_list = [
                torch.reshape(kernel_preds[i][idx].permute((1, 2, 0)), shape=(-1, self.kernel_out_channels))
                for i in range(num_levels)
            ]
            cate_pred_list = torch.cat(cate_pred_list, 0)
            kernel_pred_list = torch.cat(kernel_pred_list, 0)

            output = self.get_seg_single(
                cate_pred_list, seg_pred_list, kernel_pred_list, featmap_size,
                im_shape[idx], ori_shape[idx])
            if output is None:
                seg_masks, cate_labels, cate_scores = None, None, None
                bbox_num = 0
            else:
                seg_masks, cate_labels, cate_scores = output
                bbox_num = cate_labels.shape[0]
            seg_masks_list.append(seg_masks)
            cate_labels_list.append(cate_labels)
            cate_scores_list.append(cate_scores)
            bbox_num_list.append(bbox_num)
        return seg_masks_list, cate_labels_list, cate_scores_list, bbox_num_list

    def get_seg_single(self, cate_preds, seg_preds, kernel_preds, featmap_size,
                       im_shape, ori_shape):
        """
        The code of this function is based on:
            https://github.com/WXinlong/SOLO/blob/master/mmdet/models/anchor_heads/solov2_head.py#L385
        """
        hw = im_shape.to(torch.int32)
        h = hw[0]
        w = hw[1]
        upsampled_size_out = [featmap_size[0] * 4, featmap_size[1] * 4]

        # process.
        inds = (cate_preds > self.nms_cfg['score_threshold'])
        cate_scores = cate_preds[inds]
        if len(cate_scores) == 0:
            return None

        # cate_labels & kernel_preds
        inds = inds.nonzero()
        cate_labels = inds[:, 1]
        kernel_preds = kernel_preds[inds[:, 0]]

        # trans vector.
        size_trans = cate_labels.new_tensor(self.seg_num_grids).pow(2).cumsum(0)
        strides = kernel_preds.new_ones(size_trans[-1])

        n_stage = len(self.seg_num_grids)
        strides[:size_trans[0]] *= self.segm_strides[0]
        for ind_ in range(1, n_stage):
            strides[size_trans[ind_-1]:size_trans[ind_]] *= self.segm_strides[ind_]
        strides = strides[inds[:, 0]]

        # mask encoding.
        I, N = kernel_preds.shape
        kernel_preds = kernel_preds.view(I, N, 1, 1)
        seg_preds = F.conv2d(seg_preds, kernel_preds, stride=1).squeeze(0).sigmoid()
        # mask.
        seg_masks = seg_preds > self.mask_threshold
        seg_masks = seg_masks.to(torch.float32)
        sum_masks = seg_masks.sum((1, 2))

        # filter.
        keep = sum_masks > strides
        if keep.sum() == 0:
            return None

        seg_masks = seg_masks[keep, ...]
        seg_preds = seg_preds[keep, ...]
        sum_masks = sum_masks[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # maskness.
        seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks
        cate_scores *= seg_scores

        # sort and keep top nms_pre
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > self.nms_cfg['nms_top_k']:
            sort_inds = sort_inds[:self.nms_cfg['nms_top_k']]
        seg_masks = seg_masks[sort_inds, :, :]
        seg_preds = seg_preds[sort_inds, :, :]
        sum_masks = sum_masks[sort_inds]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        # Matrix NMS
        cate_scores = matrix_nms(seg_masks, cate_labels, cate_scores,
                                 kernel=self.nms_cfg['kernel'], sigma=self.nms_cfg['gaussian_sigma'], sum_masks=sum_masks)

        # filter.
        keep = cate_scores >= self.nms_cfg['post_threshold']
        if keep.sum() == 0:
            return None
        seg_preds = seg_preds[keep, :, :]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # sort and keep top_k
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > self.nms_cfg['keep_top_k']:
            sort_inds = sort_inds[:self.nms_cfg['keep_top_k']]
        seg_preds = seg_preds[sort_inds, :, :]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        seg_preds = F.interpolate(seg_preds.unsqueeze(0), size=upsampled_size_out, mode='bilinear')[:, :, :h, :w]
        ori_shape = ori_shape.to(torch.int32).cpu().detach().numpy()
        seg_masks = F.interpolate(seg_preds, size=(ori_shape[0], ori_shape[1]), mode='bilinear').squeeze(0)
        seg_masks = (seg_masks > self.mask_threshold).float()
        return seg_masks, cate_labels, cate_scores
