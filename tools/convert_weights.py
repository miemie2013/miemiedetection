#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger

import cv2
import torch
import paddle.fluid as fluid

# add python path of this repo to sys.path
import sys
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

from mmdet.exp import get_exp
from mmdet.utils import fuse_model, get_model_info, postprocess, vis, get_classes
from mmdet.models import *
from mmdet.models.custom_layers import *


def make_parser():
    parser = argparse.ArgumentParser("MieMieDetection convert weights")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint")
    parser.add_argument("-oc", "--output_ckpt", default=None, type=str, help="output checkpoint")
    parser.add_argument("-nc", "--num_classes", default=80, type=int, help="dataset num_classes")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    return parser


def copy_conv_bn(conv_unit, w, scale, offset, m, v, use_gpu):
    if use_gpu:
        conv_unit.conv.weight.data = torch.Tensor(w).cuda()
        conv_unit.bn.weight.data = torch.Tensor(scale).cuda()
        conv_unit.bn.bias.data = torch.Tensor(offset).cuda()
        conv_unit.bn.running_mean.data = torch.Tensor(m).cuda()
        conv_unit.bn.running_var.data = torch.Tensor(v).cuda()
    else:
        conv_unit.conv.weight.data = torch.Tensor(w)
        conv_unit.bn.weight.data = torch.Tensor(scale)
        conv_unit.bn.bias.data = torch.Tensor(offset)
        conv_unit.bn.running_mean.data = torch.Tensor(m)
        conv_unit.bn.running_var.data = torch.Tensor(v)


def copy_conv_gn(conv_unit, w, b, scale, offset, use_gpu):
    if use_gpu:
        conv_unit.conv.weight.data = torch.Tensor(w).cuda()
        conv_unit.conv.bias.data = torch.Tensor(b).cuda()
        conv_unit.gn.weight.data = torch.Tensor(scale).cuda()
        conv_unit.gn.bias.data = torch.Tensor(offset).cuda()
    else:
        conv_unit.conv.weight.data = torch.Tensor(w)
        conv_unit.conv.bias.data = torch.Tensor(b)
        conv_unit.gn.weight.data = torch.Tensor(scale)
        conv_unit.gn.bias.data = torch.Tensor(offset)

def copy_conv_af(conv_unit, w, scale, offset, use_gpu):
    if use_gpu:
        conv_unit.conv.weight.data = torch.Tensor(w).cuda()
        conv_unit.af.weight.data = torch.Tensor(scale).cuda()
        conv_unit.af.bias.data = torch.Tensor(offset).cuda()
    else:
        conv_unit.conv.weight.data = torch.Tensor(w)
        conv_unit.af.weight.data = torch.Tensor(scale)
        conv_unit.af.bias.data = torch.Tensor(offset)


def copy_conv(conv_layer, w, b, use_gpu):
    if use_gpu:
        conv_layer.weight.data = torch.Tensor(w).cuda()
        conv_layer.bias.data = torch.Tensor(b).cuda()
    else:
        conv_layer.weight.data = torch.Tensor(w)
        conv_layer.bias.data = torch.Tensor(b)


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    logger.info("Args: {}".format(args))

    # 强制改配置文件中的类别数为args.num_classes
    exp.num_classes = args.num_classes
    if getattr(exp, "head", None) is not None:
        if 'num_classes' in exp.head.keys():
            exp.head['num_classes'] = args.num_classes

    model = exp.get_model()
    # 算法名字
    model_class_name = model.__class__.__name__
    # logger.info("Model Summary: {}".format(get_model_info(model_class_name, model, exp.test_size)))

    use_gpu = False
    if args.device == "gpu":
        model.cuda()
        use_gpu = True
    model.eval()

    # 新增算法时这里也要增加elif
    if model_class_name == 'YOLOX':
        pass
    elif model_class_name == 'PPYOLO':
        state_dict = fluid.io.load_program_state(args.ckpt)
        backbone = model.backbone
        head = model.head
        if isinstance(backbone, Resnet50Vd):
            w = state_dict['conv1_1_weights']
            scale = state_dict['bnv1_1_scale']
            offset = state_dict['bnv1_1_offset']
            m = state_dict['bnv1_1_mean']
            v = state_dict['bnv1_1_variance']
            copy_conv_bn(backbone.stage1_conv1_1, w, scale, offset, m, v, use_gpu)

            w = state_dict['conv1_2_weights']
            scale = state_dict['bnv1_2_scale']
            offset = state_dict['bnv1_2_offset']
            m = state_dict['bnv1_2_mean']
            v = state_dict['bnv1_2_variance']
            copy_conv_bn(backbone.stage1_conv1_2, w, scale, offset, m, v, use_gpu)

            w = state_dict['conv1_3_weights']
            scale = state_dict['bnv1_3_scale']
            offset = state_dict['bnv1_3_offset']
            m = state_dict['bnv1_3_mean']
            v = state_dict['bnv1_3_variance']
            copy_conv_bn(backbone.stage1_conv1_3, w, scale, offset, m, v, use_gpu)

            nums = [3, 4, 6, 3]
            for nid, num in enumerate(nums):
                stage_name = 'res' + str(nid + 2)
                for kk in range(num):
                    block_name = stage_name + chr(ord("a") + kk)
                    conv_name1 = block_name + "_branch2a"
                    conv_name2 = block_name + "_branch2b"
                    conv_name3 = block_name + "_branch2c"
                    shortcut_name = block_name + "_branch1"

                    bn_name1 = 'bn' + conv_name1[3:]
                    bn_name2 = 'bn' + conv_name2[3:]
                    bn_name3 = 'bn' + conv_name3[3:]
                    shortcut_bn_name = 'bn' + shortcut_name[3:]

                    w = state_dict[conv_name1 + '_weights']
                    scale = state_dict[bn_name1 + '_scale']
                    offset = state_dict[bn_name1 + '_offset']
                    m = state_dict[bn_name1 + '_mean']
                    v = state_dict[bn_name1 + '_variance']
                    copy_conv_bn(backbone.get_block('stage%d_%d' % (2 + nid, kk)).conv1, w, scale, offset, m, v, use_gpu)

                    if nid == 3:  # DCNv2
                        conv_unit = backbone.get_block('stage%d_%d' % (2 + nid, kk)).conv2

                        offset_w = state_dict[conv_name2 + '_conv_offset.w_0']
                        offset_b = state_dict[conv_name2 + '_conv_offset.b_0']
                        if isinstance(conv_unit.conv, MyDCNv2):  # 如果是自实现的DCNv2
                            copy_conv(conv_unit.conv_offset, offset_w, offset_b, use_gpu)
                        # else:
                        #     copy_conv(conv_unit.conv.conv_offset_mask, offset_w, offset_b, use_gpu)

                        w = state_dict[conv_name2 + '_weights']
                        scale = state_dict[bn_name2 + '_scale']
                        offset = state_dict[bn_name2 + '_offset']
                        m = state_dict[bn_name2 + '_mean']
                        v = state_dict[bn_name2 + '_variance']

                        if isinstance(conv_unit.conv, MyDCNv2):  # 如果是自实现的DCNv2
                            conv_unit.conv.weight.data = torch.Tensor(w).cuda()
                            conv_unit.bn.weight.data = torch.Tensor(scale).cuda()
                            conv_unit.bn.bias.data = torch.Tensor(offset).cuda()
                            conv_unit.bn.running_mean.data = torch.Tensor(m).cuda()
                            conv_unit.bn.running_var.data = torch.Tensor(v).cuda()
                        # else:
                        #     copy_conv_bn(conv_unit, w, scale, offset, m, v, use_gpu)
                    else:
                        w = state_dict[conv_name2 + '_weights']
                        scale = state_dict[bn_name2 + '_scale']
                        offset = state_dict[bn_name2 + '_offset']
                        m = state_dict[bn_name2 + '_mean']
                        v = state_dict[bn_name2 + '_variance']
                        copy_conv_bn(backbone.get_block('stage%d_%d' % (2 + nid, kk)).conv2, w, scale, offset, m, v, use_gpu)

                    w = state_dict[conv_name3 + '_weights']
                    scale = state_dict[bn_name3 + '_scale']
                    offset = state_dict[bn_name3 + '_offset']
                    m = state_dict[bn_name3 + '_mean']
                    v = state_dict[bn_name3 + '_variance']
                    copy_conv_bn(backbone.get_block('stage%d_%d' % (2 + nid, kk)).conv3, w, scale, offset, m, v, use_gpu)

                    # 每个stage的第一个卷积块才有4个卷积层
                    if kk == 0:
                        w = state_dict[shortcut_name + '_weights']
                        scale = state_dict[shortcut_bn_name + '_scale']
                        offset = state_dict[shortcut_bn_name + '_offset']
                        m = state_dict[shortcut_bn_name + '_mean']
                        v = state_dict[shortcut_bn_name + '_variance']
                        copy_conv_bn(backbone.get_block('stage%d_%d' % (2 + nid, kk)).conv4, w, scale, offset, m, v, use_gpu)
            # head
            conv_block_num = 2
            num_classes = 80
            anchors = [[10, 13], [16, 30], [33, 23],
                       [30, 61], [62, 45], [59, 119],
                       [116, 90], [156, 198], [373, 326]]
            anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
            batch_size = 1
            norm_type = "bn"
            coord_conv = True
            iou_aware = True
            iou_aware_factor = 0.4
            block_size = 3
            scale_x_y = 1.05
            use_spp = True
            drop_block = True
            keep_prob = 0.9
            clip_bbox = True
            yolo_loss = None
            downsample = [32, 16, 8]
            in_channels = [2048, 1024, 512]
            nms_cfg = None
            is_train = False

            bn = 0
            gn = 0
            af = 0
            if norm_type == 'bn':
                bn = 1
            elif norm_type == 'gn':
                gn = 1
            elif norm_type == 'affine_channel':
                af = 1

            def copy_DetectionBlock(
                    _detection_block,
                    in_c,
                    channel,
                    coord_conv=True,
                    bn=0,
                    gn=0,
                    af=0,
                    conv_block_num=2,
                    is_first=False,
                    use_spp=True,
                    drop_block=True,
                    block_size=3,
                    keep_prob=0.9,
                    is_test=True,
                    name=''):
                kkk = 0
                for j in range(conv_block_num):
                    kkk += 1

                    conv_name = '{}.{}.0'.format(name, j)
                    w = state_dict[conv_name + '.conv.weights']
                    scale = state_dict[conv_name + '.bn.scale']
                    offset = state_dict[conv_name + '.bn.offset']
                    m = state_dict[conv_name + '.bn.mean']
                    v = state_dict[conv_name + '.bn.var']
                    copy_conv_bn(_detection_block.layers[kkk], w, scale, offset, m, v, use_gpu)
                    kkk += 1

                    if use_spp and is_first and j == 1:
                        kkk += 1

                        conv_name = '{}.{}.spp.conv'.format(name, j)
                        w = state_dict[conv_name + '.conv.weights']
                        scale = state_dict[conv_name + '.bn.scale']
                        offset = state_dict[conv_name + '.bn.offset']
                        m = state_dict[conv_name + '.bn.mean']
                        v = state_dict[conv_name + '.bn.var']
                        copy_conv_bn(_detection_block.layers[kkk], w, scale, offset, m, v, use_gpu)
                        kkk += 1

                        conv_name = '{}.{}.1'.format(name, j)
                        w = state_dict[conv_name + '.conv.weights']
                        scale = state_dict[conv_name + '.bn.scale']
                        offset = state_dict[conv_name + '.bn.offset']
                        m = state_dict[conv_name + '.bn.mean']
                        v = state_dict[conv_name + '.bn.var']
                        copy_conv_bn(_detection_block.layers[kkk], w, scale, offset, m, v, use_gpu)
                        kkk += 1
                    else:
                        conv_name = '{}.{}.1'.format(name, j)
                        w = state_dict[conv_name + '.conv.weights']
                        scale = state_dict[conv_name + '.bn.scale']
                        offset = state_dict[conv_name + '.bn.offset']
                        m = state_dict[conv_name + '.bn.mean']
                        v = state_dict[conv_name + '.bn.var']
                        copy_conv_bn(_detection_block.layers[kkk], w, scale, offset, m, v, use_gpu)
                        kkk += 1

                    if drop_block and j == 0 and not is_first:
                        kkk += 1

                if drop_block and is_first:
                    kkk += 1

                kkk += 1

                conv_name = '{}.2'.format(name)
                w = state_dict[conv_name + '.conv.weights']
                scale = state_dict[conv_name + '.bn.scale']
                offset = state_dict[conv_name + '.bn.offset']
                m = state_dict[conv_name + '.bn.mean']
                v = state_dict[conv_name + '.bn.var']
                copy_conv_bn(_detection_block.layers[kkk], w, scale, offset, m, v, use_gpu)
                kkk += 1

                conv_name = '{}.tip'.format(name)
                w = state_dict[conv_name + '.conv.weights']
                scale = state_dict[conv_name + '.bn.scale']
                offset = state_dict[conv_name + '.bn.offset']
                m = state_dict[conv_name + '.bn.mean']
                v = state_dict[conv_name + '.bn.var']
                copy_conv_bn(_detection_block.tip_layers[1], w, scale, offset, m, v, use_gpu)

            out_layer_num = len(downsample)
            for i in range(out_layer_num):
                copy_DetectionBlock(
                    head.detection_blocks[i],
                    in_c=in_channels[i],
                    channel=64 * (2 ** out_layer_num) // (2 ** i),
                    coord_conv=coord_conv,
                    bn=bn,
                    gn=gn,
                    af=af,
                    is_first=i == 0,
                    conv_block_num=conv_block_num,
                    use_spp=use_spp,
                    drop_block=drop_block,
                    block_size=block_size,
                    keep_prob=keep_prob,
                    is_test=(not is_train),
                    name="yolo_block.{}".format(i)
                )

                w = state_dict["yolo_output.{}.conv.weights".format(i)]
                b = state_dict["yolo_output.{}.conv.bias".format(i)]
                copy_conv(head.yolo_output_convs[i].conv, w, b, use_gpu)

                if i < out_layer_num - 1:
                    conv_name = "yolo_transition.{}".format(i)
                    w = state_dict[conv_name + '.conv.weights']
                    scale = state_dict[conv_name + '.bn.scale']
                    offset = state_dict[conv_name + '.bn.offset']
                    m = state_dict[conv_name + '.bn.mean']
                    v = state_dict[conv_name + '.bn.var']
                    copy_conv_bn(head.upsample_layers[i * 2], w, scale, offset, m, v, use_gpu)
        elif isinstance(backbone, Resnet18Vd):
            w = state_dict['conv1_1_weights']
            scale = state_dict['bnv1_1_scale']
            offset = state_dict['bnv1_1_offset']
            m = state_dict['bnv1_1_mean']
            v = state_dict['bnv1_1_variance']
            copy_conv_bn(backbone.stage1_conv1_1, w, scale, offset, m, v, use_gpu)

            w = state_dict['conv1_2_weights']
            scale = state_dict['bnv1_2_scale']
            offset = state_dict['bnv1_2_offset']
            m = state_dict['bnv1_2_mean']
            v = state_dict['bnv1_2_variance']
            copy_conv_bn(backbone.stage1_conv1_2, w, scale, offset, m, v, use_gpu)

            w = state_dict['conv1_3_weights']
            scale = state_dict['bnv1_3_scale']
            offset = state_dict['bnv1_3_offset']
            m = state_dict['bnv1_3_mean']
            v = state_dict['bnv1_3_variance']
            copy_conv_bn(backbone.stage1_conv1_3, w, scale, offset, m, v, use_gpu)

            nums = [2, 2, 2, 2]
            for nid, num in enumerate(nums):
                stage_name = 'res' + str(nid + 2)
                for kk in range(num):
                    block_name = stage_name + chr(ord("a") + kk)
                    conv_name1 = block_name + "_branch2a"
                    conv_name2 = block_name + "_branch2b"
                    shortcut_name = block_name + "_branch1"

                    bn_name1 = 'bn' + conv_name1[3:]
                    bn_name2 = 'bn' + conv_name2[3:]
                    shortcut_bn_name = 'bn' + shortcut_name[3:]

                    w = state_dict[conv_name1 + '_weights']
                    scale = state_dict[bn_name1 + '_scale']
                    offset = state_dict[bn_name1 + '_offset']
                    m = state_dict[bn_name1 + '_mean']
                    v = state_dict[bn_name1 + '_variance']
                    copy_conv_bn(backbone.get_block('stage%d_%d' % (2 + nid, kk)).conv1, w, scale, offset, m, v, use_gpu)

                    w = state_dict[conv_name2 + '_weights']
                    scale = state_dict[bn_name2 + '_scale']
                    offset = state_dict[bn_name2 + '_offset']
                    m = state_dict[bn_name2 + '_mean']
                    v = state_dict[bn_name2 + '_variance']
                    copy_conv_bn(backbone.get_block('stage%d_%d' % (2 + nid, kk)).conv2, w, scale, offset, m, v, use_gpu)

                    # 每个stage的第一个卷积块才有shortcut卷积层
                    if kk == 0:
                        w = state_dict[shortcut_name + '_weights']
                        scale = state_dict[shortcut_bn_name + '_scale']
                        offset = state_dict[shortcut_bn_name + '_offset']
                        m = state_dict[shortcut_bn_name + '_mean']
                        v = state_dict[shortcut_bn_name + '_variance']
                        copy_conv_bn(backbone.get_block('stage%d_%d' % (2 + nid, kk)).conv3, w, scale, offset, m, v, use_gpu)

            # head

            conv_block_num = 0
            num_classes = 80
            anchors = [[10, 14], [23, 27], [37, 58],
                       [81, 82], [135, 169], [344, 319]]
            anchor_masks = [[3, 4, 5], [0, 1, 2]]
            batch_size = 1
            norm_type = "bn"
            coord_conv = False
            iou_aware = False
            iou_aware_factor = 0.4
            block_size = 3
            scale_x_y = 1.05
            use_spp = False
            drop_block = True
            keep_prob = 0.9
            clip_bbox = True
            yolo_loss = None
            downsample = [32, 16]
            in_channels = [512, 256]
            nms_cfg = None
            is_train = False

            bn = 0
            gn = 0
            af = 0
            if norm_type == 'bn':
                bn = 1
            elif norm_type == 'gn':
                gn = 1
            elif norm_type == 'affine_channel':
                af = 1

            def copy_DetectionBlock(
                    _detection_block,
                    in_c,
                    channel,
                    coord_conv=True,
                    bn=0,
                    gn=0,
                    af=0,
                    conv_block_num=2,
                    is_first=False,
                    use_spp=True,
                    drop_block=True,
                    block_size=3,
                    keep_prob=0.9,
                    is_test=True,
                    name=''):
                kkk = 0
                for j in range(conv_block_num):
                    kkk += 1

                    conv_name = '{}.{}.0'.format(name, j)
                    w = state_dict[conv_name + '.conv.weights']
                    scale = state_dict[conv_name + '.bn.scale']
                    offset = state_dict[conv_name + '.bn.offset']
                    m = state_dict[conv_name + '.bn.mean']
                    v = state_dict[conv_name + '.bn.var']
                    copy_conv_bn(_detection_block.layers[kkk], w, scale, offset, m, v, use_gpu)
                    kkk += 1

                    if use_spp and is_first and j == 1:
                        kkk += 1

                        conv_name = '{}.{}.spp.conv'.format(name, j)
                        w = state_dict[conv_name + '.conv.weights']
                        scale = state_dict[conv_name + '.bn.scale']
                        offset = state_dict[conv_name + '.bn.offset']
                        m = state_dict[conv_name + '.bn.mean']
                        v = state_dict[conv_name + '.bn.var']
                        copy_conv_bn(_detection_block.layers[kkk], w, scale, offset, m, v, use_gpu)
                        kkk += 1

                        conv_name = '{}.{}.1'.format(name, j)
                        w = state_dict[conv_name + '.conv.weights']
                        scale = state_dict[conv_name + '.bn.scale']
                        offset = state_dict[conv_name + '.bn.offset']
                        m = state_dict[conv_name + '.bn.mean']
                        v = state_dict[conv_name + '.bn.var']
                        copy_conv_bn(_detection_block.layers[kkk], w, scale, offset, m, v, use_gpu)
                        kkk += 1
                    else:
                        conv_name = '{}.{}.1'.format(name, j)
                        w = state_dict[conv_name + '.conv.weights']
                        scale = state_dict[conv_name + '.bn.scale']
                        offset = state_dict[conv_name + '.bn.offset']
                        m = state_dict[conv_name + '.bn.mean']
                        v = state_dict[conv_name + '.bn.var']
                        copy_conv_bn(_detection_block.layers[kkk], w, scale, offset, m, v, use_gpu)
                        kkk += 1

                    if drop_block and j == 0 and not is_first:
                        kkk += 1

                if drop_block and is_first:
                    kkk += 1

                kkk += 1

                conv_name = '{}.2'.format(name)
                w = state_dict[conv_name + '.conv.weights']
                scale = state_dict[conv_name + '.bn.scale']
                offset = state_dict[conv_name + '.bn.offset']
                m = state_dict[conv_name + '.bn.mean']
                v = state_dict[conv_name + '.bn.var']
                copy_conv_bn(_detection_block.layers[kkk], w, scale, offset, m, v, use_gpu)
                kkk += 1

                conv_name = '{}.tip'.format(name)
                w = state_dict[conv_name + '.conv.weights']
                scale = state_dict[conv_name + '.bn.scale']
                offset = state_dict[conv_name + '.bn.offset']
                m = state_dict[conv_name + '.bn.mean']
                v = state_dict[conv_name + '.bn.var']
                copy_conv_bn(_detection_block.tip_layers[1], w, scale, offset, m, v, use_gpu)

            out_layer_num = len(downsample)
            for i in range(out_layer_num):
                copy_DetectionBlock(
                    head.detection_blocks[i],
                    in_c=in_channels[i],
                    channel=64 * (2 ** out_layer_num) // (2 ** i),
                    coord_conv=coord_conv,
                    bn=bn,
                    gn=gn,
                    af=af,
                    is_first=i == 0,
                    conv_block_num=conv_block_num,
                    use_spp=use_spp,
                    drop_block=drop_block,
                    block_size=block_size,
                    keep_prob=keep_prob,
                    is_test=(not is_train),
                    name="yolo_block.{}".format(i)
                )

                w = state_dict["yolo_output.{}.conv.weights".format(i)]
                b = state_dict["yolo_output.{}.conv.bias".format(i)]
                copy_conv(head.yolo_output_convs[i].conv, w, b, use_gpu)

                if i < out_layer_num - 1:
                    conv_name = "yolo_transition.{}".format(i)
                    w = state_dict[conv_name + '.conv.weights']
                    scale = state_dict[conv_name + '.bn.scale']
                    offset = state_dict[conv_name + '.bn.offset']
                    m = state_dict[conv_name + '.bn.mean']
                    v = state_dict[conv_name + '.bn.var']
                    copy_conv_bn(head.upsample_layers[i * 2], w, scale, offset, m, v, use_gpu)
    elif model_class_name == 'FCOS':
        ss = args.ckpt.split('.')
        if ss[-1] == 'pth':
            state_dict = torch.load(args.ckpt, map_location=torch.device('cpu'))
            backbone_dic = {}
            fpn_dic = {}
            fcos_head_dic = {}
            others = {}
            for key, value in state_dict.items():
                if 'tracked' in key:
                    continue
                if 'bottom_up' in key:
                    backbone_dic[key] = value.data.numpy()
                elif 'fpn' in key:
                    fpn_dic[key] = value.data.numpy()
                elif 'fcos_head' in key:
                    fcos_head_dic[key] = value.data.numpy()
                else:
                    others[key] = value.data.numpy()

            backbone = model.backbone
            fpn = model.fpn
            head = model.head
            if isinstance(backbone, Resnet50Vb):
                resnet = backbone

                # AdelaiDet里输入图片使用了BGR格式。这里做一下手脚使输入图片默认是RGB格式。
                w = backbone_dic['backbone.bottom_up.stem.conv1.weight']
                cpw = np.copy(w)
                w[:, 2, :, :] = cpw[:, 0, :, :]
                w[:, 0, :, :] = cpw[:, 2, :, :]
                scale = backbone_dic['backbone.bottom_up.stem.conv1.norm.weight']
                offset = backbone_dic['backbone.bottom_up.stem.conv1.norm.bias']
                m = backbone_dic['backbone.bottom_up.stem.conv1.norm.running_mean']
                v = backbone_dic['backbone.bottom_up.stem.conv1.norm.running_var']
                copy_conv_bn(resnet.stage1_conv1_1, w, scale, offset, m, v, use_gpu)

                nums = [3, 4, 6, 3]
                for nid, num in enumerate(nums):
                    stage_name = 'res' + str(nid + 2)
                    for kk in range(num):
                        conv_name1 = 'backbone.bottom_up.%s.%d.conv1' % (stage_name, kk)
                        w = backbone_dic[conv_name1 + '.weight']
                        scale = backbone_dic[conv_name1 + '.norm.weight']
                        offset = backbone_dic[conv_name1 + '.norm.bias']
                        m = backbone_dic[conv_name1 + '.norm.running_mean']
                        v = backbone_dic[conv_name1 + '.norm.running_var']
                        copy_conv_bn(resnet.get_block('stage%d_%d' % (2 + nid, kk)).conv1, w, scale, offset, m, v, use_gpu)

                        conv_name2 = 'backbone.bottom_up.%s.%d.conv2' % (stage_name, kk)
                        w = backbone_dic[conv_name2 + '.weight']
                        scale = backbone_dic[conv_name2 + '.norm.weight']
                        offset = backbone_dic[conv_name2 + '.norm.bias']
                        m = backbone_dic[conv_name2 + '.norm.running_mean']
                        v = backbone_dic[conv_name2 + '.norm.running_var']
                        copy_conv_bn(resnet.get_block('stage%d_%d' % (2 + nid, kk)).conv2, w, scale, offset, m, v, use_gpu)

                        conv_name3 = 'backbone.bottom_up.%s.%d.conv3' % (stage_name, kk)
                        w = backbone_dic[conv_name3 + '.weight']
                        scale = backbone_dic[conv_name3 + '.norm.weight']
                        offset = backbone_dic[conv_name3 + '.norm.bias']
                        m = backbone_dic[conv_name3 + '.norm.running_mean']
                        v = backbone_dic[conv_name3 + '.norm.running_var']
                        copy_conv_bn(resnet.get_block('stage%d_%d' % (2 + nid, kk)).conv3, w, scale, offset, m, v, use_gpu)

                        # 每个stage的第一个卷积块才有4个卷积层
                        if kk == 0:
                            shortcut_name = 'backbone.bottom_up.%s.%d.shortcut' % (stage_name, kk)
                            w = backbone_dic[shortcut_name + '.weight']
                            scale = backbone_dic[shortcut_name + '.norm.weight']
                            offset = backbone_dic[shortcut_name + '.norm.bias']
                            m = backbone_dic[shortcut_name + '.norm.running_mean']
                            v = backbone_dic[shortcut_name + '.norm.running_var']
                            copy_conv_bn(resnet.get_block('stage%d_%d' % (2 + nid, kk)).conv4, w, scale, offset, m, v, use_gpu)
                # fpn, 6个卷积层
                w = fpn_dic['backbone.fpn_lateral5.weight']
                b = fpn_dic['backbone.fpn_lateral5.bias']
                copy_conv(fpn.fpn_inner_convs[0].conv, w, b, use_gpu)

                w = fpn_dic['backbone.fpn_lateral4.weight']
                b = fpn_dic['backbone.fpn_lateral4.bias']
                copy_conv(fpn.fpn_inner_convs[1].conv, w, b, use_gpu)

                w = fpn_dic['backbone.fpn_lateral3.weight']
                b = fpn_dic['backbone.fpn_lateral3.bias']
                copy_conv(fpn.fpn_inner_convs[2].conv, w, b, use_gpu)

                w = fpn_dic['backbone.fpn_output5.weight']
                b = fpn_dic['backbone.fpn_output5.bias']
                copy_conv(fpn.fpn_convs[0].conv, w, b, use_gpu)

                w = fpn_dic['backbone.fpn_output4.weight']
                b = fpn_dic['backbone.fpn_output4.bias']
                copy_conv(fpn.fpn_convs[1].conv, w, b, use_gpu)

                w = fpn_dic['backbone.fpn_output3.weight']
                b = fpn_dic['backbone.fpn_output3.bias']
                copy_conv(fpn.fpn_convs[2].conv, w, b, use_gpu)

                # head
                num_convs = 4
                ids = [[0, 1], [3, 4], [6, 7], [9, 10]]
                for lvl in range(0, num_convs):
                    # conv + gn
                    w = fcos_head_dic['proposal_generator.fcos_head.cls_tower.%d.weight' % ids[lvl][0]]
                    b = fcos_head_dic['proposal_generator.fcos_head.cls_tower.%d.bias' % ids[lvl][0]]
                    scale = fcos_head_dic['proposal_generator.fcos_head.cls_tower.%d.weight' % ids[lvl][1]]
                    offset = fcos_head_dic['proposal_generator.fcos_head.cls_tower.%d.bias' % ids[lvl][1]]
                    copy_conv_gn(head.cls_convs[lvl], w, b, scale, offset, use_gpu)

                    # conv + gn
                    w = fcos_head_dic['proposal_generator.fcos_head.bbox_tower.%d.weight' % ids[lvl][0]]
                    b = fcos_head_dic['proposal_generator.fcos_head.bbox_tower.%d.bias' % ids[lvl][0]]
                    scale = fcos_head_dic['proposal_generator.fcos_head.bbox_tower.%d.weight' % ids[lvl][1]]
                    offset = fcos_head_dic['proposal_generator.fcos_head.bbox_tower.%d.bias' % ids[lvl][1]]
                    copy_conv_gn(head.reg_convs[lvl], w, b, scale, offset, use_gpu)

                # 类别分支最后的conv
                w = fcos_head_dic['proposal_generator.fcos_head.cls_logits.weight']
                b = fcos_head_dic['proposal_generator.fcos_head.cls_logits.bias']
                copy_conv(head.cls_pred.conv, w, b, use_gpu)

                # 坐标分支最后的conv
                w = fcos_head_dic['proposal_generator.fcos_head.bbox_pred.weight']
                b = fcos_head_dic['proposal_generator.fcos_head.bbox_pred.bias']
                copy_conv(head.reg_pred.conv, w, b, use_gpu)

                # centerness分支最后的conv
                w = fcos_head_dic['proposal_generator.fcos_head.ctrness.weight']
                b = fcos_head_dic['proposal_generator.fcos_head.ctrness.bias']
                copy_conv(head.ctn_pred.conv, w, b, use_gpu)

                # 3个scale。请注意，AdelaiDet在head部分是从小感受野到大感受野遍历，而PaddleDetection是从大感受野到小感受野遍历。所以这里scale顺序反过来。
                scale_0 = fcos_head_dic['proposal_generator.fcos_head.scales.0.scale']
                scale_1 = fcos_head_dic['proposal_generator.fcos_head.scales.1.scale']
                scale_2 = fcos_head_dic['proposal_generator.fcos_head.scales.2.scale']
                if use_gpu:
                    head.scales_on_reg[2].data = torch.Tensor(scale_0).cuda()
                    head.scales_on_reg[1].data = torch.Tensor(scale_1).cuda()
                    head.scales_on_reg[0].data = torch.Tensor(scale_2).cuda()
                else:
                    head.scales_on_reg[2].data = torch.Tensor(scale_0)
                    head.scales_on_reg[1].data = torch.Tensor(scale_1)
                    head.scales_on_reg[0].data = torch.Tensor(scale_2)
        elif ss[-1] == 'pdparams':
            state_dict = fluid.io.load_program_state(args.ckpt)
            backbone_dic = {}
            scale_on_reg_dic = {}
            fpn_dic = {}
            head_dic = {}
            others = {}
            for key, value in state_dict.items():
                # if 'tracked' in key:
                #     continue
                if 'branch' in key:
                    backbone_dic[key] = value
                elif 'scale_on_reg' in key:
                    scale_on_reg_dic[key] = value
                elif 'fpn' in key:
                    fpn_dic[key] = value
                elif 'fcos_head' in key:
                    head_dic[key] = value
                else:
                    others[key] = value

            backbone = model.backbone
            fpn = model.fpn
            head = model.head
            if isinstance(backbone, Resnet50Vb):
                resnet = backbone

                # AdelaiDet里输入图片使用了BGR格式。这里做一下手脚使输入图片默认是RGB格式。
                w = state_dict['conv1_weights']
                scale = state_dict['bn_conv1_scale']
                offset = state_dict['bn_conv1_offset']
                copy_conv_af(backbone.stage1_conv1_1, w, scale, offset, use_gpu)

                nums = [3, 4, 6, 3]
                for nid, num in enumerate(nums):
                    stage_name = 'res' + str(nid + 2)
                    for kk in range(num):
                        block_name = stage_name + chr(ord("a") + kk)

                        conv_name1 = block_name + "_branch2a"
                        bn_name1 = 'bn' + conv_name1[3:]
                        w = backbone_dic[conv_name1 + '_weights']
                        scale = backbone_dic[bn_name1 + '_scale']
                        offset = backbone_dic[bn_name1 + '_offset']
                        copy_conv_af(resnet.get_block('stage%d_%d' % (2 + nid, kk)).conv1, w, scale, offset, use_gpu)

                        conv_name2 = block_name + "_branch2b"
                        bn_name2 = 'bn' + conv_name2[3:]
                        w = state_dict[conv_name2 + '_weights']
                        scale = state_dict[bn_name2 + '_scale']
                        offset = state_dict[bn_name2 + '_offset']
                        copy_conv_af(resnet.get_block('stage%d_%d' % (2 + nid, kk)).conv2, w, scale, offset, use_gpu)

                        conv_name3 = block_name + "_branch2c"
                        bn_name3 = 'bn' + conv_name3[3:]
                        w = backbone_dic[conv_name3 + '_weights']
                        scale = backbone_dic[bn_name3 + '_scale']
                        offset = backbone_dic[bn_name3 + '_offset']
                        copy_conv_af(resnet.get_block('stage%d_%d' % (2 + nid, kk)).conv3, w, scale, offset, use_gpu)

                        # 每个stage的第一个卷积块才有4个卷积层
                        if kk == 0:
                            shortcut_name = block_name + "_branch1"
                            shortcut_bn_name = 'bn' + shortcut_name[3:]
                            w = backbone_dic[shortcut_name + '_weights']
                            scale = backbone_dic[shortcut_bn_name + '_scale']
                            offset = backbone_dic[shortcut_bn_name + '_offset']
                            copy_conv_af(resnet.get_block('stage%d_%d' % (2 + nid, kk)).conv4, w, scale, offset, use_gpu)
            # fpn
            w = fpn_dic['fpn_inner_res5_sum_w']
            b = fpn_dic['fpn_inner_res5_sum_b']
            copy_conv(fpn.fpn_inner_convs[0].conv, w, b, use_gpu)

            w = fpn_dic['fpn_inner_res4_sum_lateral_w']
            b = fpn_dic['fpn_inner_res4_sum_lateral_b']
            copy_conv(fpn.fpn_inner_convs[1].conv, w, b, use_gpu)

            w = fpn_dic['fpn_inner_res3_sum_lateral_w']
            b = fpn_dic['fpn_inner_res3_sum_lateral_b']
            copy_conv(fpn.fpn_inner_convs[2].conv, w, b, use_gpu)

            w = fpn_dic['fpn_res5_sum_w']
            b = fpn_dic['fpn_res5_sum_b']
            copy_conv(fpn.fpn_convs[0].conv, w, b, use_gpu)

            w = fpn_dic['fpn_res4_sum_w']
            b = fpn_dic['fpn_res4_sum_b']
            copy_conv(fpn.fpn_convs[1].conv, w, b, use_gpu)

            w = fpn_dic['fpn_res3_sum_w']
            b = fpn_dic['fpn_res3_sum_b']
            copy_conv(fpn.fpn_convs[2].conv, w, b, use_gpu)

            w = fpn_dic['fpn_6_w']
            b = fpn_dic['fpn_6_b']
            copy_conv(fpn.extra_convs[0].conv, w, b, use_gpu)

            w = fpn_dic['fpn_7_w']
            b = fpn_dic['fpn_7_b']
            copy_conv(fpn.extra_convs[1].conv, w, b, use_gpu)

            # head
            num_convs = 4
            for lvl in range(0, num_convs):
                # conv + gn
                conv_cls_name = 'fcos_head_cls_tower_conv_{}'.format(lvl)
                norm_name = conv_cls_name + "_norm"
                w = head_dic[conv_cls_name + "_weights"]
                b = head_dic[conv_cls_name + "_bias"]
                scale = head_dic[norm_name + "_scale"]
                offset = head_dic[norm_name + "_offset"]
                copy_conv_gn(head.cls_convs[lvl], w, b, scale, offset, use_gpu)

                # conv + gn
                conv_reg_name = 'fcos_head_reg_tower_conv_{}'.format(lvl)
                norm_name = conv_reg_name + "_norm"
                w = head_dic[conv_reg_name + "_weights"]
                b = head_dic[conv_reg_name + "_bias"]
                scale = head_dic[norm_name + "_scale"]
                offset = head_dic[norm_name + "_offset"]
                copy_conv_gn(head.reg_convs[lvl], w, b, scale, offset, use_gpu)

            # 类别分支最后的conv
            conv_cls_name = "fcos_head_cls"
            w = head_dic[conv_cls_name + "_weights"]
            b = head_dic[conv_cls_name + "_bias"]
            copy_conv(head.cls_pred.conv, w, b, use_gpu)

            # 坐标分支最后的conv
            conv_reg_name = "fcos_head_reg"
            w = head_dic[conv_reg_name + "_weights"]
            b = head_dic[conv_reg_name + "_bias"]
            copy_conv(head.reg_pred.conv, w, b, use_gpu)

            # centerness分支最后的conv
            conv_centerness_name = "fcos_head_centerness"
            w = head_dic[conv_centerness_name + "_weights"]
            b = head_dic[conv_centerness_name + "_bias"]
            copy_conv(head.ctn_pred.conv, w, b, use_gpu)

            # 5个scale
            fpn_names = ['fpn_7', 'fpn_6', 'fpn_res5_sum', 'fpn_res4_sum', 'fpn_res3_sum']
            i = 0
            for fpn_name in fpn_names:
                scale_i = scale_on_reg_dic["%s_scale_on_reg" % fpn_name]
                if use_gpu:
                    head.scales_on_reg[i].data = torch.Tensor(scale_i).cuda()
                else:
                    head.scales_on_reg[i].data = torch.Tensor(scale_i)
                i += 1
    else:
        raise NotImplementedError("Architectures \'{}\' is not implemented.".format(model_class_name))

    # save checkpoint.
    ckpt_state = {
        "start_epoch": 0,
        "model": model.state_dict(),
        "optimizer": None,
    }
    torch.save(ckpt_state, args.output_ckpt)
    logger.info("Done.")


if __name__ == "__main__":
    args = make_parser().parse_args()
    # 判断是否是调试状态
    isDebug = True if sys.gettrace() else False
    if isDebug:
        print('Debug Mode.')
        args.exp_file = '../' + args.exp_file
        args.ckpt = '../' + args.ckpt
        args.output_ckpt = '../' + args.output_ckpt
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
