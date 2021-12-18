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
    logger.info("Model Summary: {}".format(get_model_info(model_class_name, model, exp.test_size)))

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
