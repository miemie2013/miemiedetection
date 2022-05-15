#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger

import cv2
import torch
# import paddle.fluid as fluid
import pickle
import six

# add python path of this repo to sys.path
import sys
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

from mmdet.exp import get_exp
from mmdet.utils import fuse_model, get_model_info, postprocess, vis, get_classes
from mmdet.models import *
from mmdet.models.custom_layers import *
from mmdet.models.necks.yolo_fpn import PPYOLOFPN, PPYOLOPAN


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
        "--only_backbone",
        default=False,
        type=bool,
        help="only convert backbone",
    )
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

def copy(name, w, std):
    value2 = torch.Tensor(w)
    value = std[name]
    value.copy_(value2)
    std[name] = value

def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    logger.info("Args: {}".format(args))

    # 强制改配置文件中的类别数为args.num_classes
    exp.num_classes = args.num_classes
    if getattr(exp, "head", None) is not None:
        if 'num_classes' in exp.head.keys():
            exp.head['num_classes'] = args.num_classes

    # 这些预训练骨干网络没有使用DCNv2
    no_dcnv2_backbones = ['ResNet50_vd_ssld_pretrained.pdparams', 'ResNet101_vd_ssld_pretrained.pdparams']
    if args.only_backbone and args.ckpt in no_dcnv2_backbones:
        exp.backbone['dcn_v2_stages'] = [-1]

    model = exp.get_model()
    # 算法名字
    model_class_name = model.__class__.__name__
    # logger.info("Model Summary: {}".format(get_model_info(model_class_name, model, exp.test_size)))

    use_gpu = False
    if args.device == "gpu":
        model.cuda()
        use_gpu = True
    model.eval()
    model_std = model.state_dict()

    # 新增算法时这里也要增加elif
    if model_class_name == 'YOLOX':
        pass
    elif model_class_name == 'PPYOLO':
        with open(args.ckpt, 'rb') as f:
            state_dict = pickle.load(f) if six.PY2 else pickle.load(f, encoding='latin1')
        # state_dict = fluid.io.load_program_state(args.ckpt)
        backbone_dic = {}
        fpn_dic = {}
        head_dic = {}
        others = {}
        for key, value in state_dict.items():
            if 'tracked' in key:
                continue
            if 'backbone' in key:
                backbone_dic[key] = value
            elif 'neck' in key:
                fpn_dic[key] = value
            elif 'head' in key:
                head_dic[key] = value
            else:
                others[key] = value
        backbone_dic2 = {}
        fpn_dic2 = {}
        head_dic2 = {}
        others2 = {}
        for key, value in model_std.items():
            if 'tracked' in key:
                continue
            if 'backbone' in key:
                backbone_dic2[key] = value
            elif 'neck' in key:
                fpn_dic2[key] = value
            elif 'head' in key:
                head_dic2[key] = value
            else:
                others2[key] = value
        backbone = model.backbone
        fpn = model.neck
        head = model.yolo_head
        for key in state_dict.keys():
            name2 = key
            w = state_dict[key]
            if 'StructuredToParameterName@@' in key:
                continue
            else:
                if '._mean' in key:
                    name2 = name2.replace('._mean', '.running_mean')
                if '._variance' in key:
                    name2 = name2.replace('._variance', '.running_var')
                if 'yolo_block.' in key:
                    name2 = name2.replace('yolo_block.', 'yolo_block_')
                if 'yolo_transition.' in key:
                    name2 = name2.replace('yolo_transition.', 'yolo_transition_')
                if 'yolo_output.' in key:
                    name2 = name2.replace('yolo_output.', 'yolo_output_')
                if 'fpn.' in key:
                    name2 = name2.replace('fpn.', 'fpn_')
                    name2 = name2.replace('0.0', '0_0')
                    name2 = name2.replace('0.1', '0_1')
                    name2 = name2.replace('1.0', '1_0')
                    name2 = name2.replace('1.1', '1_1')
                    name2 = name2.replace('2.0', '2_0')
                    name2 = name2.replace('2.1', '2_1')
                if 'fpn_transition.' in key:
                    name2 = name2.replace('fpn_transition.', 'fpn_transition_')
                if 'pan_transition.' in key:
                    name2 = name2.replace('pan_transition.', 'pan_transition_')
                if 'pan.' in key:
                    name2 = name2.replace('pan.', 'pan_')
                    name2 = name2.replace('0.0', '0_0')
                    name2 = name2.replace('0.1', '0_1')
                    name2 = name2.replace('1.0', '1_0')
                    name2 = name2.replace('1.1', '1_1')
                    name2 = name2.replace('2.0', '2_0')
                    name2 = name2.replace('2.1', '2_1')
                copy(name2, w, model_std)
        if args.only_backbone:
            delattr(model, "neck")
            delattr(model, "yolo_head")
    elif model_class_name == 'PPYOLOE':
        temp_x = torch.randn((2, 3, 640, 640))
        temp_scale_factor = torch.ones((2, 2))
        if args.device == "gpu":
            temp_x = temp_x.cuda()
            temp_scale_factor = temp_scale_factor.cuda()
        temp_out = model(temp_x, temp_scale_factor)
        with open(args.ckpt, 'rb') as f:
            state_dict = pickle.load(f) if six.PY2 else pickle.load(f, encoding='latin1')
        # state_dict = fluid.io.load_program_state(args.ckpt)
        backbone_dic = {}
        fpn_dic = {}
        head_dic = {}
        others = {}
        for key, value in state_dict.items():
            if 'tracked' in key:
                continue
            if 'backbone' in key:
                backbone_dic[key] = value
            elif 'neck' in key:
                fpn_dic[key] = value
            elif 'head' in key:
                head_dic[key] = value
            else:
                others[key] = value
        backbone_dic2 = {}
        fpn_dic2 = {}
        head_dic2 = {}
        others2 = {}
        for key, value in model_std.items():
            if 'tracked' in key:
                continue
            if 'backbone' in key:
                backbone_dic2[key] = value
            elif 'neck' in key:
                fpn_dic2[key] = value
            elif 'head' in key:
                head_dic2[key] = value
            else:
                others2[key] = value
        backbone = model.backbone
        fpn = model.neck
        head = model.yolo_head
        for key in state_dict.keys():
            name2 = key
            w = state_dict[key]
            if 'StructuredToParameterName@@' in key:
                continue
            else:
                if '._mean' in key:
                    name2 = name2.replace('._mean', '.running_mean')
                if '._variance' in key:
                    name2 = name2.replace('._variance', '.running_var')
                if 'yolo_block.' in key:
                    name2 = name2.replace('yolo_block.', 'yolo_block_')
                if 'yolo_transition.' in key:
                    name2 = name2.replace('yolo_transition.', 'yolo_transition_')
                if 'yolo_output.' in key:
                    name2 = name2.replace('yolo_output.', 'yolo_output_')
                if 'fpn.' in key:
                    name2 = name2.replace('fpn.', 'fpn_')
                    name2 = name2.replace('0.0', '0_0')
                    name2 = name2.replace('0.1', '0_1')
                    name2 = name2.replace('1.0', '1_0')
                    name2 = name2.replace('1.1', '1_1')
                    name2 = name2.replace('2.0', '2_0')
                    name2 = name2.replace('2.1', '2_1')
                if 'fpn_transition.' in key:
                    name2 = name2.replace('fpn_transition.', 'fpn_transition_')
                if 'pan_transition.' in key:
                    name2 = name2.replace('pan_transition.', 'pan_transition_')
                if 'pan.' in key:
                    name2 = name2.replace('pan.', 'pan_')
                    name2 = name2.replace('0.0', '0_0')
                    name2 = name2.replace('0.1', '0_1')
                    name2 = name2.replace('1.0', '1_0')
                    name2 = name2.replace('1.1', '1_1')
                    name2 = name2.replace('2.0', '2_0')
                    name2 = name2.replace('2.1', '2_1')
                if args.only_backbone:
                    name2 = 'backbone.' + name2
                copy(name2, w, model_std)
        if args.only_backbone:
            delattr(model, "neck")
            delattr(model, "yolo_head")
    elif model_class_name == 'FCOS':
        pass
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
