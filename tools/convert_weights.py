#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import os
import time
from loguru import logger

import cv2
import torch
import numpy as np
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
    parser.add_argument("-pp0", "--private_purpose_0", action='store_true', help='for private purpose 0')
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    return parser

def copy(name, w, std):
    value2 = torch.Tensor(w)
    value = std[name]
    assert value.ndim == value2.ndim
    mul1 = np.prod(value.shape)
    mul2 = np.prod(value2.shape)
    assert mul1 == mul2
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
        if args.private_purpose_0:
            ckpt_file = args.ckpt
            ckpt = torch.load(ckpt_file, map_location="cpu")
            state_dict = ckpt['model']
            state_dict2 = {}
            for key, value in state_dict.items():
                if key.startswith('backbone.backbone.'):
                    state_dict2[key.replace('backbone.backbone.', 'backbone.')] = state_dict[key]
                elif key.startswith('backbone.'):
                    state_dict2[key.replace('backbone.', 'neck.')] = state_dict[key]
                elif key.startswith('head.'):
                    state_dict2[key] = state_dict[key]
                else:
                    raise NotImplementedError("not implemented.")
            new_state_dict = {}
            new_state_dict['state_dict'] = state_dict2
            torch.save(new_state_dict, args.output_ckpt)
            logger.info("Done.")
            return 0
        # 转 ppdetection 的权重
        with open(args.ckpt, 'rb') as f:
            state_dict = pickle.load(f) if six.PY2 else pickle.load(f, encoding='latin1')
        state_dict = state_dict['model']
        print('======================== convert ppdetection weights ========================')
        for key in state_dict.keys():
            name2 = key
            w = state_dict[key]   # w是一个元组，飞桨是这样保存的
            w = w[1]
            if 'StructuredToParameterName@@' in key:
                continue
            if 'num_batches_tracked' in key:
                continue
            else:
                if '._mean' in key:
                    name2 = name2.replace('._mean', '.running_mean')
                if '._variance' in key:
                    name2 = name2.replace('._variance', '.running_var')
                copy(name2, w, model_std)
        if args.only_backbone:
            delattr(model, "head")
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
    elif model_class_name == 'PicoDet':
        temp_x = torch.randn((2, 3, 416, 416))
        temp_scale_factor = torch.ones((2, 2))
        if args.device == "gpu":
            temp_x = temp_x.cuda()
            temp_scale_factor = temp_scale_factor.cuda()
        # temp_out = model(temp_x, temp_scale_factor)
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
        head = model.head
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
                for ii in range(10):
                    for jj in range(10):
                        if 'cls_conv_dw{}.{}'.format(ii, jj) in key:
                            name2 = name2.replace('cls_conv_dw{}.{}'.format(ii, jj), 'cls_conv_dw{}_{}'.format(ii, jj))
                        if 'cls_conv_pw{}.{}'.format(ii, jj) in key:
                            name2 = name2.replace('cls_conv_pw{}.{}'.format(ii, jj), 'cls_conv_pw{}_{}'.format(ii, jj))
                        if 'reg_conv_dw{}.{}'.format(ii, jj) in key:
                            name2 = name2.replace('reg_conv_dw{}.{}'.format(ii, jj), 'reg_conv_dw{}_{}'.format(ii, jj))
                        if 'reg_conv_pw{}.{}'.format(ii, jj) in key:
                            name2 = name2.replace('reg_conv_pw{}.{}'.format(ii, jj), 'reg_conv_pw{}_{}'.format(ii, jj))
                if args.only_backbone and 'backbone.' not in name2:
                    name2 = 'backbone.' + name2
                if args.only_backbone:
                    if name2 in ['backbone.last_conv.weight', 'backbone.fc.weight', 'backbone.fc.bias']:
                        continue
                copy(name2, w, model_std)
        if args.only_backbone:
            delattr(model, "neck")
            delattr(model, "head")
    elif model_class_name == 'SOLO':
        temp_x = torch.randn((1, 3, 640, 640))
        temp_im_shape = torch.ones((1, 2)) * 640
        temp_ori_shape = torch.ones((1, 2)) * 640
        if args.device == "gpu":
            temp_x = temp_x.cuda()
            temp_im_shape = temp_im_shape.cuda()
            temp_ori_shape = temp_ori_shape.cuda()
        temp_out = model(temp_x, temp_im_shape, temp_ori_shape)
        with open(args.ckpt, 'rb') as f:
            state_dict = pickle.load(f) if six.PY2 else pickle.load(f, encoding='latin1')
        # state_dict = fluid.io.load_program_state(args.ckpt)
        backbone_dic = {}
        fpn_dic = {}
        solov2_head_dic = {}
        mask_head_dic = {}
        others = {}
        for key, value in state_dict.items():
            if 'tracked' in key:
                continue
            if 'backbone.' in key:
                backbone_dic[key] = value
            elif 'neck.' in key:
                fpn_dic[key] = value
            elif 'solov2_head.' in key:
                solov2_head_dic[key] = value
            elif 'mask_head.' in key:
                mask_head_dic[key] = value
            else:
                others[key] = value
        backbone_dic2 = {}
        fpn_dic2 = {}
        solov2_head_dic2 = {}
        mask_head_dic2 = {}
        others2 = {}
        for key, value in model_std.items():
            if 'tracked' in key:
                continue
            if 'backbone.' in key:
                backbone_dic2[key] = value
            elif 'neck.' in key:
                fpn_dic2[key] = value
            elif 'solov2_head.' in key:
                solov2_head_dic2[key] = value
            elif 'mask_head.' in key:
                mask_head_dic2[key] = value
            else:
                others2[key] = value
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

                # SOLOv2Head
                if 'bbox_head.kernel_convs.' in key:
                    name2 = name2.replace('bbox_head.kernel_convs.', 'bbox_head_kernel_convs_')
                if 'bbox_head.cate_convs.' in key:
                    name2 = name2.replace('bbox_head.cate_convs.', 'bbox_head_cate_convs_')
                if 'bbox_head.solo_kernel' in key:
                    name2 = name2.replace('bbox_head.solo_kernel', 'bbox_head_solo_kernel')
                if 'bbox_head.solo_cate' in key:
                    name2 = name2.replace('bbox_head.solo_cate', 'bbox_head_solo_cate')

                # SOLOv2MaskHead
                if 'mask_feat_head.conv_pred.0' in key:
                    name2 = name2.replace('mask_feat_head.conv_pred.0', 'mask_feat_head_conv_pred_0')
                if 'mask_feat_head.convs_all_levels.0.conv' in key:
                    name2 = name2.replace('mask_feat_head.convs_all_levels.0.conv', 'mask_feat_head_convs_all_levels_0_conv')
                if 'mask_feat_head.convs_all_levels.1.conv' in key:
                    name2 = name2.replace('mask_feat_head.convs_all_levels.1.conv', 'mask_feat_head_convs_all_levels_1_conv')
                if 'mask_feat_head.convs_all_levels.2.conv' in key:
                    name2 = name2.replace('mask_feat_head.convs_all_levels.2.conv', 'mask_feat_head_convs_all_levels_2_conv')
                if 'mask_feat_head.convs_all_levels.3.conv' in key:
                    name2 = name2.replace('mask_feat_head.convs_all_levels.3.conv', 'mask_feat_head_convs_all_levels_3_conv')

                if args.only_backbone:
                    name2 = 'backbone.' + name2
                copy(name2, w, model_std)
        if args.only_backbone:
            delattr(model, "neck")
            delattr(model, "yolo_head")
    elif model_class_name == 'FCOS':
        pass
    elif model_class_name == 'DETR':
        temp_x = torch.randn((1, 3, 640, 640))
        temp_im_shape = torch.ones((1, 2)) * 640
        temp_ori_shape = torch.ones((1, 2)) * 640
        if args.device == "gpu":
            temp_x = temp_x.cuda()
            temp_im_shape = temp_im_shape.cuda()
            temp_ori_shape = temp_ori_shape.cuda()

        inputs = {}
        inputs['image'] = temp_x
        inputs['im_shape'] = temp_im_shape
        inputs['scale_factor'] = temp_ori_shape
        model.cuda()
        model.eval()
        temp_out = model(inputs)

        with open(args.ckpt, 'rb') as f:
            state_dict = pickle.load(f) if six.PY2 else pickle.load(f, encoding='latin1')
        # state_dict = fluid.io.load_program_state(args.ckpt)
        backbone_dic = {}
        fpn_dic = {}
        transformer_dic = {}
        others = {}
        for key, value in state_dict.items():
            if 'tracked' in key:
                continue
            if 'backbone' in key:
                backbone_dic[key] = value
            elif 'neck' in key:
                fpn_dic[key] = value
            elif 'transformer' in key:
                transformer_dic[key] = value
            else:
                others[key] = value
        backbone_dic2 = {}
        fpn_dic2 = {}
        transformer_dic2 = {}
        others2 = {}
        for key, value in model_std.items():
            if 'tracked' in key:
                continue
            if 'backbone' in key:
                backbone_dic2[key] = value
            elif 'neck' in key:
                fpn_dic2[key] = value
            elif 'transformer' in key:
                transformer_dic2[key] = value
            else:
                others2[key] = value
        print('======================== find weights not in transformer_dic2 ========================')
        for key, value in transformer_dic.items():
            if key not in transformer_dic2.keys():
                print(key)
        print('======================== find weights not in transformer_dic ========================')
        for key, value in transformer_dic2.items():
            if key not in transformer_dic.keys():
                print(key)
        print('======================== find Embedding Layer name and Linear name ========================')
        Embedding_Layer_names = []
        Linear_Layer_names = []
        for key, module in model.named_modules():
            if isinstance(module, nn.Embedding):
                print(key)
                Embedding_Layer_names.append(key)
            if isinstance(module, nn.Linear):
                print(key)
                Linear_Layer_names.append(key)
        print('======================== copy weights ========================')
        for key in state_dict.keys():
            name2 = key
            w = state_dict[key]
            if 'StructuredToParameterName@@' in key:
                continue
            else:
                if key.startswith('transformer.input_proj.'):
                    name2 = name2.replace('.conv.', '.0.')
                    name2 = name2.replace('.norm.', '.1.')
                if '._mean' in key:
                    name2 = name2.replace('._mean', '.running_mean')
                if '._variance' in key:
                    name2 = name2.replace('._variance', '.running_var')
                if len(w.shape) == 2:
                    # 要注意，pytorch的fc层和paddle的fc层的权重weight需要转置一下才能等价！！！
                    # 但是Embedding层的权重不需要转置
                    layer_type = -1   # 0代表是Embedding层的权重, 1代表是Linear层的权重
                    for prefix in Embedding_Layer_names:
                        if name2.startswith(prefix):
                            layer_type = 0
                            break
                    for prefix in Linear_Layer_names:
                        if name2.startswith(prefix):
                            layer_type = 1
                            break
                    if layer_type == 1:
                        print('transpose param: name=\'%s\' '%name2)
                        w = w.transpose([1, 0])
                    elif layer_type == 0:
                        pass
                    elif layer_type == -1:
                        need_zhuanzhi = False
                        if name2.endswith('.in_proj_weight'):  # MultiHeadAttention 的 in_proj_weight 需要转置
                            need_zhuanzhi = True
                        if need_zhuanzhi:
                            print('transpose param: name=\'%s\' '%name2)
                            w = w.transpose([1, 0])
                        else:
                            raise NotImplementedError(name2)
                copy(name2, w, model_std)
        if args.only_backbone:
            delattr(model, "neck")
            delattr(model, "yolo_head")
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
