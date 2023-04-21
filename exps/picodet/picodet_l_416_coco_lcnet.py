#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @miemie2013

import os
import sys

from mmdet.exp import PicoDet_Method_Exp
from mmdet.models.backbones.lcnet import make_divisible


class Exp(PicoDet_Method_Exp):
    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # learning_rate
        self.basic_lr_per_img = 0.12 / (4. * 24.0)
        self.max_epoch = 250
        self.cosinedecay_epochs = 300

        self.scale = 2.0
        self.backbone['scale'] = self.scale
        self.fpn['in_channels'] = [make_divisible(128 * self.scale), make_divisible(256 * self.scale), make_divisible(512 * self.scale)]
        self.fpn['out_channels'] = 160
        self.head['feat_in_chan'] = 160
        self.head['num_classes'] = self.num_classes
        self.conv_feat['feat_in'] = 160
        self.conv_feat['feat_out'] = 160
        self.conv_feat['num_convs'] = 4
        self.conv_feat['num_fpn_stride'] = 4
        self.conv_feat['share_cls_reg'] = True
        self.conv_feat['use_se'] = True
        self.static_assigner['num_classes'] = self.num_classes

        # ---------------- dataloader config ---------------- #
        # 默认是4。如果报错“OSError: [WinError 1455] 页面文件太小,无法完成操作”，设置为2或0解决。
        self.data_num_workers = 2
        self.eval_data_num_workers = 1

        # 判断是否是调试状态
        isDebug = True if sys.gettrace() else False
        if isDebug:
            print('Debug Mode.')
            self.data_dir = '../' + self.data_dir
            self.cls_names = '../' + self.cls_names
            self.output_dir = '../' + self.output_dir
