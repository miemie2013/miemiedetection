#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @miemie2013

import os
import sys

from mmdet.exp import PPYOLOEPlus_Method_Exp


class Exp(PPYOLOEPlus_Method_Exp):
    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # learning_rate
        self.basic_lr_per_img = 0.001 / (8. * 8.0)
        self.print_interval = 20
        self.eval_interval = 5

        self.depth_mult = 0.33
        self.width_mult = 0.50
        self.backbone['depth_mult'] = self.depth_mult
        self.backbone['width_mult'] = self.width_mult
        self.fpn['in_channels'] = [int(256 * self.width_mult), int(512 * self.width_mult), int(1024 * self.width_mult)]
        self.fpn['depth_mult'] = self.depth_mult
        self.fpn['width_mult'] = self.width_mult
        self.head['in_channels'] = [int(768 * self.width_mult), int(384 * self.width_mult), int(192 * self.width_mult)]
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
