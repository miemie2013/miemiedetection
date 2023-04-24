#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @miemie2013

import os
import sys

from mmdet.exp import PPYOLOEPlus_Method_Exp


class Exp(PPYOLOEPlus_Method_Exp):
    def __init__(self):
        super(Exp, self).__init__()
        # custom dataset
        self.num_classes = 20
        self.data_dir = '../VOCdevkit/VOC2012'
        self.cls_names = 'class_names/voc_classes.txt'
        self.ann_folder = "annotations2"
        self.train_ann = "voc2012_val2.json"
        self.val_ann = "voc2012_val2.json"
        self.train_ann = "voc2012_train.json"
        self.val_ann = "voc2012_val.json"
        self.train_image_folder = "JPEGImages"
        self.val_image_folder = "JPEGImages"

        # --------------  training config --------------------- #
        self.max_epoch = 16

        self.ema = True
        self.ema_decay = 0.9998
        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 20
        self.eval_interval = 2
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # learning_rate
        self.scheduler = "warm_cosinedecay"
        self.warmup_epochs = 1
        self.cosinedecay_epochs = 20
        self.basic_lr_per_img = 0.001 / (8. * 8.0)
        self.start_factor = 0.0
        self.head['static_assigner_epoch'] = 4

        self.depth_mult = 0.33
        self.width_mult = 0.50
        self.backbone['depth_mult'] = self.depth_mult
        self.backbone['width_mult'] = self.width_mult
        self.fpn['in_channels'] = [int(256 * self.width_mult), int(512 * self.width_mult), int(1024 * self.width_mult)]
        self.fpn['depth_mult'] = self.depth_mult
        self.fpn['width_mult'] = self.width_mult
        self.head['in_channels'] = [int(768 * self.width_mult), int(384 * self.width_mult), int(192 * self.width_mult)]
        self.head['num_classes'] = self.num_classes
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
