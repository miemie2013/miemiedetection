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

        self.max_epoch = 16
        self.print_interval = 20
        self.eval_interval = 2
        self.warmup_epochs = 1
        self.cosinedecay_epochs = 20
        self.head['static_assigner_epoch'] = 4

        # learning_rate
        self.basic_lr_per_img = 0.24 / (4. * 48.0)

        self.scale = 1.5
        self.backbone['scale'] = self.scale
        self.fpn['in_channels'] = [make_divisible(128 * self.scale), make_divisible(256 * self.scale), make_divisible(512 * self.scale)]
        self.head['num_classes'] = self.num_classes
        self.static_assigner['num_classes'] = self.num_classes
        self.randomShape['sizes'] = [576, 608, 640, 672, 704]
        self.eval_height = 640
        self.eval_width = 640
        self.test_size = [self.eval_height, self.eval_width]
        self.resizeImage['target_size'] = 640
        self.head['eval_size'] = self.test_size

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
