#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import sys

from mmdet.exp import YOLOXExp


class Exp(YOLOXExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.act = 'relu'
        self.use_focus = False
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.num_classes = 20
        self.data_dir = '../VOCdevkit/VOC2012'
        self.cls_names = 'class_names/voc_classes.txt'
        self.ann_folder = "annotations2"
        self.train_ann = "voc2012_val2.json"
        self.train_ann = "voc2012_val8imgs.json"
        self.val_ann = "voc2012_val2.json"
        self.train_ann = "voc2012_train.json"
        self.val_ann = "voc2012_val_2008_000073.json"
        self.val_ann = "voc2012_val.json"
        self.train_image_folder = "JPEGImages"
        self.val_image_folder = "JPEGImages"

        # self.num_classes = 80
        # self.data_dir = '../COCO'
        # self.cls_names = 'class_names/coco_classes.txt'
        # self.ann_folder = "annotations"
        # self.train_ann = "instances_val2017.json"
        # self.val_ann = "instances_val2017.json"
        # self.train_image_folder = "val2017"
        # self.val_image_folder = "val2017"

        self.multiscale_range = 0
        self.warmup_epochs = 1
        self.max_epoch = 16
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 64.0
        self.no_aug_epochs = 5
        self.min_lr_ratio = 0.05
        self.print_interval = 20
        self.eval_interval = 4

        # 判断是否是调试状态
        isDebug = True if sys.gettrace() else False
        if isDebug:
            print('Debug Mode.')
            self.data_dir = '../' + self.data_dir
            self.cls_names = '../' + self.cls_names
            self.output_dir = '../' + self.output_dir
