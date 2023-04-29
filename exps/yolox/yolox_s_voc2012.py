#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from mmdet.exp import YOLOXExp


class Exp(YOLOXExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

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

        self.multiscale_range = 5
        self.warmup_epochs = 1
        self.max_epoch = 16
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 64.0
        self.no_aug_epochs = 5
        self.min_lr_ratio = 0.05
        self.print_interval = 20
        self.eval_interval = 2
