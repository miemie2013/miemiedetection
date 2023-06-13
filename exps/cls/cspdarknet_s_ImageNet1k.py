#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @miemie2013

import os
import sys

from mmcls.exp import BaseCls_Method_Exp


class Exp(BaseCls_Method_Exp):
    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        # custom dataset
        self.num_classes = 1000
        self.data_dir = '../ImageNet1k/Light_ILSVRC2012'
        self.cls_names = None
        self.train_ann = "train_list.txt"    # self.train_ann should be placed in the self.data_dir directory.
        self.val_ann = "val_list.txt"        # self.val_ann should be placed in the self.data_dir directory.


        # --------------  training config --------------------- #
        self.max_epoch = 360

        self.ema = True
        self.ema_decay = 0.9998
        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 20
        self.eval_interval = 1
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # learning_rate
        self.scheduler = "yoloxwarmcos"
        self.warmup_lr = 0.
        self.warmup_epochs = 1
        self.basic_lr_per_img = 0.01 / 64.0
        self.min_lr_ratio = 0.05
        self.no_aug_epochs = 15
        self.multiscale_range = 5


        # -----------------  testing config ------------------ #
        self.input_size = (224, 224)

        # ---------------- model config ---------------- #
        self.output_dir = "BaseCls_outputs"
        self.depth = 0.33
        self.width = 0.50
        self.backbone_type = 'CSPDarknet'
        self.act = 'relu'
        self.use_focus = False

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
