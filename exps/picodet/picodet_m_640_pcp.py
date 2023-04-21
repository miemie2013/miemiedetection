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
        self.num_classes = 5
        self.data_dir = '../PersonCarPet'
        self.cls_names = 'class_names/pcp_classes.txt'
        self.ann_folder = "annotations"
        self.train_ann = "train_T40RGBIR_pet_Package_voc_coco_11_22131images_121576bbox.json"
        self.val_ann = "val_T40_RGB_PCPaP_1871images_11892bbox.json"
        self.train_image_folder = "pcp_train"
        self.val_image_folder = "pcp_val"

        self.max_epoch = 60
        self.print_interval = 20
        self.eval_interval = 4
        self.warmup_epochs = 1
        self.cosinedecay_epochs = 80
        self.head['static_assigner_epoch'] = 20

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
