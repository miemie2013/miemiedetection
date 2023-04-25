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
        self.num_classes = 5
        self.data_dir = '../PersonCarPet'
        self.cls_names = 'class_names/pcp_classes.txt'
        self.ann_folder = "annotations"
        self.train_ann = "train_T40RGBIR_pet_Package_voc_coco_11_22131images_121576bbox.json"
        self.val_ann = "val_T40_RGB_PCPaP_1871images_11892bbox.json"
        self.train_image_folder = "pcp_train"
        self.val_image_folder = "pcp_val"

        # --------------  training config --------------------- #
        self.max_epoch = 30

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
        self.cosinedecay_epochs = 40
        self.basic_lr_per_img = 0.001 / (8. * 8.0)
        self.start_factor = 0.0
        self.head['static_assigner_epoch'] = 6

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
