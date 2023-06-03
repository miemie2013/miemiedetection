#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @miemie2013

import os
import sys

from mmdet.exp import RTDETR_Method_Exp


class Exp(RTDETR_Method_Exp):
    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # self.num_classes = 80
        # self.data_dir = '../COCO'
        # self.cls_names = 'class_names/coco_classes.txt'
        # self.ann_folder = "annotations"
        # self.train_ann = "instances_val2017.json"
        # self.val_ann = "instances_val2017.json"
        # self.train_image_folder = "val2017"
        # self.val_image_folder = "val2017"

        # learning_rate
        self.basic_lr_per_img = 0.0001 / (4. * 4.0)


        self.backbone_type = 'ResNet'
        self.backbone = dict(
            depth=18,
            variant='d',
            return_idx=[1, 2, 3],
            freeze_at=-1,
            freeze_norm=False,
        )
        self.neck_type = 'HybridEncoder'
        self.neck = dict(
            in_channels=[128, 256, 512],
            hidden_dim=self.hidden_dim,
            use_encoder_idx=[2],
            num_encoder_layers=1,
            encoder_layer=dict(
                name='TransformerLayer',
                d_model=self.hidden_dim,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.,
                activation='gelu',
            ),
            expansion=0.5,
            depth_mult=1.0,
            eval_size=self.test_size,
        )
        self.transformer['eval_idx'] = -1
        self.transformer['num_decoder_layers'] = 3

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
