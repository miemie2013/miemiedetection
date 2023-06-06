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

        # --------------  training config --------------------- #
        self.max_epoch = 16
        self.aug_epochs = 16  # 前几轮进行mixup、cutmix、mosaic

        self.ema = True
        self.ema_decay = 0.9999
        self.ema_decay_type = "exponential"
        self.ema_filter_no_grad = True
        self.weight_decay = 0.0001
        self.print_interval = 20
        self.eval_interval = 2
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # learning_rate
        self.scheduler = "warm_piecewisedecay"
        self.warmup_epochs = 1
        self.basic_lr_per_img = 0.0001 / (4. * 4.0)
        self.clip_grad_by_norm = 0.1
        self.start_factor = 0.001
        self.decay_gamma = 1.0
        self.milestones_epoch = [999999, ]


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
        self.transformer['num_classes'] = self.num_classes
        self.post_cfg['num_classes'] = self.num_classes

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
