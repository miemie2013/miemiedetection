#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @miemie2013

import os
import sys

from mmdet.exp import PPYOLOE_Method_Exp


class Exp(PPYOLOE_Method_Exp):
    def __init__(self):
        super(Exp, self).__init__()
        # custom dataset
        self.num_classes = 13
        self.data_dir = '../bh3_letu_dataset'
        self.cls_names = 'class_names/bh3_letu_classes.txt'
        self.ann_folder = "annotations"
        self.train_ann = "bh3_letu_train.json"
        self.val_ann = "bh3_letu_val.json"
        self.train_image_folder = "images"
        self.val_image_folder = "images"

        # ---------------- architecture name(算法名) ---------------- #
        self.archi_name = 'PPYOLOE'

        # --------------  training config --------------------- #
        self.max_epoch = 160

        self.ema = True
        self.ema_decay = 0.9998
        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 20
        self.eval_interval = 20
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # learning_rate
        self.scheduler = "warm_cosinedecay"
        self.warmup_epochs = 10
        self.cosinedecay_epochs = 200
        self.basic_lr_per_img = 0.025 / (8. * 20.0)
        self.start_factor = 0.0

        # -----------------  testing config ------------------ #
        self.eval_height = 640
        self.eval_width = 640
        self.test_size = [self.eval_height, self.eval_width]

        # ---------------- model config ---------------- #
        self.depth_mult = 1.0
        self.width_mult = 1.0
        self.backbone_type = 'CSPResNet'
        self.backbone = dict(
            layers=[3, 6, 6, 3],
            channels=[64, 128, 256, 512, 1024],
            return_idx=[1, 2, 3],
            freeze_at=3,
            use_large_stem=True,
            depth_mult=self.depth_mult,
            width_mult=self.width_mult,
        )
        self.fpn_type = 'CustomCSPPAN'
        self.fpn = dict(
            in_channels=[int(256 * self.width_mult), int(512 * self.width_mult), int(1024 * self.width_mult)],
            out_channels=[768, 384, 192],
            stage_num=1,
            block_num=3,
            act='swish',
            spp=True,
            depth_mult=self.depth_mult,
            width_mult=self.width_mult,
        )
        self.head_type = 'PPYOLOEHead'
        self.head = dict(
            in_channels=[int(768 * self.width_mult), int(384 * self.width_mult), int(192 * self.width_mult)],
            fpn_strides=[32, 16, 8],
            grid_cell_scale=5.0,
            grid_cell_offset=0.5,
            static_assigner_epoch=4,
            use_varifocal_loss=True,
            num_classes=self.num_classes,
            loss_weight={'class': 1.0, 'iou': 2.5, 'dfl': 0.5, },
            eval_size=self.test_size,
        )
        self.static_assigner_type = 'ATSSAssigner'
        self.static_assigner = dict(
            topk=9,
            num_classes=self.num_classes,
        )
        self.assigner_type = 'TaskAlignedAssigner'
        self.assigner = dict(
            topk=13,
            alpha=1.0,
            beta=6.0,
        )
        self.nms_cfg = dict(
            nms_type='multiclass_nms',
            score_threshold=0.01,
            nms_threshold=0.6,
            nms_top_k=1000,
            keep_top_k=100,
        )

        # ---------------- 预处理相关 ---------------- #
        self.context = {'fields': ['image', 'gt_bbox', 'gt_class', 'gt_score']}
        # DecodeImage
        self.decodeImage = dict(
            to_rgb=True,
            with_mixup=False,
            with_cutmix=False,
            with_mosaic=False,
        )
        # ColorDistort
        self.colorDistort = dict()
        # RandomExpand
        self.randomExpand = dict(
            fill_value=[123.675, 116.28, 103.53],
        )
        # RandomCrop
        self.randomCrop = dict()
        # RandomFlipImage
        self.randomFlipImage = dict(
            is_normalized=False,
        )
        # RandomShape
        self.randomShape = dict(
            sizes=[320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768],
            # sizes=[640],
            random_inter=True,
            resize_box=True,
        )
        # NormalizeImage
        self.normalizeImage = dict(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            is_scale=True,
            is_channel_first=False,
        )
        # Permute
        self.permute = dict(
            to_bgr=False,
            channel_first=True,
        )
        # PadGT
        self.padGT = dict(
            num_max_boxes=200,
        )
        # ResizeImage
        self.resizeImage = dict(
            target_size=640,
            interp=2,
        )

        # 预处理顺序。增加一些数据增强时这里也要加上，否则train.py中相当于没加！
        self.sample_transforms_seq = []
        self.sample_transforms_seq.append('decodeImage')
        self.sample_transforms_seq.append('colorDistort')
        self.sample_transforms_seq.append('randomExpand')
        self.sample_transforms_seq.append('randomCrop')
        self.sample_transforms_seq.append('randomFlipImage')
        self.sample_transforms_seq.append('randomShape')
        self.sample_transforms_seq.append('normalizeImage')
        self.sample_transforms_seq.append('permute')
        self.sample_transforms_seq.append('padGT')
        self.batch_transforms_seq = []
        # self.batch_transforms_seq.append('padGT')

        # ---------------- dataloader config ---------------- #
        # 默认是4。如果报错“OSError: [WinError 1455] 页面文件太小,无法完成操作”，设置为2或0解决。
        self.data_num_workers = 1
        self.eval_data_num_workers = 0

        # 判断是否是调试状态
        isDebug = True if sys.gettrace() else False
        if isDebug:
            print('Debug Mode.')
            self.data_dir = '../' + self.data_dir
            self.cls_names = '../' + self.cls_names
            self.output_dir = '../' + self.output_dir
