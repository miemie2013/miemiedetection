#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @miemie2013

import os
import sys

from mmdet.exp import PPYOLO_Method_Exp


class Exp(PPYOLO_Method_Exp):
    def __init__(self):
        super(Exp, self).__init__()
        # custom dataset
        self.num_classes = 20
        self.data_dir = '../VOCdevkit/VOC2012'
        self.cls_names = 'class_names/voc_classes.txt'
        self.ann_folder = "annotations2"
        self.train_ann = "voc2012_train.json"
        self.val_ann = "voc2012_val.json"
        self.train_image_folder = "JPEGImages"
        self.val_image_folder = "JPEGImages"

        # ---------------- architecture name(算法名) ---------------- #
        self.archi_name = 'PPYOLO'

        # --------------  training config --------------------- #
        self.max_epoch = 16
        self.aug_epochs = 16  # 前几轮进行mixup、cutmix、mosaic

        self.ema = True
        self.ema_decay = 0.9998
        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 20
        self.eval_interval = 2
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # learning_rate
        self.scheduler = "warm_piecewisedecay"
        self.warmup_epochs = 1
        self.basic_lr_per_img = 0.004 / 128.0
        self.start_factor = 0.0
        self.decay_gamma = 0.1
        self.milestones_epoch = [12, 14]

        # -----------------  testing config ------------------ #
        self.test_size = (416, 416)

        # ---------------- model config ---------------- #
        self.backbone_type = 'ResNet'
        self.backbone = dict(
            depth=18,
            variant='d',
            return_idx=[2, 3],
            dcn_v2_stages=[-1],
            freeze_at=-1,
            freeze_norm=False,
            norm_decay=0.,
        )
        self.fpn_type = 'PPYOLOFPN'
        self.fpn = dict(
            in_channels=[256, 512],
            coord_conv=False,
            drop_block=True,
            block_size=3,
            keep_prob=0.9,
            conv_block_num=0,
            spp=False,
        )
        self.head_type = 'YOLOv3Head'
        self.head = dict(
            in_channels=[512, 256],
            num_classes=self.num_classes,
            anchor_masks=[[3, 4, 5], [0, 1, 2]],
            anchors=[[10, 14], [23, 27], [37, 58],
                     [81, 82], [135, 169], [344, 319]],
            downsample=[32, 16],
            scale_x_y=1.05,
            clip_bbox=True,
            iou_aware=False,
        )
        self.iou_loss = dict(
            loss_weight=2.5,
            loss_square=True,
        )
        self.yolo_loss = dict(
            ignore_thresh=0.7,
            downsample=[32, 16],
            label_smooth=False,
            scale_x_y=1.05,
        )
        self.nms_cfg = dict(
            nms_type='matrix_nms',
            score_threshold=0.01,
            post_threshold=0.01,
            nms_top_k=500,
            keep_top_k=100,
            use_gaussian=False,
            gaussian_sigma=2.,
        )

        # ---------------- 预处理相关 ---------------- #
        self.context = {'fields': ['image', 'gt_bbox', 'gt_class', 'gt_score']}
        # DecodeImage
        self.decodeImage = dict(
            to_rgb=True,
            with_mixup=True,
            with_cutmix=False,
            with_mosaic=False,
        )
        # MixupImage
        self.mixupImage = dict(
            alpha=1.5,
            beta=1.5,
        )
        # CutmixImage
        self.cutmixImage = dict(
            alpha=1.5,
            beta=1.5,
        )
        # MosaicImage
        self.mosaicImage = dict(
            alpha=1.5,
            beta=1.5,
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
        # NormalizeBox
        self.normalizeBox = dict()
        # PadBox
        self.padBox = dict(
            num_max_boxes=50,
        )
        # BboxXYXY2XYWH
        self.bboxXYXY2XYWH = dict()
        # RandomShape
        self.randomShape = dict(
            sizes=[320, 352, 384, 416, 448, 480, 512, 544, 576, 608],
            random_inter=True,
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
        # Gt2YoloTarget
        self.gt2YoloTarget = dict(
            anchor_masks=[[3, 4, 5], [0, 1, 2]],
            anchors=[[10, 14], [23, 27], [37, 58],
                     [81, 82], [135, 169], [344, 319]],
            downsample_ratios=[32, 16],
            num_classes=self.num_classes,
        )
        # ResizeImage
        self.resizeImage = dict(
            target_size=608,
            interp=2,
        )

        # 预处理顺序。增加一些数据增强时这里也要加上，否则train.py中相当于没加！
        self.sample_transforms_seq = []
        self.sample_transforms_seq.append('decodeImage')
        if self.decodeImage['with_mixup']:
            self.sample_transforms_seq.append('mixupImage')
        elif self.decodeImage['with_cutmix']:
            self.sample_transforms_seq.append('cutmixImage')
        elif self.decodeImage['with_mosaic']:
            self.sample_transforms_seq.append('mosaicImage')
        self.sample_transforms_seq.append('colorDistort')
        self.sample_transforms_seq.append('randomExpand')
        self.sample_transforms_seq.append('randomCrop')
        self.sample_transforms_seq.append('randomFlipImage')
        self.sample_transforms_seq.append('normalizeBox')
        self.sample_transforms_seq.append('padBox')
        self.sample_transforms_seq.append('bboxXYXY2XYWH')
        self.sample_transforms_seq.append('randomShape')
        self.sample_transforms_seq.append('normalizeImage')
        self.sample_transforms_seq.append('permute')
        self.sample_transforms_seq.append('gt2YoloTarget')
        self.batch_transforms_seq = []
        # self.batch_transforms_seq.append('randomShape')
        # self.batch_transforms_seq.append('normalizeImage')
        # self.batch_transforms_seq.append('permute')
        # self.batch_transforms_seq.append('gt2YoloTarget')

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
