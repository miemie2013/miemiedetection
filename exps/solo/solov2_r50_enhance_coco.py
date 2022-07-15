#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @miemie2013

import os
import sys

from mmdet.exp import SOLO_Method_Exp


class Exp(SOLO_Method_Exp):
    def __init__(self):
        super(Exp, self).__init__()
        # COCO2017 dataset。用来调试。
        self.num_classes = 80
        self.data_dir = '../COCO'
        self.cls_names = 'class_names/coco_classes.txt'
        self.ann_folder = "annotations"
        self.train_ann = "instances_val2017.json"
        self.val_ann = "instances_val2017.json"
        self.train_image_folder = "val2017"
        self.val_image_folder = "val2017"

        # ---------------- architecture name(算法名) ---------------- #
        self.archi_name = 'SOLO'

        # --------------  training config --------------------- #
        self.max_epoch = 16
        self.aug_epochs = 16  # 前几轮进行mixup、cutmix、mosaic

        self.ema = True
        self.ema_decay = 0.9998
        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 20
        self.eval_interval = 10
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # learning_rate
        self.scheduler = "warm_piecewisedecay"
        self.warmup_epochs = 5
        self.basic_lr_per_img = 0.01 / 192.0
        self.start_factor = 0.0
        self.decay_gamma = 0.1
        self.milestones_epoch = [649, 730]

        # -----------------  testing config ------------------ #
        self.test_size = (512, 512)

        # ---------------- model config ---------------- #
        self.output_dir = "SOLO_outputs"
        self.backbone_type = 'ResNet'
        self.backbone = dict(
            depth=50,
            variant='d',
            freeze_at=0,
            freeze_norm=False,
            norm_type='sync_bn',
            return_idx=[0, 1, 2, 3],
            dcn_v2_stages=[1, 2, 3],
            lr_mult_list=[0.05, 0.05, 0.1, 0.15],
            num_stages=4,
        )
        self.fpn_type = 'FPN'
        self.fpn = dict(
            in_channels=[256, 512, 1024, 2048],
            out_channel=256,
        )
        self.head_type = 'SOLOv2Head'
        self.head = dict(
            in_channels=256,
            seg_feat_channels=256,
            num_classes=self.num_classes,
            stacked_convs=3,
            num_grids=[40, 36, 24, 16, 12],
            kernel_out_channels=128,
            dcn_v2_stages=[2],
            drop_block=True,
        )
        self.solomaskhead_type = 'SOLOv2MaskHead'
        self.solomaskhead = dict(
            mid_channels=128,
            out_channels=128,
            start_level=0,
            end_level=3,
            use_dcn_in_tower=True,
        )
        self.iou_loss = dict(
            loss_weight=2.5,
            loss_square=True,
        )
        self.iou_aware_loss = dict(
            loss_weight=1.0,
        )
        self.solo_loss = dict(
            ins_loss_weight=3.0,
            focal_loss_gamma=2.0,
            focal_loss_alpha=0.25,
        )
        self.nms_cfg = dict(
            nms_type='mask_matrix_nms',
            score_threshold=0.1,
            post_threshold=0.05,
            nms_top_k=500,
            keep_top_k=100,
            kernel='gaussian',  # 'gaussian' or 'linear'
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
            anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
            anchors=[[10, 13], [16, 30], [33, 23],
                     [30, 61], [62, 45], [59, 119],
                     [116, 90], [156, 198], [373, 326]],
            downsample_ratios=[32, 16, 8],
            num_classes=self.num_classes,
        )
        # ResizeImage
        self.resizeImage = dict(
            target_size=512,
            max_size=852,
            interp=1,
        )
        # PadBatch
        self.padBatch = dict(
            pad_to_stride=32,
            use_padded_im_info=False,
        )
        # SOLOv2Pad
        self.sOLOv2Pad = dict(
            max_size=864,
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
