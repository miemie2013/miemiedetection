#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @miemie2013

import os
import sys

from mmdet.exp import SOLO_Method_Exp


class Exp(SOLO_Method_Exp):
    def __init__(self):
        super(Exp, self).__init__()
        # COCO2017 dataset
        self.num_classes = 80
        self.data_dir = '../COCO'
        self.cls_names = 'class_names/coco_classes.txt'
        self.ann_folder = "annotations"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.train_image_folder = "train2017"
        self.val_image_folder = "val2017"

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
        self.max_epoch = 36
        self.aug_epochs = 0  # 前几轮进行mixup、cutmix、mosaic

        self.ema = True
        self.ema_decay = 0.9998
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.print_interval = 20
        self.eval_interval = 1111
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # learning_rate
        self.scheduler = "warm_piecewisedecay"
        self.warmup_epochs = 1
        self.basic_lr_per_img = 0.01 / 16.0
        self.clip_grad_by_norm = 35.0
        self.start_factor = 0.0
        self.decay_gamma = 0.1
        self.milestones_epoch = [24, 33]

        # -----------------  testing config ------------------ #
        self.test_size = (416, 416)

        # ---------------- model config ---------------- #
        self.output_dir = "SOLO_outputs"
        self.backbone_type = 'ResNet'
        self.backbone = dict(
            depth=50,
            variant='d',
            freeze_at=3,
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
            out_channel=64,
        )
        self.head_type = 'SOLOv2Head'
        self.head = dict(
            in_channels=64,
            seg_feat_channels=128,
            num_classes=self.num_classes,
            stacked_convs=3,
            num_grids=[40, 36, 24, 16, 12],
            kernel_out_channels=64,
            dcn_v2_stages=[2],
            drop_block=True,
        )
        self.solomaskhead_type = 'SOLOv2MaskHead'
        self.solomaskhead = dict(
            in_channels=64,
            mid_channels=64,
            out_channels=64,
            start_level=0,
            end_level=3,
            use_dcn_in_tower=True,
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
        )
        # Decode
        self.decode = dict()
        # Poly2Mask
        self.poly2Mask = dict()
        # RandomDistort
        self.randomDistort = dict()
        # RandomCrop
        self.randomCrop = dict()
        # RandomResize
        self.randomResize = dict(
            target_size=[[352, 512], [384, 512], [416, 512], [448, 512], [480, 512], [512, 512]],
            keep_ratio=True,
            interp=1,
        )
        # RandomFlip
        self.randomFlip = dict()
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
        # Gt2Solov2Target
        self.gt2Solov2Target = dict(
            num_grids=[40, 36, 24, 16, 12],
            scale_ranges=[[1, 96], [48, 192], [96, 384], [192, 768], [384, 2048]],
            coord_sigma=0.2,
        )
        # ResizeImage
        self.resizeImage = dict(
            target_size=416,
            max_size=512,
            interp=1,
        )
        # PadBatch
        self.padBatch = dict(
            pad_to_stride=32,
            use_padded_im_info=False,
        )
        # SOLOv2Pad
        self.sOLOv2Pad = dict(
            max_size=512,
        )

        # 预处理顺序。增加一些数据增强时这里也要加上，否则train.py中相当于没加！
        self.sample_transforms_seq = []
        self.sample_transforms_seq.append('decode')
        self.sample_transforms_seq.append('poly2Mask')
        self.sample_transforms_seq.append('randomDistort')
        self.sample_transforms_seq.append('randomCrop')
        self.sample_transforms_seq.append('randomResize')
        self.sample_transforms_seq.append('randomFlip')
        self.sample_transforms_seq.append('normalizeImage')
        self.sample_transforms_seq.append('permute')
        self.batch_transforms_seq = []
        self.batch_transforms_seq.append('padBatch')
        self.batch_transforms_seq.append('gt2Solov2Target')

        # ---------------- dataloader config ---------------- #
        # 默认是4。如果报错“OSError: [WinError 1455] 页面文件太小,无法完成操作”，设置为2或0解决。
        # 如果报错“self = reduction.pickle.load(from_parent) EOFError: Ran out of input”，设置为0解决。
        self.data_num_workers = 0
        self.eval_data_num_workers = 0

        # 判断是否是调试状态
        isDebug = True if sys.gettrace() else False
        if isDebug:
            print('Debug Mode.')
            self.data_dir = '../' + self.data_dir
            self.cls_names = '../' + self.cls_names
            self.output_dir = '../' + self.output_dir
