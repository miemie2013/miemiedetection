#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @miemie2013

import os
import sys

from mmdet.exp.fcos.fcos_method_base import FCOS_Method_Exp


class FCOS_RT_R50_FPN_4x_Exp(FCOS_Method_Exp):
    def __init__(self):
        super().__init__()
        # ---------------- architecture name(算法名) ---------------- #
        self.archi_name = 'FCOS'

        # --------------  training config --------------------- #
        self.max_epoch = 4
        self.aug_epochs = 4  # 前几轮进行mixup、cutmix、mosaic

        self.ema = True
        self.ema_decay = 0.9998
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.print_interval = 20
        self.eval_interval = 1
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.learningRate = dict(
            base_lr=0.01 / 16,  # 最初base_lr表示的是每一张图片的学习率。代码中会自动修改为乘以批大小。
            PiecewiseDecay=dict(
                gamma=0.1,
                milestones_epoch=[2, 3],
            ),
            LinearWarmup=dict(
                start_factor=0.3333333333333333,
                steps=200,
            ),
        )

        # -----------------  testing config ------------------ #
        self.test_size = (512, 736)

        # ---------------- model config ---------------- #
        self.output_dir = "FCOS_outputs"
        self.backbone_type = 'Resnet50Vb'
        self.backbone = dict(
            norm_type='bn',
            feature_maps=[3, 4, 5],
            dcn_v2_stages=[],
            downsample_in3x3=False,  # 注意这个细节，是在1x1卷积层下采样的。即Resnet50Va。
            freeze_at=5,
            fix_bn_mean_var_at=0,
            freeze_norm=False,
            norm_decay=0.,
        )
        self.fpn_type = 'FPN'
        self.fpn = dict(
            in_channels=[2048, 1024, 512],
            num_chan=256,
            min_level=3,
            max_level=5,
            spatial_scale=[0.03125, 0.0625, 0.125],
            has_extra_convs=False,
            use_c5=False,
            reverse_out=False,
        )
        self.head_type = 'FCOSHead'
        self.head = dict(
            in_channel=256,
            num_classes=self.num_classes,
            fpn_stride=[8, 16, 32],
            num_convs=4,
            norm_type='gn',
            norm_reg_targets=True,
            thresh_with_ctr=True,
            centerness_on_reg=True,
            use_dcn_in_tower=False,
        )
        self.fcos_loss_type = 'FCOSLoss'
        self.fcos_loss = dict(
            loss_alpha=0.25,
            loss_gamma=2.0,
            iou_loss_type='giou',  # linear_iou/giou/iou/ciou
            reg_weights=1.0,
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
        self.context = {'fields': ['image', 'im_info', 'fcos_target']}
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
        # RandomFlipImage
        self.randomFlipImage = dict(
            prob=0.5,
        )
        # NormalizeImage
        self.normalizeImage = dict(
            is_channel_first=False,
            is_scale=False,
            mean=[123.675, 116.28, 103.53],
            std=[1.0, 1.0, 1.0],
        )
        # ResizeImage
        # 图片短的那一边缩放到选中的target_size，长的那一边等比例缩放；如果这时候长的那一边大于max_size，
        # 那么改成长的那一边缩放到max_size，短的那一边等比例缩放。这时候im_scale_x = im_scale， im_scale_y = im_scale。
        # resize_box=True 表示真实框（格式是x0y0x1y1）也跟着缩放，横纵坐标分别乘以im_scale_x、im_scale_y。
        # resize_box=False表示真实框（格式是x0y0x1y1）不跟着缩放，因为后面会在Gt2FCOSTarget中缩放。
        self.resizeImage = dict(
            target_size=[256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608],
            max_size=900,
            interp=1,
            use_cv2=True,
            resize_box=False,
        )
        # Permute
        self.permute = dict(
            to_bgr=False,
            channel_first=True,
        )
        # PadBatch
        self.padBatch = dict(
            pad_to_stride=32,  # 添加黑边使得图片边长能够被pad_to_stride整除。pad_to_stride代表着最大下采样倍率，这个模型最大到p5，为32。
            use_padded_im_info=False,
        )
        # Gt2FCOSTarget
        self.gt2FCOSTarget = dict(
            object_sizes_boundary=[64, 128],
            center_sampling_radius=1.5,
            downsample_ratios=[8, 16, 32],
            norm_reg_targets=True,
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
        self.sample_transforms_seq.append('randomFlipImage')
        self.sample_transforms_seq.append('normalizeImage')
        self.sample_transforms_seq.append('resizeImage')
        self.sample_transforms_seq.append('permute')
        self.batch_transforms_seq = []
        self.batch_transforms_seq.append('padBatch')
        self.batch_transforms_seq.append('gt2FCOSTarget')

        # ---------------- dataloader config ---------------- #
        # 默认是4。如果报错“OSError: [WinError 1455] 页面文件太小,无法完成操作”，设置为2或0解决。
        self.data_num_workers = 2

        # 判断是否是调试状态
        isDebug = True if sys.gettrace() else False
        if isDebug:
            print('Debug Mode.')
            self.data_dir = '../' + self.data_dir
            self.cls_names = '../' + self.cls_names
            self.output_dir = '../' + self.output_dir
