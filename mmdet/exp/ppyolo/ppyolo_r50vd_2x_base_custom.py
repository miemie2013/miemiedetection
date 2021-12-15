#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import os
import sys
import random

import torch
import torch.distributed as dist
import torch.nn as nn

from mmdet.data import *
from mmdet.exp.datasets.coco_base import COCOBaseExp


class PPYOLO_R50VD_2x_Exp(COCOBaseExp):
    def __init__(self):
        super().__init__()
        # ---------------- architecture name(算法名) ---------------- #
        self.archi_name = 'PPYOLO'

        # --------------  training config --------------------- #
        self.max_epoch = 8
        self.aug_epochs = 8  # 前几轮进行mixup、cutmix、mosaic

        self.ema = True
        self.ema_decay = 0.9998
        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 20
        self.eval_interval = 1
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.learningRate = dict(
            # base_lr=0.01 * self.train_cfg['batch_size'] / 192,
            base_lr=0.01 * 8.0 / 192,
            PiecewiseDecay=dict(
                gamma=0.1,
                milestones_epoch=[6, 7],
            ),
            LinearWarmup=dict(
                start_factor=0.,
                steps=500,
            ),
        )

        # -----------------  testing config ------------------ #
        self.test_size = (608, 608)

        # ---------------- model config ---------------- #
        self.output_dir = "PPYOLO_outputs"
        self.backbone = dict(
            norm_type='bn',
            feature_maps=[3, 4, 5],
            dcn_v2_stages=[5],
            downsample_in3x3=True,   # 注意这个细节，是在3x3卷积层下采样的。
            freeze_at=4,
            fix_bn_mean_var_at=-1,
            freeze_norm=False,
            norm_decay=0.,
        )
        self.head = dict(
            num_classes=self.num_classes,
            norm_type='bn',
            anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
            anchors=[[10, 13], [16, 30], [33, 23],
                     [30, 61], [62, 45], [59, 119],
                     [116, 90], [156, 198], [373, 326]],
            coord_conv=True,
            iou_aware=True,
            iou_aware_factor=0.4,
            scale_x_y=1.05,
            spp=True,
            drop_block=True,
            keep_prob=0.9,
            downsample=[32, 16, 8],
            in_channels=[2048, 1024, 512],
        )
        self.iou_loss = dict(
            loss_weight=2.5,
            max_height=608,
            max_width=608,
            ciou_term=False,
        )
        self.iou_aware_loss = dict(
            loss_weight=1.0,
            max_height=608,
            max_width=608,
        )
        self.yolo_loss = dict(
            ignore_thresh=0.7,
            scale_x_y=1.05,
            label_smooth=False,
            use_fine_grained_loss=True,
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
            anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
            anchors=[[10, 13], [16, 30], [33, 23],
                     [30, 61], [62, 45], [59, 119],
                     [116, 90], [156, 198], [373, 326]],
            downsample_ratios=[32, 16, 8],
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
        self.batch_transforms_seq = []
        self.batch_transforms_seq.append('randomShape')
        self.batch_transforms_seq.append('normalizeImage')
        self.batch_transforms_seq.append('permute')
        self.batch_transforms_seq.append('gt2YoloTarget')

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

    def get_model(self):
        from mmdet.models import Resnet50Vd, IouLoss, IouAwareLoss, YOLOv3Loss, YOLOv3Head, PPYOLO
        if getattr(self, "model", None) is None:
            backbone = Resnet50Vd(**self.backbone)
            # 冻结骨干网络
            backbone.freeze()
            backbone.fix_bn()
            iou_loss = IouLoss(**self.iou_loss)
            iou_aware_loss = None
            if self.head['iou_aware']:
                iou_aware_loss = IouAwareLoss(**self.iou_aware_loss)
            yolo_loss = YOLOv3Loss(iou_loss=iou_loss, iou_aware_loss=iou_aware_loss, **self.yolo_loss)
            head = YOLOv3Head(yolo_loss=yolo_loss, nms_cfg=self.nms_cfg, **self.head)
            self.model = PPYOLO(backbone, head)
        return self.model

    def get_data_loader(
        self, batch_size, is_distributed, cache_img=False
    ):
        from mmdet.data import (
            MieMieCOCOTrainDataset,
            InfiniteSampler,
            worker_init_reset_seed,
        )
        from mmdet.utils import (
            wait_for_the_master,
            get_local_rank,
        )

        local_rank = get_local_rank()

        with wait_for_the_master(local_rank):
            # 训练时的数据预处理
            sample_transforms = get_sample_transforms(self)
            batch_transforms = get_batch_transforms(self)

            train_dataset = MieMieCOCOTrainDataset(
                data_dir=self.data_dir,
                json_file=self.train_ann,
                ann_folder=self.ann_folder,
                name=self.train_image_folder,
                cfg=self,
                sample_transforms=sample_transforms,
                batch_transforms=batch_transforms,
                batch_size=batch_size,
            )

        self.dataset = train_dataset
        self.epoch_steps = train_dataset.train_steps
        self.max_iters = train_dataset.max_iters

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), shuffle=False, seed=self.seed if self.seed else 0)

        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=True,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed
        dataloader_kwargs["shuffle"] = False

        train_loader = torch.utils.data.DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader, self.epoch_steps, self.max_iters

    def random_resize(self, data_loader, epoch, rank, is_distributed):
        tensor = torch.LongTensor(2).cuda()

        if rank == 0:
            size_factor = self.input_size[1] * 1.0 / self.input_size[0]
            if not hasattr(self, 'random_size'):
                min_size = int(self.input_size[0] / 32) - self.multiscale_range
                max_size = int(self.input_size[0] / 32) + self.multiscale_range
                self.random_size = (min_size, max_size)
            size = random.randint(*self.random_size)
            size = (int(32 * size), 32 * int(size * size_factor))
            tensor[0] = size[0]
            tensor[1] = size[1]

        if is_distributed:
            dist.barrier()
            dist.broadcast(tensor, 0)

        input_size = (tensor[0].item(), tensor[1].item())
        return input_size

    def preprocess(self, inputs, targets, tsize):
        scale_y = tsize[0] / self.input_size[0]
        scale_x = tsize[1] / self.input_size[1]
        if scale_x != 1 or scale_y != 1:
            inputs = nn.functional.interpolate(
                inputs, size=tsize, mode="bilinear", align_corners=False
            )
            targets[..., 1::2] = targets[..., 1::2] * scale_x
            targets[..., 2::2] = targets[..., 2::2] * scale_y
        return inputs, targets

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            lr = self.learningRate['LinearWarmup']['start_factor']

            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    if v.bias.requires_grad:
                        pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    if v.weight.requires_grad:
                        pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    if v.weight.requires_grad:
                        pg1.append(v.weight)  # apply decay

            optimizer = torch.optim.SGD(
                pg0, lr=lr, momentum=self.momentum
            )
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})
            self.optimizer = optimizer

        return self.optimizer

    def get_lr_scheduler(self, lr, iters_per_epoch):
        return 1

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from mmdet.data import MieMieCOCOEvalDataset

        # 预测时的数据预处理
        decodeImage = DecodeImage(**self.decodeImage)
        resizeImage = ResizeImage(target_size=self.test_size[0], interp=self.resizeImage['interp'])
        normalizeImage = NormalizeImage(**self.normalizeImage)
        permute = Permute(**self.permute)
        transforms = [decodeImage, resizeImage, normalizeImage, permute]
        val_dataset = MieMieCOCOEvalDataset(
            data_dir=self.data_dir,
            json_file=self.val_ann if not testdev else "image_info_test-dev2017.json",
            ann_folder=self.ann_folder,
            name=self.val_image_folder if not testdev else "test2017",
            cfg=self,
            transforms=transforms,
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(val_dataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(val_dataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from mmdet.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=-99.0,
            nmsthre=-99.0,
            num_classes=self.num_classes,
            archi_name=self.archi_name,
            testdev=testdev,
        )
        return evaluator

    def eval(self, model, evaluator, is_distributed, half=False):
        return evaluator.evaluate_ppyolo(model, is_distributed, half)
