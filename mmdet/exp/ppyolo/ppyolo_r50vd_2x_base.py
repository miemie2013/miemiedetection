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

        # ---------------- model config ---------------- #
        self.output_dir = "PPYOLO_outputs"
        self.backbone = dict(
            norm_type='bn',
            feature_maps=[3, 4, 5],
            dcn_v2_stages=[5],
            downsample_in3x3=True,   # 注意这个细节，是在3x3卷积层下采样的。
            freeze_at=-1,
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
        # set worker to 4 for shorter dataloader init time
        self.data_num_workers = 4
        self.input_size = (640, 640)  # (height, width)
        # Actual multiscale ranges: [640-5*32, 640+5*32].
        # To disable multiscale training, set the
        # self.multiscale_range to 0.
        self.multiscale_range = 5
        # You can uncomment this line to specify a multiscale range
        # self.random_size = (14, 26)

        # --------------- transform config ----------------- #
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.degrees = 10.0
        self.translate = 0.1
        self.mosaic_scale = (0.1, 2)
        self.mixup_scale = (0.5, 1.5)
        self.shear = 2.0
        self.enable_mixup = True

        # --------------  training config --------------------- #
        self.warmup_epochs = 6
        self.max_epoch = 811
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 192.0
        self.scheduler = "yoloxwarmcos"
        self.min_lr_ratio = 0.05


        self.ema = True
        self.ema_decay = 0.9998
        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 20
        self.eval_interval = 10
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # -----------------  testing config ------------------ #
        self.test_size = (608, 608)

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
            COCODataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )
        from mmdet.utils import (
            wait_for_the_master,
            get_local_rank,
        )

        local_rank = get_local_rank()

        with wait_for_the_master(local_rank):
            dataset = COCODataset(
                data_dir=self.data_dir,
                json_file=self.train_ann,
                ann_folder=self.ann_folder,
                name=self.train_image_folder,
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob),
                cache=cache_img,
            )
        no_aug = False
        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

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
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

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
        from mmdet.utils import LRScheduler

        scheduler = LRScheduler(
            self.scheduler,
            lr,
            iters_per_epoch,
            self.max_epoch,
            warmup_epochs=self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
            no_aug_epochs=self.no_aug_epochs,
            min_lr_ratio=self.min_lr_ratio,
        )
        return scheduler

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from mmdet.data import MieMieCOCOEvalDataset, PPYOLOValTransform

        # 预测时的数据预处理
        decodeImage = DecodeImage(**self.decodeImage)
        resizeImage = ResizeImage(target_size=self.test_size[0], interp=self.resizeImage['interp'])
        normalizeImage = NormalizeImage(**self.normalizeImage)
        permute = Permute(**self.permute)
        transforms = [decodeImage, resizeImage, normalizeImage, permute]
        valdataset = MieMieCOCOEvalDataset(
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
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

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
