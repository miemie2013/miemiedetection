#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import os
import sys
import random

import torch
import torch.distributed as dist
import torch.nn as nn

from mmdet.exp.datasets.coco_base import COCOBaseExp


class YOLOXExp(COCOBaseExp):
    def __init__(self):
        super().__init__()
        # ---------------- architecture name(算法名) ---------------- #
        self.archi_name = 'YOLOX'

        # ---------------- model config ---------------- #
        self.depth = 1.00
        self.width = 1.00
        self.act = 'silu'
        self.use_focus = True

        # ---------------- dataloader config ---------------- #
        # 默认是4。如果报错“OSError: [WinError 1455] 页面文件太小,无法完成操作”，设置为2或0解决。
        self.data_num_workers = 2
        self.input_size = (640, 640)  # (height, width)
        # Actual multiscale ranges: [640-5*32, 640+5*32].
        # To disable multiscale training, set the
        # self.multiscale_range to 0.
        self.multiscale_range = 5
        # You can uncomment this line to specify a multiscale range
        # self.random_size = (14, 26)
        self.output_dir = "YOLOX_outputs"

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
        self.torch_augment = False

        # --------------  training config --------------------- #
        self.warmup_epochs = 5
        self.max_epoch = 300
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 15
        self.min_lr_ratio = 0.05
        self.ema = True
        self.ema_decay = 0.9998
        self.freeze_at = 0

        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 10
        self.eval_interval = 10
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # -----------------  testing config ------------------ #
        self.test_size = (640, 640)
        self.test_conf = 0.01
        self.nmsthre = 0.65

        # 判断是否是调试状态
        isDebug = True if sys.gettrace() else False
        if isDebug:
            print('Debug Mode.')
            self.data_dir = '../' + self.data_dir
            self.cls_names = '../' + self.cls_names
            self.output_dir = '../' + self.output_dir

    def get_model(self):
        from mmdet.models import YOLOX, YOLOPAFPN, YOLOXHead

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, use_focus=self.use_focus, act=self.act, freeze_at=self.freeze_at)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def get_data_loader(
        self, batch_size, is_distributed, no_aug=False, cache_img=False
    ):
        from mmdet.data import (
            SimpleCOCODataset,
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
        if self.torch_augment:
            with wait_for_the_master(local_rank):
                dataset = SimpleCOCODataset(
                    data_dir=self.data_dir,
                    json_file=self.train_ann,
                    ann_folder=self.ann_folder,
                    name=self.train_image_folder,
                    img_size=self.input_size,
                    max_labels=120,
                )
        else:
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

        sampler = InfiniteSampler(len(self.dataset), shuffle=True, seed=self.seed if self.seed else 0)

        if self.torch_augment:
            batch_sampler = torch.utils.data.sampler.BatchSampler(
                sampler=sampler,
                batch_size=batch_size,
                drop_last=True,
            )
            dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
            dataloader_kwargs["batch_sampler"] = batch_sampler
            dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed
            train_loader = torch.utils.data.DataLoader(self.dataset, **dataloader_kwargs)
        else:
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
                pg0, lr=lr, momentum=self.momentum, nesterov=True
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

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from mmdet.data import COCODataset, ValTransform

        valdataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann if not testdev else "image_info_test-dev2017.json",
            ann_folder=self.ann_folder,
            name=self.val_image_folder if not testdev else "test2017",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
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

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from mmdet.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            archi_name=self.archi_name,
            testdev=testdev,
        )
        return evaluator

    def eval(self, model, evaluator, is_distributed, half=False):
        return evaluator.evaluate_yolox(model, is_distributed, half)
