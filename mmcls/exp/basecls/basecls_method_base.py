#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @miemie2013

import os
import sys
import random

import torch
import torch.distributed as dist
import torch.nn as nn

from mmcls.exp.base_exp import BaseExp


class BaseCls_Method_Exp(BaseExp):
    def __init__(self):
        super().__init__()
        # ---------------- dataset ---------------- #
        # AFHQ dataset
        self.num_classes = 3
        self.data_dir = '../afhq'
        self.cls_names = 'class_names/afhq_classes.txt'
        self.train_ann = "train.txt"    # self.train_ann should be placed in the self.data_dir directory.
        self.val_ann = "val.txt"        # self.val_ann should be placed in the self.data_dir directory.

        # ---------------- architecture name(算法名) ---------------- #
        self.archi_name = 'BaseCls'

        # --------------  training config --------------------- #
        self.max_epoch = 100

        self.ema = True
        self.ema_decay = 0.9998
        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 20
        self.eval_interval = 1
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # learning_rate
        self.scheduler = "yoloxwarmcos"
        self.warmup_lr = 0.
        self.warmup_epochs = 1
        self.basic_lr_per_img = 0.01 / 64.0
        self.min_lr_ratio = 0.05
        self.no_aug_epochs = 15
        self.multiscale_range = 5


        # -----------------  testing config ------------------ #
        self.input_size = (224, 224)

        # ---------------- model config ---------------- #
        self.output_dir = "BaseCls_outputs"
        self.scale = 2.0
        self.backbone_type = 'LCNet'
        self.backbone = dict(
            scale=self.scale,
            feature_maps=[5],
        )

        # ---------------- dataloader config ---------------- #
        # 默认是4。如果报错“OSError: [WinError 1455] 页面文件太小,无法完成操作”，设置为2或0解决。
        self.data_num_workers = 2
        self.eval_data_num_workers = 2

        # 判断是否是调试状态
        isDebug = True if sys.gettrace() else False
        if isDebug:
            print('Debug Mode.')
            self.data_dir = '../' + self.data_dir
            self.cls_names = '../' + self.cls_names
            self.output_dir = '../' + self.output_dir

    def get_model(self):
        from mmcls.models import BaseCls
        from mmdet.models import LCNet, CSPDarknet, MCNet
        if getattr(self, "model", None) is None:
            in_channel = -1
            if self.backbone_type == 'LCNet':
                backbone = LCNet(**self.backbone)
                in_channel = backbone._out_channels[-1]
            elif self.backbone_type == 'CSPDarknet':
                backbone = CSPDarknet(self.depth, self.width, depthwise=False, use_focus=self.use_focus, act=self.act, freeze_at=0)
                base_channels = int(self.width * 64)  # 64
                in_channel = base_channels * 16
            elif self.backbone_type == 'MCNet':
                backbone = MCNet(self.scale, act=self.act)
                base_channels = int(self.scale * 64)  # 64
                in_channel = base_channels * 8
            self.model = BaseCls(backbone, in_channel, self.num_classes)
        return self.model

    def get_data_loader(
        self, batch_size, is_distributed
    ):
        from mmcls.data import (
            BaseClsDataset,
            InfiniteSampler,
            worker_init_reset_seed,
        )
        from mmdet.utils import (
            wait_for_the_master,
            get_local_rank,
        )

        local_rank = get_local_rank()

        with wait_for_the_master(local_rank):
            dataset = BaseClsDataset(
                data_dir=self.data_dir,
                anno=self.train_ann,
                img_size=self.input_size,
                type='Train',
            )
        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), shuffle=True, seed=self.seed if self.seed else 0)

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

        # collater = PPYOLOTrainCollater(self.context, batch_transforms, self.n_layers)
        # dataloader_kwargs["collate_fn"] = collater
        train_loader = torch.utils.data.DataLoader(self.dataset, **dataloader_kwargs)

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

    def get_eval_loader(self, batch_size, is_distributed):
        from mmcls.data import BaseClsDataset

        val_dataset = BaseClsDataset(
            data_dir=self.data_dir,
            anno=self.val_ann,
            img_size=self.input_size,
            type='Val',
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(val_dataset)

        dataloader_kwargs = {
            "num_workers": self.eval_data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(val_dataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed):
        from mmcls.evaluators import CLSEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed)
        evaluator = CLSEvaluator(
            dataloader=val_loader,
            num_classes=self.num_classes,
            archi_name=self.archi_name,
        )
        return evaluator

    def eval(self, model, evaluator, is_distributed, half=False):
        return evaluator.evaluate_basecls(model, is_distributed, half)
