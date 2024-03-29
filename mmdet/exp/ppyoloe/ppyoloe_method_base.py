#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @miemie2013

import os
import sys
import random

import torch
import torch.distributed as dist
import torch.nn as nn

from mmdet.data import *
from mmdet.exp.datasets.coco_base import COCOBaseExp


class PPYOLOETrainCollater():
    def __init__(self, context, batch_transforms, n_layers):
        self.context = context
        self.batch_transforms = batch_transforms
        self.n_layers = n_layers

    def __call__(self, batch):
        # 重组samples
        samples = []
        for i, item in enumerate(batch):
            sample = {}
            sample['image'] = item[0]
            sample['im_info'] = item[1]
            sample['im_id'] = item[2]
            sample['h'] = item[3]
            sample['w'] = item[4]
            sample['is_crowd'] = item[5]
            sample['gt_class'] = item[6]
            sample['gt_bbox'] = item[7]
            sample['gt_score'] = item[8]
            # sample['curr_iter'] = item[9]
            samples.append(sample)

        # batch_transforms
        for batch_transform in self.batch_transforms:
            samples = batch_transform(samples, self.context)

        # 取出感兴趣的项
        images = []
        gt_class = []
        gt_bbox = []
        pad_gt_mask = []
        im_ids = []
        for i, sample in enumerate(samples):
            images.append(sample['image'].astype(np.float32))
            gt_class.append(sample['gt_class'].astype(np.int32))
            gt_bbox.append(sample['gt_bbox'].astype(np.float32))
            pad_gt_mask.append(sample['pad_gt_mask'].astype(np.float32))
            im_ids.append(sample['im_id'].astype(np.int32))
        images = np.stack(images, axis=0)
        gt_class = np.stack(gt_class, axis=0)
        gt_bbox = np.stack(gt_bbox, axis=0)
        pad_gt_mask = np.stack(pad_gt_mask, axis=0)
        im_ids = np.stack(im_ids, axis=0)

        images = torch.Tensor(images)
        gt_class = torch.Tensor(gt_class)
        gt_bbox = torch.Tensor(gt_bbox)
        pad_gt_mask = torch.Tensor(pad_gt_mask)
        im_ids = torch.Tensor(im_ids)
        return images, gt_class, gt_bbox, pad_gt_mask, im_ids


class PPYOLOE_Method_Exp(COCOBaseExp):
    def __init__(self):
        super().__init__()
        # ---------------- architecture name(算法名) ---------------- #
        self.archi_name = 'PPYOLOE'

        # --------------  training config --------------------- #
        self.max_epoch = 300

        self.ema = True
        self.ema_decay = 0.9998
        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 20
        self.eval_interval = 10
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # learning_rate
        self.scheduler = "warm_cosinedecay"
        self.warmup_epochs = 5
        self.cosinedecay_epochs = 360
        self.basic_lr_per_img = 0.04 / (8. * 32.0)
        self.start_factor = 0.0

        # -----------------  testing config ------------------ #
        self.eval_height = 640
        self.eval_width = 640
        self.test_size = [self.eval_height, self.eval_width]

        # ---------------- model config ---------------- #
        self.output_dir = "outputs/PPYOLOE_outputs"
        self.depth_mult = 0.33
        self.width_mult = 0.50
        self.backbone_type = 'CSPResNet'
        self.backbone = dict(
            layers=[3, 6, 6, 3],
            channels=[64, 128, 256, 512, 1024],
            return_idx=[1, 2, 3],
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
            static_assigner_epoch=100,
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
        self.for_distill = False
        self.feat_distill_place = 'neck_feats'

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
        self.data_num_workers = 2
        self.eval_data_num_workers = 2

    def get_model(self):
        from mmdet.models import CSPResNet, ATSSAssigner, TaskAlignedAssigner, PPYOLOEHead, PPYOLOE
        from mmdet.models.necks.custom_pan import CustomCSPPAN
        if getattr(self, "model", None) is None:
            Backbone = None
            if self.backbone_type == 'CSPResNet':
                Backbone = CSPResNet
            backbone = Backbone(**self.backbone)
            Fpn = None
            if self.fpn_type == 'CustomCSPPAN':
                Fpn = CustomCSPPAN
            fpn = Fpn(**self.fpn)
            static_assigner = ATSSAssigner(**self.static_assigner)
            assigner = TaskAlignedAssigner(**self.assigner)
            head = PPYOLOEHead(static_assigner=static_assigner, assigner=assigner, for_distill=self.for_distill, nms_cfg=self.nms_cfg, **self.head)
            self.model = PPYOLOE(backbone, fpn, head, self.for_distill, self.feat_distill_place)
        return self.model

    def get_distill_loss(self):
        from mmdet.slim import DistillPPYOLOELoss
        distill_loss = DistillPPYOLOELoss(**self.distill_loss_cfg)
        return distill_loss

    def get_data_loader(
        self, batch_size, is_distributed, num_gpus, cache_img=False
    ):
        from mmdet.data import (
            PPYOLOE_COCOTrainDataset,
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

            train_dataset = PPYOLOE_COCOTrainDataset(
                data_dir=self.data_dir,
                json_file=self.train_ann,
                ann_folder=self.ann_folder,
                name=self.train_image_folder,
                max_epoch=self.max_epoch,
                num_gpus=num_gpus,
                cfg=self,
                sample_transforms=sample_transforms,
                batch_size=batch_size,
            )

        self.dataset = train_dataset
        self.n_layers = train_dataset.n_layers

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

        # collater = PPYOLOETrainCollater(self.context, batch_transforms, self.n_layers)
        # dataloader_kwargs["collate_fn"] = collater
        train_loader = torch.utils.data.DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def random_resize(self, data_loader, epoch, rank, is_distributed):
        return 1

    def preprocess(self, inputs, targets, tsize):
        return 1

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.basic_lr_per_img * batch_size * self.start_factor
            else:
                lr = self.basic_lr_per_img * batch_size

            # 不可以加正则化的参数：norm层(比如bn层、affine_channel层、gn层)的scale、offset；卷积层的偏移参数。
            no_decay, use_decay = {}, {}
            eps = 1e-9
            for name, param in self.model.named_parameters():
                # 只加入需要梯度的参数。
                if not param.requires_grad:
                    continue
                param_lr = 1.0
                if hasattr(param, 'param_lr'):
                    param_lr = getattr(param, 'param_lr')
                param_weight_decay = -1.   # -1 means use decay.  0 means no decay.
                if hasattr(param, 'weight_decay'):
                    param_weight_decay = getattr(param, 'weight_decay')
                    assert abs(param_weight_decay - 0.0) < eps
                if param_weight_decay < -0.5:
                    if param_lr not in use_decay.keys():
                        use_decay[param_lr] = []
                    use_decay[param_lr].append(param)
                else:
                    if param_lr not in no_decay.keys():
                        no_decay[param_lr] = []
                    no_decay[param_lr].append(param)
            optimizer = torch.optim.SGD(no_decay[1.0], lr=lr, momentum=self.momentum)
            for param_group in optimizer.param_groups:
                param_group["lr_factor"] = 1.0   # 设置 no_decay[1.0] 的学习率
            for param_lr_ in no_decay.keys():
                if param_lr_ == 1.0:
                    continue
                optimizer.add_param_group({"params": no_decay[param_lr_], "lr_factor": param_lr_})
            for param_lr_ in use_decay.keys():
                optimizer.add_param_group({"params": use_decay[param_lr_], "lr_factor": param_lr_, "weight_decay": self.weight_decay})
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
            cosinedecay_epochs=self.cosinedecay_epochs,
            warmup_lr_start=lr * self.start_factor,
        )
        return scheduler

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from mmdet.data import PPYOLOE_COCOEvalDataset

        # 预测时的数据预处理
        decodeImage = DecodeImage(**self.decodeImage)
        resizeImage = ResizeImage(target_size=self.test_size[0], interp=self.resizeImage['interp'])
        normalizeImage = NormalizeImage(**self.normalizeImage)
        permute = Permute(**self.permute)
        transforms = [decodeImage, resizeImage, normalizeImage, permute]
        val_dataset = PPYOLOE_COCOEvalDataset(
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
            "num_workers": self.eval_data_num_workers,
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
        return evaluator.evaluate_ppyoloe(model, is_distributed, half)
