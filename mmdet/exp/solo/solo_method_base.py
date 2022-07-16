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


class SOLOTrainCollater():
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
            sample['im_shape'] = item[1]
            sample['scale_factor'] = item[2]
            sample['im_id'] = item[3]
            sample['h'] = item[4]
            sample['w'] = item[5]
            sample['is_crowd'] = item[6]
            sample['gt_class'] = item[7]
            sample['gt_bbox'] = item[8]
            sample['gt_segm'] = item[9]
            sample['gt_poly'] = item[10]
            sample['gt_score'] = item[11]
            samples.append(sample)

        # batch_transforms
        for batch_transform in self.batch_transforms:
            samples = batch_transform(samples, self.context)

        # 取出感兴趣的项
        images = []
        # gt_ins_labels = [[]] * self.n_layers
        # gt_cate_labels = [[]] * self.n_layers
        # gt_grid_orders = [[]] * self.n_layers
        gt_ins_labels = []
        gt_cate_labels = []
        gt_grid_orders = []
        for lvl in range(self.n_layers):
            gt_ins_labels.append([])
            gt_cate_labels.append([])
            gt_grid_orders.append([])
        fg_nums = []
        im_ids = []
        for i, sample in enumerate(samples):
            images.append(sample['image'].astype(np.float32))
            for lvl in range(self.n_layers):
                ins_label = 'ins_label{}'.format(lvl)
                gt_ins_labels[lvl].append(sample[ins_label].astype(np.float32))
                cate_label = 'cate_label{}'.format(lvl)
                gt_cate_labels[lvl].append(sample[cate_label].astype(np.int32))
                grid_order = 'grid_order{}'.format(lvl)
                gt_grid_orders[lvl].append(sample[grid_order].astype(np.int32))
            fg_nums.append(np.array([sample['fg_num']]).astype(np.int32))
            im_ids.append(sample['im_id'].astype(np.int32))
        images = np.stack(images, axis=0)
        images = torch.Tensor(images)
        for lvl in range(self.n_layers):
            gt_ins_labels[lvl] = np.stack(gt_ins_labels[lvl], axis=0)
            gt_cate_labels[lvl] = np.stack(gt_cate_labels[lvl], axis=0)
            gt_grid_orders[lvl] = np.stack(gt_grid_orders[lvl], axis=0)
            gt_ins_labels[lvl] = torch.Tensor(gt_ins_labels[lvl])
            gt_cate_labels[lvl] = torch.Tensor(gt_cate_labels[lvl])
            gt_grid_orders[lvl] = torch.Tensor(gt_grid_orders[lvl])
        fg_nums = np.stack(fg_nums, axis=0)
        im_ids = np.stack(im_ids, axis=0)
        fg_nums = torch.Tensor(fg_nums)
        im_ids = torch.Tensor(im_ids)
        return [images, ] + gt_ins_labels + gt_cate_labels + gt_grid_orders + [fg_nums, im_ids]


class SOLOEvalCollater():
    def __init__(self, context, batch_transforms):
        self.context = context
        self.batch_transforms = batch_transforms

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
            samples.append(sample)

        # batch_transforms
        for batch_transform in self.batch_transforms:
            samples = batch_transform(samples, self.context)

        # 取出感兴趣的项
        images = []
        im_sizes = []
        ori_shapes = []
        im_ids = []
        for i, sample in enumerate(samples):
            images.append(sample['image'].astype(np.float32))
            im_sizes.append(np.array([sample['im_info'][0], sample['im_info'][1]]).astype(np.int32))
            ori_shapes.append(np.array([sample['h'], sample['w']]).astype(np.int32))
            im_ids.append(sample['im_id'].astype(np.int32))
        images = np.stack(images, axis=0)
        im_sizes = np.stack(im_sizes, axis=0)
        ori_shapes = np.stack(ori_shapes, axis=0)
        im_ids = np.stack(im_ids, axis=0)

        images = torch.Tensor(images)
        im_sizes = torch.Tensor(im_sizes)
        ori_shapes = torch.Tensor(ori_shapes)
        im_ids = torch.Tensor(im_ids)
        return images, im_sizes, ori_shapes, im_ids


class SOLO_Method_Exp(COCOBaseExp):
    def __init__(self):
        super().__init__()
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
        self.eval_interval = 2
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
        self.test_size = (800, 800)

        # ---------------- model config ---------------- #
        self.output_dir = "SOLO_outputs"
        self.backbone_type = 'ResNet'
        self.backbone = dict(
            depth=50,
            freeze_at=0,
            return_idx=[0, 1, 2, 3],
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
            seg_feat_channels=512,
            num_classes=self.num_classes,
            stacked_convs=4,
            num_grids=[40, 36, 24, 16, 12],
            kernel_out_channels=256,
        )
        self.solomaskhead_type = 'SOLOv2MaskHead'
        self.solomaskhead = dict(
            mid_channels=128,
            out_channels=256,
            start_level=0,
            end_level=3,
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
            target_size=[[640, 1333], [672, 1333], [704, 1333], [736, 1333], [768, 1333], [800, 1333]],
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
            target_size=800,
            max_size=1333,
            interp=1,
        )
        # PadBatch
        self.padBatch = dict(
            pad_to_stride=32,
            use_padded_im_info=False,
        )
        # SOLOv2Pad
        self.sOLOv2Pad = dict(
            max_size=1344,
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
        self.data_num_workers = 2
        self.eval_data_num_workers = 2

    def get_model(self):
        from mmdet.models import ResNet, SOLOv2Loss, SOLOv2Head, SOLOv2MaskHead, SOLO
        from mmdet.models.necks.fpn import FPN
        if getattr(self, "model", None) is None:
            Backbone = None
            if self.backbone_type == 'ResNet':
                Backbone = ResNet
            backbone = Backbone(**self.backbone)
            # 冻结骨干网络
            backbone.fix_bn()
            Fpn = None
            if self.fpn_type == 'FPN':
                Fpn = FPN
            fpn = Fpn(**self.fpn)
            solov2_loss = SOLOv2Loss(**self.solo_loss)
            head = SOLOv2Head(solov2_loss=solov2_loss, nms_cfg=self.nms_cfg, **self.head)
            mask_head = SOLOv2MaskHead(**self.solomaskhead)
            self.model = SOLO(backbone, fpn, head, mask_head)
        return self.model

    def get_data_loader(
        self, batch_size, is_distributed, num_gpus, cache_img=False
    ):
        from mmdet.data import (
            SOLO_COCOTrainDataset,
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

            train_dataset = SOLO_COCOTrainDataset(
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

        collater = SOLOTrainCollater(self.context, batch_transforms, self.n_layers)
        dataloader_kwargs["collate_fn"] = collater
        train_loader = torch.utils.data.DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def random_resize(self, data_loader, epoch, rank, is_distributed):
        return 1

    def preprocess(self, inputs, targets, tsize):
        return 1

    def get_optimizer(self, batch_size, param_groups, momentum, weight_decay):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.basic_lr_per_img * batch_size * self.start_factor
            else:
                lr = self.basic_lr_per_img * batch_size

            optimizer = torch.optim.SGD(
                param_groups, lr=lr, momentum=momentum, weight_decay=weight_decay
            )
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
            warmup_lr_start=lr * self.start_factor,
            milestones=self.milestones_epoch,
            gamma=self.decay_gamma,
        )
        return scheduler

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from mmdet.data import SOLO_COCOEvalDataset

        # 预测时的数据预处理
        decodeImage = DecodeImage(**self.decodeImage)
        target_size = self.test_size[0]
        resizeImage_cfg = copy.deepcopy(self.resizeImage)
        resizeImage_cfg['target_size'] = target_size
        resizeImage = ResizeImage(**resizeImage_cfg)
        normalizeImage = NormalizeImage(**self.normalizeImage)
        permute = Permute(**self.permute)

        # 方案1，DataLoader里使用collate_fn参数，慢
        transforms = [decodeImage, normalizeImage, resizeImage, permute]
        padBatch = PadBatch(**self.padBatch)
        batch_transforms = [padBatch, ]

        # 方案2，用SOLOv2Pad
        # sOLOv2Pad = SOLOv2Pad(**self.sOLOv2Pad)
        # transforms = [decodeImage, normalizeImage, resizeImage, permute, sOLOv2Pad]

        val_dataset = SOLO_COCOEvalDataset(
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

        # 方案1，DataLoader里使用collate_fn参数，慢
        collater = SOLOEvalCollater(self.context, batch_transforms)
        dataloader_kwargs["collate_fn"] = collater

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
