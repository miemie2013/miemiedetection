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
from mmdet.models.backbones.lcnet import make_divisible


class PicoDetTrainCollater():
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


class PicoDet_Method_Exp(COCOBaseExp):
    def __init__(self):
        super().__init__()
        # ---------------- architecture name(算法名) ---------------- #
        self.archi_name = 'PicoDet'

        # --------------  training config --------------------- #
        self.max_epoch = 300

        self.ema = True
        self.ema_decay = 0.9998
        self.weight_decay = 0.00004
        self.momentum = 0.9
        self.print_interval = 20
        self.eval_interval = 10
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # learning_rate
        self.scheduler = "warm_cosinedecay"
        self.warmup_epochs = 1
        self.cosinedecay_epochs = 300
        self.basic_lr_per_img = 0.32 / (4. * 64.0)
        self.start_factor = 0.1

        # -----------------  testing config ------------------ #
        self.eval_height = 416
        self.eval_width = 416
        self.test_size = [self.eval_height, self.eval_width]

        # ---------------- model config ---------------- #
        self.output_dir = "PicoDet_outputs"
        self.scale = 1.5
        self.backbone_type = 'LCNet'
        self.backbone = dict(
            scale=self.scale,
            feature_maps=[3, 4, 5],
        )
        self.fpn_type = 'LCPAN'
        self.fpn = dict(
            in_channels=[make_divisible(128 * self.scale), make_divisible(256 * self.scale), make_divisible(512 * self.scale)],
            out_channels=128,
            use_depthwise=True,
            num_features=4,
        )
        self.head_type = 'PicoHeadV2'
        self.head = dict(
            fpn_stride=[8, 16, 32, 64],
            feat_in_chan=128,
            prior_prob=0.01,
            reg_max=7,   # 起源于Generalized Focal Loss， 每个anchor预测的ltrb的最大值是reg_max个格子边长。
            cell_offset=0.5,
            grid_cell_scale=5.0,
            static_assigner_epoch=100,
            use_align_head=True,
            num_classes=self.num_classes,
            eval_size=self.test_size,
        )
        self.conv_feat_type = 'PicoFeat'
        self.conv_feat = dict(
            feat_in=128,
            feat_out=128,
            num_convs=4,
            num_fpn_stride=4,
            norm_type='bn',
            share_cls_reg=True,
            use_se=True,
        )
        self.static_assigner_type = 'ATSSAssigner'
        self.static_assigner = dict(
            topk=9,
            force_gt_matching=False,
            num_classes=self.num_classes,
        )
        self.assigner_type = 'TaskAlignedAssigner'
        self.assigner = dict(
            topk=13,
            alpha=1.0,
            beta=6.0,
        )
        self.loss_class_type = 'VarifocalLoss'
        self.loss_class = dict(
            use_sigmoid=False,
            iou_weighted=True,
            loss_weight=1.0,
        )
        self.loss_dfl_type = 'DistributionFocalLoss'
        self.loss_dfl = dict(
            loss_weight=0.5,
        )
        self.loss_bbox_type = 'GIoULoss'
        self.loss_bbox = dict(
            loss_weight=2.5,
        )
        self.nms_cfg = dict(
            nms_type='multiclass_nms',
            score_threshold=0.025,
            nms_threshold=0.6,
            nms_top_k=1000,
            keep_top_k=100,
        )

        # ---------------- 预处理相关 ---------------- #
        self.context = {'fields': ['image', 'gt_bbox', 'gt_class', 'gt_score']}
        # DecodeImage
        self.decodeImage = dict(
            to_rgb=True,
            with_mixup=False,
            with_cutmix=False,
            with_mosaic=False,
        )
        # RandomCrop
        self.randomCrop = dict()
        # RandomFlipImage
        self.randomFlipImage = dict(
            is_normalized=False,
        )
        # ColorDistort
        self.colorDistort = dict()
        # RandomShape
        self.randomShape = dict(
            sizes=[352, 384, 416, 448, 480],
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
            target_size=416,
            interp=2,
        )

        # 预处理顺序。增加一些数据增强时这里也要加上，否则train.py中相当于没加！
        self.sample_transforms_seq = []
        self.sample_transforms_seq.append('decodeImage')
        self.sample_transforms_seq.append('randomCrop')
        self.sample_transforms_seq.append('randomFlipImage')
        self.sample_transforms_seq.append('colorDistort')
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
        from mmdet.models import LCNet, ATSSAssigner, TaskAlignedAssigner, PositionAssigner, PicoHeadV2, PicoDet, PicoFeat
        from mmdet.models.necks.lc_pan import LCPAN
        from mmdet.models.losses.iou_losses import GIoULoss
        from mmdet.models.losses.varifocal_loss import VarifocalLoss
        from mmdet.models.losses.gfocal_loss import DistributionFocalLoss
        if getattr(self, "model", None) is None:
            Backbone = None
            if self.backbone_type == 'LCNet':
                Backbone = LCNet
            backbone = Backbone(**self.backbone)
            Fpn = None
            if self.fpn_type == 'LCPAN':
                Fpn = LCPAN
            fpn = Fpn(**self.fpn)
            conv_feat = PicoFeat(**self.conv_feat)
            static_assigner = ATSSAssigner(**self.static_assigner)
            Assigner = None
            if self.assigner_type == 'TaskAlignedAssigner':
                Assigner = TaskAlignedAssigner
            elif self.assigner_type == 'PositionAssigner':
                Assigner = PositionAssigner
            assigner = Assigner(**self.assigner)
            loss_class = VarifocalLoss(**self.loss_class)
            loss_dfl = DistributionFocalLoss(**self.loss_dfl)
            loss_bbox = GIoULoss(**self.loss_bbox)
            head = PicoHeadV2(conv_feat=conv_feat, static_assigner=static_assigner,
                              assigner_type=self.assigner_type, assigner=assigner, loss_class=loss_class, loss_dfl=loss_dfl,
                              loss_bbox=loss_bbox, nms_cfg=self.nms_cfg, **self.head)
            self.model = PicoDet(backbone, fpn, head)
        return self.model

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

        # collater = PicoDetTrainCollater(self.context, batch_transforms, self.n_layers)
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
