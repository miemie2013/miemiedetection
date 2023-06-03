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


class RTDETR_Method_Exp(COCOBaseExp):
    def __init__(self):
        super().__init__()
        # ---------------- architecture name(算法名) ---------------- #
        self.archi_name = 'RTDETR'

        # --------------  training config --------------------- #
        self.max_epoch = 72

        self.ema = True
        self.ema_decay = 0.9999
        self.ema_decay_type = "exponential"
        self.ema_filter_no_grad = True
        self.weight_decay = 0.0001
        self.print_interval = 20
        self.eval_interval = 10
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # RTDETR 有梯度裁剪，参数全部都不使用L2正则化，而且使用的是AdamW优化器 weight_decay=0.0001
        # learning_rate
        self.scheduler = "warm_piecewisedecay"
        self.warmup_epochs = 1
        self.basic_lr_per_img = 0.0001 / (4. * 4.0)
        self.clip_grad_by_norm = 0.1
        self.start_factor = 0.001
        self.decay_gamma = 1.0
        self.milestones_epoch = [999999, ]

        # -----------------  testing config ------------------ #
        self.eval_height = 640
        self.eval_width = 640
        self.test_size = [self.eval_height, self.eval_width]

        # ---------------- model config ---------------- #
        self.output_dir = "RTDETR_outputs"
        self.hidden_dim = 256
        self.use_focal_loss = True
        self.with_mask = False
        self.backbone_type = 'ResNet'
        self.backbone = dict(
            depth=50,
            variant='d',
            freeze_at=0,
            return_idx=[1, 2, 3],
            lr_mult_list=[0.1, 0.1, 0.1, 0.1],
            num_stages=4,
            freeze_stem_only=True,
        )
        self.neck_type = 'HybridEncoder'
        self.neck = dict(
            hidden_dim=self.hidden_dim,
            use_encoder_idx=[2],
            num_encoder_layers=1,
            encoder_layer=dict(
                name='TransformerLayer',
                d_model=self.hidden_dim,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.,
                activation='gelu',
            ),
            expansion=1.,
            eval_size=self.test_size,
        )
        self.transformer_type = 'RTDETRTransformer'
        self.transformer = dict(
            num_queries=300,
            position_embed_type='sine',
            backbone_feat_channels=[self.hidden_dim, self.hidden_dim, self.hidden_dim],
            feat_strides=[8, 16, 32],
            num_levels=3,
            nhead=8,
            num_decoder_layers=6,
            dim_feedforward=1024,
            dropout=0.0,
            activation='relu',
            num_denoising=100,
            label_noise_ratio=0.5,
            box_noise_scale=1.0,
            learnt_init_query=False,
            num_classes=self.num_classes,
            hidden_dim=self.hidden_dim,
            eval_size=self.test_size,
        )
        self.detr_head_type = 'DINOHead'
        self.detr_head = dict(
            loss=dict(
                name='DINOLoss',
                loss_coeff={'class': 1.0, 'bbox': 5.0, 'giou': 2.0},
                aux_loss=True,
                use_vfl=True,
                matcher=dict(
                    name='HungarianMatcher',
                    matcher_coeff={'class': 2.0, 'bbox': 5.0, 'giou': 2.0},
                ),
            ),
        )
        self.post_process_type = 'DETRPostProcess'
        self.post_cfg = dict(
            num_top_queries=300,
            num_classes=self.num_classes,
            use_focal_loss=self.use_focal_loss,
            with_mask=self.with_mask,
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
            sizes=[480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800],
            random_inter=True,
            resize_box=True,
        )
        # NormalizeImage
        self.normalizeImage = dict(
            mean=[0., 0., 0.],
            std=[1., 1., 1.],
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
        # NormalizeBox
        self.normalizeBox = dict()
        # BboxXYXY2XYWH
        self.bboxXYXY2XYWH = dict()
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
        self.sample_transforms_seq.append('normalizeBox')
        self.sample_transforms_seq.append('bboxXYXY2XYWH')
        self.batch_transforms_seq = []
        # self.batch_transforms_seq.append('padGT')

        # ---------------- dataloader config ---------------- #
        # 默认是4。如果报错“OSError: [WinError 1455] 页面文件太小,无法完成操作”，设置为2或0解决。
        self.data_num_workers = 2
        self.eval_data_num_workers = 2

    def get_model(self):
        from mmdet.models import ResNet, HybridEncoder, RTDETRTransformer, DINOHead, DETR
        from mmdet.models.transformers.hybrid_encoder import TransformerLayer
        from mmdet.models.post_process import DETRPostProcess
        if getattr(self, "model", None) is None:
            Backbone = None
            if self.backbone_type == 'ResNet':
                Backbone = ResNet
            backbone = Backbone(**self.backbone)
            Neck = None
            encoder_layer = None
            if self.neck_type == 'HybridEncoder':
                Neck = HybridEncoder
                encoder_layer_cfg = self.neck.pop('encoder_layer')
                name = encoder_layer_cfg.pop('name')
                if name == 'TransformerLayer':
                    encoder_layer = TransformerLayer(**encoder_layer_cfg)
            neck = Neck(encoder_layer=encoder_layer, **self.neck)
            Transformer = None
            if self.transformer_type == 'RTDETRTransformer':
                Transformer = RTDETRTransformer
            transformer = Transformer(**self.transformer)
            Detr_head = None
            if self.detr_head_type == 'DINOHead':
                Detr_head = DINOHead
            detr_head = Detr_head(**self.detr_head)
            Post_process = None
            if self.post_process_type == 'DETRPostProcess':
                Post_process = DETRPostProcess
            post_process = Post_process(**self.post_cfg)
            self.model = DETR(backbone, transformer, detr_head, neck, post_process=post_process, with_mask=False, exclude_post_process=False)
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
            optimizer = torch.optim.AdamW(no_decay[1.0], betas=(0.9, 0.999), lr=lr, eps=1e-8, amsgrad=True)
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
            warmup_lr_start=lr * self.start_factor,
            milestones=self.milestones_epoch,
            gamma=self.decay_gamma,
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
            return_hw=True,
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
        return evaluator.evaluate_rtdetr(model, is_distributed, half)
