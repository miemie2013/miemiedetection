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


class FCOSEvalCollater():
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
            samples.append(sample)

        # batch_transforms
        for batch_transform in self.batch_transforms:
            samples = batch_transform(samples, self.context)

        # 取出感兴趣的项
        images = []
        im_scales = []
        im_ids = []
        for i, sample in enumerate(samples):
            images.append(sample['image'])
            im_scales.append(sample['im_info'][2:3])
            im_ids.append(sample['im_id'])
        images = np.stack(images, axis=0)
        im_scales = np.stack(im_scales, axis=0)
        im_ids = np.stack(im_ids, axis=0)

        images = torch.Tensor(images)
        im_scales = torch.Tensor(im_scales)
        im_ids = torch.Tensor(im_ids)
        return images, im_scales, im_ids


class FCOSTrainCollater():
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
            samples.append(sample)

        # batch_transforms
        for batch_transform in self.batch_transforms:
            samples = batch_transform(samples, self.context)

        # 取出感兴趣的项
        images = []
        labels0 = []
        reg_target0 = []
        centerness0 = []
        labels1 = []
        reg_target1 = []
        centerness1 = []
        labels2 = []
        reg_target2 = []
        centerness2 = []
        labels3 = []
        reg_target3 = []
        centerness3 = []
        labels4 = []
        reg_target4 = []
        centerness4 = []
        for i, sample in enumerate(samples):
            images.append(sample['image'].astype(np.float32))
            labels0.append(sample['labels0'].astype(np.int32))
            reg_target0.append(sample['reg_target0'].astype(np.float32))
            centerness0.append(sample['centerness0'].astype(np.float32))
            labels1.append(sample['labels1'].astype(np.int32))
            reg_target1.append(sample['reg_target1'].astype(np.float32))
            centerness1.append(sample['centerness1'].astype(np.float32))
            labels2.append(sample['labels2'].astype(np.int32))
            reg_target2.append(sample['reg_target2'].astype(np.float32))
            centerness2.append(sample['centerness2'].astype(np.float32))
            if self.n_layers == 5:
                labels3.append(sample['labels3'].astype(np.int32))
                reg_target3.append(sample['reg_target3'].astype(np.float32))
                centerness3.append(sample['centerness3'].astype(np.float32))
                labels4.append(sample['labels4'].astype(np.int32))
                reg_target4.append(sample['reg_target4'].astype(np.float32))
                centerness4.append(sample['centerness4'].astype(np.float32))
        images = np.stack(images, axis=0)
        labels0 = np.stack(labels0, axis=0)
        reg_target0 = np.stack(reg_target0, axis=0)
        centerness0 = np.stack(centerness0, axis=0)
        labels1 = np.stack(labels1, axis=0)
        reg_target1 = np.stack(reg_target1, axis=0)
        centerness1 = np.stack(centerness1, axis=0)
        labels2 = np.stack(labels2, axis=0)
        reg_target2 = np.stack(reg_target2, axis=0)
        centerness2 = np.stack(centerness2, axis=0)

        images = torch.Tensor(images)
        labels0 = torch.Tensor(labels0)
        reg_target0 = torch.Tensor(reg_target0)
        centerness0 = torch.Tensor(centerness0)
        labels1 = torch.Tensor(labels1)
        reg_target1 = torch.Tensor(reg_target1)
        centerness1 = torch.Tensor(centerness1)
        labels2 = torch.Tensor(labels2)
        reg_target2 = torch.Tensor(reg_target2)
        centerness2 = torch.Tensor(centerness2)
        if self.n_layers == 5:
            labels3 = np.stack(labels3, axis=0)
            reg_target3 = np.stack(reg_target3, axis=0)
            centerness3 = np.stack(centerness3, axis=0)
            labels4 = np.stack(labels4, axis=0)
            reg_target4 = np.stack(reg_target4, axis=0)
            centerness4 = np.stack(centerness4, axis=0)

            labels3 = torch.Tensor(labels3)
            reg_target3 = torch.Tensor(reg_target3)
            centerness3 = torch.Tensor(centerness3)
            labels4 = torch.Tensor(labels4)
            reg_target4 = torch.Tensor(reg_target4)
            centerness4 = torch.Tensor(centerness4)
            return images, labels0, reg_target0, centerness0, labels1, reg_target1, centerness1, labels2, reg_target2, centerness2, labels3, reg_target3, centerness3, labels4, reg_target4, centerness4
        return images, labels0, reg_target0, centerness0, labels1, reg_target1, centerness1, labels2, reg_target2, centerness2


class FCOS_Method_Exp(COCOBaseExp):
    def __init__(self):
        super().__init__()
        # ---------------- architecture name(算法名) ---------------- #
        self.archi_name = 'FCOS'

        # --------------  training config --------------------- #
        self.max_epoch = 48
        self.aug_epochs = 48  # 前几轮进行mixup、cutmix、mosaic

        self.ema = True
        self.ema_decay = 0.9998
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.print_interval = 20
        self.eval_interval = 2
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.learningRate = dict(
            base_lr=0.01 / 16,  # 最初base_lr表示的是每一张图片的学习率。代码中会自动修改为乘以批大小。
            PiecewiseDecay=dict(
                gamma=0.1,
                milestones_epoch=[32, 44],
            ),
            LinearWarmup=dict(
                start_factor=0.3333333333333333,
                steps=500,
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
            freeze_at=2,
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

    def get_model(self):
        from mmdet.models import Resnet50Vd, Resnet18Vd, Resnet50Vb
        from mmdet.models.necks.fpn import FPN
        from mmdet.models import FCOS, FCOSHead
        from mmdet.models import FCOSLoss
        if getattr(self, "model", None) is None:
            Backbone = None
            if self.backbone_type == 'Resnet50Vd':
                Backbone = Resnet50Vd
            elif self.backbone_type == 'Resnet18Vd':
                Backbone = Resnet18Vd
            elif self.backbone_type == 'Resnet50Vb':
                Backbone = Resnet50Vb
            backbone = Backbone(**self.backbone)
            # 冻结骨干网络
            backbone.freeze()
            backbone.fix_bn()
            Fpn = None
            if self.fpn_type == 'FPN':
                Fpn = FPN
            fpn = Fpn(**self.fpn)
            fcos_loss = FCOSLoss(**self.fcos_loss)
            head = FCOSHead(fcos_loss=fcos_loss, nms_cfg=self.nms_cfg, **self.head)
            self.model = FCOS(backbone, fpn, head)
        return self.model

    def get_data_loader(
        self, batch_size, start_epoch, is_distributed, cache_img=False
    ):
        from mmdet.data import (
            FCOS_COCOTrainDataset,
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

            train_dataset = FCOS_COCOTrainDataset(
                data_dir=self.data_dir,
                json_file=self.train_ann,
                ann_folder=self.ann_folder,
                name=self.train_image_folder,
                cfg=self,
                sample_transforms=sample_transforms,
                batch_size=batch_size,
                start_epoch=start_epoch,
            )

        self.dataset = train_dataset
        self.epoch_steps = train_dataset.train_steps
        self.max_iters = train_dataset.max_iters
        self.n_layers = train_dataset.n_layers

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

        collater = FCOSTrainCollater(self.context, batch_transforms, self.n_layers)
        train_loader = torch.utils.data.DataLoader(self.dataset, collate_fn=collater, **dataloader_kwargs)

        return train_loader

    def random_resize(self, data_loader, epoch, rank, is_distributed):
        return 1

    def preprocess(self, inputs, targets, tsize):
        return 1

    def get_optimizer(self, param_groups, lr, momentum, weight_decay):
        if "optimizer" not in self.__dict__:
            optimizer = torch.optim.SGD(
                param_groups, lr=lr, momentum=momentum, weight_decay=weight_decay
            )
            self.optimizer = optimizer

        return self.optimizer

    def get_lr_scheduler(self, lr, iters_per_epoch):
        return 1

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from mmdet.data import FCOS_COCOEvalDataset

        # 预测时的数据预处理

        # sample_transforms
        decodeImage = DecodeImage(**self.decodeImage)
        normalizeImage = NormalizeImage(**self.normalizeImage)
        target_size = self.test_size[0]
        max_size = self.test_size[1]
        resizeImage = ResizeImage(target_size=target_size, max_size=max_size, interp=self.resizeImage['interp'],
                                  use_cv2=self.resizeImage['use_cv2'])
        permute = Permute(**self.permute)

        # batch_transforms
        padBatch = PadBatch(**self.padBatch)

        sample_transforms = [decodeImage, normalizeImage, resizeImage, permute]
        batch_transforms = [padBatch]
        val_dataset = FCOS_COCOEvalDataset(
            data_dir=self.data_dir,
            json_file=self.val_ann if not testdev else "image_info_test-dev2017.json",
            ann_folder=self.ann_folder,
            name=self.val_image_folder if not testdev else "test2017",
            cfg=self,
            sample_transforms=sample_transforms,
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

        collater = FCOSEvalCollater(self.context, batch_transforms)
        val_loader = torch.utils.data.DataLoader(val_dataset, collate_fn=collater, **dataloader_kwargs)

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
