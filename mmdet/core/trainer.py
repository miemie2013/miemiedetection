#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import datetime
import os
import math
import copy
import time
import numpy as np
import threading
from loguru import logger

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from mmdet.data import DataPrefetcher, PPYOLODataPrefetcher, PPYOLOEDataPrefetcher, SOLODataPrefetcher, yolox_torch_aug, yolox_torch_aug2
from mmdet.data.data_prefetcher import FCOSDataPrefetcher
from mmdet.slim import PPYOLOEDistillModel
from mmdet.utils import (
    MeterBuffer,
    ModelEMA,
    PPdetModelEMA,
    all_reduce_norm,
    get_local_rank,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    is_parallel,
    load_ckpt,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    synchronize
)


def print_diff(dic, key, tensor):
    if tensor is not None:  # 有的梯度张量可能是None
        ddd = np.sum((dic[key] - tensor.cpu().detach().numpy()) ** 2)
        print('diff=%.6f (%s)' % (ddd, key))


class Trainer:
    def __init__(self, exp, args):
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # before_train methods.
        self.exp = exp
        self.slim_exp = exp.slim_exp
        # 算法名字
        self.archi_name = self.exp.archi_name
        self.args = args

        # training related attr
        self.max_epoch = exp.max_epoch
        self.amp_training = args.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.is_distributed = get_world_size() > 1
        self.world_size = get_world_size()
        self.rank = get_rank()     # 单机2卡训练时，0号卡的rank==0, local_rank==0，1号卡的rank==1, local_rank==1
        self.local_rank = get_local_rank()
        self.device = "cuda:{}".format(self.local_rank)
        # self.device = "cpu"
        self.use_model_ema = exp.ema
        self.align_grad = True
        self.align_2gpu_1gpu = True
        self.save_npz = (args.save_npz == 1)

        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32

        # YOLOX多尺度训练，初始尺度。
        if self.archi_name == 'YOLOX':
            self.input_size = exp.input_size
        elif self.archi_name == 'PPYOLO':
            pass
        elif self.archi_name in ['PPYOLOE', 'PicoDet']:
            pass
        elif self.archi_name == 'SOLO':
            pass
        elif self.archi_name == 'FCOS':
            pass
        else:
            raise NotImplementedError("Architectures \'{}\' is not implemented.".format(self.archi_name))
        self.best_ap = 0

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(
            self.file_name,
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )

    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception as e:
            logger.error(e)
            raise
        finally:
            self.after_train()

    def before_train(self):
        self.exp.data_num_workers = self.args.worker_num
        self.exp.eval_data_num_workers = self.args.eval_worker_num
        self.exp.basic_lr_per_img *= self.args.lr_scale
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        # model related init
        torch.cuda.set_device(self.local_rank)

        if self.archi_name in ['PPYOLO', 'PPYOLOE', 'PicoDet', 'SOLO', 'FCOS']:
            torch.backends.cudnn.benchmark = True  # Improves training speed.
            torch.backends.cuda.matmul.allow_tf32 = False  # Allow PyTorch to internally use tf32 for matmul
            torch.backends.cudnn.allow_tf32 = False  # Allow PyTorch to internally use tf32 for convolutions

        slim_model = None
        distill_loss = None
        if self.slim_exp:
            self.exp.for_distill = self.slim_exp.for_distill
            self.exp.feat_distill_place = self.slim_exp.feat_distill_place
            self.exp.__delattr__('slim_exp')
            slim_model = self.slim_exp.get_model()
            slim_model.to(self.device)
            slim_model.is_teacher = True
            slim_model.yolo_head.is_teacher = True
            distill_loss = self.slim_exp.get_distill_loss()
            distill_loss.to(self.device)
        model = self.exp.get_model()
        logger.info("Model Summary: {}".format(get_model_info(self.archi_name, model, self.exp.test_size)))
        model.to(self.device)

        # 是否进行梯度裁剪
        self.need_clip = False

        if self.archi_name == 'YOLOX':
            # solver related init
            self.optimizer = self.exp.get_optimizer(self.args.batch_size)

            # value of epoch will be set in `resume_train`
            model = self.resume_train(model)

            self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
            self.train_loader = self.exp.get_data_loader(
                batch_size=self.args.batch_size,
                is_distributed=self.is_distributed,
                no_aug=self.no_aug,
                cache_img=self.args.cache,
            )
            logger.info("init prefetcher, this might take one minute or less...")
            self.prefetcher = DataPrefetcher(self.train_loader)
        elif self.archi_name == 'PPYOLO':
            # 不可以加正则化的参数：norm层(比如bn层、affine_channel层、gn层)的scale、offset；卷积层的偏移参数。
            self.base_lr = self.exp.basic_lr_per_img * self.args.batch_size
            param_groups = []
            base_wd = self.exp.weight_decay
            momentum = self.exp.momentum
            # 是否进行梯度裁剪
            self.need_clip = hasattr(self.exp, 'clip_grad_by_norm')
            self.clip_norm = 1000000.0
            if self.need_clip:
                self.clip_norm = getattr(self.exp, 'clip_grad_by_norm')
            model.add_param_group(param_groups, self.base_lr, base_wd, self.need_clip, self.clip_norm)

            # solver related init
            self.optimizer = self.exp.get_optimizer(self.args.batch_size, param_groups, momentum=momentum, weight_decay=base_wd)

            # value of epoch will be set in `resume_train`
            model = self.resume_train(model)


            self.train_loader = self.exp.get_data_loader(
                batch_size=self.args.batch_size,
                is_distributed=self.is_distributed,
                num_gpus=self.world_size,
                cache_img=self.args.cache,
            )
            self.n_layers = self.exp.n_layers

            logger.info("init prefetcher, this might take one minute or less...")
            self.prefetcher = PPYOLODataPrefetcher(self.train_loader, self.n_layers)
        elif self.archi_name == 'PPYOLOE':
            # 不可以加正则化的参数：norm层(比如bn层、affine_channel层、gn层)的scale、offset；卷积层的偏移参数。
            self.base_lr = self.exp.basic_lr_per_img * self.args.batch_size
            param_groups = []
            base_wd = self.exp.weight_decay
            momentum = self.exp.momentum
            # 是否进行梯度裁剪
            self.need_clip = hasattr(self.exp, 'clip_grad_by_norm')
            self.clip_norm = 1000000.0
            if self.need_clip:
                self.clip_norm = getattr(self.exp, 'clip_grad_by_norm')
            model.add_param_group(param_groups, self.base_lr, base_wd, self.need_clip, self.clip_norm)

            # solver related init
            self.optimizer = self.exp.get_optimizer(self.args.batch_size, param_groups, momentum=momentum, weight_decay=base_wd)

            # value of epoch will be set in `resume_train`
            model = self.resume_train(model, distill_loss)
            if self.slim_exp:
                slim_model = self.resume_slim_model(slim_model)
                model = PPYOLOEDistillModel(model, slim_model, distill_loss)


            self.train_loader = self.exp.get_data_loader(
                batch_size=self.args.batch_size,
                is_distributed=self.is_distributed,
                num_gpus=self.world_size,
                cache_img=self.args.cache,
            )
            self.n_layers = self.exp.n_layers

            logger.info("init prefetcher, this might take one minute or less...")
            self.prefetcher = PPYOLOEDataPrefetcher(self.train_loader, self.n_layers)
        elif self.archi_name == 'PicoDet':
            # 不可以加正则化的参数：norm层(比如bn层、affine_channel层、gn层)的scale、offset；卷积层的偏移参数。
            self.optimizer = self.exp.get_optimizer(self.args.batch_size)

            # value of epoch will be set in `resume_train`
            model = self.resume_train(model)


            self.train_loader = self.exp.get_data_loader(
                batch_size=self.args.batch_size,
                is_distributed=self.is_distributed,
                num_gpus=self.world_size,
                cache_img=self.args.cache,
            )
            self.n_layers = self.exp.n_layers

            logger.info("init prefetcher, this might take one minute or less...")
            self.prefetcher = PPYOLOEDataPrefetcher(self.train_loader, self.n_layers)
        elif self.archi_name == 'SOLO':
            # 不可以加正则化的参数：norm层(比如bn层、affine_channel层、gn层)的scale、offset；卷积层的偏移参数。
            self.base_lr = self.exp.basic_lr_per_img * self.args.batch_size
            param_groups = []
            base_wd = self.exp.weight_decay
            momentum = self.exp.momentum
            # 是否进行梯度裁剪
            self.need_clip = hasattr(self.exp, 'clip_grad_by_norm')
            self.clip_norm = 1000000.0
            if self.need_clip:
                self.clip_norm = getattr(self.exp, 'clip_grad_by_norm')
            model.add_param_group(param_groups, self.base_lr, base_wd, self.need_clip, self.clip_norm)

            # solver related init
            self.optimizer = self.exp.get_optimizer(self.args.batch_size, param_groups, momentum=momentum, weight_decay=base_wd)

            # value of epoch will be set in `resume_train`
            model = self.resume_train(model)


            self.train_loader = self.exp.get_data_loader(
                batch_size=self.args.batch_size,
                is_distributed=self.is_distributed,
                num_gpus=self.world_size,
                cache_img=self.args.cache,
            )
            self.n_layers = self.exp.n_layers

            logger.info("init prefetcher, this might take one minute or less...")
            self.prefetcher = SOLODataPrefetcher(self.train_loader, self.n_layers)
        elif self.archi_name == 'FCOS':
            # 不可以加正则化的参数：norm层(比如bn层、affine_channel层、gn层)的scale、offset；卷积层的偏移参数。
            self.base_lr = self.exp.basic_lr_per_img * self.args.batch_size
            param_groups = []
            base_wd = self.exp.weight_decay
            momentum = self.exp.momentum
            # 是否进行梯度裁剪
            self.need_clip = hasattr(self.exp, 'clip_grad_by_norm')
            self.clip_norm = 1000000.0
            if self.need_clip:
                self.clip_norm = getattr(self.exp, 'clip_grad_by_norm')
            model.add_param_group(param_groups, self.base_lr, base_wd, self.need_clip, self.clip_norm)

            # solver related init
            self.optimizer = self.exp.get_optimizer(self.args.batch_size, param_groups, momentum=momentum, weight_decay=base_wd)

            # value of epoch will be set in `resume_train`
            model = self.resume_train(model)


            self.train_loader = self.exp.get_data_loader(
                batch_size=self.args.batch_size,
                is_distributed=self.is_distributed,
                cache_img=self.args.cache,
            )
            self.n_layers = self.exp.n_layers

            logger.info("init prefetcher, this might take one minute or less...")
            self.prefetcher = FCOSDataPrefetcher(self.train_loader, self.n_layers)
        else:
            raise NotImplementedError("Architectures \'{}\' is not implemented.".format(self.archi_name))

        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )
        if self.args.occupy:
            occupy_mem(self.local_rank)

        if self.archi_name in ['PPYOLO', 'PPYOLOE', 'PicoDet', 'SOLO', 'FCOS']:
            # 多卡训练时，使用同步bn。
            # torch.nn.SyncBatchNorm.convert_sync_batchnorm()的使用一定要在创建优化器之后，创建DDP之前。
            if self.is_distributed:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                logger.info('Using SyncBatchNorm()')

        if self.is_distributed:
            find_unused_parameters = False
            if self.archi_name in ['PicoDet']:
                find_unused_parameters = True
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False, find_unused_parameters=find_unused_parameters)

        if self.use_model_ema:
            if self.archi_name == 'YOLOX':
                self.ema_model = ModelEMA(model, self.exp.ema_decay)
                self.ema_model.updates = self.max_iter * self.start_epoch
            elif self.archi_name in ['PPYOLO', 'PPYOLOE', 'PicoDet', 'SOLO', 'FCOS']:
                ema_decay = getattr(self.exp, 'ema_decay', 0.9998)
                cycle_epoch = getattr(self.exp, 'cycle_epoch', -1)
                ema_decay_type = getattr(self.exp, 'ema_decay_type', 'threshold')
                self.ema_model = PPdetModelEMA(
                    model,
                    decay=ema_decay,
                    ema_decay_type=ema_decay_type,
                    cycle_epoch=cycle_epoch)
            else:
                raise NotImplementedError("Architectures \'{}\' is not implemented.".format(self.archi_name))

        self.model = model
        self.model.train()
        if self.slim_exp:
            if self.is_distributed:
                self.model.module.teacher_model.eval()   # 防止老师bn层的均值方差发生变化
            else:
                self.model.teacher_model.eval()   # 防止老师bn层的均值方差发生变化

        self.evaluator = self.exp.get_evaluator(
            batch_size=self.args.eval_batch_size, is_distributed=self.is_distributed
        )

        # Tensorboard logger
        if self.rank == 0:
            self.tblogger = SummaryWriter(self.file_name)

        logger.info("Training start...")
        # logger.info("\n{}".format(model))
        trainable_params = 0
        nontrainable_params = 0
        for name_, param_ in model.named_parameters():
            mul = np.prod(param_.shape)
            if param_.requires_grad is True:
                trainable_params += mul
            else:
                nontrainable_params += mul
        total_params = trainable_params + nontrainable_params
        logger.info('Total params: %s' % format(total_params, ","))
        logger.info('Trainable params: %s' % format(trainable_params, ","))
        logger.info('Non-trainable params: %s' % format(nontrainable_params, ","))
        if self.archi_name == 'YOLOX':
            logger.info("use torch_augment:")
            logger.info(self.exp.torch_augment)
            if self.exp.torch_augment:
                # Mosaic cache
                self.mosaic_max_cached_images = 40
                self.random_pop = self.exp.width > 0.4999  # ['s', 'm', 'l', 'x']
                self.mosaic_cache = []
                # Mixup cache
                self.mixup_max_cached_images = 20
                self.mixup_cache = []
                if self.exp.width <= 0.4999:  # ['nano', 'tiny']
                    self.mosaic_max_cached_images = self.mosaic_max_cached_images // 2
                    self.mixup_max_cached_images = self.mixup_max_cached_images // 2

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            train_start = time.time()
            self.train_in_iter()
            if self.rank == 0:
                cost = time.time() - train_start
                logger.info('Train epoch %d cost time: %.1f s.' % (self.epoch + 1, cost))
            self.after_epoch()

    def after_train(self):
        logger.info(
            "Training of experiment is done and the best AP is {:.2f}".format(self.best_ap * 100)
        )

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()
            # if self.iter == 10:
            #     break

    def train_one_iter(self):
        iter_start_time = time.time()
        batch_idx = self.iter
        rank = self.rank

        if self.archi_name == 'YOLOX':
            inps, targets = self.prefetcher.next()
            if self.exp.torch_augment:
                # 先转fp16再增强会掉精度，所以用fp32做增强
                with torch.no_grad():
                    inps, targets = yolox_torch_aug(inps, targets, self.mosaic_cache, self.mixup_cache,
                                                    self.mosaic_max_cached_images, self.mixup_max_cached_images,
                                                    self.random_pop, self.exp, self.use_mosaic, self.rank)
            inps = inps.to(self.data_type)
            targets = targets.to(self.data_type)
            targets.requires_grad = False
            inps, targets = self.exp.preprocess(inps, targets, self.input_size)
            data_end_time = time.time()

            with torch.cuda.amp.autocast(enabled=self.amp_training):
                outputs = self.model(inps, targets)
        elif self.archi_name == 'PPYOLO':
            if self.n_layers == 3:
                inps, gt_bbox, target0, target1, target2, im_ids = self.prefetcher.next()
            elif self.n_layers == 2:
                inps, gt_bbox, target0, target1, im_ids = self.prefetcher.next()
            inps = inps.to(self.data_type)
            gt_bbox = gt_bbox.to(self.data_type)
            target0 = target0.to(self.data_type)
            target1 = target1.to(self.data_type)
            if self.n_layers == 3:
                target2 = target2.to(self.data_type)
                target2.requires_grad = False
            gt_bbox.requires_grad = False
            target0.requires_grad = False
            target1.requires_grad = False
            # 用张量操作实现预处理，因为DataLoader的collate_fn太费时了
            # inps, targets = self.exp.preprocess(inps, targets, self.input_size)
            data_end_time = time.time()

            with torch.cuda.amp.autocast(enabled=self.amp_training):
                if self.n_layers == 3:
                    targets = [target0, target1, target2]
                elif self.n_layers == 2:
                    targets = [target0, target1]
                '''
                获得损失（训练）、推理 都要放在forward()中进行，否则DDP会计算错误结果。
                '''
                outputs = self.model(inps, None, gt_bbox, targets)
        elif self.archi_name in ['PPYOLOE', 'PicoDet']:
            inps, gt_class, gt_bbox, pad_gt_mask, im_ids = self.prefetcher.next()
            inps = inps.to(self.data_type)
            gt_class = gt_class.to(self.data_type)
            gt_bbox = gt_bbox.to(self.data_type)
            pad_gt_mask = pad_gt_mask.to(self.data_type)

            # miemie2013: 剪掉填充的gt
            num_boxes = pad_gt_mask.sum([1, 2])
            num_max_boxes = num_boxes.max().to(torch.int32)
            pad_gt_mask = pad_gt_mask[:, :num_max_boxes, :]
            gt_class = gt_class[:, :num_max_boxes, :]
            gt_bbox = gt_bbox[:, :num_max_boxes, :]
            gt_class.requires_grad = False
            gt_bbox.requires_grad = False
            pad_gt_mask.requires_grad = False
            data_end_time = time.time()

            with torch.cuda.amp.autocast(enabled=self.amp_training):
                targets = dict(
                    gt_class=gt_class,
                    gt_bbox=gt_bbox,
                    pad_gt_mask=pad_gt_mask,
                    epoch_id=self.epoch,
                )
                '''
                获得损失（训练）、推理 都要放在forward()中进行，否则DDP会计算错误结果。
                '''
                outputs = self.model(inps, None, targets)
        elif self.archi_name == 'SOLO':
            inps, *labels, fg_nums, im_ids = self.prefetcher.next()
            inps = inps.to(self.data_type)
            data_end_time = time.time()

            with torch.cuda.amp.autocast(enabled=self.amp_training):
                '''
                获得损失（训练）、推理 都要放在forward()中进行，否则DDP会计算错误结果。
                '''
                outputs = self.model(inps, None, None, labels, fg_nums)
        else:
            raise NotImplementedError("Architectures \'{}\' is not implemented.".format(self.archi_name))


        loss = outputs["total_loss"]

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        # 梯度裁剪
        if self.need_clip:
            for param_group in self.optimizer.param_groups:
                if param_group['need_clip']:
                    torch.nn.utils.clip_grad_norm_(param_group['params'], max_norm=param_group['clip_norm'], norm_type=2)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.use_model_ema:
            self.ema_model.update(self.model)

        # 修改学习率
        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        if self.archi_name == 'YOLOX':
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
        elif self.archi_name == 'PicoDet':
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
        elif self.archi_name == 'PPYOLO':
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr * param_group['base_lr'] / self.base_lr   # = lr * 参数自己的学习率
        elif self.archi_name == 'PPYOLOE':
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr * param_group['base_lr'] / self.base_lr   # = lr * 参数自己的学习率
        elif self.archi_name == 'SOLO':
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr * param_group['base_lr'] / self.base_lr   # = lr * 参数自己的学习率
        elif self.archi_name == 'FCOS':
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr * param_group['base_lr'] / self.base_lr   # = lr * 参数自己的学习率
        else:
            raise NotImplementedError("Architectures \'{}\' is not implemented.".format(self.archi_name))

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )

    def before_epoch(self):
        logger.info("---> start train epoch{}".format(self.epoch + 1))
        if self.archi_name == 'YOLOX':
            if self.exp.torch_augment:
                self.use_mosaic = True
                if self.epoch >= self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
                    while len(self.mosaic_cache) > 0:
                        self.mosaic_cache.pop(0)
                    while len(self.mixup_cache) > 0:
                        self.mixup_cache.pop(0)
                    self.use_mosaic = False
            if self.epoch == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
                logger.info("--->No mosaic aug now!")
                if self.exp.torch_augment:
                    pass
                else:
                    self.train_loader.close_mosaic()
                logger.info("--->Add additional L1 loss now!")
                if self.is_distributed:
                    self.model.module.head.use_l1 = True
                else:
                    self.model.head.use_l1 = True
                if self.exp.num_classes == 80 and self.exp.val_ann == "instances_val2017.json":
                    self.exp.eval_interval = 1
                    logger.info("--->For COCO dataset, we modify eval_interval==1 now!")
                else:
                    logger.info("--->For other dataset, we dont modify eval_interval!")
                if not self.no_aug:
                    self.save_ckpt(ckpt_name="last_mosaic_epoch")
        elif self.archi_name == 'PPYOLO':
            pass
        elif self.archi_name == 'PPYOLOE':
            pass
        elif self.archi_name == 'PicoDet':
            pass
        elif self.archi_name == 'SOLO':
            pass
        elif self.archi_name == 'FCOS':
            self.train_loader.dataset.set_epoch(self.epoch)
        else:
            raise NotImplementedError("Architectures \'{}\' is not implemented.".format(self.archi_name))

    def after_epoch(self):
        self.save_ckpt(ckpt_name="%d" % (self.epoch + 1))

        if (self.epoch + 1) % self.exp.eval_interval == 0:
            # 除了YOLOX，其它模型都用torch.nn.SyncBatchNorm.convert_sync_batchnorm()把所有bn转换成同步bn，所以不需要调用all_reduce_norm()；
            # 里面的中间变量states会变成一个空字典，会导致all_reduce()方法报错。
            if self.archi_name == 'YOLOX':
                all_reduce_norm(self.model)
            self.evaluate_and_save_model()

    def before_iter(self):
        pass

    def after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        if (self.iter + 1) % self.exp.print_interval == 0:
            # TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            log_msg = "{}, lr: {:.6f}, mem: {:.0f}Mb, {}, {}".format(progress_str, self.meter["lr"].latest, gpu_mem_usage(), time_str, loss_str, )
            if self.archi_name == 'YOLOX':
                log_msg += (", size: {:d}, {}".format(self.input_size[0], eta_str))
            elif self.archi_name == 'PPYOLO':
                log_msg += (", {}".format(eta_str))
            elif self.archi_name == 'PPYOLOE':
                log_msg += (", {}".format(eta_str))
            elif self.archi_name == 'PicoDet':
                log_msg += (", {}".format(eta_str))
            elif self.archi_name == 'SOLO':
                log_msg += (", {}".format(eta_str))
            elif self.archi_name == 'FCOS':
                log_msg += (", {}".format(eta_str))
            else:
                raise NotImplementedError("Architectures \'{}\' is not implemented.".format(self.archi_name))
            logger.info(log_msg)
            self.meter.clear_meters()

        if self.archi_name == 'YOLOX':
            # random resizing
            if (self.progress_in_iter + 1) % 10 == 0:
                self.input_size = self.exp.random_resize(
                    self.train_loader, self.epoch, self.rank, self.is_distributed
                )
        elif self.archi_name == 'PPYOLO':
            pass
        elif self.archi_name == 'PPYOLOE':
            pass
        elif self.archi_name == 'PicoDet':
            pass
        elif self.archi_name == 'SOLO':
            pass
        elif self.archi_name == 'FCOS':
            pass
        else:
            raise NotImplementedError("Architectures \'{}\' is not implemented.".format(self.archi_name))

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model, distill_loss=None):
        if self.args.resume:
            logger.info("resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.file_name, "last_ckpt.pth")
            else:
                ckpt_file = self.args.ckpt
                if '/' not in ckpt_file and not os.path.exists(ckpt_file):
                    ckpt_file = os.path.join(self.file_name, ckpt_file)

            ckpt = torch.load(ckpt_file, map_location=self.device)
            # resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])
            if distill_loss and "distill_loss" in ckpt.keys():
                logger.info("resume distill_loss weights!!!")
                distill_loss.load_state_dict(ckpt["distill_loss"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            # resume the training states variables
            start_epoch = ckpt["start_epoch"]
            self.start_epoch = start_epoch
            logger.info(
                "loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, self.start_epoch
                )
            )  # noqa
        else:
            if self.args.ckpt is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.args.ckpt
                ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
                model = load_ckpt(model, ckpt)
            self.start_epoch = 0

        return model

    def resume_slim_model(self, model):
        if self.args.slim_ckpt is not None:
            logger.info("loading slim checkpoint!!!")
            ckpt_file = self.args.slim_ckpt
            ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
            model = load_ckpt(model, ckpt)
        return model

    def evaluate_and_save_model(self):
        if self.use_model_ema:
            if self.archi_name == 'YOLOX':
                evalmodel = self.ema_model.ema
            elif self.archi_name in ['PPYOLO', 'PPYOLOE', 'PicoDet', 'SOLO', 'FCOS']:
                cur_weight = copy.deepcopy(self.model.state_dict())
                if self.is_distributed:
                    self.model.module.load_state_dict(self.ema_model.apply(), strict=False)
                else:
                    self.model.load_state_dict(self.ema_model.apply(), strict=False)
                evalmodel = self.model
                if is_parallel(evalmodel):
                    evalmodel = evalmodel.module
            else:
                raise NotImplementedError("Architectures \'{}\' is not implemented.".format(self.archi_name))
        else:
            evalmodel = self.model
            if is_parallel(evalmodel):
                evalmodel = evalmodel.module

        ap50_95, ap50, summary = self.exp.eval(
            evalmodel, self.evaluator, self.is_distributed
        )
        self.model.train()
        if self.slim_exp:
            if self.is_distributed:
                self.model.module.teacher_model.eval()   # 防止老师bn层的均值方差发生变化
            else:
                self.model.teacher_model.eval()   # 防止老师bn层的均值方差发生变化
        if self.use_model_ema:
            if self.archi_name in ['PPYOLO', 'PPYOLOE', 'PicoDet', 'SOLO', 'FCOS']:
                self.model.load_state_dict(cur_weight)
                del cur_weight
            elif self.archi_name in ['YOLOX']:
                pass
            else:
                raise NotImplementedError("Architectures \'{}\' is not implemented.".format(self.archi_name))
        if self.rank == 0:
            self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
            self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)
            logger.info("\n" + summary)
        synchronize()

        self.save_ckpt("last_ckpt", ap50_95 > self.best_ap)
        self.best_ap = max(self.best_ap, ap50_95)

    def save_ckpt(self, ckpt_name, update_best_ckpt=False):
        if self.rank == 0:
            if self.archi_name == 'YOLOX':
                save_model = self.ema_model.ema if self.use_model_ema else self.model
                if self.is_distributed and not self.use_model_ema:
                    save_model = save_model.module
            elif self.archi_name in ['PPYOLO', 'PPYOLOE', 'PicoDet', 'SOLO', 'FCOS']:
                if self.use_model_ema:
                    cur_weight = copy.deepcopy(self.model.state_dict())
                    if self.is_distributed:
                        self.model.module.load_state_dict(self.ema_model.apply(), strict=False)
                    else:
                        self.model.load_state_dict(self.ema_model.apply(), strict=False)
                save_model = self.model
                if is_parallel(save_model):
                    save_model = save_model.module
            else:
                raise NotImplementedError("Architectures \'{}\' is not implemented.".format(self.archi_name))

            logger.info("Save weights to {}".format(self.file_name))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "optimizer": self.optimizer.state_dict(),
            }
            # 知识蒸馏时，只保存 学生模型 和 distill_loss
            if self.slim_exp:
                ckpt_state["model"] = save_model.student_model.state_dict()
                ckpt_state["distill_loss"] = save_model.distill_loss.state_dict()
            else:
                ckpt_state["model"] = save_model.state_dict()
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )
            if self.archi_name == 'YOLOX':
                pass
            elif self.archi_name in ['PPYOLO', 'PPYOLOE', 'PicoDet', 'SOLO', 'FCOS']:
                if self.use_model_ema:
                    self.model.load_state_dict(cur_weight)
                    del cur_weight
            else:
                raise NotImplementedError("Architectures \'{}\' is not implemented.".format(self.archi_name))
