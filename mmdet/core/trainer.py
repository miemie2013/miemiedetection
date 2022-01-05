#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import datetime
import os
import math
import time
import numpy as np
from loguru import logger

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from mmdet.data import DataPrefetcher, PPYOLODataPrefetcher
from mmdet.data.data_prefetcher import FCOSDataPrefetcher
from mmdet.utils import (
    MeterBuffer,
    ModelEMA,
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


class Trainer:
    def __init__(self, exp, args):
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # before_train methods.
        self.exp = exp
        # 算法名字
        self.archi_name = self.exp.archi_name
        self.args = args

        # training related attr
        self.max_epoch = exp.max_epoch
        self.amp_training = args.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.device = "cuda:{}".format(self.local_rank)
        self.use_model_ema = exp.ema

        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32

        # YOLOX多尺度训练，初始尺度。
        if self.archi_name == 'YOLOX':
            self.input_size = exp.input_size
        elif self.archi_name == 'PPYOLO':
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
        except Exception:
            raise
        finally:
            self.after_train()

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()

    def train_one_iter(self):
        iter_start_time = time.time()

        if self.archi_name == 'YOLOX':
            inps, targets = self.prefetcher.next()
            inps = inps.to(self.data_type)
            targets = targets.to(self.data_type)
            targets.requires_grad = False
            inps, targets = self.exp.preprocess(inps, targets, self.input_size)
            data_end_time = time.time()

            with torch.cuda.amp.autocast(enabled=self.amp_training):
                outputs = self.model(inps, targets)
        elif self.archi_name == 'PPYOLO':
            if self.n_layers == 3:
                inps, gt_bbox, target0, target1, target2 = self.prefetcher.next()
            elif self.n_layers == 2:
                inps, gt_bbox, target0, target1 = self.prefetcher.next()
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
            data_end_time = time.time()

            with torch.cuda.amp.autocast(enabled=self.amp_training):
                if self.n_layers == 3:
                    targets = [target0, target1, target2]
                elif self.n_layers == 2:
                    targets = [target0, target1]
                outputs = self.model.train_model(inps, gt_bbox, targets)
        elif self.archi_name == 'FCOS':
            if self.n_layers == 5:
                inps, labels0, reg_target0, centerness0, labels1, reg_target1, centerness1, labels2, reg_target2, centerness2, labels3, reg_target3, centerness3, labels4, reg_target4, centerness4 = self.prefetcher.next()
            elif self.n_layers == 3:
                inps, labels0, reg_target0, centerness0, labels1, reg_target1, centerness1, labels2, reg_target2, centerness2 = self.prefetcher.next()
            inps = inps.to(self.data_type)
            labels0 = labels0.to(self.data_type)
            reg_target0 = reg_target0.to(self.data_type)
            centerness0 = centerness0.to(self.data_type)
            labels1 = labels1.to(self.data_type)
            reg_target1 = reg_target1.to(self.data_type)
            centerness1 = centerness1.to(self.data_type)
            labels2 = labels2.to(self.data_type)
            reg_target2 = reg_target2.to(self.data_type)
            centerness2 = centerness2.to(self.data_type)
            if self.n_layers == 5:
                labels3 = labels3.to(self.data_type)
                reg_target3 = reg_target3.to(self.data_type)
                centerness3 = centerness3.to(self.data_type)
                labels4 = labels4.to(self.data_type)
                reg_target4 = reg_target4.to(self.data_type)
                centerness4 = centerness4.to(self.data_type)
                labels3.requires_grad = False
                reg_target3.requires_grad = False
                centerness3.requires_grad = False
                labels4.requires_grad = False
                reg_target4.requires_grad = False
                centerness4.requires_grad = False
            labels0.requires_grad = False
            reg_target0.requires_grad = False
            centerness0.requires_grad = False
            labels1.requires_grad = False
            reg_target1.requires_grad = False
            centerness1.requires_grad = False
            labels2.requires_grad = False
            reg_target2.requires_grad = False
            centerness2.requires_grad = False
            data_end_time = time.time()

            with torch.cuda.amp.autocast(enabled=self.amp_training):
                if self.n_layers == 5:
                    tag_labels = [labels0, labels1, labels2, labels3, labels4]
                    tag_bboxes = [reg_target0, reg_target1, reg_target2, reg_target3, reg_target4]
                    tag_center = [centerness0, centerness1, centerness2, centerness3, centerness4]
                elif self.n_layers == 3:
                    tag_labels = [labels0, labels1, labels2]
                    tag_bboxes = [reg_target0, reg_target1, reg_target2]
                    tag_center = [centerness0, centerness1, centerness2]
                outputs = self.model.train_model(inps, tag_labels, tag_bboxes, tag_center)
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
        elif self.archi_name == 'PPYOLO':
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

    def before_train(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        # model related init
        torch.cuda.set_device(self.local_rank)
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
            # 多卡训练时，使用同步bn
            if self.is_distributed:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                logger.info('Using SyncBatchNorm()')

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
            self.prefetcher = PPYOLODataPrefetcher(self.train_loader, self.n_layers)
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

        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, self.exp.ema_decay)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.model = model
        self.model.train()

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

    def after_train(self):
        logger.info(
            "Training of experiment is done and the best AP is {:.2f}".format(self.best_ap * 100)
        )

    def before_epoch(self):
        logger.info("---> start train epoch{}".format(self.epoch + 1))
        if self.archi_name == 'YOLOX':
            if self.epoch == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
                logger.info("--->No mosaic aug now!")
                self.train_loader.close_mosaic()
                logger.info("--->Add additional L1 loss now!")
                if self.is_distributed:
                    self.model.module.head.use_l1 = True
                else:
                    self.model.head.use_l1 = True
                self.exp.eval_interval = 1
                if not self.no_aug:
                    self.save_ckpt(ckpt_name="last_mosaic_epoch")
        elif self.archi_name == 'PPYOLO':
            self.train_loader.dataset.set_epoch(self.epoch)
        elif self.archi_name == 'FCOS':
            self.train_loader.dataset.set_epoch(self.epoch)
        else:
            raise NotImplementedError("Architectures \'{}\' is not implemented.".format(self.archi_name))

    def after_epoch(self):
        self.save_ckpt(ckpt_name="%d" % (self.epoch + 1))

        if (self.epoch + 1) % self.exp.eval_interval == 0:
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

            log_msg = "{}, mem: {:.0f}Mb, {}, {}, lr: {:.6f}".format(progress_str, gpu_mem_usage(), time_str, loss_str, self.meter["lr"].latest, )
            if self.archi_name == 'YOLOX':
                log_msg += (", size: {:d}, {}".format(self.input_size[0], eta_str))
            elif self.archi_name == 'PPYOLO':
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
        elif self.archi_name == 'FCOS':
            pass
        else:
            raise NotImplementedError("Architectures \'{}\' is not implemented.".format(self.archi_name))

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model):
        if self.args.resume:
            logger.info("resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
            else:
                ckpt_file = self.args.ckpt

            ckpt = torch.load(ckpt_file, map_location=self.device)
            # resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])
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

    def evaluate_and_save_model(self):
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
            evalmodel = self.model
            if is_parallel(evalmodel):
                evalmodel = evalmodel.module

        ap50_95, ap50, summary = self.exp.eval(
            evalmodel, self.evaluator, self.is_distributed
        )
        self.model.train()
        if self.rank == 0:
            self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
            self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)
            logger.info("\n" + summary)
        synchronize()

        self.save_ckpt("last_epoch", ap50_95 > self.best_ap)
        self.best_ap = max(self.best_ap, ap50_95)

    def save_ckpt(self, ckpt_name, update_best_ckpt=False):
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model
            logger.info("Save weights to {}".format(self.file_name))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )
