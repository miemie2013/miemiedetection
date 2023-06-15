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

from mmcls.data import BaseClsDataPrefetcher
from mmcls.data.data_aug import data_aug
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
        self.world_size = get_world_size()
        self.rank = get_rank()     # 单机2卡训练时，0号卡的rank==0, local_rank==0，1号卡的rank==1, local_rank==1
        self.local_rank = get_local_rank()
        self.device = "cuda:{}".format(self.local_rank)
        # self.device = "cpu"
        self.use_model_ema = exp.ema

        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32

        # BaseCls多尺度训练，初始尺度。
        if self.archi_name == 'BaseCls':
            self.input_size = exp.input_size
        self.best_top1_acc = 0

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
            if self.args.only_eval:
                self.evaluate_and_save_model(save=False)
            else:
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

        torch.backends.cudnn.benchmark = True  # Improves training speed.
        torch.backends.cuda.matmul.allow_tf32 = False  # Allow PyTorch to internally use tf32 for matmul
        torch.backends.cudnn.allow_tf32 = False  # Allow PyTorch to internally use tf32 for convolutions

        model = self.exp.get_model()
        model.to(self.device)

        # 是否进行梯度裁剪
        self.need_clip = False

        if self.archi_name == 'BaseCls':
            # solver related init
            self.optimizer = self.exp.get_optimizer(self.args.batch_size)

            # value of epoch will be set in `resume_train`
            model = self.resume_train(model)

            self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
            self.train_loader = self.exp.get_data_loader(
                batch_size=self.args.batch_size,
                is_distributed=self.is_distributed,
            )
            logger.info("init prefetcher, this might take one minute or less...")
            self.prefetcher = BaseClsDataPrefetcher(self.train_loader)
        else:
            raise NotImplementedError("Architectures \'{}\' is not implemented.".format(self.archi_name))

        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )
        if self.args.occupy:
            occupy_mem(self.local_rank)

        # 多卡训练时，使用同步bn。
        # torch.nn.SyncBatchNorm.convert_sync_batchnorm()的使用一定要在创建优化器之后，创建DDP之前。
        if self.is_distributed:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            logger.info('Using SyncBatchNorm()')

        if self.is_distributed:
            find_unused_parameters = False
            if self.archi_name in ['PicoDet', 'RTDETR']:
                find_unused_parameters = True
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False, find_unused_parameters=find_unused_parameters)

        if self.use_model_ema:
            if self.archi_name == 'BaseCls':
                self.ema_model = ModelEMA(model, self.exp.ema_decay)
                self.ema_model.updates = self.max_iter * self.start_epoch
            elif self.archi_name in ['PPYOLO', 'PPYOLOE', 'PicoDet', 'RTDETR']:
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
            "Training of experiment is done and the best top1_acc is {:.2f}".format(self.best_top1_acc * 100)
        )

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()
            # if self.iter == 1:
            #     break

    def train_one_iter(self):
        iter_start_time = time.time()

        if self.archi_name == 'BaseCls':
            inps, targets = self.prefetcher.next()
            with torch.no_grad():
                inps = data_aug(inps)
            inps = inps.to(self.data_type)
            targets = targets.to(torch.int64)
            targets.requires_grad = False
            data_end_time = time.time()

            with torch.cuda.amp.autocast(enabled=self.amp_training):
                outputs = self.model(inps, targets)
        else:
            raise NotImplementedError("Architectures \'{}\' is not implemented.".format(self.archi_name))


        loss = outputs["total_loss"]

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        # 梯度裁剪
        if self.need_clip:
            for param_group in self.optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(param_group['params'], max_norm=self.clip_norm, norm_type=2)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.use_model_ema:
            self.ema_model.update(self.model)

        # 修改学习率
        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        if self.archi_name in ['BaseCls']:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
        elif self.archi_name in ['PPYOLO', 'PPYOLOE', 'RTDETR']:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr * param_group['lr_factor']   # = lr * 参数自己的学习率
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
        if self.archi_name == 'BaseCls':
            if self.epoch == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
                logger.info("--->No data aug now!")
                self.exp.eval_interval = 1
                logger.info("--->We modify eval_interval==1 now!")
                if not self.no_aug:
                    self.save_ckpt(ckpt_name="last_mosaic_epoch")

    def after_epoch(self):
        self.save_ckpt(ckpt_name="%d" % (self.epoch + 1))

        if (self.epoch + 1) % self.exp.eval_interval == 0:
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

            log_msg = "{}, lr: {:.6f}, {}".format(progress_str, self.meter["lr"].latest, loss_str, )
            if self.archi_name == 'BaseCls':
                log_msg += (", size: {:d}, {}".format(self.input_size[0], eta_str))
            else:
                log_msg += (", {}".format(eta_str))
            logger.info(log_msg)
            self.meter.clear_meters()

        if self.archi_name == 'BaseCls':
            # random resizing
            if (self.progress_in_iter + 1) % 10 == 0:
                self.input_size = self.exp.random_resize(
                    self.train_loader, self.epoch, self.rank, self.is_distributed
                )

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

    def evaluate_and_save_model(self, save=True):
        if self.use_model_ema:
            if self.archi_name == 'BaseCls':
                evalmodel = self.ema_model.ema
            elif self.archi_name in ['PPYOLO', 'PPYOLOE', 'PicoDet', 'RTDETR']:
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

        top1_acc, top5_acc = self.exp.eval(
            evalmodel, self.evaluator, self.is_distributed
        )
        self.model.train()
        if self.use_model_ema:
            if self.archi_name in ['PPYOLO', 'PPYOLOE', 'PicoDet', 'RTDETR']:
                self.model.load_state_dict(cur_weight)
                del cur_weight
            elif self.archi_name in ['BaseCls']:
                pass
            else:
                raise NotImplementedError("Architectures \'{}\' is not implemented.".format(self.archi_name))
        if self.rank == 0:
            if save:
                self.tblogger.add_scalar("val/Top-1 Acc", top1_acc, self.epoch + 1)
                self.tblogger.add_scalar("val/Top-5 Acc", top5_acc, self.epoch + 1)
            logger.info("top1_acc=%.6f, top5_acc=%.6f"%(top1_acc, top5_acc))
        synchronize()

        if save:
            self.save_ckpt("last_ckpt", top1_acc > self.best_top1_acc)
            self.best_top1_acc = max(self.best_top1_acc, top1_acc)

    def save_ckpt(self, ckpt_name, update_best_ckpt=False):
        if self.rank == 0:
            if self.archi_name == 'BaseCls':
                save_model = self.ema_model.ema if self.use_model_ema else self.model
                if self.is_distributed and not self.use_model_ema:
                    save_model = save_model.module
            elif self.archi_name in ['PPYOLO', 'PPYOLOE', 'PicoDet', 'RTDETR']:
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
            ckpt_state["model"] = save_model.state_dict()
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )
            if self.archi_name == 'BaseCls':
                pass
            elif self.archi_name in ['PPYOLO', 'PPYOLOE', 'PicoDet', 'RTDETR']:
                if self.use_model_ema:
                    self.model.load_state_dict(cur_weight)
                    del cur_weight
            else:
                raise NotImplementedError("Architectures \'{}\' is not implemented.".format(self.archi_name))
