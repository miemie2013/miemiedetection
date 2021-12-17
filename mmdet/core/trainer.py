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
        if self.archi_name == 'YOLOX':
            for self.epoch in range(self.start_epoch, self.max_epoch):
                self.before_epoch()
                self.train_in_iter()
                self.after_epoch()
        elif self.archi_name == 'PPYOLO':
            self.train_in_all_iter()   # 所有的epoch被合并成1个大的epoch
        else:
            raise NotImplementedError("Architectures \'{}\' is not implemented.".format(self.archi_name))

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter_yolox()

    def train_in_all_iter(self):   # 所有的epoch被合并成1个大的epoch
        for self.iter in range(self.max_iter):
            self.before_iter()
            train_iter = self.train_one_iter()
            if train_iter:
                self.after_iter_mmdet()
                if (self.iter + 1) % self.epoch_steps == 0:
                    self.epoch = self.iter // self.epoch_steps
                    self.after_epoch()

    def train_one_iter(self):
        iter_start_time = time.time()

        train_iter = True
        if self.archi_name == 'YOLOX':
            inps, targets = self.prefetcher.next()
            inps = inps.to(self.data_type)
            targets = targets.to(self.data_type)
            targets.requires_grad = False
            inps, targets = self.exp.preprocess(inps, targets, self.input_size)
            data_end_time = time.time()

            with torch.cuda.amp.autocast(enabled=self.amp_training):
                outputs = self.model(inps, targets)

            loss = outputs["total_loss"]

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.use_model_ema:
                self.ema_model.update(self.model)

            # 修改学习率
            lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
        elif self.archi_name == 'PPYOLO':
            inps, gt_bbox, gt_score, gt_class, target0, target1, target2 = self.prefetcher.next()
            if self.iter < self.init_iter_id:  # 恢复训练时跳过。
                train_iter = False
                return train_iter
            inps = inps.to(self.data_type)
            gt_bbox = gt_bbox.to(self.data_type)
            gt_score = gt_score.to(self.data_type)
            gt_class = gt_class.to(self.data_type)
            target0 = target0.to(self.data_type)
            target1 = target1.to(self.data_type)
            target2 = target2.to(self.data_type)
            gt_bbox.requires_grad = False
            gt_score.requires_grad = False
            gt_class.requires_grad = False
            target0.requires_grad = False
            target1.requires_grad = False
            target2.requires_grad = False
            data_end_time = time.time()

            with torch.cuda.amp.autocast(enabled=self.amp_training):
                targets = [target0, target1, target2]
                outputs = self.model.train_model(inps, gt_bbox, gt_class, gt_score, targets)

            loss = outputs["total_loss"]

            # 修改学习率
            lr = self.calc_lr(self.iter, self.epoch_steps, self.max_iters, self.exp)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr * param_group['base_lr'] / self.base_lr

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.use_model_ema:
                self.ema_model.update(self.model)
        else:
            raise NotImplementedError("Architectures \'{}\' is not implemented.".format(self.archi_name))

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )
        return train_iter

    def calc_lr(self, iter_id, train_steps, max_iters, cfg):
        base_lr = cfg.learningRate['base_lr']
        piecewiseDecay = cfg.learningRate.get('PiecewiseDecay', None)
        cosineDecay = cfg.learningRate.get('CosineDecay', None)
        linearWarmup = cfg.learningRate.get('LinearWarmup', None)

        cur_lr = base_lr

        linearWarmup_end_iter_id = 0
        skip = False
        if linearWarmup is not None:
            start_factor = linearWarmup['start_factor']
            steps = linearWarmup.get('steps', -1)
            epochs = linearWarmup.get('epochs', -1)

            if steps <= 0 and epochs <= 0:
                raise ValueError(
                    "\'steps\' or \'epochs\' should be positive in {}.learningRate[\'LinearWarmup\']".format(cfg))
            if steps > 0 and epochs > 0:
                steps = -1  # steps和epochs都设置为正整数时，优先选择epochs
            if steps <= 0 and epochs > 0:
                steps = epochs * train_steps

            linearWarmup_end_iter_id = steps
            if iter_id < steps:
                k = (1.0 - start_factor) / steps
                factor = start_factor + k * iter_id
                cur_lr = base_lr * factor
                skip = True

        if skip:
            return cur_lr

        if piecewiseDecay is not None:
            gamma = piecewiseDecay['gamma']
            milestones = piecewiseDecay.get('milestones', None)
            milestones_epoch = piecewiseDecay.get('milestones_epoch', None)

            if milestones is not None:
                pass
            elif milestones_epoch is not None:
                milestones = [f * train_steps for f in milestones_epoch]
            n = len(milestones)
            cur_lr = base_lr
            for i in range(n, 0, -1):
                if iter_id >= milestones[i - 1]:
                    cur_lr = base_lr * gamma ** i
                    break

        if cosineDecay is not None:
            start_iter_id = linearWarmup_end_iter_id
            dx = (iter_id - start_iter_id) / (max_iters - start_iter_id) * math.pi
            cur_lr = base_lr * (1.0 + np.cos(dx)) * 0.5
        return cur_lr

    def before_train(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        # model related init
        torch.cuda.set_device(self.local_rank)
        model = self.exp.get_model()
        logger.info("Model Summary: {}".format(get_model_info(self.archi_name, model, self.exp.test_size)))
        model.to(self.device)

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
        elif self.archi_name == 'PPYOLO':
            # 修改基础学习率
            base_lr = self.exp.learningRate['base_lr']
            base_lr *= self.args.batch_size
            self.exp.learningRate['base_lr'] = base_lr
            self.base_lr = base_lr

            # 不可以加正则化的参数：norm层(比如bn层、affine_channel层、gn层)的scale、offset；卷积层的偏移参数。
            param_groups = []
            base_wd = self.exp.weight_decay
            momentum = self.exp.momentum
            model.add_param_group(param_groups, base_lr, base_wd)

            # solver related init
            self.optimizer = self.exp.get_optimizer(param_groups, lr=base_lr, momentum=momentum, weight_decay=base_wd)

            # value of epoch will be set in `resume_train`
            model = self.resume_train(model)


            self.train_loader, self.epoch_steps, self.max_iters = self.exp.get_data_loader(
                batch_size=self.args.batch_size,
                start_epoch=self.start_epoch,
                is_distributed=self.is_distributed,
                cache_img=self.args.cache,
            )
            # 初始化开始的迭代id
            self.init_iter_id = self.start_epoch * self.epoch_steps

            logger.info("init prefetcher, this might take one minute or less...")
            self.prefetcher = PPYOLODataPrefetcher(self.train_loader)
            # max_iter means iters per epoch
            self.max_iter = len(self.train_loader)

            # self.lr_scheduler = self.exp.get_lr_scheduler(
            #     self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
            # )
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
        else:
            raise NotImplementedError("Architectures \'{}\' is not implemented.".format(self.archi_name))
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

    def after_epoch(self):
        self.save_ckpt(ckpt_name="%d" % (self.epoch + 1))

        if (self.epoch + 1) % self.exp.eval_interval == 0:
            all_reduce_norm(self.model)
            self.evaluate_and_save_model()

    def before_iter(self):
        pass

    def after_iter_yolox(self):
        """
        `after_iter_yolox` contains two parts of logic:
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

            logger.info(
                "{}, mem: {:.0f}Mb, {}, {}, lr: {:.6f}".format(
                    progress_str,
                    gpu_mem_usage(),
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", size: {:d}, {}".format(self.input_size[0], eta_str))
            )
            self.meter.clear_meters()

        # random resizing
        if (self.progress_in_iter + 1) % 10 == 0:
            self.input_size = self.exp.random_resize(
                self.train_loader, self.epoch, self.rank, self.is_distributed
            )


    def after_iter_mmdet(self):
        """
        `after_iter_mmdet` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        if (self.iter + 1) % self.exp.print_interval == 0:
            # TODO check ETA logic
            left_iters = self.max_iters - (self.iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            self.epoch = self.iter // self.epoch_steps
            it_ = self.iter % self.epoch_steps
            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, it_ + 1, self.epoch_steps
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            logger.info(
                "{}, mem: {:.0f}Mb, {}, {}, lr: {:.6f}".format(
                    progress_str,
                    gpu_mem_usage(),
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", {}".format(eta_str))
            )
            self.meter.clear_meters()

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
