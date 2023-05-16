#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import math
import weakref
from copy import deepcopy
from loguru import logger

import torch
import torch.nn as nn

__all__ = ["ModelEMA", "PPdetModelEMA", "is_parallel"]


def is_parallel(model):
    """check if model is in parallel mode."""
    parallel_type = (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel,
    )
    return isinstance(model, parallel_type)


class ModelEMA:
    """
    Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        """
        Args:
            model (nn.Module): model to apply EMA.
            decay (float): ema decay reate.
            updates (int): counter of EMA updates.
        """
        # Create EMA(FP32)
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()
        self.updates = updates   # 这是ema已经更新的步数
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1   # 这是ema已经更新的步数

            # 计算衰减率，衰减率会从 (0 * decay) 递增到 (1 * decay)
            d = self.decay(self.updates)

            model_std = None
            if is_parallel(model):
                model_std = model.module.state_dict()
            else:
                model_std = model.state_dict()

            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    # new_v = d * v + (1.0 - d) * model_std[k].detach()
                    new_v = model_std[k].lerp(v, d)
                    v.copy_(new_v)


class PPdetModelEMA(object):
    """
    ppdet中的EMA
    Exponential Weighted Average for Deep Neutal Networks
    Args:
        model (nn.Layer): Detector of model.
        decay (int):  The decay used for updating ema parameter.
            Ema's parameter are updated with the formula:
           `ema_param = decay * ema_param + (1 - decay) * cur_param`.
            Defaults is 0.9998.
        ema_decay_type (str): type in ['threshold', 'normal', 'exponential'],
            'threshold' as default.
        cycle_epoch (int): The epoch of interval to reset ema_param and
            step. Defaults is -1, which means not reset. Its function is to
            add a regular effect to ema, which is set according to experience
            and is effective when the total training epoch is large.
    """

    def __init__(self,
                 model,
                 decay=0.9998,
                 ema_decay_type='threshold',
                 cycle_epoch=-1):
        self.step = 0
        self.epoch = 0
        self.decay = decay
        self.state_dict = dict()
        model_real = model.module if is_parallel(model) else model
        for k, v in model_real.state_dict().items():   # bn层的均值、方差也会参与ema
            if '.num_batches_tracked' in k:
                continue
            if k.startswith('teacher_model.'):
                logger.info("skip teacher weight '%s'" % k)
                continue
            self.state_dict[k] = torch.zeros_like(v)
            self.state_dict[k].requires_grad_(False)
        self.ema_decay_type = ema_decay_type
        self.cycle_epoch = cycle_epoch

        self._model_state = {
            k: weakref.ref(p)
            for k, p in model_real.state_dict().items()
        }

    def reset(self):
        self.step = 0
        self.epoch = 0
        for k, v in self.state_dict.items():
            self.state_dict[k] = torch.zeros_like(v)
            self.state_dict[k].requires_grad_(False)

    def resume(self, state_dict, step=0):
        for k, v in state_dict.items():
            if k in self.state_dict:
                self.state_dict[k].copy_(v)
        self.step = step

    def update(self, model=None):
        with torch.no_grad():
            if self.ema_decay_type == 'threshold':
                decay = min(self.decay, (1 + self.step) / (10 + self.step))
            elif self.ema_decay_type == 'exponential':
                decay = self.decay * (1 - math.exp(-(self.step + 1) / 2000))
            else:
                decay = self.decay
            self._decay = decay

            if model is not None:
                model_real = model.module if is_parallel(model) else model
                model_dict = model_real.state_dict()
            else:
                model_dict = {k: p() for k, p in self._model_state.items()}
                assert all(
                    [v is not None for _, v in model_dict.items()]), 'python gc.'

            for k, v in self.state_dict.items():
                v = decay * v + (1 - decay) * model_dict[k]
                v.requires_grad_(False)
                self.state_dict[k].copy_(v)
            self.step += 1

    def apply(self):
        if self.step == 0:
            return self.state_dict
        state_dict = dict()
        for k, v in self.state_dict.items():
            if self.ema_decay_type != 'exponential':
                new_v = v / (1 - self._decay**self.step)
                v.copy_(new_v)
            v.requires_grad_(False)
            state_dict[k] = v
        self.epoch += 1
        if self.cycle_epoch > 0 and self.epoch == self.cycle_epoch:
            self.reset()

        return state_dict

