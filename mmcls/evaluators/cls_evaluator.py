#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import contextlib
import io
import itertools
import json
import tempfile
import time
import numpy as np
from loguru import logger
from tabulate import tabulate
from tqdm import tqdm
import pycocotools.mask as maskUtils
import torch

from mmdet.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
)


class CLSEvaluator:

    def __init__(
        self, dataloader, num_classes, archi_name=''
    ):
        self.dataloader = dataloader
        self.num_classes = num_classes
        self.archi_name = archi_name

    def evaluate_basecls(
        self,
        model,
        distributed=False,
        half=False,
    ):
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        data_list = []
        progress_bar = iter if is_main_process() else iter

        steps = len(self.dataloader)
        print_interval = max(steps // 5, 1) + 1
        num_imgs = self.dataloader.dataset.num_record

        eval_start = time.time()
        for cur_iter, (pimages, targets) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                pimages = pimages.type(tensor_type)     # [N, 3, 256, 256]
                targets = targets.to(pimages.device).to(torch.int64)  # [N, 1]
                preds = model(pimages)
                _, pred_labels = torch.topk(preds, k=1, dim=1, largest=True, sorted=True)
                # correct2 = (targets == pred_labels)
                correct = (targets == pred_labels).sum()
                if cur_iter % print_interval == 0:
                    progress_str = "Eval iter: {}/{}".format(cur_iter + 1, steps)
                    logger.info(progress_str)

            pred_data = {
                "correct": int(correct),
            }
            data_list.append(pred_data)
        cost = time.time() - eval_start
        logger.info('Eval time: %.1f s;  Speed: %.1f FPS.'%(cost, (num_imgs / cost)))

        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))

        eval_results = self.evaluate_prediction(data_list, num_imgs)
        synchronize()
        return eval_results

    def evaluate_prediction(self, data_dict, num_imgs):
        if not is_main_process():
            return 0., 0.

        logger.info("Evaluate in main process...")

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            correct = 0
            for data_ in data_dict:
                correct += data_['correct']
            top1_acc = correct / num_imgs
            top5_acc = 0.
            return top1_acc, top5_acc
        else:
            return 0., 0.

