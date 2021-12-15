#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch


class DataPrefetcher:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = DataPrefetcher._record_stream_for_image
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target, _, _ = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.input_cuda()
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            self.record_stream(input)
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target

    def _input_cuda_for_image(self):
        self.next_input = self.next_input.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())


class PPYOLODataPrefetcher:
    """
    PPYOLODataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = PPYOLODataPrefetcher._record_stream_for_image
        self.preload()

    def preload(self):
        try:
            self.next_input, self.gt_bbox, self.gt_score, self.gt_class, self.target0, self.target1, self.target2 = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.gt_bbox = None
            self.gt_score = None
            self.gt_class = None
            self.target0 = None
            self.target1 = None
            self.target2 = None
            return

        with torch.cuda.stream(self.stream):
            self.input_cuda()
            self.gt_bbox = self.gt_bbox.cuda(non_blocking=True)
            self.gt_score = self.gt_score.cuda(non_blocking=True)
            self.gt_class = self.gt_class.cuda(non_blocking=True)
            self.target0 = self.target0.cuda(non_blocking=True)
            self.target1 = self.target1.cuda(non_blocking=True)
            self.target2 = self.target2.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        gt_bbox = self.gt_bbox
        gt_score = self.gt_score
        gt_class = self.gt_class
        target0 = self.target0
        target1 = self.target1
        target2 = self.target2
        if input is not None:
            self.record_stream(input)
        if gt_bbox is not None:
            gt_bbox.record_stream(torch.cuda.current_stream())
        if gt_score is not None:
            gt_score.record_stream(torch.cuda.current_stream())
        if gt_class is not None:
            gt_class.record_stream(torch.cuda.current_stream())
        if target0 is not None:
            target0.record_stream(torch.cuda.current_stream())
        if target1 is not None:
            target1.record_stream(torch.cuda.current_stream())
        if target2 is not None:
            target2.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, gt_bbox, gt_score, gt_class, target0, target1, target2

    def _input_cuda_for_image(self):
        self.next_input = self.next_input.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())
