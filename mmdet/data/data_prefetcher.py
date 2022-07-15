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
    xxxDataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader, n_layers):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.n_layers = n_layers
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = PPYOLODataPrefetcher._record_stream_for_image
        self.preload()

    def preload(self):
        try:
            if self.n_layers == 3:
                self.next_input, self.gt_bbox, self.target0, self.target1, self.target2, self.im_ids = next(self.loader)
            elif self.n_layers == 2:
                self.next_input, self.gt_bbox, self.target0, self.target1, self.im_ids = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.gt_bbox = None
            self.target0 = None
            self.target1 = None
            self.target2 = None
            self.im_ids = None
            return

        with torch.cuda.stream(self.stream):
            self.input_cuda()
            self.gt_bbox = self.gt_bbox.cuda(non_blocking=True)
            self.target0 = self.target0.cuda(non_blocking=True)
            self.target1 = self.target1.cuda(non_blocking=True)
            self.im_ids = self.im_ids.cuda(non_blocking=True)
            if self.n_layers == 3:
                self.target2 = self.target2.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        gt_bbox = self.gt_bbox
        target0 = self.target0
        target1 = self.target1
        im_ids = self.im_ids
        if self.n_layers == 3:
            target2 = self.target2
        if input is not None:
            self.record_stream(input)
        if gt_bbox is not None:
            gt_bbox.record_stream(torch.cuda.current_stream())
        if target0 is not None:
            target0.record_stream(torch.cuda.current_stream())
        if target1 is not None:
            target1.record_stream(torch.cuda.current_stream())
        if im_ids is not None:
            im_ids.record_stream(torch.cuda.current_stream())
        if self.n_layers == 3:
            if target2 is not None:
                target2.record_stream(torch.cuda.current_stream())
        self.preload()
        if self.n_layers == 3:
            return input, gt_bbox, target0, target1, target2, im_ids
        return input, gt_bbox, target0, target1, im_ids

    def _input_cuda_for_image(self):
        self.next_input = self.next_input.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())


class SOLODataPrefetcher:
    """
    xxxDataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader, n_layers):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.n_layers = n_layers
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = SOLODataPrefetcher._record_stream_for_image
        self.preload()

    def preload(self):
        try:
            self.next_input, self.gt_bbox, self.target0, self.target1, self.target2, self.im_ids = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.gt_bbox = None
            self.target0 = None
            self.target1 = None
            self.target2 = None
            self.im_ids = None
            return

        with torch.cuda.stream(self.stream):
            self.input_cuda()
            self.gt_bbox = self.gt_bbox.cuda(non_blocking=True)
            self.target0 = self.target0.cuda(non_blocking=True)
            self.target1 = self.target1.cuda(non_blocking=True)
            self.im_ids = self.im_ids.cuda(non_blocking=True)
            self.target2 = self.target2.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        gt_bbox = self.gt_bbox
        target0 = self.target0
        target1 = self.target1
        im_ids = self.im_ids
        target2 = self.target2
        if input is not None:
            self.record_stream(input)
        if gt_bbox is not None:
            gt_bbox.record_stream(torch.cuda.current_stream())
        if target0 is not None:
            target0.record_stream(torch.cuda.current_stream())
        if target1 is not None:
            target1.record_stream(torch.cuda.current_stream())
        if im_ids is not None:
            im_ids.record_stream(torch.cuda.current_stream())
        if target2 is not None:
            target2.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, gt_bbox, target0, target1, target2, im_ids

    def _input_cuda_for_image(self):
        self.next_input = self.next_input.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())


class PPYOLOEDataPrefetcher:
    """
    xxxDataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader, n_layers):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.n_layers = n_layers
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = PPYOLOEDataPrefetcher._record_stream_for_image
        self.preload()

    def preload(self):
        try:
            self.next_input, self.gt_class, self.gt_bbox, self.pad_gt_mask, self.im_ids = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.gt_class = None
            self.gt_bbox = None
            self.pad_gt_mask = None
            self.im_ids = None
            return

        with torch.cuda.stream(self.stream):
            self.input_cuda()
            self.gt_class = self.gt_class.cuda(non_blocking=True)
            self.gt_bbox = self.gt_bbox.cuda(non_blocking=True)
            self.pad_gt_mask = self.pad_gt_mask.cuda(non_blocking=True)
            self.im_ids = self.im_ids.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        gt_class = self.gt_class
        gt_bbox = self.gt_bbox
        pad_gt_mask = self.pad_gt_mask
        im_ids = self.im_ids
        if input is not None:
            self.record_stream(input)
        if gt_class is not None:
            gt_class.record_stream(torch.cuda.current_stream())
        if gt_bbox is not None:
            gt_bbox.record_stream(torch.cuda.current_stream())
        if pad_gt_mask is not None:
            pad_gt_mask.record_stream(torch.cuda.current_stream())
        if im_ids is not None:
            im_ids.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, gt_class, gt_bbox, pad_gt_mask, im_ids

    def _input_cuda_for_image(self):
        self.next_input = self.next_input.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())


class FCOSDataPrefetcher:
    """
    xxxDataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader, n_layers):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.n_layers = n_layers
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = FCOSDataPrefetcher._record_stream_for_image
        self.preload()

    def preload(self):
        try:
            if self.n_layers == 5:
                self.next_input, self.labels0, self.reg_target0, self.centerness0, self.labels1, self.reg_target1, self.centerness1, self.labels2, self.reg_target2, self.centerness2, self.labels3, self.reg_target3, self.centerness3, self.labels4, self.reg_target4, self.centerness4 = next(self.loader)
            elif self.n_layers == 3:
                self.next_input, self.labels0, self.reg_target0, self.centerness0, self.labels1, self.reg_target1, self.centerness1, self.labels2, self.reg_target2, self.centerness2 = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.labels0 = None
            self.reg_target0 = None
            self.centerness0 = None
            self.labels1 = None
            self.reg_target1 = None
            self.centerness1 = None
            self.labels2 = None
            self.reg_target2 = None
            self.centerness2 = None
            self.labels3 = None
            self.reg_target3 = None
            self.centerness3 = None
            self.labels4 = None
            self.reg_target4 = None
            self.centerness4 = None
            return

        with torch.cuda.stream(self.stream):
            self.input_cuda()
            self.labels0 = self.labels0.cuda(non_blocking=True)
            self.reg_target0 = self.reg_target0.cuda(non_blocking=True)
            self.centerness0 = self.centerness0.cuda(non_blocking=True)
            self.labels1 = self.labels1.cuda(non_blocking=True)
            self.reg_target1 = self.reg_target1.cuda(non_blocking=True)
            self.centerness1 = self.centerness1.cuda(non_blocking=True)
            self.labels2 = self.labels2.cuda(non_blocking=True)
            self.reg_target2 = self.reg_target2.cuda(non_blocking=True)
            self.centerness2 = self.centerness2.cuda(non_blocking=True)
            if self.n_layers == 5:
                self.labels3 = self.labels3.cuda(non_blocking=True)
                self.reg_target3 = self.reg_target3.cuda(non_blocking=True)
                self.centerness3 = self.centerness3.cuda(non_blocking=True)
                self.labels4 = self.labels4.cuda(non_blocking=True)
                self.reg_target4 = self.reg_target4.cuda(non_blocking=True)
                self.centerness4 = self.centerness4.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        labels0 = self.labels0
        reg_target0 = self.reg_target0
        centerness0 = self.centerness0
        labels1 = self.labels1
        reg_target1 = self.reg_target1
        centerness1 = self.centerness1
        labels2 = self.labels2
        reg_target2 = self.reg_target2
        centerness2 = self.centerness2
        if self.n_layers == 5:
            labels3 = self.labels3
            reg_target3 = self.reg_target3
            centerness3 = self.centerness3
            labels4 = self.labels4
            reg_target4 = self.reg_target4
            centerness4 = self.centerness4

        if input is not None:
            self.record_stream(input)
        if labels0 is not None:
            labels0.record_stream(torch.cuda.current_stream())
        if reg_target0 is not None:
            reg_target0.record_stream(torch.cuda.current_stream())
        if centerness0 is not None:
            centerness0.record_stream(torch.cuda.current_stream())

        if labels1 is not None:
            labels1.record_stream(torch.cuda.current_stream())
        if reg_target1 is not None:
            reg_target1.record_stream(torch.cuda.current_stream())
        if centerness1 is not None:
            centerness1.record_stream(torch.cuda.current_stream())

        if labels2 is not None:
            labels2.record_stream(torch.cuda.current_stream())
        if reg_target2 is not None:
            reg_target2.record_stream(torch.cuda.current_stream())
        if centerness2 is not None:
            centerness2.record_stream(torch.cuda.current_stream())
        if self.n_layers == 5:
            if labels3 is not None:
                labels3.record_stream(torch.cuda.current_stream())
            if reg_target3 is not None:
                reg_target3.record_stream(torch.cuda.current_stream())
            if centerness3 is not None:
                centerness3.record_stream(torch.cuda.current_stream())

            if labels4 is not None:
                labels4.record_stream(torch.cuda.current_stream())
            if reg_target4 is not None:
                reg_target4.record_stream(torch.cuda.current_stream())
            if centerness4 is not None:
                centerness4.record_stream(torch.cuda.current_stream())
        self.preload()
        if self.n_layers == 5:
            return input, labels0, reg_target0, centerness0, labels1, reg_target1, centerness1, labels2, reg_target2, centerness2, labels3, reg_target3, centerness3, labels4, reg_target4, centerness4
        return input, labels0, reg_target0, centerness0, labels1, reg_target1, centerness1, labels2, reg_target2, centerness2

    def _input_cuda_for_image(self):
        self.next_input = self.next_input.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())
