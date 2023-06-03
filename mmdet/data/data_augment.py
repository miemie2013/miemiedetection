#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""

import math
import random

import torch
import torch.nn.functional as F
import cv2
import copy
import numpy as np
from loguru import logger
from numbers import Number, Integral

from mmdet.utils import xyxy2cxcywh


def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge(
        (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
    ).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2):
    # box1(4,n), box2(4,n)
    # Compute candidate boxes which include follwing 5 things:
    # box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (
        (w2 > wh_thr)
        & (h2 > wh_thr)
        & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr)
        & (ar < ar_thr)
    )  # candidates


def random_perspective(
    img,
    targets=(),
    degrees=10,
    translate=0.1,
    scale=0.1,
    shear=10,
    perspective=0.0,
    border=(0, 0),
):
    # targets = [cls, xyxy]
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(scale[0], scale[1])
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * width
    )  # x translation (pixels)
    T[1, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * height
    )  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ C  # order of operations (right to left) is IMPORTANT

    ###########################
    # For Aug out of Mosaic
    # s = 1.
    # M = np.eye(3)
    ###########################

    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(
                img, M, dsize=(width, height), borderValue=(114, 114, 114)
            )
        else:  # affine
            img = cv2.warpAffine(
                img, M[:2], dsize=(width, height), borderValue=(114, 114, 114)
            )

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
            n * 4, 2
        )  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, :4].T * s, box2=xy.T)
        targets = targets[i]
        targets[:, :4] = xy[i]

    return img, targets


def _mirror(image, boxes, prob=0.5):
    _, width, _ = image.shape
    if random.random() < prob:
        image = image[:, ::-1]
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes


def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def preproc_ppyolo(img, input_size):
    im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # resizeImage
    im_shape = im.shape
    selected_size = input_size[0]
    im_scale_x = float(selected_size) / float(im_shape[1])
    im_scale_y = float(selected_size) / float(im_shape[0])
    im = cv2.resize(
        im,
        None,
        None,
        fx=im_scale_x,
        fy=im_scale_y,
        interpolation=2)

    # normalizeImage
    im = im.astype(np.float32, copy=False)
    mean = np.array([0.485, 0.456, 0.406])[np.newaxis, np.newaxis, :]
    std = np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis, :]
    im = im / 255.0
    im -= mean
    im /= std

    # permute
    im = np.swapaxes(im, 1, 2)
    im = np.swapaxes(im, 1, 0)

    pimage = np.expand_dims(im, axis=0)
    im_size = np.array([[im_shape[0], im_shape[1]]]).astype(np.float32)
    return pimage, im_size


class TrainTransform:
    def __init__(self, max_labels=50, flip_prob=0.5, hsv_prob=1.0):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob
        self.init_bbox = [0.0, -9999.0, -9999.0, 10.0, 10.0]

    def __call__(self, image, targets, input_dim):
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, 5), dtype=np.float32)
            image, r_o = preproc(image, input_dim)
            return image, targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4]
        labels_o = targets_o[:, 4]
        # bbox_o: [xyxy] to [c_x,c_y,w,h]
        boxes_o = xyxy2cxcywh(boxes_o)

        if random.random() < self.hsv_prob:
            augment_hsv(image)
        image_t, boxes = _mirror(image, boxes, self.flip_prob)
        height, width, _ = image_t.shape
        image_t, r_ = preproc(image_t, input_dim)
        # boxes [xyxy] 2 [cx,cy,w,h]
        boxes = xyxy2cxcywh(boxes)
        boxes *= r_

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]

        if len(boxes_t) == 0:
            image_t, r_o = preproc(image_o, input_dim)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o

        labels_t = np.expand_dims(labels_t, 1)

        targets_t = np.hstack((labels_t, boxes_t))
        # padded_labels = np.zeros((self.max_labels, 5))
        # 一定要用self.init_bbox初始化填充的假gt
        padded_labels = np.ones((self.max_labels, 5), dtype=np.float64) * self.init_bbox
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
            : self.max_labels
        ]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        return image_t, padded_labels


class ValTransform:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, swap=(2, 0, 1), legacy=False):
        self.swap = swap
        self.legacy = legacy

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size):
        img, _ = preproc(img, input_size, self.swap)
        if self.legacy:
            img = img[::-1, :, :].copy()
            img /= 255.0
            img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        return img, np.zeros((1, 5))


class PPYOLOValTransform:
    def __init__(self, context, to_rgb, resizeImage, normalizeImage, permute):
        self.context = context
        self.to_rgb = to_rgb
        self.resizeImage = resizeImage
        self.normalizeImage = normalizeImage
        self.permute = permute

    def __call__(self, img):
        if self.to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        context = self.context
        sample = {}
        sample['image'] = img
        sample['h'] = img.shape[0]
        sample['w'] = img.shape[1]

        sample = self.resizeImage(sample, context)
        sample = self.normalizeImage(sample, context)
        sample = self.permute(sample, context)

        pimage = np.expand_dims(sample['image'], axis=0)
        im_size = np.array([[img.shape[0], img.shape[1]]]).astype(np.int32)
        return pimage, im_size


class SOLOValTransform:
    def __init__(self, context, to_rgb, resizeImage, normalizeImage, permute, padBatch):
        self.context = context
        self.to_rgb = to_rgb
        self.resizeImage = resizeImage
        self.normalizeImage = normalizeImage
        self.permute = permute
        self.padBatch = padBatch

    def __call__(self, img):
        if self.to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        context = self.context
        sample = {}
        sample['image'] = img
        sample['h'] = img.shape[0]
        sample['w'] = img.shape[1]

        sample = self.normalizeImage(sample, context)
        sample = self.resizeImage(sample, context)
        sample = self.permute(sample, context)
        samples = self.padBatch([sample, ], context)
        sample = samples[0]

        pimage = np.expand_dims(sample['image'], axis=0)
        im_size = np.array([[sample['im_info'][0], sample['im_info'][1]]]).astype(np.int32)
        ori_shape = np.array([[img.shape[0], img.shape[1]]]).astype(np.int32)
        return pimage, im_size, ori_shape


class PPYOLOEValTransform:
    def __init__(self, context, to_rgb, resizeImage, normalizeImage, permute):
        self.context = context
        self.to_rgb = to_rgb
        self.resizeImage = resizeImage
        self.normalizeImage = normalizeImage
        self.permute = permute

    def __call__(self, img):
        if self.to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        context = self.context
        sample = {}
        sample['image'] = img
        sample['h'] = img.shape[0]
        sample['w'] = img.shape[1]

        sample = self.resizeImage(sample, context)
        sample = self.normalizeImage(sample, context)
        sample = self.permute(sample, context)

        pimage = np.expand_dims(sample['image'], axis=0)
        scale_factor = np.array([[sample['scale_factor'][1], sample['scale_factor'][0]]]).astype(np.float32)
        return pimage, scale_factor

class RTDETRValTransform:
    def __init__(self, context, to_rgb, resizeImage, normalizeImage, permute):
        self.context = context
        self.to_rgb = to_rgb
        self.resizeImage = resizeImage
        self.normalizeImage = normalizeImage
        self.permute = permute

    def __call__(self, img):
        if self.to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        context = self.context
        sample = {}
        sample['image'] = img
        sample['h'] = img.shape[0]
        sample['w'] = img.shape[1]

        sample = self.resizeImage(sample, context)
        sample = self.normalizeImage(sample, context)
        sample = self.permute(sample, context)

        pimage = np.expand_dims(sample['image'], axis=0)
        # scale_factor = np.array([[sample['scale_factor'][1], sample['scale_factor'][0]]]).astype(np.float32)
        # im_shape = np.array([[sample['w'], sample['h']]]).astype(np.float32)
        scale_factor = np.array([[sample['scale_factor'][1], sample['scale_factor'][0]]]).astype(np.float32)
        im_shape = np.array([[pimage.shape[2], pimage.shape[3]]]).astype(np.float32)
        return pimage, scale_factor, im_shape


class FCOSValTransform:
    def __init__(self, context, to_rgb, normalizeImage, resizeImage, permute, padBatch):
        self.context = context
        self.to_rgb = to_rgb
        self.normalizeImage = normalizeImage
        self.resizeImage = resizeImage
        self.permute = permute
        self.padBatch = padBatch

    def __call__(self, img):
        if self.to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        context = self.context
        sample = {}
        sample['image'] = img
        sample['h'] = img.shape[0]
        sample['w'] = img.shape[1]

        sample = self.normalizeImage(sample, context)
        sample = self.resizeImage(sample, context)
        sample = self.permute(sample, context)

        # batch_transforms
        samples = self.padBatch([sample], context)
        sample = samples[0]

        pimage = np.expand_dims(sample['image'], axis=0)
        im_scale = np.expand_dims(sample['im_info'][2:3], axis=0)
        return pimage, im_scale



# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-06-05 15:35:27
#   Description : 数据增强。凑不要脸地搬运了百度PaddleDetection的部分代码。
#
# ================================================================
import uuid
from scipy import ndimage
from PIL import Image, ImageEnhance, ImageDraw

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence


class BboxError(ValueError):
    pass


class ImageError(ValueError):
    pass


def is_poly(segm):
    assert isinstance(segm, (list, dict)), \
        "Invalid segm type: {}".format(type(segm))
    return isinstance(segm, list)


class BaseOperator(object):
    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        self._id = name + '_' + str(uuid.uuid4())[-6:]

    def __call__(self, sample, context=None):
        """ Process a sample.
        Args:
            sample (dict): a dict of sample, eg: {'image':xx, 'label': xxx}
            context (dict): info about this sample processing
        Returns:
            result (dict): a processed sample
        """
        return sample

    def __str__(self):
        return str(self._id)


class DecodeImage(BaseOperator):
    def __init__(self, to_rgb=True, with_mixup=False, with_cutmix=False, with_mosaic=False):
        """ Transform the image data to numpy format.
        Args:
            to_rgb (bool): whether to convert BGR to RGB
            with_mixup (bool): whether or not to mixup image and gt_bbbox/gt_score
            with_cutmix (bool): whether or not to cutmix image and gt_bbbox/gt_score
            with_mosaic (bool): whether or not to mosaic image and gt_bbbox/gt_score
        """

        super(DecodeImage, self).__init__()
        self.to_rgb = to_rgb
        self.with_mixup = with_mixup
        self.with_cutmix = with_cutmix
        self.with_mosaic = with_mosaic
        if not isinstance(self.to_rgb, bool):
            raise TypeError("{}: input type is invalid.".format(self))
        if not isinstance(self.with_mixup, bool):
            raise TypeError("{}: input type is invalid.".format(self))

    def __call__(self, sample, context=None):
        """ load image if 'im_file' field is not empty but 'image' is"""
        if 'image' not in sample:
            with open(sample['im_file'], 'rb') as f:
                sample['image'] = f.read()

        im = sample['image']
        data = np.frombuffer(im, dtype='uint8')
        im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode

        if self.to_rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        sample['image'] = im

        if 'h' not in sample:
            sample['h'] = im.shape[0]
        elif sample['h'] != im.shape[0]:
            logger.warning(
                "The actual image height: {} is not equal to the "
                "height: {} in annotation, and update sample['h'] by actual "
                "image height.".format(im.shape[0], sample['h']))
            sample['h'] = im.shape[0]
        if 'w' not in sample:
            sample['w'] = im.shape[1]
        elif sample['w'] != im.shape[1]:
            logger.warning(
                "The actual image width: {} is not equal to the "
                "width: {} in annotation, and update sample['w'] by actual "
                "image width.".format(im.shape[1], sample['w']))
            sample['w'] = im.shape[1]

        # make default im_info with [h, w, 1]
        sample['im_info'] = np.array(
            [im.shape[0], im.shape[1], 1.], dtype=np.float32)

        # decode mixup image
        if self.with_mixup and 'mixup' in sample:
            self.__call__(sample['mixup'], context)

        # decode cutmix image
        if self.with_cutmix and 'cutmix' in sample:
            self.__call__(sample['cutmix'], context)

        # decode mosaic image
        if self.with_mosaic and 'mosaic1' in sample:
            self.__call__(sample['mosaic1'], context)
            self.__call__(sample['mosaic2'], context)
            self.__call__(sample['mosaic3'], context)

        # decode semantic label
        if 'semantic' in sample.keys() and sample['semantic'] is not None:
            sem_file = sample['semantic']
            sem = cv2.imread(sem_file, cv2.IMREAD_GRAYSCALE)
            sample['semantic'] = sem.astype('int32')

        return sample

class Decode(BaseOperator):
    def __init__(self):
        """ Transform the image data to numpy format following the rgb format
        """
        super(Decode, self).__init__()

    def __call__(self, sample, context=None):
        """ load image if 'im_file' field is not empty but 'image' is"""
        if 'image' not in sample:
            with open(sample['im_file'], 'rb') as f:
                sample['image'] = f.read()
            sample.pop('im_file')

        im = sample['image']
        data = np.frombuffer(im, dtype='uint8')
        im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
        if 'keep_ori_im' in sample and sample['keep_ori_im']:
            sample['ori_image'] = im
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        sample['image'] = im
        if 'h' not in sample:
            sample['h'] = im.shape[0]
        elif sample['h'] != im.shape[0]:
            logger.warning(
                "The actual image height: {} is not equal to the "
                "height: {} in annotation, and update sample['h'] by actual "
                "image height.".format(im.shape[0], sample['h']))
            sample['h'] = im.shape[0]
        if 'w' not in sample:
            sample['w'] = im.shape[1]
        elif sample['w'] != im.shape[1]:
            logger.warning(
                "The actual image width: {} is not equal to the "
                "width: {} in annotation, and update sample['w'] by actual "
                "image width.".format(im.shape[1], sample['w']))
            sample['w'] = im.shape[1]

        sample['im_shape'] = np.array(im.shape[:2], dtype=np.float32)
        sample['scale_factor'] = np.array([1., 1.], dtype=np.float32)
        return sample


class MixupImage(BaseOperator):
    def __init__(self, alpha=1.5, beta=1.5):
        """ Mixup image and gt_bbbox/gt_score
        Args:
            alpha (float): alpha parameter of beta distribute
            beta (float): beta parameter of beta distribute
        """
        super(MixupImage, self).__init__()
        self.alpha = alpha
        self.beta = beta
        if self.alpha <= 0.0:
            raise ValueError("alpha should be positive in {}".format(self))
        if self.beta <= 0.0:
            raise ValueError("beta should be positive in {}".format(self))

    def _mixup_img(self, img1, img2, factor):
        h = max(img1.shape[0], img2.shape[0])
        w = max(img1.shape[1], img2.shape[1])
        img = np.zeros((h, w, img1.shape[2]), 'float32')
        img[:img1.shape[0], :img1.shape[1], :] = \
            img1.astype('float32') * factor
        img[:img2.shape[0], :img2.shape[1], :] += \
            img2.astype('float32') * (1.0 - factor)
        return img.astype('uint8')

    def __call__(self, sample, context=None):
        if 'mixup' not in sample:
            return sample
        factor = np.random.beta(self.alpha, self.beta)
        factor = max(0.0, min(1.0, factor))
        if factor >= 1.0:
            sample.pop('mixup')
            return sample
        if factor <= 0.0:
            return sample['mixup']
        im = self._mixup_img(sample['image'], sample['mixup']['image'], factor)
        gt_bbox1 = sample['gt_bbox'].reshape((-1, 4))
        gt_bbox2 = sample['mixup']['gt_bbox'].reshape((-1, 4))
        gt_bbox = np.concatenate((gt_bbox1, gt_bbox2), axis=0)
        gt_class1 = sample['gt_class']
        gt_class2 = sample['mixup']['gt_class']
        gt_class = np.concatenate((gt_class1, gt_class2), axis=0)

        gt_score1 = sample['gt_score']
        gt_score2 = sample['mixup']['gt_score']
        gt_score = np.concatenate(
            (gt_score1 * factor, gt_score2 * (1. - factor)), axis=0)

        is_crowd1 = sample['is_crowd']
        is_crowd2 = sample['mixup']['is_crowd']
        is_crowd = np.concatenate((is_crowd1, is_crowd2), axis=0)

        sample['image'] = im
        sample['gt_bbox'] = gt_bbox
        sample['gt_score'] = gt_score
        sample['gt_class'] = gt_class
        sample['is_crowd'] = is_crowd
        sample['h'] = im.shape[0]
        sample['w'] = im.shape[1]
        sample.pop('mixup')
        return sample


class CutmixImage(BaseOperator):
    def __init__(self, alpha=1.5, beta=1.5):
        """
        CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features, see https://https://arxiv.org/abs/1905.04899
        Cutmix image and gt_bbbox/gt_score
        Args:
             alpha (float): alpha parameter of beta distribute
             beta (float): beta parameter of beta distribute
        """
        super(CutmixImage, self).__init__()
        self.alpha = alpha
        self.beta = beta
        if self.alpha <= 0.0:
            raise ValueError("alpha shold be positive in {}".format(self))
        if self.beta <= 0.0:
            raise ValueError("beta shold be positive in {}".format(self))

    def _rand_bbox(self, img1, img2, factor):
        """ _rand_bbox """
        h = max(img1.shape[0], img2.shape[0])
        w = max(img1.shape[1], img2.shape[1])
        cut_rat = np.sqrt(1. - factor)

        cut_w = np.int(w * cut_rat)
        cut_h = np.int(h * cut_rat)

        # uniform
        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        img_1 = np.zeros((h, w, img1.shape[2]), 'float32')
        img_1[:img1.shape[0], :img1.shape[1], :] = \
            img1.astype('float32')
        img_2 = np.zeros((h, w, img2.shape[2]), 'float32')
        img_2[:img2.shape[0], :img2.shape[1], :] = \
            img2.astype('float32')
        img_1[bby1:bby2, bbx1:bbx2, :] = img2[bby1:bby2, bbx1:bbx2, :]
        return img_1

    def __call__(self, sample, context=None):
        if 'cutmix' not in sample:
            return sample
        factor = np.random.beta(self.alpha, self.beta)
        factor = max(0.0, min(1.0, factor))
        if factor >= 1.0:
            sample.pop('cutmix')
            return sample
        if factor <= 0.0:
            return sample['cutmix']
        img1 = sample['image']
        img2 = sample['cutmix']['image']
        img = self._rand_bbox(img1, img2, factor)
        gt_bbox1 = sample['gt_bbox'].reshape((-1, 4))
        gt_bbox2 = sample['cutmix']['gt_bbox'].reshape((-1, 4))
        gt_bbox = np.concatenate((gt_bbox1, gt_bbox2), axis=0)
        gt_class1 = sample['gt_class']
        gt_class2 = sample['cutmix']['gt_class']
        gt_class = np.concatenate((gt_class1, gt_class2), axis=0)
        gt_score1 = sample['gt_score']
        gt_score2 = sample['cutmix']['gt_score']
        gt_score = np.concatenate(
            (gt_score1 * factor, gt_score2 * (1. - factor)), axis=0)
        sample['image'] = img
        sample['gt_bbox'] = gt_bbox
        sample['gt_score'] = gt_score
        sample['gt_class'] = gt_class
        sample['h'] = img.shape[0]
        sample['w'] = img.shape[1]
        sample.pop('cutmix')
        return sample


class MosaicImage(BaseOperator):
    def __init__(self, alpha=1.5, beta=1.5, thr=0.3):
        """
        CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features, see https://https://arxiv.org/abs/1905.04899
        Cutmix image and gt_bbbox/gt_score
        Args:
             alpha (float): alpha parameter of beta distribute
             beta (float): beta parameter of beta distribute
        """
        super(MosaicImage, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.thr = thr
        if self.alpha <= 0.0:
            raise ValueError("alpha shold be positive in {}".format(self))
        if self.beta <= 0.0:
            raise ValueError("beta shold be positive in {}".format(self))

    def _rand_bbox(self, img1, img2, img3, img4, factor_y, factor_x):
        """ _rand_bbox """
        h = max(img1.shape[0], img2.shape[0], img3.shape[0], img4.shape[0])
        w = max(img1.shape[1], img2.shape[1], img3.shape[1], img4.shape[1])
        scale = np.random.uniform(0, 1) * 0.5 + 1.0  # 取值范围[1.0, 1.5]
        h = int(h * scale)
        w = int(w * scale)

        cx = np.int(w * factor_x)
        cy = np.int(h * factor_y)
        return h, w, cx, cy

    def overlap(self, box1_x0, box1_y0, box1_x1, box1_y1, box2_x0, box2_y0, box2_x1, box2_y1):
        # 两个矩形的面积
        # box1_area = (box1_x1 - box1_x0) * (box1_y1 - box1_y0)
        box2_area = (box2_x1 - box2_x0) * (box2_y1 - box2_y0)

        # 相交矩形的左下角坐标、右上角坐标
        cx0 = max(box1_x0, box2_x0)
        cy0 = max(box1_y0, box2_y0)
        cx1 = min(box1_x1, box2_x1)
        cy1 = min(box1_y1, box2_y1)

        # 相交矩形的面积inter_area。
        inter_w = max(cx1 - cx0, 0.0)
        inter_h = max(cy1 - cy0, 0.0)
        inter_area = inter_w * inter_h
        _overlap = inter_area / (box2_area + 1e-9)
        return _overlap

    def __call__(self, sample, context=None):
        img1 = sample['image']
        img2 = sample['mosaic1']['image']
        img3 = sample['mosaic2']['image']
        img4 = sample['mosaic3']['image']
        factor_y = np.random.uniform(0, 1) * 0.5 + 0.25  # 取值范围[0.25, 0.75]
        factor_x = np.random.uniform(0, 1) * 0.5 + 0.25  # 取值范围[0.25, 0.75]
        # cv2.imwrite('aaaaaa1.jpg', cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
        # cv2.imwrite('aaaaaa2.jpg', cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
        # cv2.imwrite('aaaaaa3.jpg', cv2.cvtColor(img3, cv2.COLOR_RGB2BGR))
        # cv2.imwrite('aaaaaa4.jpg', cv2.cvtColor(img4, cv2.COLOR_RGB2BGR))
        h, w, cx, cy = self._rand_bbox(img1, img2, img3, img4, factor_y, factor_x)
        img = np.zeros((h, w, img1.shape[2]), 'float32')

        img1_box_xyxy = [0, 0, min(img1.shape[1], cx), min(img1.shape[0], cy)]
        img1_inner_xyxy = [0, 0, min(img1.shape[1], cx), min(img1.shape[0], cy)]

        img2_box_xyxy = [max(w - img2.shape[1], cx), 0, w, min(img2.shape[0], cy)]
        img2_inner_xyxy = [img2.shape[1] - (img2_box_xyxy[2] - img2_box_xyxy[0]), 0, img2.shape[1],
                           min(img2.shape[0], cy)]

        img3_box_xyxy = [0, max(h - img3.shape[0], cy), min(img3.shape[1], cx), h]
        img3_inner_xyxy = [0, img3.shape[0] - (img3_box_xyxy[3] - img3_box_xyxy[1]), min(img3.shape[1], cx),
                           img3.shape[0]]

        img4_box_xyxy = [max(w - img4.shape[1], cx), max(h - img4.shape[0], cy), w, h]
        img4_inner_xyxy = [img4.shape[1] - (img4_box_xyxy[2] - img4_box_xyxy[0]),
                           img4.shape[0] - (img4_box_xyxy[3] - img4_box_xyxy[1]), img4.shape[1], img4.shape[0]]

        img[img1_box_xyxy[1]:img1_box_xyxy[3], img1_box_xyxy[0]:img1_box_xyxy[2], :] = \
            img1.astype('float32')[img1_inner_xyxy[1]:img1_inner_xyxy[3], img1_inner_xyxy[0]:img1_inner_xyxy[2], :]
        img[img2_box_xyxy[1]:img2_box_xyxy[3], img2_box_xyxy[0]:img2_box_xyxy[2], :] = \
            img2.astype('float32')[img2_inner_xyxy[1]:img2_inner_xyxy[3], img2_inner_xyxy[0]:img2_inner_xyxy[2], :]
        img[img3_box_xyxy[1]:img3_box_xyxy[3], img3_box_xyxy[0]:img3_box_xyxy[2], :] = \
            img3.astype('float32')[img3_inner_xyxy[1]:img3_inner_xyxy[3], img3_inner_xyxy[0]:img3_inner_xyxy[2], :]
        img[img4_box_xyxy[1]:img4_box_xyxy[3], img4_box_xyxy[0]:img4_box_xyxy[2], :] = \
            img4.astype('float32')[img4_inner_xyxy[1]:img4_inner_xyxy[3], img4_inner_xyxy[0]:img4_inner_xyxy[2], :]

        # cv2.imwrite('aaaaaa5.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        gt_bbox1 = sample['gt_bbox'].reshape((-1, 4))
        gt_bbox2 = sample['mosaic1']['gt_bbox'].reshape((-1, 4))
        gt_bbox3 = sample['mosaic2']['gt_bbox'].reshape((-1, 4))
        gt_bbox4 = sample['mosaic3']['gt_bbox'].reshape((-1, 4))
        gt_class1 = sample['gt_class'].reshape((-1,))
        gt_class2 = sample['mosaic1']['gt_class'].reshape((-1,))
        gt_class3 = sample['mosaic2']['gt_class'].reshape((-1,))
        gt_class4 = sample['mosaic3']['gt_class'].reshape((-1,))
        gt_score1 = sample['gt_score'].reshape((-1,))
        gt_score2 = sample['mosaic1']['gt_score'].reshape((-1,))
        gt_score3 = sample['mosaic2']['gt_score'].reshape((-1,))
        gt_score4 = sample['mosaic3']['gt_score'].reshape((-1,))
        gt_is_crowd1 = sample['is_crowd'].reshape((-1,))
        gt_is_crowd2 = sample['mosaic1']['is_crowd'].reshape((-1,))
        gt_is_crowd3 = sample['mosaic2']['is_crowd'].reshape((-1,))
        gt_is_crowd4 = sample['mosaic3']['is_crowd'].reshape((-1,))
        # gt_bbox4222222 = np.copy(gt_bbox4)
        # gt_score4222222 = np.copy(gt_score4)
        # gt_class4222222 = np.copy(gt_class4)

        # img1
        for i, box in enumerate(gt_bbox1):
            ov = self.overlap(img1_box_xyxy[0], img1_box_xyxy[1], img1_box_xyxy[2], img1_box_xyxy[3],
                              box[0], box[1], box[2], box[3])
            if ov < self.thr:
                gt_score1[i] -= 99.0
            else:
                x0 = np.clip(box[0], img1_box_xyxy[0], img1_box_xyxy[2])
                y0 = np.clip(box[1], img1_box_xyxy[1], img1_box_xyxy[3])
                x1 = np.clip(box[2], img1_box_xyxy[0], img1_box_xyxy[2])
                y1 = np.clip(box[3], img1_box_xyxy[1], img1_box_xyxy[3])
                gt_bbox1[i, :] = np.array([x0, y0, x1, y1])
        keep = np.where(gt_score1 >= 0.0)
        gt_bbox1 = gt_bbox1[keep]  # [M, 4]
        gt_score1 = gt_score1[keep]  # [M, ]
        gt_class1 = gt_class1[keep]  # [M, ]
        gt_is_crowd1 = gt_is_crowd1[keep]  # [M, ]

        # img2
        for i, box in enumerate(gt_bbox2):
            offset_x = img2_box_xyxy[0]
            if img2.shape[1] >= w - cx:
                offset_x = w - img2.shape[1]
            box[0] += offset_x
            box[1] += 0
            box[2] += offset_x
            box[3] += 0
            ov = self.overlap(img2_box_xyxy[0], img2_box_xyxy[1], img2_box_xyxy[2], img2_box_xyxy[3],
                              box[0], box[1], box[2], box[3])
            if ov < self.thr:
                gt_score2[i] -= 99.0
            else:
                x0 = np.clip(box[0], img2_box_xyxy[0], img2_box_xyxy[2])
                y0 = np.clip(box[1], img2_box_xyxy[1], img2_box_xyxy[3])
                x1 = np.clip(box[2], img2_box_xyxy[0], img2_box_xyxy[2])
                y1 = np.clip(box[3], img2_box_xyxy[1], img2_box_xyxy[3])
                gt_bbox2[i, :] = np.array([x0, y0, x1, y1])
        keep = np.where(gt_score2 >= 0.0)
        gt_bbox2 = gt_bbox2[keep]  # [M, 4]
        gt_score2 = gt_score2[keep]  # [M, ]
        gt_class2 = gt_class2[keep]  # [M, ]
        gt_is_crowd2 = gt_is_crowd2[keep]  # [M, ]

        # img3
        for i, box in enumerate(gt_bbox3):
            offset_y = img3_box_xyxy[1]
            if img3.shape[0] >= h - cy:
                offset_y = h - img3.shape[0]
            box[0] += 0
            box[1] += offset_y
            box[2] += 0
            box[3] += offset_y
            ov = self.overlap(img3_box_xyxy[0], img3_box_xyxy[1], img3_box_xyxy[2], img3_box_xyxy[3],
                              box[0], box[1], box[2], box[3])
            if ov < self.thr:
                gt_score3[i] -= 99.0
            else:
                x0 = np.clip(box[0], img3_box_xyxy[0], img3_box_xyxy[2])
                y0 = np.clip(box[1], img3_box_xyxy[1], img3_box_xyxy[3])
                x1 = np.clip(box[2], img3_box_xyxy[0], img3_box_xyxy[2])
                y1 = np.clip(box[3], img3_box_xyxy[1], img3_box_xyxy[3])
                gt_bbox3[i, :] = np.array([x0, y0, x1, y1])
        keep = np.where(gt_score3 >= 0.0)
        gt_bbox3 = gt_bbox3[keep]  # [M, 4]
        gt_score3 = gt_score3[keep]  # [M, ]
        gt_class3 = gt_class3[keep]  # [M, ]
        gt_is_crowd3 = gt_is_crowd3[keep]  # [M, ]

        # img4
        for i, box in enumerate(gt_bbox4):
            offset_x = img4_box_xyxy[0]
            if img4.shape[1] >= w - cx:
                offset_x = w - img4.shape[1]
            offset_y = img4_box_xyxy[1]
            if img4.shape[0] >= h - cy:
                offset_y = h - img4.shape[0]
            box[0] += offset_x
            box[1] += offset_y
            box[2] += offset_x
            box[3] += offset_y
            ov = self.overlap(img4_box_xyxy[0], img4_box_xyxy[1], img4_box_xyxy[2], img4_box_xyxy[3],
                              box[0], box[1], box[2], box[3])
            if ov < self.thr:
                gt_score4[i] -= 99.0
            else:
                x0 = np.clip(box[0], img4_box_xyxy[0], img4_box_xyxy[2])
                y0 = np.clip(box[1], img4_box_xyxy[1], img4_box_xyxy[3])
                x1 = np.clip(box[2], img4_box_xyxy[0], img4_box_xyxy[2])
                y1 = np.clip(box[3], img4_box_xyxy[1], img4_box_xyxy[3])
                gt_bbox4[i, :] = np.array([x0, y0, x1, y1])
        keep = np.where(gt_score4 >= 0.0)
        gt_bbox4 = gt_bbox4[keep]  # [M, 4]
        gt_score4 = gt_score4[keep]  # [M, ]
        gt_class4 = gt_class4[keep]  # [M, ]
        gt_is_crowd4 = gt_is_crowd4[keep]  # [M, ]

        # gt_bbox4222222 = gt_bbox4222222[keep]     # [M, 4]
        # gt_score4222222 = gt_score4222222[keep]   # [M, ]
        # gt_class4222222 = gt_class4222222[keep]   # [M, ]

        gt_bbox = np.concatenate((gt_bbox1, gt_bbox2, gt_bbox3, gt_bbox4), axis=0)
        gt_class = np.concatenate((gt_class1, gt_class2, gt_class3, gt_class4), axis=0)
        gt_is_crowd = np.concatenate((gt_is_crowd1, gt_is_crowd2, gt_is_crowd3, gt_is_crowd4), axis=0)
        gt_score = np.concatenate((gt_score1, gt_score2, gt_score3, gt_score4), axis=0)
        # gt_score = np.concatenate((gt_score1 * factor, gt_score2 * (1. - factor)), axis=0)
        gt_class = np.reshape(gt_class, (-1, 1))
        gt_score = np.reshape(gt_score, (-1, 1))
        gt_is_crowd = np.reshape(gt_is_crowd, (-1, 1))
        sample['image'] = img
        sample['gt_bbox'] = gt_bbox
        sample['gt_score'] = gt_score
        sample['gt_class'] = gt_class
        sample['is_crowd'] = gt_is_crowd
        sample['h'] = img.shape[0]
        sample['w'] = img.shape[1]
        sample.pop('mosaic1')
        sample.pop('mosaic2')
        sample.pop('mosaic3')
        return sample


class YOLOXMosaicImage(BaseOperator):
    def __init__(self,
                 prob=0.5,
                 degrees=10.0,
                 translate=0.1,
                 scale=(0.1, 2),
                 shear=2.0,
                 perspective=0.0,
                 input_dim=(640, 640),
                 enable_mixup=True,
                 mixup_prob=1.0,
                 mixup_scale=(0.5, 1.5), ):
        """
        CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features, see https://https://arxiv.org/abs/1905.04899
        Cutmix image and gt_bbbox/gt_score
        Args:
             alpha (float): alpha parameter of beta distribute
             beta (float): beta parameter of beta distribute
        """
        super(YOLOXMosaicImage, self).__init__()
        self.prob = prob
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.input_dim = input_dim
        self.enable_mixup = enable_mixup
        self.mixup_prob = mixup_prob
        self.mixup_scale = mixup_scale

    def get_mosaic_coordinate(self, mosaic_image, mosaic_index, xc, yc, w, h, input_h, input_w):
        # TODO update doc
        # index0 to top left part of image
        if mosaic_index == 0:
            x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
            small_coord = w - (x2 - x1), h - (y2 - y1), w, h
        # index1 to top right part of image
        elif mosaic_index == 1:
            x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
            small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
        # index2 to bottom left part of image
        elif mosaic_index == 2:
            x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
            small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
        # index2 to bottom right part of image
        elif mosaic_index == 3:
            x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)  # noqa
            small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
        return (x1, y1, x2, y2), small_coord

    def random_perspective(
            self,
            img,
            targets=(),
            degrees=10,
            translate=0.1,
            scale=0.1,
            shear=10,
            perspective=0.0,
            border=(0, 0),
    ):
        # targets = [cls, xyxy]
        height = img.shape[0] + border[0] * 2  # shape(h,w,c)
        width = img.shape[1] + border[1] * 2

        # Center
        C = np.eye(3)
        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-degrees, degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(scale[0], scale[1])
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3)
        T[0, 2] = (
                random.uniform(0.5 - translate, 0.5 + translate) * width
        )  # x translation (pixels)
        T[1, 2] = (
                random.uniform(0.5 - translate, 0.5 + translate) * height
        )  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ C  # order of operations (right to left) is IMPORTANT

        ###########################
        # For Aug out of Mosaic
        # s = 1.
        # M = np.eye(3)
        ###########################

        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if perspective:
                img = cv2.warpPerspective(
                    img, M, dsize=(width, height), borderValue=(114, 114, 114)
                )
            else:  # affine
                img = cv2.warpAffine(
                    img, M[:2], dsize=(width, height), borderValue=(114, 114, 114)
                )

        # Transform label coordinates
        n = len(targets)
        if n:
            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
                n * 4, 2
            )  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            if perspective:
                xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
            else:  # affine
                xy = xy[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip boxes
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

            # filter candidates
            i = self.box_candidates(box1=targets[:, :4].T * s, box2=xy.T)
            targets = targets[i]
            targets[:, :4] = xy[i]

        return img, targets

    def box_candidates(self, box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2):
        # box1(4,n), box2(4,n)
        # Compute candidate boxes which include follwing 5 things:
        # box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
        return (
                (w2 > wh_thr)
                & (h2 > wh_thr)
                & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr)
                & (ar < ar_thr)
        )  # candidates

    def __call__(self, sample, context=None):
        if 'mosaic1' not in sample:  # 最后15个epoch没有马赛克增强
            return sample

        if np.random.uniform(0., 1.) > self.prob:
            sample.pop('mosaic1')
            sample.pop('mosaic2')
            sample.pop('mosaic3')
            sample.pop('mixup')
            return sample

        sample1 = sample.pop('mosaic1')
        sample2 = sample.pop('mosaic2')
        sample3 = sample.pop('mosaic3')
        sample_mixup = sample.pop('mixup')
        sample0 = sample

        mosaic_gt_class = []
        mosaic_gt_bbox = []
        input_dim = self.input_dim
        input_h, input_w = input_dim[0], input_dim[1]

        # yc, xc = s, s  # mosaic center x, y
        yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
        xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

        for i_mosaic, sp in enumerate([sample0, sample1, sample2, sample3]):
            # img, _labels, _, img_id = self._dataset.pull_item(index)
            img = sp['image']
            im_id = sp['im_id']  # [1, ]
            gt_class = sp['gt_class']  # [?, 1]
            gt_score = sp['gt_score']  # [?, 1]
            gt_bbox = sp['gt_bbox']  # [?, 4]
            im_info = sp['im_info']  # [3, ]   value = [h, w, 1]
            # h = sp['h']        # [1, ]
            # w = sp['h']        # [1, ]
            h0, w0 = img.shape[:2]  # orig hw
            scale = min(1. * input_h / h0, 1. * input_w / w0)
            img = cv2.resize(
                img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR
            )
            # generate output mosaic image
            (h, w, c) = img.shape[:3]
            if i_mosaic == 0:
                mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)

            # suffix l means large image, while s means small image in mosaic aug.
            (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = self.get_mosaic_coordinate(
                mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w
            )

            mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
            padw, padh = l_x1 - s_x1, l_y1 - s_y1

            _gt_bbox = gt_bbox.copy()
            # Normalized xywh to pixel xyxy format
            if len(gt_bbox) > 0:
                _gt_bbox[:, 0] = scale * gt_bbox[:, 0] + padw
                _gt_bbox[:, 1] = scale * gt_bbox[:, 1] + padh
                _gt_bbox[:, 2] = scale * gt_bbox[:, 2] + padw
                _gt_bbox[:, 3] = scale * gt_bbox[:, 3] + padh
            mosaic_gt_bbox.append(_gt_bbox)
            mosaic_gt_class.append(gt_class)
        # cv2.imwrite('%d.jpg'%im_id, mosaic_img)
        # print()

        if len(mosaic_gt_bbox):
            mosaic_gt_bbox = np.concatenate(mosaic_gt_bbox, 0)
            mosaic_gt_class = np.concatenate(mosaic_gt_class, 0)
            np.clip(mosaic_gt_bbox[:, 0], 0, 2 * input_w, out=mosaic_gt_bbox[:, 0])
            np.clip(mosaic_gt_bbox[:, 1], 0, 2 * input_h, out=mosaic_gt_bbox[:, 1])
            np.clip(mosaic_gt_bbox[:, 2], 0, 2 * input_w, out=mosaic_gt_bbox[:, 2])
            np.clip(mosaic_gt_bbox[:, 3], 0, 2 * input_h, out=mosaic_gt_bbox[:, 3])

        mosaic_labels = np.concatenate([mosaic_gt_bbox, mosaic_gt_class.astype(mosaic_gt_bbox.dtype)], 1)
        mosaic_img, mosaic_labels = self.random_perspective(
            mosaic_img,
            mosaic_labels,
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            border=[-input_h // 2, -input_w // 2],
        )  # border to remove
        # cv2.imwrite('%d2.jpg'%im_id, mosaic_img)
        # print()

        # -----------------------------------------------------------------
        # CopyPaste: https://arxiv.org/abs/2012.07177
        # -----------------------------------------------------------------
        if (
                self.enable_mixup
                and not len(mosaic_labels) == 0
                and random.random() < self.mixup_prob
        ):
            img_mixup = sample_mixup['image']
            cp_labels = np.concatenate([sample_mixup['gt_bbox'], sample_mixup['gt_class'].astype(mosaic_labels.dtype)],
                                       1)
            mosaic_img, mosaic_labels = self.mixup(mosaic_img, mosaic_labels, self.input_dim, cp_labels, img_mixup)
            # cv2.imwrite('%d3.jpg' % im_id, mosaic_img)
            # print()
        sample['image'] = mosaic_img
        sample['h'] = float(mosaic_img.shape[0])
        sample['w'] = float(mosaic_img.shape[1])
        sample['im_info'][0] = sample['h']
        sample['im_info'][1] = sample['w']
        sample['gt_class'] = mosaic_labels[:, 4:5].astype(np.int32)
        sample['gt_bbox'] = mosaic_labels[:, :4].astype(np.float32)
        sample['gt_score'] = np.ones(sample['gt_class'].shape, np.float32)
        return sample

    def adjust_box_anns(self, bbox, scale_ratio, padw, padh, w_max, h_max):
        bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
        bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
        return bbox

    def mixup(self, origin_img, origin_labels, input_dim, cp_labels, img):
        jit_factor = random.uniform(*self.mixup_scale)
        FLIP = random.uniform(0, 1) > 0.5
        # cp_labels = []
        # while len(cp_labels) == 0:
        #     cp_index = random.randint(0, self.__len__() - 1)
        #     cp_labels = self._dataset.load_anno(cp_index)
        # img, cp_labels, _, _ = self._dataset.pull_item(cp_index)

        if len(img.shape) == 3:
            cp_img = np.ones((input_dim[0], input_dim[1], 3), dtype=np.uint8) * 114
        else:
            cp_img = np.ones(input_dim, dtype=np.uint8) * 114

        cp_scale_ratio = min(input_dim[0] / img.shape[0], input_dim[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        )

        cp_img[
        : int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)
        ] = resized_img

        cp_img = cv2.resize(
            cp_img,
            (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
        )
        cp_scale_ratio *= jit_factor

        if FLIP:
            cp_img = cp_img[:, ::-1, :]

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3), dtype=np.uint8
        )
        padded_img[:origin_h, :origin_w] = cp_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[
                             y_offset: y_offset + target_h, x_offset: x_offset + target_w
                             ]

        cp_bboxes_origin_np = self.adjust_box_anns(
            cp_labels[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h
        )
        if FLIP:
            cp_bboxes_origin_np[:, 0::2] = (
                    origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1]
            )
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        cp_bboxes_transformed_np[:, 0::2] = np.clip(
            cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w
        )
        cp_bboxes_transformed_np[:, 1::2] = np.clip(
            cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h
        )
        keep_list = self.box_candidates(cp_bboxes_origin_np.T, cp_bboxes_transformed_np.T, 5)

        if keep_list.sum() >= 1.0:
            cls_labels = cp_labels[keep_list, 4:5].copy()
            box_labels = cp_bboxes_transformed_np[keep_list]
            labels = np.hstack((box_labels, cls_labels))
            origin_labels = np.vstack((origin_labels, labels))
            origin_img = origin_img.astype(np.float32)
            origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)

        return origin_img.astype(np.uint8), origin_labels


class PhotometricDistort(BaseOperator):
    def __init__(self):
        super(PhotometricDistort, self).__init__()

    def __call__(self, sample, context=None):
        im = sample['image']

        image = im.astype(np.float32)

        # RandomBrightness
        if np.random.randint(2):
            delta = 32
            delta = np.random.uniform(-delta, delta)
            image += delta

        state = np.random.randint(2)
        if state == 0:
            if np.random.randint(2):
                lower = 0.5
                upper = 1.5
                alpha = np.random.uniform(lower, upper)
                image *= alpha

        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        if np.random.randint(2):
            lower = 0.5
            upper = 1.5
            image[:, :, 1] *= np.random.uniform(lower, upper)

        if np.random.randint(2):
            delta = 18.0
            image[:, :, 0] += np.random.uniform(-delta, delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0

        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

        if state == 1:
            if np.random.randint(2):
                lower = 0.5
                upper = 1.5
                alpha = np.random.uniform(lower, upper)
                image *= alpha

        sample['image'] = image
        return sample


class RandomCrop(BaseOperator):
    """Random crop image and bboxes.
    Args:
        aspect_ratio (list): aspect ratio of cropped region.
            in [min, max] format.
        thresholds (list): iou thresholds for decide a valid bbox crop.
        scaling (list): ratio between a cropped region and the original image.
             in [min, max] format.
        num_attempts (int): number of tries before giving up.
        allow_no_crop (bool): allow return without actually cropping them.
        cover_all_box (bool): ensure all bboxes are covered in the final crop.
        is_mask_crop(bool): whether crop the segmentation.
    """

    def __init__(self,
                 aspect_ratio=[.5, 2.],
                 thresholds=[.0, .1, .3, .5, .7, .9],
                 scaling=[.3, 1.],
                 num_attempts=50,
                 allow_no_crop=True,
                 cover_all_box=False,
                 is_mask_crop=False):
        super(RandomCrop, self).__init__()
        self.aspect_ratio = aspect_ratio
        self.thresholds = thresholds
        self.scaling = scaling
        self.num_attempts = num_attempts
        self.allow_no_crop = allow_no_crop
        self.cover_all_box = cover_all_box
        self.is_mask_crop = is_mask_crop

    def crop_segms(self, segms, valid_ids, crop, height, width):
        def _crop_poly(segm, crop):
            xmin, ymin, xmax, ymax = crop
            crop_coord = [xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]
            crop_p = np.array(crop_coord).reshape(4, 2)
            crop_p = Polygon(crop_p)

            crop_segm = list()
            for poly in segm:
                poly = np.array(poly).reshape(len(poly) // 2, 2)
                polygon = Polygon(poly)
                if not polygon.is_valid:
                    exterior = polygon.exterior
                    multi_lines = exterior.intersection(exterior)
                    polygons = shapely.ops.polygonize(multi_lines)
                    polygon = MultiPolygon(polygons)
                multi_polygon = list()
                if isinstance(polygon, MultiPolygon):
                    multi_polygon = copy.deepcopy(polygon)
                else:
                    multi_polygon.append(copy.deepcopy(polygon))
                for per_polygon in multi_polygon:
                    inter = per_polygon.intersection(crop_p)
                    if not inter:
                        continue
                    if isinstance(inter, (MultiPolygon, GeometryCollection)):
                        for part in inter:
                            if not isinstance(part, Polygon):
                                continue
                            part = np.squeeze(
                                np.array(part.exterior.coords[:-1]).reshape(1,
                                                                            -1))
                            part[0::2] -= xmin
                            part[1::2] -= ymin
                            crop_segm.append(part.tolist())
                    elif isinstance(inter, Polygon):
                        crop_poly = np.squeeze(
                            np.array(inter.exterior.coords[:-1]).reshape(1, -1))
                        crop_poly[0::2] -= xmin
                        crop_poly[1::2] -= ymin
                        crop_segm.append(crop_poly.tolist())
                    else:
                        continue
            return crop_segm

        def _crop_rle(rle, crop, height, width):
            if 'counts' in rle and type(rle['counts']) == list:
                rle = mask_util.frPyObjects(rle, height, width)
            mask = mask_util.decode(rle)
            mask = mask[crop[1]:crop[3], crop[0]:crop[2]]
            rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
            return rle

        crop_segms = []
        for id in valid_ids:
            segm = segms[id]
            if is_poly(segm):
                import copy
                import shapely.ops
                from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
                # logging.getLogger("shapely").setLevel(logging.WARNING)
                # Polygon format
                crop_segms.append(_crop_poly(segm, crop))
            else:
                # RLE format
                import pycocotools.mask as mask_util
                crop_segms.append(_crop_rle(segm, crop, height, width))
        return crop_segms

    def __call__(self, sample, context=None):
        if 'gt_bbox' in sample and len(sample['gt_bbox']) == 0:
            return sample

        h = sample['h']
        w = sample['w']
        gt_bbox = sample['gt_bbox']

        # NOTE Original method attempts to generate one candidate for each
        # threshold then randomly sample one from the resulting list.
        # Here a short circuit approach is taken, i.e., randomly choose a
        # threshold and attempt to find a valid crop, and simply return the
        # first one found.
        # The probability is not exactly the same, kinda resembling the
        # "Monty Hall" problem. Actually carrying out the attempts will affect
        # observability (just like opening doors in the "Monty Hall" game).
        thresholds = list(self.thresholds)
        if self.allow_no_crop:
            thresholds.append('no_crop')
        np.random.shuffle(thresholds)

        for thresh in thresholds:
            if thresh == 'no_crop':
                return sample

            found = False
            for i in range(self.num_attempts):
                scale = np.random.uniform(*self.scaling)
                if self.aspect_ratio is not None:
                    min_ar, max_ar = self.aspect_ratio
                    aspect_ratio = np.random.uniform(
                        max(min_ar, scale ** 2), min(max_ar, scale ** -2))
                    h_scale = scale / np.sqrt(aspect_ratio)
                    w_scale = scale * np.sqrt(aspect_ratio)
                else:
                    h_scale = np.random.uniform(*self.scaling)
                    w_scale = np.random.uniform(*self.scaling)
                crop_h = h * h_scale
                crop_w = w * w_scale
                if self.aspect_ratio is None:
                    if crop_h / crop_w < 0.5 or crop_h / crop_w > 2.0:
                        continue

                crop_h = int(crop_h)
                crop_w = int(crop_w)
                crop_y = np.random.randint(0, h - crop_h)
                crop_x = np.random.randint(0, w - crop_w)
                crop_box = [crop_x, crop_y, crop_x + crop_w, crop_y + crop_h]
                iou = self._iou_matrix(
                    gt_bbox, np.array(
                        [crop_box], dtype=np.float32))
                if iou.max() < thresh:
                    continue

                if self.cover_all_box and iou.min() < thresh:
                    continue

                cropped_box, valid_ids = self._crop_box_with_center_constraint(
                    gt_bbox, np.array(
                        crop_box, dtype=np.float32))
                if valid_ids.size > 0:
                    found = True
                    break

            if found:
                if self.is_mask_crop and 'gt_poly' in sample and len(sample['gt_poly']) > 0:
                    crop_polys = self.crop_segms(
                        sample['gt_poly'],
                        valid_ids,
                        np.array(
                            crop_box, dtype=np.int64),
                        h,
                        w)
                    if [] in crop_polys:
                        delete_id = list()
                        valid_polys = list()
                        for id, crop_poly in enumerate(crop_polys):
                            if crop_poly == []:
                                delete_id.append(id)
                            else:
                                valid_polys.append(crop_poly)
                        valid_ids = np.delete(valid_ids, delete_id)
                        if len(valid_polys) == 0:
                            return sample
                        sample['gt_poly'] = valid_polys
                    else:
                        sample['gt_poly'] = crop_polys
                sample['image'] = self._crop_image(sample['image'], crop_box)
                # 掩码也被删去与裁剪
                if 'gt_segm' in sample.keys() and sample['gt_segm'] is not None:
                    gt_segm = sample['gt_segm']
                    gt_segm = gt_segm.transpose(1, 2, 0)
                    gt_segm = np.take(gt_segm, valid_ids, axis=-1)
                    gt_segm = self._crop_image(gt_segm, crop_box)
                    gt_segm = gt_segm.transpose(2, 0, 1)
                    sample['gt_segm'] = gt_segm
                sample['gt_bbox'] = np.take(cropped_box, valid_ids, axis=0)
                sample['gt_class'] = np.take(
                    sample['gt_class'], valid_ids, axis=0)
                sample['w'] = crop_box[2] - crop_box[0]
                sample['h'] = crop_box[3] - crop_box[1]
                if 'gt_score' in sample:
                    sample['gt_score'] = np.take(
                        sample['gt_score'], valid_ids, axis=0)

                if 'is_crowd' in sample:
                    sample['is_crowd'] = np.take(
                        sample['is_crowd'], valid_ids, axis=0)
                return sample

        return sample

    def _iou_matrix(self, a, b):
        tl_i = np.maximum(a[:, np.newaxis, :2], b[:, :2])
        br_i = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

        area_i = np.prod(br_i - tl_i, axis=2) * (tl_i < br_i).all(axis=2)
        area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
        area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
        area_o = (area_a[:, np.newaxis] + area_b - area_i)
        return area_i / (area_o + 1e-10)

    def _crop_box_with_center_constraint(self, box, crop):
        cropped_box = box.copy()

        cropped_box[:, :2] = np.maximum(box[:, :2], crop[:2])
        cropped_box[:, 2:] = np.minimum(box[:, 2:], crop[2:])
        cropped_box[:, :2] -= crop[:2]
        cropped_box[:, 2:] -= crop[:2]

        centers = (box[:, :2] + box[:, 2:]) / 2
        valid = np.logical_and(crop[:2] <= centers,
                               centers < crop[2:]).all(axis=1)
        valid = np.logical_and(
            valid, (cropped_box[:, :2] < cropped_box[:, 2:]).all(axis=1))

        return cropped_box, np.where(valid)[0]

    def _crop_image(self, img, crop):
        x1, y1, x2, y2 = crop
        return img[y1:y2, x1:x2, :]


class GridMaskOp(BaseOperator):
    def __init__(self,
                 use_h=True,
                 use_w=True,
                 rotate=1,
                 offset=False,
                 ratio=0.5,
                 mode=1,
                 prob=0.7,
                 upper_iter=360000):
        """
        GridMask Data Augmentation, see https://arxiv.org/abs/2001.04086
        Args:
            use_h (bool): whether to mask vertically
            use_w (boo;): whether to mask horizontally
            rotate (float): angle for the mask to rotate
            offset (float): mask offset
            ratio (float): mask ratio
            mode (int): gridmask mode
            prob (float): max probability to carry out gridmask
            upper_iter (int): suggested to be equal to global max_iter
        """
        super(GridMaskOp, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.prob = prob
        self.upper_iter = upper_iter

        from .gridmask_utils import GridMask
        self.gridmask_op = GridMask(
            use_h,
            use_w,
            rotate=rotate,
            offset=offset,
            ratio=ratio,
            mode=mode,
            prob=prob,
            upper_iter=upper_iter)

    def __call__(self, sample, context=None):
        samples = sample
        batch_input = True
        if not isinstance(samples, Sequence):
            batch_input = False
            samples = [samples]
        for sample in samples:
            sample['image'] = self.gridmask_op(sample['image'],
                                               sample['curr_iter'])
        if not batch_input:
            samples = samples[0]
        return samples


class Poly2Mask(BaseOperator):
    """
    gt poly to mask annotations
    """

    def __init__(self):
        super(Poly2Mask, self).__init__()
        import pycocotools.mask as maskUtils
        self.maskutils = maskUtils

    def _poly2mask(self, mask_ann, img_h, img_w):
        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = self.maskutils.frPyObjects(mask_ann, img_h, img_w)
            rle = self.maskutils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = self.maskutils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = self.maskutils.decode(rle)
        return mask

    def __call__(self, sample, context=None):
        assert 'gt_poly' in sample
        im_h = sample['h']
        im_w = sample['w']
        masks = [
            self._poly2mask(gt_poly, im_h, im_w)
            for gt_poly in sample['gt_poly']
        ]
        sample['gt_segm'] = np.asarray(masks).astype(np.uint8)
        return sample


class ColorDistort(BaseOperator):
    """Random color distortion.
    Args:
        hue (list): hue settings.
            in [lower, upper, probability] format.
        saturation (list): saturation settings.
            in [lower, upper, probability] format.
        contrast (list): contrast settings.
            in [lower, upper, probability] format.
        brightness (list): brightness settings.
            in [lower, upper, probability] format.
        random_apply (bool): whether to apply in random (yolo) or fixed (SSD)
            order.
        hsv_format (bool): whether to convert color from BGR to HSV
        random_channel (bool): whether to swap channels randomly
    """

    def __init__(self,
                 hue=[-18, 18, 0.5],
                 saturation=[0.5, 1.5, 0.5],
                 contrast=[0.5, 1.5, 0.5],
                 brightness=[0.5, 1.5, 0.5],
                 random_apply=True,
                 hsv_format=False,
                 random_channel=False):
        super(ColorDistort, self).__init__()
        self.hue = hue
        self.saturation = saturation
        self.contrast = contrast
        self.brightness = brightness
        self.random_apply = random_apply
        self.hsv_format = hsv_format
        self.random_channel = random_channel

    def apply_hue(self, img):
        low, high, prob = self.hue
        if np.random.uniform(0., 1.) < prob:
            return img

        img = img.astype(np.float32)
        if self.hsv_format:
            img[..., 0] += random.uniform(low, high)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360
            return img

        # XXX works, but result differ from HSV version
        delta = np.random.uniform(low, high)
        u = np.cos(delta * np.pi)
        w = np.sin(delta * np.pi)
        bt = np.array([[1.0, 0.0, 0.0], [0.0, u, -w], [0.0, w, u]])
        tyiq = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.321],
                         [0.211, -0.523, 0.311]])
        ityiq = np.array([[1.0, 0.956, 0.621], [1.0, -0.272, -0.647],
                          [1.0, -1.107, 1.705]])
        t = np.dot(np.dot(ityiq, bt), tyiq).T
        img = np.dot(img, t)
        return img

    def apply_saturation(self, img):
        low, high, prob = self.saturation
        if np.random.uniform(0., 1.) < prob:
            return img
        delta = np.random.uniform(low, high)
        img = img.astype(np.float32)
        if self.hsv_format:
            img[..., 1] *= delta
            return img
        gray = img * np.array([[[0.299, 0.587, 0.114]]], dtype=np.float32)
        gray = gray.sum(axis=2, keepdims=True)
        gray *= (1.0 - delta)
        img *= delta
        img += gray
        return img

    def apply_contrast(self, img):
        low, high, prob = self.contrast
        if np.random.uniform(0., 1.) < prob:
            return img
        delta = np.random.uniform(low, high)

        img = img.astype(np.float32)
        img *= delta
        return img

    def apply_brightness(self, img):
        low, high, prob = self.brightness
        if np.random.uniform(0., 1.) < prob:
            return img
        delta = np.random.uniform(low, high)

        img = img.astype(np.float32)
        img += delta
        return img

    def __call__(self, sample, context=None):
        img = sample['image']
        if self.random_apply:
            functions = [
                self.apply_brightness,
                self.apply_contrast,
                self.apply_saturation,
                self.apply_hue,
            ]
            distortions = np.random.permutation(functions)
            for func in distortions:
                img = func(img)
            sample['image'] = img
            return sample

        img = self.apply_brightness(img)

        if np.random.randint(0, 2):
            img = self.apply_contrast(img)
            if self.hsv_format:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            img = self.apply_saturation(img)
            img = self.apply_hue(img)
            if self.hsv_format:
                img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        else:
            if self.hsv_format:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            img = self.apply_saturation(img)
            img = self.apply_hue(img)
            if self.hsv_format:
                img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
            img = self.apply_contrast(img)

        if self.random_channel:
            if np.random.randint(0, 2):
                img = img[..., np.random.permutation(3)]
        sample['image'] = img
        return sample


from numbers import Number


class RandomExpand(BaseOperator):
    """Random expand the canvas.
    Args:
        ratio (float): maximum expansion ratio.
        prob (float): probability to expand.
        fill_value (list): color value used to fill the canvas. in RGB order.
        is_mask_expand(bool): whether expand the segmentation.
    """

    def __init__(self,
                 ratio=4.,
                 prob=0.5,
                 fill_value=(127.5,) * 3,
                 is_mask_expand=False):
        super(RandomExpand, self).__init__()
        assert ratio > 1.01, "expand ratio must be larger than 1.01"
        self.ratio = ratio
        self.prob = prob
        assert isinstance(fill_value, (Number, Sequence)), \
            "fill value must be either float or sequence"
        if isinstance(fill_value, Number):
            fill_value = (fill_value,) * 3
        if not isinstance(fill_value, tuple):
            fill_value = tuple(fill_value)
        self.fill_value = fill_value
        self.is_mask_expand = is_mask_expand

    def expand_segms(self, segms, x, y, height, width, ratio):
        def _expand_poly(poly, x, y):
            expanded_poly = np.array(poly)
            expanded_poly[0::2] += x
            expanded_poly[1::2] += y
            return expanded_poly.tolist()

        def _expand_rle(rle, x, y, height, width, ratio):
            if 'counts' in rle and type(rle['counts']) == list:
                rle = mask_util.frPyObjects(rle, height, width)
            mask = mask_util.decode(rle)
            expanded_mask = np.full((int(height * ratio), int(width * ratio)),
                                    0).astype(mask.dtype)
            expanded_mask[y:y + height, x:x + width] = mask
            rle = mask_util.encode(
                np.array(
                    expanded_mask, order='F', dtype=np.uint8))
            return rle

        expanded_segms = []
        for segm in segms:
            if is_poly(segm):
                # Polygon format
                expanded_segms.append(
                    [_expand_poly(poly, x, y) for poly in segm])
            else:
                # RLE format
                import pycocotools.mask as mask_util
                expanded_segms.append(
                    _expand_rle(segm, x, y, height, width, ratio))
        return expanded_segms

    def __call__(self, sample, context=None):
        if np.random.uniform(0., 1.) < self.prob:
            return sample

        img = sample['image']
        height = int(sample['h'])
        width = int(sample['w'])

        expand_ratio = np.random.uniform(1., self.ratio)
        h = int(height * expand_ratio)
        w = int(width * expand_ratio)
        if not h > height or not w > width:
            return sample
        y = np.random.randint(0, h - height)
        x = np.random.randint(0, w - width)
        canvas = np.ones((h, w, 3), dtype=np.uint8)
        canvas *= np.array(self.fill_value, dtype=np.uint8)
        canvas[y:y + height, x:x + width, :] = img.astype(np.uint8)

        sample['h'] = h
        sample['w'] = w
        sample['image'] = canvas
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            sample['gt_bbox'] += np.array([x, y] * 2, dtype=np.float32)
        if self.is_mask_expand and 'gt_poly' in sample and len(sample[
                                                                   'gt_poly']) > 0:
            sample['gt_poly'] = self.expand_segms(sample['gt_poly'], x, y,
                                                  height, width, expand_ratio)
        return sample


class RandomFlipImage(BaseOperator):
    def __init__(self, prob=0.5, is_normalized=False, is_mask_flip=False):
        """
        Args:
            prob (float): the probability of flipping image
            is_normalized (bool): whether the bbox scale to [0,1]
            is_mask_flip (bool): whether flip the segmentation
        """
        super(RandomFlipImage, self).__init__()
        self.prob = prob
        self.is_normalized = is_normalized
        self.is_mask_flip = is_mask_flip
        if not (isinstance(self.prob, float) and
                isinstance(self.is_normalized, bool) and
                isinstance(self.is_mask_flip, bool)):
            raise TypeError("{}: input type is invalid.".format(self))

    def flip_segms(self, segms, height, width):
        def _flip_poly(poly, width):
            flipped_poly = np.array(poly)
            flipped_poly[0::2] = width - np.array(poly[0::2]) - 1
            return flipped_poly.tolist()

        def _flip_rle(rle, height, width):
            if 'counts' in rle and type(rle['counts']) == list:
                rle = mask_util.frPyObjects(rle, height, width)
            mask = mask_util.decode(rle)
            mask = mask[:, ::-1]
            rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
            return rle

        flipped_segms = []
        for segm in segms:
            if is_poly(segm):
                # Polygon format
                flipped_segms.append([_flip_poly(poly, width) for poly in segm])
            else:
                # RLE format
                import pycocotools.mask as mask_util
                flipped_segms.append(_flip_rle(segm, height, width))
        return flipped_segms

    def flip_keypoint(self, gt_keypoint, width):
        for i in range(gt_keypoint.shape[1]):
            if i % 2 == 0:
                old_x = gt_keypoint[:, i].copy()
                if self.is_normalized:
                    gt_keypoint[:, i] = 1 - old_x
                else:
                    gt_keypoint[:, i] = width - old_x - 1
        return gt_keypoint

    def __call__(self, sample, context=None):
        """Filp the image and bounding box.
        Operators:
            1. Flip the image numpy.
            2. Transform the bboxes' x coordinates.
              (Must judge whether the coordinates are normalized!)
            3. Transform the segmentations' x coordinates.
              (Must judge whether the coordinates are normalized!)
        Output:
            sample: the image, bounding box and segmentation part
                    in sample are flipped.
        """

        samples = sample
        batch_input = True
        if not isinstance(samples, Sequence):
            batch_input = False
            samples = [samples]
        for sample in samples:
            gt_bbox = sample['gt_bbox']
            im = sample['image']
            if not isinstance(im, np.ndarray):
                raise TypeError("{}: image is not a numpy array.".format(self))
            if len(im.shape) != 3:
                raise ImageError("{}: image is not 3-dimensional.".format(self))
            height, width, _ = im.shape
            if np.random.uniform(0, 1) < self.prob:
                im = im[:, ::-1, :]
                if gt_bbox.shape[0] == 0:
                    return sample
                oldx1 = gt_bbox[:, 0].copy()
                oldx2 = gt_bbox[:, 2].copy()
                if self.is_normalized:
                    gt_bbox[:, 0] = 1 - oldx2
                    gt_bbox[:, 2] = 1 - oldx1
                else:
                    gt_bbox[:, 0] = width - oldx2 - 1
                    gt_bbox[:, 2] = width - oldx1 - 1
                if gt_bbox.shape[0] != 0 and (
                        gt_bbox[:, 2] < gt_bbox[:, 0]).all():
                    m = "{}: invalid box, x2 should be greater than x1".format(
                        self)
                    raise BboxError(m)
                sample['gt_bbox'] = gt_bbox
                if self.is_mask_flip and len(sample['gt_poly']) != 0:
                    sample['gt_poly'] = self.flip_segms(sample['gt_poly'],
                                                        height, width)
                if 'gt_keypoint' in sample.keys():
                    sample['gt_keypoint'] = self.flip_keypoint(
                        sample['gt_keypoint'], width)

                if 'semantic' in sample.keys() and sample[
                    'semantic'] is not None:
                    sample['semantic'] = sample['semantic'][:, ::-1]

                if 'gt_segm' in sample.keys() and sample['gt_segm'] is not None:
                    sample['gt_segm'] = sample['gt_segm'][:, :, ::-1]

                sample['flipped'] = True
                sample['image'] = im
        sample = samples if batch_input else samples[0]
        return sample


class NormalizeBox(BaseOperator):
    """Transform the bounding box's coornidates to [0,1]."""

    def __init__(self):
        super(NormalizeBox, self).__init__()

    def __call__(self, sample, context):
        gt_bbox = sample['gt_bbox']
        width = sample['w']
        height = sample['h']
        for i in range(gt_bbox.shape[0]):
            gt_bbox[i][0] = gt_bbox[i][0] / width
            gt_bbox[i][1] = gt_bbox[i][1] / height
            gt_bbox[i][2] = gt_bbox[i][2] / width
            gt_bbox[i][3] = gt_bbox[i][3] / height
        sample['gt_bbox'] = gt_bbox

        if 'gt_keypoint' in sample.keys():
            gt_keypoint = sample['gt_keypoint']

            for i in range(gt_keypoint.shape[1]):
                if i % 2:
                    gt_keypoint[:, i] = gt_keypoint[:, i] / height
                else:
                    gt_keypoint[:, i] = gt_keypoint[:, i] / width
            sample['gt_keypoint'] = gt_keypoint

        return sample


class BboxXYXY2XYWH(BaseOperator):
    """
    Convert bbox XYXY format to XYWH format.
    """

    def __init__(self):
        super(BboxXYXY2XYWH, self).__init__()

    def __call__(self, sample, context=None):
        assert 'gt_bbox' in sample
        bbox = sample['gt_bbox']
        bbox[:, 2:4] = bbox[:, 2:4] - bbox[:, :2]
        bbox[:, :2] = bbox[:, :2] + bbox[:, 2:4] / 2.
        sample['gt_bbox'] = bbox
        return sample


class NormalizeImage(BaseOperator):
    def __init__(self,
                 mean=[0.485, 0.456, 0.406],
                 std=[1, 1, 1],
                 is_scale=True,
                 is_channel_first=True):
        """
        Args:
            mean (list): the pixel mean
            std (list): the pixel variance
        """
        super(NormalizeImage, self).__init__()
        self.mean = mean
        self.std = std
        self.is_scale = is_scale
        self.is_channel_first = is_channel_first
        if not (isinstance(self.mean, list) and isinstance(self.std, list) and
                isinstance(self.is_scale, bool)):
            raise TypeError("{}: input type is invalid.".format(self))
        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def __call__(self, sample, context=None):
        """Normalize the image.
        Operators:
            1.(optional) Scale the image to [0,1]
            2. Each pixel minus mean and is divided by std
        """
        samples = sample
        batch_input = True
        if not isinstance(samples, Sequence):
            batch_input = False
            samples = [samples]
        for sample in samples:
            for k in sample.keys():
                # hard code
                if k.startswith('image'):
                    im = sample[k]
                    im = im.astype(np.float32, copy=False)
                    if self.is_channel_first:
                        mean = np.array(self.mean)[:, np.newaxis, np.newaxis]
                        std = np.array(self.std)[:, np.newaxis, np.newaxis]
                    else:
                        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
                        std = np.array(self.std)[np.newaxis, np.newaxis, :]
                    if self.is_scale:
                        im = im / 255.0
                    im -= mean
                    im /= std
                    sample[k] = im
        if not batch_input:
            samples = samples[0]
        return samples


class SquareImage(BaseOperator):
    def __init__(self,
                 fill_value=0,
                 is_channel_first=True):
        """
        Args:
            fill_value (int): the filled pixel value
            is_channel_first (bool): ...
        """
        super(SquareImage, self).__init__()
        if not isinstance(fill_value, int):
            raise ValueError('fill_value must be int!')
        if fill_value < 0 or fill_value > 255:
            raise ValueError('fill_value must in 0 ~ 255')
        self.fill_value = fill_value
        self.is_channel_first = is_channel_first

    def __call__(self, sample, context=None):
        """Square the image.
        Operators:
            1. ...
            2. ...
        """
        samples = sample
        batch_input = True
        if not isinstance(samples, Sequence):
            batch_input = False
            samples = [samples]
        for sample in samples:
            for k in sample.keys():
                # hard code
                if k.startswith('image'):
                    im = sample[k]
                    im = im.astype(np.float32, copy=False)
                    sample[k] = im  # 类型转换
                    if self.is_channel_first:
                        C, H, W = im.shape
                        if H != W:
                            max_ = max(H, W)
                            padded_img = np.ones((C, max_, max_), dtype=np.uint8) * self.fill_value
                            padded_img = padded_img.astype(np.float32)
                            padded_img[:C, :H, :W] = im
                            sample[k] = padded_img
                    else:
                        H, W, C = im.shape
                        if H != W:
                            max_ = max(H, W)
                            padded_img = np.ones((max_, max_, C), dtype=np.uint8) * self.fill_value
                            padded_img = padded_img.astype(np.float32)
                            padded_img[:H, :W, :C] = im
                            sample[k] = padded_img
                    break
        if not batch_input:
            samples = samples[0]
        return samples


class ResizeImage(BaseOperator):
    def __init__(self,
                 target_size=0,
                 max_size=0,
                 interp=cv2.INTER_LINEAR,
                 use_cv2=True,
                 resize_box=False):
        """
        Rescale image to the specified target size, and capped at max_size
        if max_size != 0.
        If target_size is list, selected a scale randomly as the specified
        target size.
        Args:
            target_size (int|list): the target size of image's short side,
                multi-scale training is adopted when type is list.
            max_size (int): the max size of image
            interp (int): the interpolation method
            use_cv2 (bool): use the cv2 interpolation method or use PIL
                interpolation method
            resize_box (bool): whether resize ground truth bbox annotations.
        """
        super(ResizeImage, self).__init__()
        self.max_size = int(max_size)
        self.interp = int(interp)
        self.use_cv2 = use_cv2
        self.resize_box = resize_box
        if not (isinstance(target_size, int) or isinstance(target_size, list)):
            raise TypeError(
                "Type of target_size is invalid. Must be Integer or List, now is {}".
                    format(type(target_size)))
        self.target_size = target_size
        if not (isinstance(self.max_size, int) and isinstance(self.interp,
                                                              int)):
            raise TypeError("{}: input type is invalid.".format(self))

    def __call__(self, sample, context=None):
        """ Resize the image numpy.
        """
        im = sample['image']
        if not isinstance(im, np.ndarray):
            raise TypeError("{}: image type is not numpy.".format(self))
        if len(im.shape) != 3:
            raise ImageError('{}: image is not 3-dimensional.'.format(self))
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        if isinstance(self.target_size, list):
            # Case for multi-scale training
            selected_size = random.choice(self.target_size)
        else:
            selected_size = self.target_size
        if float(im_size_min) == 0:
            raise ZeroDivisionError('{}: min size of image is 0'.format(self))
        if self.max_size != 0:
            im_scale = float(selected_size) / float(im_size_min)
            # Prevent the biggest axis from being more than max_size
            if np.round(im_scale * im_size_max) > self.max_size:
                im_scale = float(self.max_size) / float(im_size_max)
            im_scale_x = im_scale
            im_scale_y = im_scale

            resize_w = im_scale_x * float(im_shape[1])
            resize_h = im_scale_y * float(im_shape[0])
            im_info = [resize_h, resize_w, im_scale]
            if 'im_info' in sample and sample['im_info'][2] != 1.:
                sample['im_info'] = np.append(
                    list(sample['im_info']), im_info).astype(np.float32)
            else:
                sample['im_info'] = np.array(im_info).astype(np.float32)
        else:
            im_scale_x = float(selected_size) / float(im_shape[1])
            im_scale_y = float(selected_size) / float(im_shape[0])

            resize_w = selected_size
            resize_h = selected_size

        if self.use_cv2:
            im = cv2.resize(
                im,
                None,
                None,
                fx=im_scale_x,
                fy=im_scale_y,
                interpolation=self.interp)
        else:
            if self.max_size != 0:
                raise TypeError(
                    'If you set max_size to cap the maximum size of image,'
                    'please set use_cv2 to True to resize the image.')
            im = im.astype('uint8')
            im = Image.fromarray(im)
            im = im.resize((int(resize_w), int(resize_h)), self.interp)
            im = np.array(im)
        sample['image'] = im
        sample['scale_factor'] = [im_scale_x, im_scale_y] * 2
        if 'gt_bbox' in sample and self.resize_box and len(sample[
                                                               'gt_bbox']) > 0:
            bboxes = sample['gt_bbox'] * sample['scale_factor']
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, resize_w - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, resize_h - 1)
            sample['gt_bbox'] = bboxes
        if 'semantic' in sample.keys() and sample['semantic'] is not None:
            semantic = sample['semantic']
            semantic = cv2.resize(
                semantic.astype('float32'),
                None,
                None,
                fx=im_scale_x,
                fy=im_scale_y,
                interpolation=self.interp)
            semantic = np.asarray(semantic).astype('int32')
            semantic = np.expand_dims(semantic, 0)
            sample['semantic'] = semantic
        if 'gt_segm' in sample and len(sample['gt_segm']) > 0:
            masks = [
                cv2.resize(
                    gt_segm,
                    None,
                    None,
                    fx=im_scale_x,
                    fy=im_scale_y,
                    interpolation=cv2.INTER_NEAREST)
                for gt_segm in sample['gt_segm']
            ]
            sample['gt_segm'] = np.asarray(masks).astype(np.uint8)

        return sample


class YOLOXResizeImage(BaseOperator):
    def __init__(self,
                 target_size=0,
                 interp=cv2.INTER_LINEAR,
                 use_cv2=True,
                 resize_box=False):
        """
        """
        super(YOLOXResizeImage, self).__init__()
        self.interp = int(interp)
        self.use_cv2 = use_cv2
        self.resize_box = resize_box
        if not (isinstance(target_size, int) or isinstance(target_size, list)):
            raise TypeError("Type of target_size is invalid. Must be Integer or List, now is {}".format(type(target_size)))
        self.target_size = target_size

    def __call__(self, sample, target_size, context=None):
        """ Resize the image numpy.
        """
        im = sample['image']
        if not isinstance(im, np.ndarray):
            raise TypeError("{}: image type is not numpy.".format(self))
        if len(im.shape) != 3:
            raise ImageError('{}: image is not 3-dimensional.'.format(self))
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        # 根据当前输入target_size设置max_size
        selected_size = target_size
        max_size = target_size
        if float(im_size_min) == 0:
            raise ZeroDivisionError('{}: min size of image is 0'.format(self))

        im_scale = float(max_size) / float(im_size_max)
        im_scale_x = im_scale
        im_scale_y = im_scale

        resize_w = im_scale_x * float(im_shape[1])
        resize_h = im_scale_y * float(im_shape[0])
        im_info = [resize_h, resize_w, im_scale]
        if 'im_info' in sample and sample['im_info'][2] != 1.:
            sample['im_info'] = np.append(
                list(sample['im_info']), im_info).astype(np.float32)
        else:
            sample['im_info'] = np.array(im_info).astype(np.float32)

        if self.use_cv2:
            im = cv2.resize(
                im,
                None,
                None,
                fx=im_scale_x,
                fy=im_scale_y,
                interpolation=self.interp)
        else:
            if max_size != 0:
                raise TypeError(
                    'If you set max_size to cap the maximum size of image,'
                    'please set use_cv2 to True to resize the image.')
            im = im.astype('uint8')
            im = Image.fromarray(im)
            im = im.resize((int(resize_w), int(resize_h)), self.interp)
            im = np.array(im)
        sample['image'] = im
        sample['scale_factor'] = [im_scale_x, im_scale_y] * 2
        if 'gt_bbox' in sample and self.resize_box and len(sample['gt_bbox']) > 0:
            bboxes = sample['gt_bbox'] * sample['scale_factor']
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, resize_w - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, resize_h - 1)
            sample['gt_bbox'] = bboxes

        return sample


class PadBatch(BaseOperator):
    """
    Pad a batch of samples so they can be divisible by a stride.
    The layout of each image should be 'CHW'.
    Args:
        pad_to_stride (int): If `pad_to_stride > 0`, pad zeros to ensure
            height and width is divisible by `pad_to_stride`.
    """

    def __init__(self, pad_to_stride=0, use_padded_im_info=True):
        super(PadBatch, self).__init__()
        self.pad_to_stride = pad_to_stride
        self.use_padded_im_info = use_padded_im_info

    def __call__(self, samples, context=None):
        """
        Args:
            samples (list): a batch of sample, each is dict.
        """
        coarsest_stride = self.pad_to_stride
        if coarsest_stride == 0:
            return samples
        max_shape = np.array([data['image'].shape for data in samples]).max(
            axis=0)

        if coarsest_stride > 0:
            max_shape[1] = int(
                np.ceil(max_shape[1] / coarsest_stride) * coarsest_stride)
            max_shape[2] = int(
                np.ceil(max_shape[2] / coarsest_stride) * coarsest_stride)

        padding_batch = []
        for data in samples:
            im = data['image']
            im_c, im_h, im_w = im.shape[:]
            padding_im = np.zeros(
                (im_c, max_shape[1], max_shape[2]), dtype=np.float32)
            padding_im[:, :im_h, :im_w] = im
            data['image'] = padding_im
            if self.use_padded_im_info:
                data['im_info'][:2] = max_shape[1:3]
            if 'semantic' in data.keys() and data['semantic'] is not None:
                semantic = data['semantic']
                padding_sem = np.zeros(
                    (1, max_shape[1], max_shape[2]), dtype=np.float32)
                padding_sem[:, :im_h, :im_w] = semantic
                data['semantic'] = padding_sem
            if 'gt_segm' in data.keys() and data['gt_segm'] is not None and len(data['gt_segm']) > 0:
                gt_segm = data['gt_segm']
                padding_segm = np.zeros(
                    (gt_segm.shape[0], max_shape[1], max_shape[2]),
                    dtype=np.uint8)
                padding_segm[:, :im_h, :im_w] = gt_segm
                data['gt_segm'] = padding_segm

        return samples


class SOLOv2Pad(BaseOperator):
    def __init__(self, max_size=0):
        super(SOLOv2Pad, self).__init__()
        self.max_size = max_size

    def __call__(self, sample, context=None):
        max_size = self.max_size

        im = sample['image']
        im_c, im_h, im_w = im.shape[:]
        padding_im = np.zeros((im_c, max_size, max_size), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = im
        sample['image'] = padding_im
        return sample


class PadBatchSingle(BaseOperator):
    """
    一张图片的PadBatch
    """

    def __init__(self, use_padded_im_info=True):
        super(PadBatchSingle, self).__init__()
        self.use_padded_im_info = use_padded_im_info

    def __call__(self, max_shape, sample, context=None):
        '''
        :param max_shape:  max_shape=[3, max_h, max_w]
        :param sample:
        :param context:
        :return:
        '''
        im = sample['image']
        im_c, im_h, im_w = im.shape[:]
        padding_im = np.zeros(
            (im_c, max_shape[1], max_shape[2]), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = im  # im贴在padding_im的左上部分实现对齐
        sample['image'] = padding_im
        if self.use_padded_im_info:
            sample['im_info'][:2] = max_shape[1:3]

        return sample


class Permute(BaseOperator):
    def __init__(self, to_bgr=True, channel_first=True):
        """
        Change the channel.
        Args:
            to_bgr (bool): confirm whether to convert RGB to BGR
            channel_first (bool): confirm whether to change channel
        """
        super(Permute, self).__init__()
        self.to_bgr = to_bgr
        self.channel_first = channel_first
        if not (isinstance(self.to_bgr, bool) and
                isinstance(self.channel_first, bool)):
            raise TypeError("{}: input type is invalid.".format(self))

    def __call__(self, sample, context=None):
        samples = sample
        batch_input = True
        if not isinstance(samples, Sequence):
            batch_input = False
            samples = [samples]
        for sample in samples:
            assert 'image' in sample, "image data not found"
            for k in sample.keys():
                # hard code
                if k.startswith('image'):
                    im = sample[k]
                    if self.channel_first:
                        im = np.swapaxes(im, 1, 2)
                        im = np.swapaxes(im, 1, 0)
                    if self.to_bgr:
                        im = im[[2, 1, 0], :, :]
                    sample[k] = im
        if not batch_input:
            samples = samples[0]
        return samples


class RandomShape(BaseOperator):
    """
    Randomly reshape a batch. If random_inter is True, also randomly
    select one an interpolation algorithm [cv2.INTER_NEAREST, cv2.INTER_LINEAR,
    cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]. If random_inter is
    False, use cv2.INTER_NEAREST.
    Args:
        sizes (list): list of int, random choose a size from these
        random_inter (bool): whether to randomly interpolation, defalut true.
    """

    def __init__(self, sizes=[], random_inter=False, resize_box=False):
        super(RandomShape, self).__init__()
        self.sizes = sizes
        self.random_inter = random_inter
        self.interps = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ] if random_inter else []
        self.resize_box = resize_box

    def __call__(self, samples, context=None):
        shape = np.random.choice(self.sizes)
        method = np.random.choice(self.interps) if self.random_inter \
            else cv2.INTER_NEAREST
        for i in range(len(samples)):
            im = samples[i]['image']
            h, w = im.shape[:2]
            scale_x = float(shape) / w
            scale_y = float(shape) / h
            im = cv2.resize(
                im, None, None, fx=scale_x, fy=scale_y, interpolation=method)
            samples[i]['image'] = im
            if self.resize_box and 'gt_bbox' in samples[i] and len(samples[0][
                                                                       'gt_bbox']) > 0:
                scale_array = np.array([scale_x, scale_y] * 2, dtype=np.float32)
                samples[i]['gt_bbox'] = np.clip(samples[i]['gt_bbox'] *
                                                scale_array, 0,
                                                float(shape) - 1)
        return samples


class RandomShapeSingle(BaseOperator):
    """
    一张图片的RandomShape
    """

    def __init__(self, random_inter=False, resize_box=False):
        super(RandomShapeSingle, self).__init__()
        self.random_inter = random_inter
        self.interps = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ] if random_inter else []
        self.resize_box = resize_box

    def __call__(self, shape, sample, context=None):
        method = np.random.choice(self.interps) if self.random_inter \
            else cv2.INTER_NEAREST
        im = sample['image']
        h, w = im.shape[:2]
        scale_x = float(shape) / w
        scale_y = float(shape) / h
        im = cv2.resize(
            im, None, None, fx=scale_x, fy=scale_y, interpolation=method)
        sample['image'] = im
        if self.resize_box and 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            scale_array = np.array([scale_x, scale_y] * 2, dtype=np.float32)
            # 注意，旧版本的ppdet中float(shape)需要-1，但是PPYOLOE（新版本的ppdet）中不需要-1
            # sample['gt_bbox'] = np.clip(sample['gt_bbox'] * scale_array, 0, float(shape) - 1)
            sample['gt_bbox'] = np.clip(sample['gt_bbox'] * scale_array, 0, float(shape))
        return sample


class PadBox(BaseOperator):
    def __init__(self, num_max_boxes=50, init_bbox=None):
        """
        Pad zeros to bboxes if number of bboxes is less than num_max_boxes.
        Args:
            num_max_boxes (int): the max number of bboxes
        """
        self.num_max_boxes = num_max_boxes
        self.init_bbox = init_bbox
        super(PadBox, self).__init__()

    def __call__(self, sample, context=None):
        assert 'gt_bbox' in sample
        bbox = sample['gt_bbox']
        gt_num = min(self.num_max_boxes, len(bbox))
        num_max = self.num_max_boxes
        fields = context['fields'] if context else []
        pad_bbox = np.zeros((num_max, 4), dtype=np.float32)
        if self.init_bbox is not None:
            pad_bbox = np.ones((num_max, 4), dtype=np.float32) * self.init_bbox
        if gt_num > 0:
            pad_bbox[:gt_num, :] = bbox[:gt_num, :]
        sample['gt_bbox'] = pad_bbox
        if 'gt_class' in fields:
            pad_class = np.zeros((num_max), dtype=np.int32)
            if gt_num > 0:
                pad_class[:gt_num] = sample['gt_class'][:gt_num, 0]
            sample['gt_class'] = pad_class
        if 'gt_score' in fields:
            pad_score = np.zeros((num_max), dtype=np.float32)
            if gt_num > 0:
                pad_score[:gt_num] = sample['gt_score'][:gt_num, 0]
            sample['gt_score'] = pad_score
        # in training, for example in op ExpandImage,
        # the bbox and gt_class is expandded, but the difficult is not,
        # so, judging by it's length
        if 'is_difficult' in fields:
            pad_diff = np.zeros((num_max), dtype=np.int32)
            if gt_num > 0:
                pad_diff[:gt_num] = sample['difficult'][:gt_num, 0]
            sample['difficult'] = pad_diff
        return sample


def bbox_area(src_bbox):
    if src_bbox[2] < src_bbox[0] or src_bbox[3] < src_bbox[1]:
        return 0.
    else:
        width = src_bbox[2] - src_bbox[0]
        height = src_bbox[3] - src_bbox[1]
        return width * height


def jaccard_overlap(sample_bbox, object_bbox):
    if sample_bbox[0] >= object_bbox[2] or \
            sample_bbox[2] <= object_bbox[0] or \
            sample_bbox[1] >= object_bbox[3] or \
            sample_bbox[3] <= object_bbox[1]:
        return 0
    intersect_xmin = max(sample_bbox[0], object_bbox[0])
    intersect_ymin = max(sample_bbox[1], object_bbox[1])
    intersect_xmax = min(sample_bbox[2], object_bbox[2])
    intersect_ymax = min(sample_bbox[3], object_bbox[3])
    intersect_size = (intersect_xmax - intersect_xmin) * (
            intersect_ymax - intersect_ymin)
    sample_bbox_size = bbox_area(sample_bbox)
    object_bbox_size = bbox_area(object_bbox)
    overlap = intersect_size / (
            sample_bbox_size + object_bbox_size - intersect_size)
    return overlap


class Gt2YoloTarget(BaseOperator):
    """
    Generate YOLOv3 targets by groud truth data, this operator is only used in
    fine grained YOLOv3 loss mode
    """

    def __init__(self,
                 anchors,
                 anchor_masks,
                 downsample_ratios,
                 num_classes=80,
                 iou_thresh=1.):
        super(Gt2YoloTarget, self).__init__()
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.downsample_ratios = downsample_ratios
        self.num_classes = num_classes
        self.iou_thresh = iou_thresh

    def __call__(self, samples, context=None):
        assert len(self.anchor_masks) == len(self.downsample_ratios), \
            "anchor_masks', and 'downsample_ratios' should have same length."

        h, w = samples[0]['image'].shape[1:3]
        an_hw = np.array(self.anchors) / np.array([[w, h]])
        for sample in samples:
            # im, gt_bbox, gt_class, gt_score = sample
            im = sample['image']
            gt_bbox = sample['gt_bbox']
            gt_class = sample['gt_class']
            gt_score = sample['gt_score']
            for i, (
                    mask, downsample_ratio
            ) in enumerate(zip(self.anchor_masks, self.downsample_ratios)):
                grid_h = int(h / downsample_ratio)
                grid_w = int(w / downsample_ratio)
                target = np.zeros(
                    (len(mask), 6 + self.num_classes, grid_h, grid_w),
                    dtype=np.float32)
                for b in range(gt_bbox.shape[0]):
                    gx, gy, gw, gh = gt_bbox[b, :]
                    cls = gt_class[b]
                    score = gt_score[b]
                    if gw <= 0. or gh <= 0. or score <= 0.:
                        continue

                    # find best match anchor index
                    best_iou = 0.
                    best_idx = -1
                    for an_idx in range(an_hw.shape[0]):
                        iou = jaccard_overlap(
                            [0., 0., gw, gh],
                            [0., 0., an_hw[an_idx, 0], an_hw[an_idx, 1]])
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = an_idx

                    gi = int(gx * grid_w)
                    gj = int(gy * grid_h)

                    # gtbox should be regresed in this layes if best match
                    # anchor index in anchor mask of this layer
                    if best_idx in mask:
                        best_n = mask.index(best_idx)

                        # x, y, w, h, scale
                        target[best_n, 0, gj, gi] = gx * grid_w - gi
                        target[best_n, 1, gj, gi] = gy * grid_h - gj
                        target[best_n, 2, gj, gi] = np.log(
                            gw * w / self.anchors[best_idx][0])
                        target[best_n, 3, gj, gi] = np.log(
                            gh * h / self.anchors[best_idx][1])
                        target[best_n, 4, gj, gi] = 2.0 - gw * gh

                        # objectness record gt_score
                        target[best_n, 5, gj, gi] = score

                        # classification
                        target[best_n, 6 + cls, gj, gi] = 1.

                    # For non-matched anchors, calculate the target if the iou
                    # between anchor and gt is larger than iou_thresh
                    if self.iou_thresh < 1:
                        for idx, mask_i in enumerate(mask):
                            if mask_i == best_idx: continue
                            iou = jaccard_overlap(
                                [0., 0., gw, gh],
                                [0., 0., an_hw[mask_i, 0], an_hw[mask_i, 1]])
                            if iou > self.iou_thresh:
                                # x, y, w, h, scale
                                target[idx, 0, gj, gi] = gx * grid_w - gi
                                target[idx, 1, gj, gi] = gy * grid_h - gj
                                target[idx, 2, gj, gi] = np.log(
                                    gw * w / self.anchors[mask_i][0])
                                target[idx, 3, gj, gi] = np.log(
                                    gh * h / self.anchors[mask_i][1])
                                target[idx, 4, gj, gi] = 2.0 - gw * gh

                                # objectness record gt_score
                                target[idx, 5, gj, gi] = score

                                # classification
                                target[idx, 6 + cls, gj, gi] = 1.
                sample['target{}'.format(i)] = target
        return samples


class Gt2YoloTargetSingle(BaseOperator):
    """
    一张图片的Gt2YoloTarget
    """

    def __init__(self,
                 anchors,
                 anchor_masks,
                 downsample_ratios,
                 num_classes=80,
                 iou_thresh=1.):
        super(Gt2YoloTargetSingle, self).__init__()
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.downsample_ratios = downsample_ratios
        self.num_classes = num_classes
        self.iou_thresh = iou_thresh

    def __call__(self, sample, context=None):
        assert len(self.anchor_masks) == len(self.downsample_ratios), \
            "anchor_masks', and 'downsample_ratios' should have same length."

        h, w = sample['image'].shape[1:3]
        an_hw = np.array(self.anchors) / np.array([[w, h]])

        # im, gt_bbox, gt_class, gt_score = sample
        im = sample['image']
        gt_bbox = sample['gt_bbox']
        gt_class = sample['gt_class']
        gt_score = sample['gt_score']
        for i, (
                mask, downsample_ratio
        ) in enumerate(zip(self.anchor_masks, self.downsample_ratios)):
            grid_h = int(h / downsample_ratio)
            grid_w = int(w / downsample_ratio)
            target = np.zeros(
                (len(mask), 6 + self.num_classes, grid_h, grid_w),
                dtype=np.float32)
            for b in range(gt_bbox.shape[0]):
                gx, gy, gw, gh = gt_bbox[b, :]
                cls = gt_class[b]
                score = gt_score[b]
                if gw <= 0. or gh <= 0. or score <= 0.:
                    continue

                # find best match anchor index
                best_iou = 0.
                best_idx = -1
                for an_idx in range(an_hw.shape[0]):
                    iou = jaccard_overlap(
                        [0., 0., gw, gh],
                        [0., 0., an_hw[an_idx, 0], an_hw[an_idx, 1]])
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = an_idx

                gi = int(gx * grid_w)
                gj = int(gy * grid_h)

                # gtbox should be regresed in this layes if best match
                # anchor index in anchor mask of this layer
                if best_idx in mask:
                    best_n = mask.index(best_idx)

                    # x, y, w, h, scale
                    target[best_n, 0, gj, gi] = gx * grid_w - gi
                    target[best_n, 1, gj, gi] = gy * grid_h - gj
                    target[best_n, 2, gj, gi] = np.log(
                        gw * w / self.anchors[best_idx][0])
                    target[best_n, 3, gj, gi] = np.log(
                        gh * h / self.anchors[best_idx][1])
                    target[best_n, 4, gj, gi] = 2.0 - gw * gh

                    # objectness record gt_score
                    target[best_n, 5, gj, gi] = score

                    # classification
                    target[best_n, 6 + cls, gj, gi] = 1.

                # For non-matched anchors, calculate the target if the iou
                # between anchor and gt is larger than iou_thresh
                if self.iou_thresh < 1:
                    for idx, mask_i in enumerate(mask):
                        if mask_i == best_idx: continue
                        iou = jaccard_overlap(
                            [0., 0., gw, gh],
                            [0., 0., an_hw[mask_i, 0], an_hw[mask_i, 1]])
                        if iou > self.iou_thresh:
                            # x, y, w, h, scale
                            target[idx, 0, gj, gi] = gx * grid_w - gi
                            target[idx, 1, gj, gi] = gy * grid_h - gj
                            target[idx, 2, gj, gi] = np.log(
                                gw * w / self.anchors[mask_i][0])
                            target[idx, 3, gj, gi] = np.log(
                                gh * h / self.anchors[mask_i][1])
                            target[idx, 4, gj, gi] = 2.0 - gw * gh

                            # objectness record gt_score
                            target[idx, 5, gj, gi] = score

                            # classification
                            target[idx, 6 + cls, gj, gi] = 1.
            sample['target{}'.format(i)] = target
        return sample


class PadGT(BaseOperator):
    def __init__(self, return_gt_mask=True):
        super(PadGT, self).__init__()
        self.return_gt_mask = return_gt_mask

    def __call__(self, samples, context=None):
        num_max_boxes = max([len(s['gt_bbox']) for s in samples])
        for sample in samples:
            if self.return_gt_mask:
                sample['pad_gt_mask'] = np.zeros(
                    (num_max_boxes, 1), dtype=np.float32)
            if num_max_boxes == 0:
                continue

            num_gt = len(sample['gt_bbox'])
            pad_gt_class = np.zeros((num_max_boxes, 1), dtype=np.int32)
            pad_gt_bbox = np.zeros((num_max_boxes, 4), dtype=np.float32)
            if num_gt > 0:
                pad_gt_class[:num_gt] = sample['gt_class']
                pad_gt_bbox[:num_gt] = sample['gt_bbox']
            sample['gt_class'] = pad_gt_class
            sample['gt_bbox'] = pad_gt_bbox
            # pad_gt_mask
            if 'pad_gt_mask' in sample:
                sample['pad_gt_mask'][:num_gt] = 1
            # gt_score
            if 'gt_score' in sample:
                pad_gt_score = np.zeros((num_max_boxes, 1), dtype=np.float32)
                if num_gt > 0:
                    pad_gt_score[:num_gt] = sample['gt_score']
                sample['gt_score'] = pad_gt_score
            if 'is_crowd' in sample:
                pad_is_crowd = np.zeros((num_max_boxes, 1), dtype=np.int32)
                if num_gt > 0:
                    pad_is_crowd[:num_gt] = sample['is_crowd']
                sample['is_crowd'] = pad_is_crowd
            if 'difficult' in sample:
                pad_diff = np.zeros((num_max_boxes, 1), dtype=np.int32)
                if num_gt > 0:
                    pad_diff[:num_gt] = sample['difficult']
                sample['difficult'] = pad_diff
        return samples


class PadGTSingle(BaseOperator):
    def __init__(self, num_max_boxes=200, return_gt_mask=True):
        super(PadGTSingle, self).__init__()
        self.num_max_boxes = num_max_boxes
        self.return_gt_mask = return_gt_mask

    def __call__(self, sample, context=None):
        samples = [sample]
        num_max_boxes = self.num_max_boxes
        for sample in samples:
            if self.return_gt_mask:
                sample['pad_gt_mask'] = np.zeros(
                    (num_max_boxes, 1), dtype=np.float32)
            if num_max_boxes == 0:
                continue

            num_gt = len(sample['gt_bbox'])
            # miemie2013 add it.
            num_gt = min(num_gt, num_max_boxes)
            pad_gt_class = np.zeros((num_max_boxes, 1), dtype=np.int32)
            pad_gt_bbox = np.zeros((num_max_boxes, 4), dtype=np.float32)
            if num_gt > 0:
                pad_gt_class[:num_gt] = sample['gt_class'][:num_gt]
                pad_gt_bbox[:num_gt] = sample['gt_bbox'][:num_gt]
            sample['gt_class'] = pad_gt_class
            sample['gt_bbox'] = pad_gt_bbox
            # pad_gt_mask
            if 'pad_gt_mask' in sample:
                sample['pad_gt_mask'][:num_gt] = 1
            # gt_score
            if 'gt_score' in sample:
                pad_gt_score = np.zeros((num_max_boxes, 1), dtype=np.float32)
                if num_gt > 0:
                    pad_gt_score[:num_gt] = sample['gt_score'][:num_gt]
                sample['gt_score'] = pad_gt_score
            if 'is_crowd' in sample:
                pad_is_crowd = np.zeros((num_max_boxes, 1), dtype=np.int32)
                if num_gt > 0:
                    pad_is_crowd[:num_gt] = sample['is_crowd'][:num_gt]
                sample['is_crowd'] = pad_is_crowd
            if 'difficult' in sample:
                pad_diff = np.zeros((num_max_boxes, 1), dtype=np.int32)
                if num_gt > 0:
                    pad_diff[:num_gt] = sample['difficult'][:num_gt]
                sample['difficult'] = pad_diff
        return samples[0]


class Gt2FCOSTarget(BaseOperator):
    """
    Generate FCOS targets by groud truth data
    """

    def __init__(self,
                 object_sizes_boundary,
                 center_sampling_radius,
                 downsample_ratios,
                 norm_reg_targets=False):
        super(Gt2FCOSTarget, self).__init__()
        self.center_sampling_radius = center_sampling_radius
        self.downsample_ratios = downsample_ratios
        self.INF = np.inf
        self.object_sizes_boundary = [-1] + object_sizes_boundary + [self.INF]
        object_sizes_of_interest = []
        for i in range(len(self.object_sizes_boundary) - 1):
            object_sizes_of_interest.append([
                self.object_sizes_boundary[i], self.object_sizes_boundary[i + 1]
            ])
        self.object_sizes_of_interest = object_sizes_of_interest
        self.norm_reg_targets = norm_reg_targets

    def _compute_points(self, w, h):
        """
        compute the corresponding points in each feature map
        :param h: image height
        :param w: image width
        :return: points from all feature map
        """
        locations = []
        # 从小感受野stride=8遍历到大感受野stride=128。location.shape=[格子行数*格子列数, 2]，存放的是每个格子的中心点的坐标。格子顺序是第一行从左到右，第二行从左到右，...
        for stride in self.downsample_ratios:
            shift_x = np.arange(0, w, stride).astype(np.float32)
            shift_y = np.arange(0, h, stride).astype(np.float32)
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            shift_x = shift_x.flatten()
            shift_y = shift_y.flatten()
            location = np.stack([shift_x, shift_y], axis=1) + stride // 2
            locations.append(location)
        num_points_each_level = [len(location) for location in
                                 locations]  # num_points_each_level=[stride=8感受野格子数, ..., stride=128感受野格子数]
        locations = np.concatenate(locations, axis=0)
        return locations, num_points_each_level

    def _convert_xywh2xyxy(self, gt_bbox, w, h):
        """
        convert the bounding box from style xywh to xyxy
        :param gt_bbox: bounding boxes normalized into [0, 1]
        :param w: image width
        :param h: image height
        :return: bounding boxes in xyxy style
        """
        bboxes = gt_bbox.copy()
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * w
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * h
        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
        return bboxes

    def _check_inside_boxes_limited(self, gt_bbox, xs, ys,
                                    num_points_each_level):
        """
        check if points is within the clipped boxes
        :param gt_bbox: bounding boxes
        :param xs: horizontal coordinate of points
        :param ys: vertical coordinate of points
        :return: the mask of points is within gt_box or not
        """
        bboxes = np.reshape(  # [gt数, 4] -> [1, gt数, 4]
            gt_bbox, newshape=[1, gt_bbox.shape[0], gt_bbox.shape[1]])
        bboxes = np.tile(bboxes, reps=[xs.shape[0], 1, 1])  # [所有格子数, gt数, 4]   gt坐标
        ct_x = (bboxes[:, :, 0] + bboxes[:, :, 2]) / 2  # [所有格子数, gt数]      gt中心点x
        ct_y = (bboxes[:, :, 1] + bboxes[:, :, 3]) / 2  # [所有格子数, gt数]      gt中心点y
        beg = 0  # 开始=0
        clipped_box = bboxes.copy()  # [所有格子数, gt数, 4]   gt坐标，限制gt的边长，最大只能是1.5 * 2 = 3个格子边长
        for lvl, stride in enumerate(self.downsample_ratios):  # 遍历每个感受野，从 stride=8的感受野 到 stride=128的感受野
            end = beg + num_points_each_level[lvl]  # 结束=开始+这个感受野的格子数
            stride_exp = self.center_sampling_radius * stride  # stride_exp = 1.5 * 这个感受野的stride(的格子边长)
            clipped_box[beg:end, :, 0] = np.maximum(
                bboxes[beg:end, :, 0], ct_x[beg:end, :] - stride_exp)  # 限制gt的边长，最大只能是1.5 * 2 = 3个格子边长
            clipped_box[beg:end, :, 1] = np.maximum(
                bboxes[beg:end, :, 1], ct_y[beg:end, :] - stride_exp)  # 限制gt的边长，最大只能是1.5 * 2 = 3个格子边长
            clipped_box[beg:end, :, 2] = np.minimum(
                bboxes[beg:end, :, 2], ct_x[beg:end, :] + stride_exp)  # 限制gt的边长，最大只能是1.5 * 2 = 3个格子边长
            clipped_box[beg:end, :, 3] = np.minimum(
                bboxes[beg:end, :, 3], ct_y[beg:end, :] + stride_exp)  # 限制gt的边长，最大只能是1.5 * 2 = 3个格子边长
            beg = end
        # xs  [所有格子数, gt数]， 所有格子中心点的横坐标重复 gt数 次
        l_res = xs - clipped_box[:, :, 0]  # [所有格子数, gt数]  所有格子需要学习 gt数 个l
        r_res = clipped_box[:, :, 2] - xs  # [所有格子数, gt数]  所有格子需要学习 gt数 个r
        t_res = ys - clipped_box[:, :, 1]  # [所有格子数, gt数]  所有格子需要学习 gt数 个t
        b_res = clipped_box[:, :, 3] - ys  # [所有格子数, gt数]  所有格子需要学习 gt数 个b
        clipped_box_reg_targets = np.stack([l_res, t_res, r_res, b_res], axis=2)  # [所有格子数, gt数, 4]  所有格子需要学习 gt数 个lrtb
        inside_gt_box = np.min(clipped_box_reg_targets,
                               axis=2) > 0  # [所有格子数, gt数]  需要学习的lrtb如果都>0，表示格子被选中。即只选取中心点落在gt内的格子。
        return inside_gt_box

    def __call__(self, samples, context=None):
        assert len(self.object_sizes_of_interest) == len(self.downsample_ratios), \
            "object_sizes_of_interest', and 'downsample_ratios' should have same length."

        for sample in samples:
            # im, gt_bbox, gt_class, gt_score = sample
            im = sample['image']
            im_info = sample['im_info']
            bboxes = sample['gt_bbox']
            gt_class = sample['gt_class']
            gt_score = sample['gt_score']
            no_gt = False
            if len(bboxes) == 0:  # 如果没有gt，虚构一个gt为了后面不报错。
                no_gt = True
                bboxes = np.array([[0, 0, 100, 100]]).astype(np.float32)
                gt_class = np.array([[0]]).astype(np.int32)
                gt_score = np.array([[1]]).astype(np.float32)
                # print('nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnone')
            # bboxes的横坐标变成缩放后图片中对应物体的横坐标
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * np.floor(im_info[1]) / \
                                np.floor(im_info[1] / im_info[2])
            # bboxes的纵坐标变成缩放后图片中对应物体的纵坐标
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * np.floor(im_info[0]) / \
                                np.floor(im_info[0] / im_info[2])
            # calculate the locations
            h, w = sample['image'].shape[1:3]  # h w是这一批所有图片对齐后的高宽。
            points, num_points_each_level = self._compute_points(w,
                                                                 h)  # points是所有格子中心点的坐标，num_points_each_level=[stride=8感受野格子数, ..., stride=128感受野格子数]
            object_scale_exp = []
            for i, num_pts in enumerate(num_points_each_level):  # 遍历每个感受野格子数
                object_scale_exp.append(  # 边界self.object_sizes_of_interest[i] 重复 num_pts=格子数 次
                    np.tile(
                        np.array([self.object_sizes_of_interest[i]]),
                        reps=[num_pts, 1]))
            object_scale_exp = np.concatenate(object_scale_exp, axis=0)

            gt_area = (bboxes[:, 2] - bboxes[:, 0]) * (  # [gt数, ]   所有gt的面积
                    bboxes[:, 3] - bboxes[:, 1])
            xs, ys = points[:, 0], points[:, 1]  # 所有格子中心点的横坐标、纵坐标
            xs = np.reshape(xs, newshape=[xs.shape[0], 1])  # [所有格子数, 1]
            xs = np.tile(xs, reps=[1, bboxes.shape[0]])  # [所有格子数, gt数]， 所有格子中心点的横坐标重复 gt数 次
            ys = np.reshape(ys, newshape=[ys.shape[0], 1])  # [所有格子数, 1]
            ys = np.tile(ys, reps=[1, bboxes.shape[0]])  # [所有格子数, gt数]， 所有格子中心点的纵坐标重复 gt数 次

            l_res = xs - bboxes[:,
                         0]  # [所有格子数, gt数] - [gt数, ] = [所有格子数, gt数]     结果是所有格子中心点的横坐标 分别减去 所有gt左上角的横坐标，即所有格子需要学习 gt数 个l
            r_res = bboxes[:, 2] - xs  # 所有格子需要学习 gt数 个r
            t_res = ys - bboxes[:, 1]  # 所有格子需要学习 gt数 个t
            b_res = bboxes[:, 3] - ys  # 所有格子需要学习 gt数 个b
            reg_targets = np.stack([l_res, t_res, r_res, b_res], axis=2)  # [所有格子数, gt数, 4]   所有格子需要学习 gt数 个lrtb
            if self.center_sampling_radius > 0:
                # [所有格子数, gt数]    True表示格子中心点（锚点）落在gt内（gt是被限制边长后的gt）。
                # FCOS首先将gt框内的锚点（格子中心点）视为候选正样本，然后根据为每个金字塔等级定义的比例范围从候选中选择最终的正样本（而且是负责预测gt里面积最小的），最后那些未选择的锚点为负样本。
                # (1)第1个正负样本判断依据
                is_inside_box = self._check_inside_boxes_limited(
                    bboxes, xs, ys, num_points_each_level)
            else:
                # [所有格子数, gt数]    True表示格子中心点（锚点）落在gt内。
                # FCOS首先将gt框内的锚点（格子中心点）视为候选正样本，然后根据为每个金字塔等级定义的比例范围从候选中选择最终的正样本（而且是负责预测gt里面积最小的），最后那些未选择的锚点为负样本。
                # (1)第1个正负样本判断依据
                is_inside_box = np.min(reg_targets, axis=2) > 0
            # check if the targets is inside the corresponding level
            max_reg_targets = np.max(reg_targets, axis=2)  # [所有格子数, gt数]   所有格子需要学习 gt数 个lrtb   中的最大值
            lower_bound = np.tile(  # [所有格子数, gt数]   下限重复 gt数 次
                np.expand_dims(
                    object_scale_exp[:, 0], axis=1),
                reps=[1, max_reg_targets.shape[1]])
            high_bound = np.tile(  # [所有格子数, gt数]   上限重复 gt数 次
                np.expand_dims(
                    object_scale_exp[:, 1], axis=1),
                reps=[1, max_reg_targets.shape[1]])

            # [所有格子数, gt数]   最大回归值如果位于区间内，就为True
            # (2)第2个正负样本判断依据
            is_match_current_level = \
                (max_reg_targets > lower_bound) & \
                (max_reg_targets < high_bound)
            # [所有格子数, gt数]   所有gt的面积
            points2gtarea = np.tile(
                np.expand_dims(
                    gt_area, axis=0), reps=[xs.shape[0], 1])
            points2gtarea[
                is_inside_box == 0] = self.INF  # 格子中心点落在gt外的（即负样本），需要学习的面积置为无穷。     这是为了points2gtarea.min(axis=1)时，若某格子有最终正样本，那么就应该不让负样本的面积影响到判断。
            points2gtarea[
                is_match_current_level == 0] = self.INF  # 最大回归值如果位于区间外（即负样本），需要学习的面积置为无穷。 这是为了points2gtarea.min(axis=1)时，若某格子有最终正样本，那么就应该不让负样本的面积影响到判断。
            points2min_area = points2gtarea.min(axis=1)  # [所有格子数, ]   所有格子需要学习 gt数 个面积  中的最小值
            points2min_area_ind = points2gtarea.argmin(axis=1)  # [所有格子数, ]   所有格子需要学习 gt数 个面积  中的最小值的下标
            labels = gt_class[points2min_area_ind] + 1  # [所有格子数, 1]   所有格子需要学习 的类别id，学习的是gt中面积最小值的的类别id
            labels[points2min_area == self.INF] = 0  # [所有格子数, 1]   负样本的points2min_area肯定是self.INF，这里将负样本需要学习 的类别id 置为0
            reg_targets = reg_targets[
                range(xs.shape[0]), points2min_area_ind]  # [所有格子数, 4]   所有格子需要学习 的 lrtb（负责预测gt里面积最小的）
            ctn_targets = np.sqrt((reg_targets[:, [0, 2]].min(axis=1) / \
                                   reg_targets[:, [0, 2]].max(axis=1)) * \
                                  (reg_targets[:, [1, 3]].min(axis=1) / \
                                   reg_targets[:, [1, 3]].max(axis=1))).astype(
                np.float32)  # [所有格子数, ]  所有格子需要学习的centerness
            ctn_targets = np.reshape(
                ctn_targets, newshape=[ctn_targets.shape[0], 1])  # [所有格子数, 1]  所有格子需要学习的centerness
            ctn_targets[labels <= 0] = 0  # 负样本需要学习的centerness置为0
            pos_ind = np.nonzero(
                labels != 0)  # tuple=( ndarray(shape=[正样本数, ]), ndarray(shape=[正样本数, ]) )   即正样本在labels中的下标，因为labels是2维的，所以一个正样本有2个下标。
            reg_targets_pos = reg_targets[pos_ind[0], :]  # [正样本数, 4]   正样本格子需要学习 的 lrtb
            split_sections = []  # 每一个感受野 最后一个格子 在reg_targets中的位置（第一维的位置）
            beg = 0
            for lvl in range(len(num_points_each_level)):
                end = beg + num_points_each_level[lvl]
                split_sections.append(end)
                beg = end
            if no_gt:  # 如果没有gt，labels里全部置为0（背景的类别id是0）即表示所有格子都是负样本
                labels[:, :] = 0
            labels_by_level = np.split(labels, split_sections, axis=0)  # 一个list，根据split_sections切分，各个感受野的target切分开来。
            reg_targets_by_level = np.split(reg_targets, split_sections,
                                            axis=0)  # 一个list，根据split_sections切分，各个感受野的target切分开来。
            ctn_targets_by_level = np.split(ctn_targets, split_sections,
                                            axis=0)  # 一个list，根据split_sections切分，各个感受野的target切分开来。

            # 最后一步是reshape，和格子的位置对应上。
            for lvl in range(len(self.downsample_ratios)):
                grid_w = int(np.ceil(w / self.downsample_ratios[lvl]))  # 格子列数
                grid_h = int(np.ceil(h / self.downsample_ratios[lvl]))  # 格子行数
                if self.norm_reg_targets:  # 是否将reg目标归一化，配置里是True
                    sample['reg_target{}'.format(lvl)] = \
                        np.reshape(
                            reg_targets_by_level[lvl] / \
                            self.downsample_ratios[lvl],  # 归一化方式是除以格子边长（即下采样倍率）
                            newshape=[grid_h, grid_w, 4])  # reshape成[grid_h, grid_w, 4]
                else:
                    sample['reg_target{}'.format(lvl)] = np.reshape(
                        reg_targets_by_level[lvl],
                        newshape=[grid_h, grid_w, 4])
                sample['labels{}'.format(lvl)] = np.reshape(
                    labels_by_level[lvl], newshape=[grid_h, grid_w, 1])  # reshape成[grid_h, grid_w, 1]
                sample['centerness{}'.format(lvl)] = np.reshape(
                    ctn_targets_by_level[lvl], newshape=[grid_h, grid_w, 1])  # reshape成[grid_h, grid_w, 1]
        return samples


class Gt2FCOSTargetSingle(BaseOperator):
    """
    一张图片的Gt2FCOSTarget
    """

    def __init__(self,
                 object_sizes_boundary,
                 center_sampling_radius,
                 downsample_ratios,
                 norm_reg_targets=False):
        super(Gt2FCOSTargetSingle, self).__init__()
        self.center_sampling_radius = center_sampling_radius
        self.downsample_ratios = downsample_ratios
        self.INF = np.inf
        self.object_sizes_boundary = [-1] + object_sizes_boundary + [self.INF]
        object_sizes_of_interest = []
        for i in range(len(self.object_sizes_boundary) - 1):
            object_sizes_of_interest.append([
                self.object_sizes_boundary[i], self.object_sizes_boundary[i + 1]
            ])
        self.object_sizes_of_interest = object_sizes_of_interest
        self.norm_reg_targets = norm_reg_targets

    def _compute_points(self, w, h):
        """
        compute the corresponding points in each feature map
        :param h: image height
        :param w: image width
        :return: points from all feature map
        """
        locations = []
        # 从小感受野stride=8遍历到大感受野stride=128。location.shape=[格子行数*格子列数, 2]，存放的是每个格子的中心点的坐标。格子顺序是第一行从左到右，第二行从左到右，...
        for stride in self.downsample_ratios:
            shift_x = np.arange(0, w, stride).astype(np.float32)
            shift_y = np.arange(0, h, stride).astype(np.float32)
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            shift_x = shift_x.flatten()
            shift_y = shift_y.flatten()

            '''
            location.shape = [grid_h*grid_w, 2]
            如果stride=8，
            location = [[4, 4], [12, 4], [20, 4], ...],  这一个输出层的格子的中心点的xy坐标。格子顺序是第一行从左到右，第二行从左到右，...
            即location = [[0.5*stride, 0.5*stride], [1.5*stride, 0.5*stride], [2.5*stride, 0.5*stride], ...]

            如果stride=16，
            location = [[8, 8], [24, 8], [40, 8], ...],  这一个输出层的格子的中心点的xy坐标。格子顺序是第一行从左到右，第二行从左到右，...
            即location = [[0.5*stride, 0.5*stride], [1.5*stride, 0.5*stride], [2.5*stride, 0.5*stride], ...]

            ...
            '''
            location = np.stack([shift_x, shift_y], axis=1) + stride // 2
            locations.append(location)
        num_points_each_level = [len(location) for location in
                                 locations]  # num_points_each_level=[stride=8感受野格子数, ..., stride=128感受野格子数]
        locations = np.concatenate(locations, axis=0)
        return locations, num_points_each_level

    def _convert_xywh2xyxy(self, gt_bbox, w, h):
        """
        convert the bounding box from style xywh to xyxy
        :param gt_bbox: bounding boxes normalized into [0, 1]
        :param w: image width
        :param h: image height
        :return: bounding boxes in xyxy style
        """
        bboxes = gt_bbox.copy()
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * w
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * h
        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
        return bboxes

    def _check_inside_boxes_limited(self, gt_bbox, xs, ys,
                                    num_points_each_level):
        """
        check if points is within the clipped boxes
        :param gt_bbox: bounding boxes
        :param xs: horizontal coordinate of points
        :param ys: vertical coordinate of points
        :return: the mask of points is within gt_box or not
        """
        bboxes = np.reshape(  # [gt数, 4] -> [1, gt数, 4]
            gt_bbox, newshape=[1, gt_bbox.shape[0], gt_bbox.shape[1]])
        bboxes = np.tile(bboxes, reps=[xs.shape[0], 1, 1])  # [所有格子数, gt数, 4]   gt坐标。可以看出，每1个gt都会参与到fpn的所有输出特征图。
        ct_x = (bboxes[:, :, 0] + bboxes[:, :, 2]) / 2  # [所有格子数, gt数]      gt中心点x
        ct_y = (bboxes[:, :, 1] + bboxes[:, :, 3]) / 2  # [所有格子数, gt数]      gt中心点y
        beg = 0  # 开始=0

        # clipped_box即修改之后的gt，和原始gt（bboxes）的中心点相同，但是边长却修改成最大只能是1.5 * 2 = 3个格子边长
        clipped_box = bboxes.copy()  # [所有格子数, gt数, 4]   gt坐标，限制gt的边长，最大只能是1.5 * 2 = 3个格子边长
        for lvl, stride in enumerate(self.downsample_ratios):  # 遍历每个感受野，从 stride=8的感受野 到 stride=128的感受野
            end = beg + num_points_each_level[lvl]  # 结束=开始+这个感受野的格子数
            stride_exp = self.center_sampling_radius * stride  # stride_exp = 1.5 * 这个感受野的stride(的格子边长)
            clipped_box[beg:end, :, 0] = np.maximum(
                bboxes[beg:end, :, 0], ct_x[beg:end, :] - stride_exp)  # 限制gt的边长，最大只能是1.5 * 2 = 3个格子边长
            clipped_box[beg:end, :, 1] = np.maximum(
                bboxes[beg:end, :, 1], ct_y[beg:end, :] - stride_exp)  # 限制gt的边长，最大只能是1.5 * 2 = 3个格子边长
            clipped_box[beg:end, :, 2] = np.minimum(
                bboxes[beg:end, :, 2], ct_x[beg:end, :] + stride_exp)  # 限制gt的边长，最大只能是1.5 * 2 = 3个格子边长
            clipped_box[beg:end, :, 3] = np.minimum(
                bboxes[beg:end, :, 3], ct_y[beg:end, :] + stride_exp)  # 限制gt的边长，最大只能是1.5 * 2 = 3个格子边长
            beg = end

        # 如果格子中心点落在clipped_box代表的gt框内，那么这个格子就被选为候选正样本。

        # xs  [所有格子数, gt数]， 所有格子中心点的横坐标重复 gt数 次
        l_res = xs - clipped_box[:, :, 0]  # [所有格子数, gt数]  所有格子需要学习 gt数 个l
        r_res = clipped_box[:, :, 2] - xs  # [所有格子数, gt数]  所有格子需要学习 gt数 个r
        t_res = ys - clipped_box[:, :, 1]  # [所有格子数, gt数]  所有格子需要学习 gt数 个t
        b_res = clipped_box[:, :, 3] - ys  # [所有格子数, gt数]  所有格子需要学习 gt数 个b
        clipped_box_reg_targets = np.stack([l_res, t_res, r_res, b_res], axis=2)  # [所有格子数, gt数, 4]  所有格子需要学习 gt数 个lrtb
        inside_gt_box = np.min(clipped_box_reg_targets,
                               axis=2) > 0  # [所有格子数, gt数]  需要学习的lrtb如果都>0，表示格子被选中。即只选取中心点落在gt内的格子。
        return inside_gt_box

    def __call__(self, sample, context=None):
        assert len(self.object_sizes_of_interest) == len(self.downsample_ratios), \
            "object_sizes_of_interest', and 'downsample_ratios' should have same length."

        # im, gt_bbox, gt_class, gt_score = sample
        im = sample['image']  # [3, pad_h, pad_w]
        im_info = sample['im_info']  # [3, ]  分别是resize_h, resize_w, im_scale
        bboxes = sample['gt_bbox']  # [m, 4]  x0y0x1y1格式
        gt_class = sample['gt_class']  # [m, 1]
        gt_score = sample['gt_score']  # [m, 1]
        no_gt = False
        if len(bboxes) == 0:  # 如果没有gt，虚构一个gt为了后面不报错。
            no_gt = True
            bboxes = np.array([[0, 0, 100, 100]]).astype(np.float32)
            gt_class = np.array([[0]]).astype(np.int32)
            gt_score = np.array([[1]]).astype(np.float32)
            # print('nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnone')
        # bboxes的横坐标变成缩放后图片中对应物体的横坐标
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * np.floor(im_info[1]) / \
                            np.floor(im_info[1] / im_info[2])
        # bboxes的纵坐标变成缩放后图片中对应物体的纵坐标
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * np.floor(im_info[0]) / \
                            np.floor(im_info[0] / im_info[2])
        # calculate the locations
        h, w = sample['image'].shape[1:3]  # h w是这一批所有图片对齐后的高宽。
        points, num_points_each_level = self._compute_points(w,
                                                             h)  # points是所有格子中心点的坐标，num_points_each_level=[stride=8感受野格子数, ..., stride=128感受野格子数]
        object_scale_exp = []
        for i, num_pts in enumerate(num_points_each_level):  # 遍历每个感受野格子数
            object_scale_exp.append(  # 边界self.object_sizes_of_interest[i] 重复 num_pts=格子数 次
                np.tile(
                    np.array([self.object_sizes_of_interest[i]]),
                    reps=[num_pts, 1]))
        object_scale_exp = np.concatenate(object_scale_exp, axis=0)

        gt_area = (bboxes[:, 2] - bboxes[:, 0]) * (  # [gt数, ]   所有gt的面积
                bboxes[:, 3] - bboxes[:, 1])
        xs, ys = points[:, 0], points[:, 1]  # 所有格子中心点的横坐标、纵坐标
        xs = np.reshape(xs, newshape=[xs.shape[0], 1])  # [所有格子数, 1]
        xs = np.tile(xs, reps=[1, bboxes.shape[0]])  # [所有格子数, gt数]， 所有格子中心点的横坐标重复 gt数 次
        ys = np.reshape(ys, newshape=[ys.shape[0], 1])  # [所有格子数, 1]
        ys = np.tile(ys, reps=[1, bboxes.shape[0]])  # [所有格子数, gt数]， 所有格子中心点的纵坐标重复 gt数 次

        l_res = xs - bboxes[:,
                     0]  # [所有格子数, gt数] - [gt数, ] = [所有格子数, gt数]     结果是所有格子中心点的横坐标 分别减去 所有gt左上角的横坐标，即所有格子需要学习 gt数 个l
        r_res = bboxes[:, 2] - xs  # 所有格子需要学习 gt数 个r
        t_res = ys - bboxes[:, 1]  # 所有格子需要学习 gt数 个t
        b_res = bboxes[:, 3] - ys  # 所有格子需要学习 gt数 个b
        reg_targets = np.stack([l_res, t_res, r_res, b_res], axis=2)  # [所有格子数, gt数, 4]   所有格子需要学习 gt数 个lrtb
        if self.center_sampling_radius > 0:
            # [所有格子数, gt数]    True表示格子中心点（锚点）落在gt内（gt是被限制边长后的gt）。
            # FCOS首先将gt框内的锚点（格子中心点）视为候选正样本，然后根据为每个金字塔等级定义的比例范围从候选中选择最终的正样本（而且是负责预测gt里面积最小的），最后那些未选择的锚点为负样本。
            # (1)第1个正负样本判断依据

            # 这里是使用gt的中心区域判断格子中心点是否在gt框内。这样做会减少很多中心度很低的低质量正样本。
            is_inside_box = self._check_inside_boxes_limited(
                bboxes, xs, ys, num_points_each_level)
        else:
            # [所有格子数, gt数]    True表示格子中心点（锚点）落在gt内。
            # FCOS首先将gt框内的锚点（格子中心点）视为候选正样本，然后根据为每个金字塔等级定义的比例范围从候选中选择最终的正样本（而且是负责预测gt里面积最小的），最后那些未选择的锚点为负样本。
            # (1)第1个正负样本判断依据

            # 这里是使用gt的完整区域判断格子中心点是否在gt框内。这样做会增加很多中心度很低的低质量正样本。
            is_inside_box = np.min(reg_targets, axis=2) > 0
        # check if the targets is inside the corresponding level
        max_reg_targets = np.max(reg_targets, axis=2)  # [所有格子数, gt数]   所有格子需要学习 gt数 个lrtb   中的最大值
        lower_bound = np.tile(  # [所有格子数, gt数]   下限重复 gt数 次
            np.expand_dims(
                object_scale_exp[:, 0], axis=1),
            reps=[1, max_reg_targets.shape[1]])
        high_bound = np.tile(  # [所有格子数, gt数]   上限重复 gt数 次
            np.expand_dims(
                object_scale_exp[:, 1], axis=1),
            reps=[1, max_reg_targets.shape[1]])

        # [所有格子数, gt数]   最大回归值如果位于区间内，就为True
        # (2)第2个正负样本判断依据
        is_match_current_level = \
            (max_reg_targets > lower_bound) & \
            (max_reg_targets < high_bound)
        # [所有格子数, gt数]   所有gt的面积
        points2gtarea = np.tile(
            np.expand_dims(
                gt_area, axis=0), reps=[xs.shape[0], 1])
        points2gtarea[
            is_inside_box == 0] = self.INF  # 格子中心点落在gt外的（即负样本），需要学习的面积置为无穷。     这是为了points2gtarea.min(axis=1)时，若某格子有最终正样本，那么就应该不让负样本的面积影响到判断。
        points2gtarea[
            is_match_current_level == 0] = self.INF  # 最大回归值如果位于区间外（即负样本），需要学习的面积置为无穷。 这是为了points2gtarea.min(axis=1)时，若某格子有最终正样本，那么就应该不让负样本的面积影响到判断。
        points2min_area = points2gtarea.min(axis=1)  # [所有格子数, ]   所有格子需要学习 gt数 个面积  中的最小值
        points2min_area_ind = points2gtarea.argmin(axis=1)  # [所有格子数, ]   所有格子需要学习 gt数 个面积  中的最小值的下标
        labels = gt_class[points2min_area_ind] + 1  # [所有格子数, 1]   所有格子需要学习 的类别id，学习的是gt中面积最小值的的类别id
        labels[points2min_area == self.INF] = 0  # [所有格子数, 1]   负样本的points2min_area肯定是self.INF，这里将负样本需要学习 的类别id 置为0
        reg_targets = reg_targets[range(xs.shape[0]), points2min_area_ind]  # [所有格子数, 4]   所有格子需要学习 的 lrtb（负责预测gt里面积最小的）
        ctn_targets = np.sqrt((reg_targets[:, [0, 2]].min(axis=1) / \
                               reg_targets[:, [0, 2]].max(axis=1)) * \
                              (reg_targets[:, [1, 3]].min(axis=1) / \
                               reg_targets[:, [1, 3]].max(axis=1))).astype(np.float32)  # [所有格子数, ]  所有格子需要学习的centerness
        ctn_targets = np.reshape(
            ctn_targets, newshape=[ctn_targets.shape[0], 1])  # [所有格子数, 1]  所有格子需要学习的centerness
        ctn_targets[labels <= 0] = 0  # 负样本需要学习的centerness置为0
        pos_ind = np.nonzero(
            labels != 0)  # tuple=( ndarray(shape=[正样本数, ]), ndarray(shape=[正样本数, ]) )   即正样本在labels中的下标，因为labels是2维的，所以一个正样本有2个下标。
        reg_targets_pos = reg_targets[pos_ind[0], :]  # [正样本数, 4]   正样本格子需要学习 的 lrtb
        split_sections = []  # 每一个感受野 最后一个格子 在reg_targets中的位置（第一维的位置）
        beg = 0
        for lvl in range(len(num_points_each_level)):
            end = beg + num_points_each_level[lvl]
            split_sections.append(end)
            beg = end
        if no_gt:  # 如果没有gt，labels里全部置为0（背景的类别id是0）即表示所有格子都是负样本
            labels[:, :] = 0
        labels_by_level = np.split(labels, split_sections, axis=0)  # 一个list，根据split_sections切分，各个感受野的target切分开来。
        reg_targets_by_level = np.split(reg_targets, split_sections,
                                        axis=0)  # 一个list，根据split_sections切分，各个感受野的target切分开来。
        ctn_targets_by_level = np.split(ctn_targets, split_sections,
                                        axis=0)  # 一个list，根据split_sections切分，各个感受野的target切分开来。

        # 最后一步是reshape，和格子的位置对应上。
        for lvl in range(len(self.downsample_ratios)):
            grid_w = int(np.ceil(w / self.downsample_ratios[lvl]))  # 格子列数
            grid_h = int(np.ceil(h / self.downsample_ratios[lvl]))  # 格子行数
            if self.norm_reg_targets:  # 是否将reg目标归一化，配置里是True
                sample['reg_target{}'.format(lvl)] = \
                    np.reshape(
                        reg_targets_by_level[lvl] / \
                        self.downsample_ratios[lvl],  # 归一化方式是除以格子边长（即下采样倍率）
                        newshape=[grid_h, grid_w, 4])  # reshape成[grid_h, grid_w, 4]
            else:
                sample['reg_target{}'.format(lvl)] = np.reshape(
                    reg_targets_by_level[lvl],
                    newshape=[grid_h, grid_w, 4])
            sample['labels{}'.format(lvl)] = np.reshape(
                labels_by_level[lvl], newshape=[grid_h, grid_w, 1])  # reshape成[grid_h, grid_w, 1]
            sample['centerness{}'.format(lvl)] = np.reshape(
                ctn_targets_by_level[lvl], newshape=[grid_h, grid_w, 1])  # reshape成[grid_h, grid_w, 1]
        return sample


class Gt2Solov2Target(BaseOperator):
    """Assign mask target and labels in SOLOv2 network.
    The code of this function is based on:
        https://github.com/WXinlong/SOLO/blob/master/mmdet/models/anchor_heads/solov2_head.py#L271
    Args:
        num_grids (list): The list of feature map grids size.
        scale_ranges (list): The list of mask boundary range.
        coord_sigma (float): The coefficient of coordinate area length.
        sampling_ratio (float): The ratio of down sampling.
    """

    def __init__(self,
                 num_grids=[40, 36, 24, 16, 12],
                 scale_ranges=[[1, 96], [48, 192], [96, 384], [192, 768],
                               [384, 2048]],
                 coord_sigma=0.2,
                 sampling_ratio=4.0):
        super(Gt2Solov2Target, self).__init__()
        self.num_grids = num_grids
        self.scale_ranges = scale_ranges
        self.coord_sigma = coord_sigma
        self.sampling_ratio = sampling_ratio

    def _scale_size(self, im, scale):
        h, w = im.shape[:2]
        new_size = (int(w * float(scale) + 0.5), int(h * float(scale) + 0.5))
        resized_img = cv2.resize(
            im, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        return resized_img

    def __call__(self, samples, context=None):
        sample_id = 0
        max_ins_num = [0] * len(self.num_grids)
        for sample in samples:
            gt_bboxes_raw = sample['gt_bbox']
            gt_labels_raw = sample['gt_class'] + 1   # 类别id+1
            im_c, im_h, im_w = sample['image'].shape[:]
            gt_masks_raw = sample['gt_segm'].astype(np.uint8)
            mask_feat_size = [
                int(im_h / self.sampling_ratio), int(im_w / self.sampling_ratio)
            ]
            gt_areas = np.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) *
                               (gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))  # gt的平均边长
            ins_ind_label_list = []
            idx = 0
            for (lower_bound, upper_bound), num_grid \
                    in zip(self.scale_ranges, self.num_grids):
                # gt的平均边长位于指定范围内，这个感受野的特征图负责预测这些满足条件的gt
                hit_indices = ((gt_areas >= lower_bound) &
                               (gt_areas <= upper_bound)).nonzero()[0]
                num_ins = len(hit_indices)

                ins_label = []
                grid_order = []
                cate_label = np.zeros([num_grid, num_grid], dtype=np.int64)
                ins_ind_label = np.zeros([num_grid**2], dtype=np.bool)

                if num_ins == 0:
                    ins_label = np.zeros([1, mask_feat_size[0], mask_feat_size[1]], dtype=np.uint8)
                    ins_ind_label_list.append(ins_ind_label)
                    sample['cate_label{}'.format(idx)] = cate_label.flatten()
                    sample['ins_label{}'.format(idx)] = ins_label
                    sample['grid_order{}'.format(idx)] = np.asarray([sample_id * num_grid * num_grid + 0], dtype=np.int32)
                    idx += 1
                    continue
                gt_bboxes = gt_bboxes_raw[hit_indices]   # [M, 4] 这个感受野的gt
                gt_labels = gt_labels_raw[hit_indices]   # [M, 1] 这个感受野的类别id(+1)
                gt_masks = gt_masks_raw[hit_indices, ...]   # [M, h, w] 这个感受野的gt_mask

                # 这个感受野的gt的宽的一半 * self.coord_sigma
                half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.coord_sigma
                # 这个感受野的gt的高的一半 * self.coord_sigma
                half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.coord_sigma

                # 遍历这个感受野的每一个gt
                for seg_mask, gt_label, half_h, half_w in zip(
                        gt_masks, gt_labels, half_hs, half_ws):
                    if seg_mask.sum() == 0:
                        continue
                    # mass center
                    upsampled_size = (mask_feat_size[0] * 4, mask_feat_size[1] * 4)
                    center_h, center_w = ndimage.measurements.center_of_mass(seg_mask)
                    coord_w = int((center_w / upsampled_size[1]) // (1. / num_grid))
                    coord_h = int((center_h / upsampled_size[0]) // (1. / num_grid))

                    # left, top, right, down
                    top_box = max(0, int(((center_h - half_h) / upsampled_size[0]) // (1. / num_grid)))
                    down_box = min(num_grid - 1, int(((center_h + half_h) / upsampled_size[0]) // (1. / num_grid)))
                    left_box = max(0, int(((center_w - half_w) / upsampled_size[1]) // (1. / num_grid)))
                    right_box = min(num_grid - 1, int(((center_w + half_w) / upsampled_size[1]) // (1. / num_grid)))

                    top = max(top_box, coord_h - 1)
                    down = min(down_box, coord_h + 1)
                    left = max(coord_w - 1, left_box)
                    right = min(right_box, coord_w + 1)

                    cate_label[top:(down + 1), left:(right + 1)] = gt_label
                    seg_mask = self._scale_size(
                        seg_mask, scale=1. / self.sampling_ratio)
                    for i in range(top, down + 1):
                        for j in range(left, right + 1):
                            label = int(i * num_grid + j)
                            cur_ins_label = np.zeros(
                                [mask_feat_size[0], mask_feat_size[1]],
                                dtype=np.uint8)
                            cur_ins_label[:seg_mask.shape[0], :seg_mask.shape[
                                1]] = seg_mask
                            ins_label.append(cur_ins_label)
                            ins_ind_label[label] = True
                            grid_order.append(sample_id * num_grid * num_grid +
                                              label)
                if ins_label == []:
                    ins_label = np.zeros(
                        [1, mask_feat_size[0], mask_feat_size[1]],
                        dtype=np.uint8)
                    ins_ind_label_list.append(ins_ind_label)
                    sample['cate_label{}'.format(idx)] = cate_label.flatten()
                    sample['ins_label{}'.format(idx)] = ins_label
                    sample['grid_order{}'.format(idx)] = np.asarray(
                        [sample_id * num_grid * num_grid + 0], dtype=np.int32)
                else:
                    ins_label = np.stack(ins_label, axis=0)
                    ins_ind_label_list.append(ins_ind_label)
                    sample['cate_label{}'.format(idx)] = cate_label.flatten()
                    sample['ins_label{}'.format(idx)] = ins_label
                    sample['grid_order{}'.format(idx)] = np.asarray(
                        grid_order, dtype=np.int32)
                    assert len(grid_order) > 0
                max_ins_num[idx] = max(
                    max_ins_num[idx],
                    sample['ins_label{}'.format(idx)].shape[0])
                idx += 1
            ins_ind_labels = np.concatenate([
                ins_ind_labels_level_img
                for ins_ind_labels_level_img in ins_ind_label_list
            ])
            fg_num = np.sum(ins_ind_labels)
            sample['fg_num'] = fg_num
            sample_id += 1

            sample.pop('is_crowd')
            sample.pop('gt_class')
            sample.pop('gt_bbox')
            sample.pop('gt_poly')
            sample.pop('gt_segm')

        # padding batch
        for data in samples:
            for idx in range(len(self.num_grids)):
                gt_ins_data = np.zeros(
                    [
                        max_ins_num[idx],
                        data['ins_label{}'.format(idx)].shape[1],
                        data['ins_label{}'.format(idx)].shape[2]
                    ],
                    dtype=np.uint8)
                gt_ins_data[0:data['ins_label{}'.format(idx)].shape[
                    0], :, :] = data['ins_label{}'.format(idx)]
                gt_grid_order = np.zeros([max_ins_num[idx]], dtype=np.int32)
                gt_grid_order[0:data['grid_order{}'.format(idx)].shape[
                    0]] = data['grid_order{}'.format(idx)]
                data['ins_label{}'.format(idx)] = gt_ins_data
                data['grid_order{}'.format(idx)] = gt_grid_order

        return samples



class Gt2RepPointsTargetSingle(BaseOperator):
    """
    一张图片的Gt2RepPointsTarget
    """

    def __init__(self,
                 object_sizes_boundary,
                 center_sampling_radius,
                 downsample_ratios,
                 norm_reg_targets=False):
        super(Gt2RepPointsTargetSingle, self).__init__()
        self.center_sampling_radius = center_sampling_radius
        self.downsample_ratios = downsample_ratios
        self.INF = np.inf
        self.object_sizes_boundary = [-1] + object_sizes_boundary + [self.INF]
        object_sizes_of_interest = []
        for i in range(len(self.object_sizes_boundary) - 1):
            object_sizes_of_interest.append([
                self.object_sizes_boundary[i], self.object_sizes_boundary[i + 1]
            ])
        self.object_sizes_of_interest = object_sizes_of_interest
        self.norm_reg_targets = norm_reg_targets

    def _compute_points(self, w, h):
        """
        compute the corresponding points in each feature map
        :param h: image height
        :param w: image width
        :return: points from all feature map
        """
        locations = []
        # 从小感受野stride=8遍历到大感受野stride=128。location.shape=[格子行数*格子列数, 2]，存放的是每个格子的中心点的坐标。格子顺序是第一行从左到右，第二行从左到右，...
        for stride in self.downsample_ratios:
            shift_x = np.arange(0, w, stride).astype(np.float32)
            shift_y = np.arange(0, h, stride).astype(np.float32)
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            shift_x = shift_x.flatten()
            shift_y = shift_y.flatten()

            '''
            location.shape = [grid_h*grid_w, 2]
            如果stride=8，
            location = [[4, 4], [12, 4], [20, 4], ...],  这一个输出层的格子的中心点的xy坐标。格子顺序是第一行从左到右，第二行从左到右，...
            即location = [[0.5*stride, 0.5*stride], [1.5*stride, 0.5*stride], [2.5*stride, 0.5*stride], ...]

            如果stride=16，
            location = [[8, 8], [24, 8], [40, 8], ...],  这一个输出层的格子的中心点的xy坐标。格子顺序是第一行从左到右，第二行从左到右，...
            即location = [[0.5*stride, 0.5*stride], [1.5*stride, 0.5*stride], [2.5*stride, 0.5*stride], ...]

            ...
            '''
            location = np.stack([shift_x, shift_y], axis=1) + stride // 2
            locations.append(location)
        num_points_each_level = [len(location) for location in
                                 locations]  # num_points_each_level=[stride=8感受野格子数, ..., stride=128感受野格子数]
        locations = np.concatenate(locations, axis=0)
        return locations, num_points_each_level

    def _convert_xywh2xyxy(self, gt_bbox, w, h):
        """
        convert the bounding box from style xywh to xyxy
        :param gt_bbox: bounding boxes normalized into [0, 1]
        :param w: image width
        :param h: image height
        :return: bounding boxes in xyxy style
        """
        bboxes = gt_bbox.copy()
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * w
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * h
        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
        return bboxes

    def _check_inside_boxes_limited(self, gt_bbox, xs, ys,
                                    num_points_each_level):
        """
        check if points is within the clipped boxes
        :param gt_bbox: bounding boxes
        :param xs: horizontal coordinate of points
        :param ys: vertical coordinate of points
        :return: the mask of points is within gt_box or not
        """
        bboxes = np.reshape(  # [gt数, 4] -> [1, gt数, 4]
            gt_bbox, newshape=[1, gt_bbox.shape[0], gt_bbox.shape[1]])
        bboxes = np.tile(bboxes, reps=[xs.shape[0], 1, 1])  # [所有格子数, gt数, 4]   gt坐标。可以看出，每1个gt都会参与到fpn的所有输出特征图。
        ct_x = (bboxes[:, :, 0] + bboxes[:, :, 2]) / 2  # [所有格子数, gt数]      gt中心点x
        ct_y = (bboxes[:, :, 1] + bboxes[:, :, 3]) / 2  # [所有格子数, gt数]      gt中心点y
        beg = 0  # 开始=0

        # clipped_box即修改之后的gt，和原始gt（bboxes）的中心点相同，但是边长却修改成最大只能是1.5 * 2 = 3个格子边长
        clipped_box = bboxes.copy()  # [所有格子数, gt数, 4]   gt坐标，限制gt的边长，最大只能是1.5 * 2 = 3个格子边长
        for lvl, stride in enumerate(self.downsample_ratios):  # 遍历每个感受野，从 stride=8的感受野 到 stride=128的感受野
            end = beg + num_points_each_level[lvl]  # 结束=开始+这个感受野的格子数
            stride_exp = self.center_sampling_radius * stride  # stride_exp = 1.5 * 这个感受野的stride(的格子边长)
            clipped_box[beg:end, :, 0] = np.maximum(
                bboxes[beg:end, :, 0], ct_x[beg:end, :] - stride_exp)  # 限制gt的边长，最大只能是1.5 * 2 = 3个格子边长
            clipped_box[beg:end, :, 1] = np.maximum(
                bboxes[beg:end, :, 1], ct_y[beg:end, :] - stride_exp)  # 限制gt的边长，最大只能是1.5 * 2 = 3个格子边长
            clipped_box[beg:end, :, 2] = np.minimum(
                bboxes[beg:end, :, 2], ct_x[beg:end, :] + stride_exp)  # 限制gt的边长，最大只能是1.5 * 2 = 3个格子边长
            clipped_box[beg:end, :, 3] = np.minimum(
                bboxes[beg:end, :, 3], ct_y[beg:end, :] + stride_exp)  # 限制gt的边长，最大只能是1.5 * 2 = 3个格子边长
            beg = end

        # 如果格子中心点落在clipped_box代表的gt框内，那么这个格子就被选为候选正样本。

        # xs  [所有格子数, gt数]， 所有格子中心点的横坐标重复 gt数 次
        l_res = xs - clipped_box[:, :, 0]  # [所有格子数, gt数]  所有格子需要学习 gt数 个l
        r_res = clipped_box[:, :, 2] - xs  # [所有格子数, gt数]  所有格子需要学习 gt数 个r
        t_res = ys - clipped_box[:, :, 1]  # [所有格子数, gt数]  所有格子需要学习 gt数 个t
        b_res = clipped_box[:, :, 3] - ys  # [所有格子数, gt数]  所有格子需要学习 gt数 个b
        clipped_box_reg_targets = np.stack([l_res, t_res, r_res, b_res], axis=2)  # [所有格子数, gt数, 4]  所有格子需要学习 gt数 个lrtb
        inside_gt_box = np.min(clipped_box_reg_targets,
                               axis=2) > 0  # [所有格子数, gt数]  需要学习的lrtb如果都>0，表示格子被选中。即只选取中心点落在gt内的格子。
        return inside_gt_box

    def __call__(self, sample, context=None):
        assert len(self.object_sizes_of_interest) == len(self.downsample_ratios), \
            "object_sizes_of_interest', and 'downsample_ratios' should have same length."

        # im, gt_bbox, gt_class, gt_score = sample
        im = sample['image']  # [3, pad_h, pad_w]
        im_info = sample['im_info']  # [3, ]  分别是resize_h, resize_w, im_scale
        bboxes = sample['gt_bbox']  # [m, 4]  x0y0x1y1格式
        gt_class = sample['gt_class']  # [m, 1]
        gt_score = sample['gt_score']  # [m, 1]
        no_gt = False
        if len(bboxes) == 0:  # 如果没有gt，虚构一个gt为了后面不报错。
            no_gt = True
            bboxes = np.array([[0, 0, 100, 100]]).astype(np.float32)
            gt_class = np.array([[0]]).astype(np.int32)
            gt_score = np.array([[1]]).astype(np.float32)
            # print('nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnone')
        # bboxes的横坐标变成缩放后图片中对应物体的横坐标
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * np.floor(im_info[1]) / \
                            np.floor(im_info[1] / im_info[2])
        # bboxes的纵坐标变成缩放后图片中对应物体的纵坐标
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * np.floor(im_info[0]) / \
                            np.floor(im_info[0] / im_info[2])
        # calculate the locations
        h, w = sample['image'].shape[1:3]  # h w是这一批所有图片对齐后的高宽。
        points, num_points_each_level = self._compute_points(w,
                                                             h)  # points是所有格子中心点的坐标，num_points_each_level=[stride=8感受野格子数, ..., stride=128感受野格子数]
        object_scale_exp = []
        for i, num_pts in enumerate(num_points_each_level):  # 遍历每个感受野格子数
            object_scale_exp.append(  # 边界self.object_sizes_of_interest[i] 重复 num_pts=格子数 次
                np.tile(
                    np.array([self.object_sizes_of_interest[i]]),
                    reps=[num_pts, 1]))
        object_scale_exp = np.concatenate(object_scale_exp, axis=0)

        gt_area = (bboxes[:, 2] - bboxes[:, 0]) * (  # [gt数, ]   所有gt的面积
                bboxes[:, 3] - bboxes[:, 1])
        xs, ys = points[:, 0], points[:, 1]  # 所有格子中心点的横坐标、纵坐标
        xs = np.reshape(xs, newshape=[xs.shape[0], 1])  # [所有格子数, 1]
        xs = np.tile(xs, reps=[1, bboxes.shape[0]])  # [所有格子数, gt数]， 所有格子中心点的横坐标重复 gt数 次
        ys = np.reshape(ys, newshape=[ys.shape[0], 1])  # [所有格子数, 1]
        ys = np.tile(ys, reps=[1, bboxes.shape[0]])  # [所有格子数, gt数]， 所有格子中心点的纵坐标重复 gt数 次

        l_res = xs - bboxes[:,
                     0]  # [所有格子数, gt数] - [gt数, ] = [所有格子数, gt数]     结果是所有格子中心点的横坐标 分别减去 所有gt左上角的横坐标，即所有格子需要学习 gt数 个l
        r_res = bboxes[:, 2] - xs  # 所有格子需要学习 gt数 个r
        t_res = ys - bboxes[:, 1]  # 所有格子需要学习 gt数 个t
        b_res = bboxes[:, 3] - ys  # 所有格子需要学习 gt数 个b
        reg_targets = np.stack([l_res, t_res, r_res, b_res], axis=2)  # [所有格子数, gt数, 4]   所有格子需要学习 gt数 个lrtb
        if self.center_sampling_radius > 0:
            # [所有格子数, gt数]    True表示格子中心点（锚点）落在gt内（gt是被限制边长后的gt）。
            # FCOS首先将gt框内的锚点（格子中心点）视为候选正样本，然后根据为每个金字塔等级定义的比例范围从候选中选择最终的正样本（而且是负责预测gt里面积最小的），最后那些未选择的锚点为负样本。
            # (1)第1个正负样本判断依据

            # 这里是使用gt的中心区域判断格子中心点是否在gt框内。这样做会减少很多中心度很低的低质量正样本。
            is_inside_box = self._check_inside_boxes_limited(
                bboxes, xs, ys, num_points_each_level)
        else:
            # [所有格子数, gt数]    True表示格子中心点（锚点）落在gt内。
            # FCOS首先将gt框内的锚点（格子中心点）视为候选正样本，然后根据为每个金字塔等级定义的比例范围从候选中选择最终的正样本（而且是负责预测gt里面积最小的），最后那些未选择的锚点为负样本。
            # (1)第1个正负样本判断依据

            # 这里是使用gt的完整区域判断格子中心点是否在gt框内。这样做会增加很多中心度很低的低质量正样本。
            is_inside_box = np.min(reg_targets, axis=2) > 0
        # check if the targets is inside the corresponding level
        max_reg_targets = np.max(reg_targets, axis=2)  # [所有格子数, gt数]   所有格子需要学习 gt数 个lrtb   中的最大值
        lower_bound = np.tile(  # [所有格子数, gt数]   下限重复 gt数 次
            np.expand_dims(
                object_scale_exp[:, 0], axis=1),
            reps=[1, max_reg_targets.shape[1]])
        high_bound = np.tile(  # [所有格子数, gt数]   上限重复 gt数 次
            np.expand_dims(
                object_scale_exp[:, 1], axis=1),
            reps=[1, max_reg_targets.shape[1]])

        # [所有格子数, gt数]   最大回归值如果位于区间内，就为True
        # (2)第2个正负样本判断依据
        is_match_current_level = \
            (max_reg_targets > lower_bound) & \
            (max_reg_targets < high_bound)
        # [所有格子数, gt数]   所有gt的面积
        points2gtarea = np.tile(
            np.expand_dims(
                gt_area, axis=0), reps=[xs.shape[0], 1])
        points2gtarea[
            is_inside_box == 0] = self.INF  # 格子中心点落在gt外的（即负样本），需要学习的面积置为无穷。     这是为了points2gtarea.min(axis=1)时，若某格子有最终正样本，那么就应该不让负样本的面积影响到判断。
        points2gtarea[
            is_match_current_level == 0] = self.INF  # 最大回归值如果位于区间外（即负样本），需要学习的面积置为无穷。 这是为了points2gtarea.min(axis=1)时，若某格子有最终正样本，那么就应该不让负样本的面积影响到判断。
        points2min_area = points2gtarea.min(axis=1)  # [所有格子数, ]   所有格子需要学习 gt数 个面积  中的最小值
        points2min_area_ind = points2gtarea.argmin(axis=1)  # [所有格子数, ]   所有格子需要学习 gt数 个面积  中的最小值的下标
        labels = gt_class[points2min_area_ind] + 1  # [所有格子数, 1]   所有格子需要学习 的类别id，学习的是gt中面积最小值的的类别id
        labels[points2min_area == self.INF] = 0  # [所有格子数, 1]   负样本的points2min_area肯定是self.INF，这里将负样本需要学习 的类别id 置为0
        reg_targets = reg_targets[range(xs.shape[0]), points2min_area_ind]  # [所有格子数, 4]   所有格子需要学习 的 lrtb（负责预测gt里面积最小的）
        ctn_targets = np.sqrt((reg_targets[:, [0, 2]].min(axis=1) / \
                               reg_targets[:, [0, 2]].max(axis=1)) * \
                              (reg_targets[:, [1, 3]].min(axis=1) / \
                               reg_targets[:, [1, 3]].max(axis=1))).astype(np.float32)  # [所有格子数, ]  所有格子需要学习的centerness
        ctn_targets = np.reshape(
            ctn_targets, newshape=[ctn_targets.shape[0], 1])  # [所有格子数, 1]  所有格子需要学习的centerness
        ctn_targets[labels <= 0] = 0  # 负样本需要学习的centerness置为0
        pos_ind = np.nonzero(
            labels != 0)  # tuple=( ndarray(shape=[正样本数, ]), ndarray(shape=[正样本数, ]) )   即正样本在labels中的下标，因为labels是2维的，所以一个正样本有2个下标。
        reg_targets_pos = reg_targets[pos_ind[0], :]  # [正样本数, 4]   正样本格子需要学习 的 lrtb
        split_sections = []  # 每一个感受野 最后一个格子 在reg_targets中的位置（第一维的位置）
        beg = 0
        for lvl in range(len(num_points_each_level)):
            end = beg + num_points_each_level[lvl]
            split_sections.append(end)
            beg = end
        if no_gt:  # 如果没有gt，labels里全部置为0（背景的类别id是0）即表示所有格子都是负样本
            labels[:, :] = 0
        labels_by_level = np.split(labels, split_sections, axis=0)  # 一个list，根据split_sections切分，各个感受野的target切分开来。
        reg_targets_by_level = np.split(reg_targets, split_sections,
                                        axis=0)  # 一个list，根据split_sections切分，各个感受野的target切分开来。
        ctn_targets_by_level = np.split(ctn_targets, split_sections,
                                        axis=0)  # 一个list，根据split_sections切分，各个感受野的target切分开来。

        # 最后一步是reshape，和格子的位置对应上。
        for lvl in range(len(self.downsample_ratios)):
            grid_w = int(np.ceil(w / self.downsample_ratios[lvl]))  # 格子列数
            grid_h = int(np.ceil(h / self.downsample_ratios[lvl]))  # 格子行数
            if self.norm_reg_targets:  # 是否将reg目标归一化，配置里是True
                sample['reg_target{}'.format(lvl)] = \
                    np.reshape(
                        reg_targets_by_level[lvl] / \
                        self.downsample_ratios[lvl],  # 归一化方式是除以格子边长（即下采样倍率）
                        newshape=[grid_h, grid_w, 4])  # reshape成[grid_h, grid_w, 4]
            else:
                sample['reg_target{}'.format(lvl)] = np.reshape(
                    reg_targets_by_level[lvl],
                    newshape=[grid_h, grid_w, 4])
            sample['labels{}'.format(lvl)] = np.reshape(
                labels_by_level[lvl], newshape=[grid_h, grid_w, 1])  # reshape成[grid_h, grid_w, 1]
            sample['centerness{}'.format(lvl)] = np.reshape(
                ctn_targets_by_level[lvl], newshape=[grid_h, grid_w, 1])  # reshape成[grid_h, grid_w, 1]
        return sample



class RandomDistort(BaseOperator):
    """Random color distortion.
    Args:
        hue (list): hue settings. in [lower, upper, probability] format.
        saturation (list): saturation settings. in [lower, upper, probability] format.
        contrast (list): contrast settings. in [lower, upper, probability] format.
        brightness (list): brightness settings. in [lower, upper, probability] format.
        random_apply (bool): whether to apply in random (yolo) or fixed (SSD)
            order.
        count (int): the number of doing distrot
        random_channel (bool): whether to swap channels randomly
    """

    def __init__(self,
                 hue=[-18, 18, 0.5],
                 saturation=[0.5, 1.5, 0.5],
                 contrast=[0.5, 1.5, 0.5],
                 brightness=[0.5, 1.5, 0.5],
                 random_apply=True,
                 count=4,
                 random_channel=False):
        super(RandomDistort, self).__init__()
        self.hue = hue
        self.saturation = saturation
        self.contrast = contrast
        self.brightness = brightness
        self.random_apply = random_apply
        self.count = count
        self.random_channel = random_channel

    def apply_hue(self, img):
        low, high, prob = self.hue
        if np.random.uniform(0., 1.) < prob:
            return img

        img = img.astype(np.float32)
        # it works, but result differ from HSV version
        delta = np.random.uniform(low, high)
        u = np.cos(delta * np.pi)
        w = np.sin(delta * np.pi)
        bt = np.array([[1.0, 0.0, 0.0], [0.0, u, -w], [0.0, w, u]])
        tyiq = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.321],
                         [0.211, -0.523, 0.311]])
        ityiq = np.array([[1.0, 0.956, 0.621], [1.0, -0.272, -0.647],
                          [1.0, -1.107, 1.705]])
        t = np.dot(np.dot(ityiq, bt), tyiq).T
        img = np.dot(img, t)
        return img

    def apply_saturation(self, img):
        low, high, prob = self.saturation
        if np.random.uniform(0., 1.) < prob:
            return img
        delta = np.random.uniform(low, high)
        img = img.astype(np.float32)
        # it works, but result differ from HSV version
        gray = img * np.array([[[0.299, 0.587, 0.114]]], dtype=np.float32)
        gray = gray.sum(axis=2, keepdims=True)
        gray *= (1.0 - delta)
        img *= delta
        img += gray
        return img

    def apply_contrast(self, img):
        low, high, prob = self.contrast
        if np.random.uniform(0., 1.) < prob:
            return img
        delta = np.random.uniform(low, high)
        img = img.astype(np.float32)
        img *= delta
        return img

    def apply_brightness(self, img):
        low, high, prob = self.brightness
        if np.random.uniform(0., 1.) < prob:
            return img
        delta = np.random.uniform(low, high)
        img = img.astype(np.float32)
        img += delta
        return img

    def __call__(self, sample, context=None):
        img = sample['image']
        if self.random_apply:
            functions = [
                self.apply_brightness, self.apply_contrast,
                self.apply_saturation, self.apply_hue
            ]
            distortions = np.random.permutation(functions)[:self.count]
            for func in distortions:
                img = func(img)
            sample['image'] = img
            return sample

        img = self.apply_brightness(img)
        mode = np.random.randint(0, 2)

        if mode:
            img = self.apply_contrast(img)

        img = self.apply_saturation(img)
        img = self.apply_hue(img)

        if not mode:
            img = self.apply_contrast(img)

        if self.random_channel:
            if np.random.randint(0, 2):
                img = img[..., np.random.permutation(3)]
        sample['image'] = img
        return sample


class Resize(BaseOperator):
    def __init__(self, target_size, keep_ratio, interp=cv2.INTER_LINEAR):
        """
        Resize image to target size. if keep_ratio is True,
        resize the image's long side to the maximum of target_size
        if keep_ratio is False, resize the image to target size(h, w)
        Args:
            target_size (int|list): image target size
            keep_ratio (bool): whether keep_ratio or not, default true
            interp (int): the interpolation method
        """
        super(Resize, self).__init__()
        self.keep_ratio = keep_ratio
        self.interp = interp
        if not isinstance(target_size, (Integral, Sequence)):
            raise TypeError(
                "Type of target_size is invalid. Must be Integer or List or Tuple, now is {}".
                format(type(target_size)))
        if isinstance(target_size, Integral):
            target_size = [target_size, target_size]
        self.target_size = target_size

    def apply_image(self, image, scale):
        im_scale_x, im_scale_y = scale

        return cv2.resize(
            image,
            None,
            None,
            fx=im_scale_x,
            fy=im_scale_y,
            interpolation=self.interp)

    def apply_bbox(self, bbox, scale, size):
        im_scale_x, im_scale_y = scale
        resize_w, resize_h = size
        bbox[:, 0::2] *= im_scale_x
        bbox[:, 1::2] *= im_scale_y
        bbox[:, 0::2] = np.clip(bbox[:, 0::2], 0, resize_w)
        bbox[:, 1::2] = np.clip(bbox[:, 1::2], 0, resize_h)
        return bbox

    def apply_segm(self, segms, im_size, scale):
        def _resize_poly(poly, im_scale_x, im_scale_y):
            resized_poly = np.array(poly).astype('float32')
            resized_poly[0::2] *= im_scale_x
            resized_poly[1::2] *= im_scale_y
            return resized_poly.tolist()

        def _resize_rle(rle, im_h, im_w, im_scale_x, im_scale_y):
            if 'counts' in rle and type(rle['counts']) == list:
                rle = mask_util.frPyObjects(rle, im_h, im_w)

            mask = mask_util.decode(rle)
            mask = cv2.resize(
                mask,
                None,
                None,
                fx=im_scale_x,
                fy=im_scale_y,
                interpolation=self.interp)
            rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
            return rle

        im_h, im_w = im_size
        im_scale_x, im_scale_y = scale
        resized_segms = []
        for segm in segms:
            if is_poly(segm):
                # Polygon format
                resized_segms.append([
                    _resize_poly(poly, im_scale_x, im_scale_y) for poly in segm
                ])
            else:
                # RLE format
                import pycocotools.mask as mask_util
                resized_segms.append(
                    _resize_rle(segm, im_h, im_w, im_scale_x, im_scale_y))

        return resized_segms

    def __call__(self, sample, context=None):
        """ Resize the image numpy.
        """
        im = sample['image']
        if not isinstance(im, np.ndarray):
            raise TypeError("{}: image type is not numpy.".format(self))
        if len(im.shape) != 3:
            raise ImageError('{}: image is not 3-dimensional.'.format(self))

        # apply image
        im_shape = im.shape
        if self.keep_ratio:

            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])

            target_size_min = np.min(self.target_size)
            target_size_max = np.max(self.target_size)

            im_scale = min(target_size_min / im_size_min,
                           target_size_max / im_size_max)

            resize_h = im_scale * float(im_shape[0])
            resize_w = im_scale * float(im_shape[1])

            im_scale_x = im_scale
            im_scale_y = im_scale
        else:
            resize_h, resize_w = self.target_size
            im_scale_y = resize_h / im_shape[0]
            im_scale_x = resize_w / im_shape[1]

        im = self.apply_image(sample['image'], [im_scale_x, im_scale_y])
        sample['image'] = im
        sample['im_shape'] = np.asarray([resize_h, resize_w], dtype=np.float32)
        if 'scale_factor' in sample:
            scale_factor = sample['scale_factor']
            sample['scale_factor'] = np.asarray(
                [scale_factor[0] * im_scale_y, scale_factor[1] * im_scale_x],
                dtype=np.float32)
        else:
            sample['scale_factor'] = np.asarray(
                [im_scale_y, im_scale_x], dtype=np.float32)

        # apply bbox
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'],
                                                [im_scale_x, im_scale_y],
                                                [resize_w, resize_h])

        # apply rbox
        if 'gt_rbox2poly' in sample:
            if np.array(sample['gt_rbox2poly']).shape[1] != 8:
                logger.warning(
                    "gt_rbox2poly's length shoule be 8, but actually is {}".
                    format(len(sample['gt_rbox2poly'])))
            sample['gt_rbox2poly'] = self.apply_bbox(sample['gt_rbox2poly'],
                                                     [im_scale_x, im_scale_y],
                                                     [resize_w, resize_h])

        # apply polygon
        if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
            sample['gt_poly'] = self.apply_segm(sample['gt_poly'], im_shape[:2],
                                                [im_scale_x, im_scale_y])

        # apply semantic
        if 'semantic' in sample and sample['semantic']:
            semantic = sample['semantic']
            semantic = cv2.resize(
                semantic.astype('float32'),
                None,
                None,
                fx=im_scale_x,
                fy=im_scale_y,
                interpolation=self.interp)
            semantic = np.asarray(semantic).astype('int32')
            semantic = np.expand_dims(semantic, 0)
            sample['semantic'] = semantic

        # apply gt_segm
        if 'gt_segm' in sample and len(sample['gt_segm']) > 0:
            masks = [
                cv2.resize(
                    gt_segm,
                    None,
                    None,
                    fx=im_scale_x,
                    fy=im_scale_y,
                    interpolation=cv2.INTER_NEAREST)
                for gt_segm in sample['gt_segm']
            ]
            sample['gt_segm'] = np.asarray(masks).astype(np.uint8)

        return sample


class RandomResize(BaseOperator):
    def __init__(self,
                 target_size,
                 keep_ratio=True,
                 interp=cv2.INTER_LINEAR,
                 random_size=True,
                 random_interp=False):
        """
        Resize image to target size randomly. random target_size and interpolation method
        Args:
            target_size (int, list, tuple): image target size, if random size is True, must be list or tuple
            keep_ratio (bool): whether keep_raio or not, default true
            interp (int): the interpolation method
            random_size (bool): whether random select target size of image
            random_interp (bool): whether random select interpolation method
        """
        super(RandomResize, self).__init__()
        self.keep_ratio = keep_ratio
        self.interp = interp
        self.interps = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ]
        assert isinstance(target_size, (
            Integral, Sequence)), "target_size must be Integer, List or Tuple"
        if random_size and not isinstance(target_size, Sequence):
            raise TypeError(
                "Type of target_size is invalid when random_size is True. Must be List or Tuple, now is {}".
                format(type(target_size)))
        self.target_size = target_size
        self.random_size = random_size
        self.random_interp = random_interp

    def __call__(self, sample, context=None):
        """ Resize the image numpy.
        """
        if self.random_size:
            target_size = random.choice(self.target_size)
        else:
            target_size = self.target_size

        if self.random_interp:
            interp = random.choice(self.interps)
        else:
            interp = self.interp

        resizer = Resize(target_size, self.keep_ratio, interp)
        return resizer(sample, context=context)


def cal_line_length(point1, point2):
    import math
    return math.sqrt(
        math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))


def get_best_begin_point_single(coordinate):
    x1, y1, x2, y2, x3, y3, x4, y4 = coordinate
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combinate = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
                 [[x4, y4], [x1, y1], [x2, y2], [x3, y3]],
                 [[x3, y3], [x4, y4], [x1, y1], [x2, y2]],
                 [[x2, y2], [x3, y3], [x4, y4], [x1, y1]]]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = cal_line_length(combinate[i][0], dst_coordinate[0]) \
                     + cal_line_length(combinate[i][1], dst_coordinate[1]) \
                     + cal_line_length(combinate[i][2], dst_coordinate[2]) \
                     + cal_line_length(combinate[i][3], dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        pass
    return np.array(combinate[force_flag]).reshape(8)



class RandomFlip(BaseOperator):
    def __init__(self, prob=0.5):
        """
        Args:
            prob (float): the probability of flipping image
        """
        super(RandomFlip, self).__init__()
        self.prob = prob
        if not (isinstance(self.prob, float)):
            raise TypeError("{}: input type is invalid.".format(self))

    def apply_segm(self, segms, height, width):
        def _flip_poly(poly, width):
            flipped_poly = np.array(poly)
            flipped_poly[0::2] = width - np.array(poly[0::2])
            return flipped_poly.tolist()

        def _flip_rle(rle, height, width):
            if 'counts' in rle and type(rle['counts']) == list:
                rle = mask_util.frPyObjects(rle, height, width)
            mask = mask_util.decode(rle)
            mask = mask[:, ::-1]
            rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
            return rle

        flipped_segms = []
        for segm in segms:
            if is_poly(segm):
                # Polygon format
                flipped_segms.append([_flip_poly(poly, width) for poly in segm])
            else:
                # RLE format
                import pycocotools.mask as mask_util
                flipped_segms.append(_flip_rle(segm, height, width))
        return flipped_segms

    def apply_keypoint(self, gt_keypoint, width):
        for i in range(gt_keypoint.shape[1]):
            if i % 2 == 0:
                old_x = gt_keypoint[:, i].copy()
                gt_keypoint[:, i] = width - old_x
        return gt_keypoint

    def apply_image(self, image):
        return image[:, ::-1, :]

    def apply_bbox(self, bbox, width):
        oldx1 = bbox[:, 0].copy()
        oldx2 = bbox[:, 2].copy()
        bbox[:, 0] = width - oldx2
        bbox[:, 2] = width - oldx1
        return bbox

    def apply_rbox(self, bbox, width):
        oldx1 = bbox[:, 0].copy()
        oldx2 = bbox[:, 2].copy()
        oldx3 = bbox[:, 4].copy()
        oldx4 = bbox[:, 6].copy()
        bbox[:, 0] = width - oldx1
        bbox[:, 2] = width - oldx2
        bbox[:, 4] = width - oldx3
        bbox[:, 6] = width - oldx4
        bbox = [get_best_begin_point_single(e) for e in bbox]
        return bbox

    def __call__(self, sample, context=None):
        """Filp the image and bounding box.
        Operators:
            1. Flip the image numpy.
            2. Transform the bboxes' x coordinates.
              (Must judge whether the coordinates are normalized!)
            3. Transform the segmentations' x coordinates.
              (Must judge whether the coordinates are normalized!)
        Output:
            sample: the image, bounding box and segmentation part
                    in sample are flipped.
        """
        if np.random.uniform(0, 1) < self.prob:
            im = sample['image']
            height, width = im.shape[:2]
            im = self.apply_image(im)
            if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
                sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'], width)
            if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
                sample['gt_poly'] = self.apply_segm(sample['gt_poly'], height,
                                                    width)
            if 'gt_keypoint' in sample and len(sample['gt_keypoint']) > 0:
                sample['gt_keypoint'] = self.apply_keypoint(
                    sample['gt_keypoint'], width)

            if 'semantic' in sample and sample['semantic']:
                sample['semantic'] = sample['semantic'][:, ::-1]

            if 'gt_segm' in sample and sample['gt_segm'].any():
                sample['gt_segm'] = sample['gt_segm'][:, :, ::-1]

            if 'gt_rbox2poly' in sample and sample['gt_rbox2poly'].any():
                sample['gt_rbox2poly'] = self.apply_rbox(sample['gt_rbox2poly'],
                                                         width)

            sample['flipped'] = True
            sample['image'] = im
        return sample


def get_sample_transforms(cfg):
    # sample_transforms
    sample_transforms = []
    for preprocess_name in cfg.sample_transforms_seq:
        if preprocess_name == 'decodeImage':
            preprocess = DecodeImage(**cfg.decodeImage)   # 对图片解码。最开始的一步。
        elif preprocess_name == 'decode':
            preprocess = Decode(**cfg.decode)   # 对图片解码。最开始的一步。
        elif preprocess_name == 'poly2Mask':
            preprocess = Poly2Mask(**cfg.poly2Mask)   #
        elif preprocess_name == 'randomDistort':
            preprocess = RandomDistort(**cfg.randomDistort)   #
        elif preprocess_name == 'randomResize':
            preprocess = RandomResize(**cfg.randomResize)   #
        elif preprocess_name == 'randomFlip':
            preprocess = RandomFlip(**cfg.randomFlip)   #
        elif preprocess_name == 'mixupImage':
            preprocess = MixupImage(**cfg.mixupImage)      # mixup增强
        elif preprocess_name == 'cutmixImage':
            preprocess = CutmixImage(**cfg.cutmixImage)    # cutmix增强
        elif preprocess_name == 'mosaicImage':
            preprocess = MosaicImage(**cfg.mosaicImage)    # mosaic增强
        elif preprocess_name == 'yOLOXMosaicImage':
            preprocess = YOLOXMosaicImage(**cfg.yOLOXMosaicImage)  # YOLOX mosaic增强
        elif preprocess_name == 'colorDistort':
            preprocess = ColorDistort(**cfg.colorDistort)  # 颜色扰动
        elif preprocess_name == 'randomExpand':
            preprocess = RandomExpand(**cfg.randomExpand)  # 随机填充
        elif preprocess_name == 'randomCrop':
            preprocess = RandomCrop(**cfg.randomCrop)        # 随机裁剪
        elif preprocess_name == 'gridMaskOp':
            preprocess = GridMaskOp(**cfg.gridMaskOp)        # GridMaskOp
        elif preprocess_name == 'poly2Mask':
            preprocess = Poly2Mask(**cfg.poly2Mask)         # 多边形变掩码
        elif preprocess_name == 'resizeImage':
            preprocess = ResizeImage(**cfg.resizeImage)        # 多尺度训练
        elif preprocess_name == 'yOLOXResizeImage':
            preprocess = YOLOXResizeImage(**cfg.yOLOXResizeImage)  # YOLOX多尺度训练
        elif preprocess_name == 'randomFlipImage':
            preprocess = RandomFlipImage(**cfg.randomFlipImage)  # 随机翻转
        elif preprocess_name == 'normalizeImage':
            preprocess = NormalizeImage(**cfg.normalizeImage)     # 图片归一化。
        elif preprocess_name == 'normalizeBox':
            preprocess = NormalizeBox(**cfg.normalizeBox)        # 将物体的左上角坐标、右下角坐标中的横坐标/图片宽、纵坐标/图片高 以归一化坐标。
        elif preprocess_name == 'padBox':
            preprocess = PadBox(**cfg.padBox)         # 如果gt_bboxes的数量少于num_max_boxes，那么填充坐标是0的bboxes以凑够num_max_boxes。
        elif preprocess_name == 'bboxXYXY2XYWH':
            preprocess = BboxXYXY2XYWH(**cfg.bboxXYXY2XYWH)     # sample['gt_bbox']被改写为cx_cy_w_h格式。
        elif preprocess_name == 'permute':
            preprocess = Permute(**cfg.permute)    # 图片从HWC格式变成CHW格式
        elif preprocess_name == 'randomShape':
            resize_box = False
            if 'resize_box' in cfg.randomShape.keys():
                resize_box = cfg.randomShape['resize_box']
            preprocess = RandomShapeSingle(random_inter=cfg.randomShape['random_inter'], resize_box=resize_box)  # 多尺度训练。随机选一个尺度。也随机选一种插值方式。
        elif preprocess_name == 'gt2YoloTarget':
            preprocess = Gt2YoloTargetSingle(**cfg.gt2YoloTarget)   # 填写target张量。
        elif preprocess_name == 'padGT':
            preprocess = PadGTSingle(**cfg.padGT)   #
        else:
            raise NotImplementedError("Transform \'{}\' is not implemented.".format(preprocess_name))
        sample_transforms.append(preprocess)
    return sample_transforms


def get_batch_transforms(cfg):
    # batch_transforms
    batch_transforms = []
    for preprocess_name in cfg.batch_transforms_seq:
        if preprocess_name == 'randomShape':
            preprocess = RandomShape(**cfg.randomShape)     # 多尺度训练。随机选一个尺度。也随机选一种插值方式。
        elif preprocess_name == 'normalizeImage':
            preprocess = NormalizeImage(**cfg.normalizeImage)     # 图片归一化。先除以255归一化，再减均值除以标准差
        elif preprocess_name == 'permute':
            preprocess = Permute(**cfg.permute)    # 图片从HWC格式变成CHW格式
        elif preprocess_name == 'squareImage':
            preprocess = SquareImage(**cfg.squareImage)    # 图片变正方形。
        elif preprocess_name == 'gt2YoloTarget':
            preprocess = Gt2YoloTarget(**cfg.gt2YoloTarget)   # 填写target张量。
        elif preprocess_name == 'padBatchSingle':
            use_padded_im_info = cfg.padBatchSingle['use_padded_im_info'] if 'use_padded_im_info' in cfg.padBatchSingle else True
            preprocess = PadBatchSingle(use_padded_im_info=use_padded_im_info)   # 填充黑边。使这一批图片有相同的大小。
        elif preprocess_name == 'padBatch':
            preprocess = PadBatch(**cfg.padBatch)                         # 填充黑边。使这一批图片有相同的大小。
        elif preprocess_name == 'gt2FCOSTarget':
            preprocess = Gt2FCOSTarget(**cfg.gt2FCOSTarget)   # 填写target张量。
        elif preprocess_name == 'gt2Solov2Target':
            preprocess = Gt2Solov2Target(**cfg.gt2Solov2Target)     # 填写target张量。
        elif preprocess_name == 'gt2RepPointsTargetSingle':
            preprocess = Gt2RepPointsTargetSingle(**cfg.gt2RepPointsTargetSingle)     # 填写target张量。
        elif preprocess_name == 'padGT':
            preprocess = PadGT(**cfg.padGT)   #
        else:
            raise NotImplementedError("Transform \'{}\' is not implemented.".format(preprocess_name))
        batch_transforms.append(preprocess)
    return batch_transforms



