#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date:
#   Description :
#
#
# ================================================================

import math
import random
import time
from loguru import logger

import torch
import torch.nn.functional as F
import cv2
import numpy as np


_constant_cache = dict()


def torch_warpAffine(img, transform_inverse_matrix, dsize, borderValue):
    N, ch, h, w = img.shape
    out_h = dsize[0]
    out_w = dsize[1]
    device = img.device
    dtype = img.dtype

    key = (N, out_h, out_w, dtype, device)
    xy = _constant_cache.get(key, None)
    if xy is None:
        logger.info('cache xy...')
        yv, xv = torch.meshgrid(
            [torch.arange(out_h, dtype=dtype, device=device), torch.arange(out_w, dtype=dtype, device=device)])
        grid = torch.stack((xv, yv), 0).view(1, 2, out_h, out_w).repeat([N, 1, 1, 1])  # [N, 2, out_h, out_w]
        xy = torch.ones((N, 3, out_h, out_w), dtype=dtype, device=device)  # [N, 3, out_h, out_w]
        xy[:, :2, :, :] = grid
        xy = xy.reshape((1, N * 3, out_h, out_w))  # [1, N*3, out_h, out_w]
        _constant_cache[key] = xy

    weight = transform_inverse_matrix.reshape((N * 3, 3, 1, 1))  # [N*3, 3, 1, 1]
    ori_xy = F.conv2d(xy, weight, groups=N)  # [1, N*3, out_h, out_w]    matmul, 变换后的坐标和逆矩阵运算，得到变换之前的坐标
    ori_xy = ori_xy.reshape((N, 3, out_h, out_w))  # [N, 3, out_h, out_w]
    ori_xy = ori_xy[:, :2, :, :]  # [N, 2, out_h, out_w]
    ori_xy = ori_xy.permute((0, 2, 3, 1))  # [N, out_h, out_w, 2]

    # ori_xy_2 = F.affine_grid(theta=transform_inverse_matrix[:, :2, :], size=(N, ch, out_h, out_w), align_corners=False)
    # 映射到-1到1之间，迎合 F.grid_sample() 双线性插值
    ori_xy[:, :, :, 0] = ori_xy[:, :, :, 0] / (w - 1) * 2.0 - 1.0
    ori_xy[:, :, :, 1] = ori_xy[:, :, :, 1] / (h - 1) * 2.0 - 1.0

    transform_img = F.grid_sample(img, ori_xy, mode='bilinear', padding_mode='zeros',
                                  align_corners=True)  # [N, in_C, out_h, out_w]
    return transform_img

def torch_LUT(index, value):
    assert index.dtype == torch.int64
    assert index.ndim == 4  # [N, 1, H, W]
    assert value.ndim == 2  # [N, 256]
    N, _, H, W = index.shape
    _, _256 = value.shape
    offsets = torch.arange(end=N, dtype=torch.int64, device=index.device) * _256
    offsets = torch.reshape(offsets, (-1, 1, 1, 1))
    index2 = index + offsets
    outs = value.flatten()[index2]
    return outs


def torch_BGR2HSV(img, max_angle=180.):
    angle_center = max_angle / 3.
    angle_radius = max_angle / 6.

    B = img[:, 0:1, :, :]
    G = img[:, 1:2, :, :]
    R = img[:, 2:3, :, :]
    max_BGR, arg_max = torch.max(img, dim=1, keepdim=True)
    min_BGR, _ = torch.min(img, dim=1, keepdim=True)
    val = max_BGR
    val = val.to(torch.int64)
    sat = torch.where(max_BGR > 0., 255. * (max_BGR - min_BGR) / max_BGR, torch.zeros_like(max_BGR))
    sat = (sat + 0.5).to(torch.int64)
    '''
    把 max_angle 分成3份，BGR每种颜色占用1/3的角度，
    当最大颜色值是B时，hue的取值范围是[angle_center*2 - angle_radius, angle_center*2 + angle_radius]
    当最大颜色值是G时，hue的取值范围是[angle_center   - angle_radius, angle_center   + angle_radius]
    当最大颜色值是R时，hue的取值范围是[0, angle_radius] U [max_angle - angle_radius, max_angle]
    '''
    hue = torch.where(arg_max == 0, (R - G) / (max_BGR - min_BGR + 1e-9) * angle_radius + angle_center * 2.,
                      torch.zeros_like(max_BGR))  # B
    hue = torch.where(arg_max == 1, (B - R) / (max_BGR - min_BGR + 1e-9) * angle_radius + angle_center, hue)  # G
    hue = torch.where((arg_max == 2) & (G >= B), (G - B) / (max_BGR - min_BGR + 1e-9) * angle_radius, hue)  # R
    hue = torch.where((arg_max == 2) & (G < B), (G - B) / (max_BGR - min_BGR + 1e-9) * angle_radius + max_angle,
                      hue)  # R
    hue = torch.where(torch.abs(max_BGR - min_BGR) < 0.001, torch.zeros_like(max_BGR), hue)
    hue = (hue + 0.5).to(torch.int64)
    return hue, sat, val


def torch_HSV2BGR(H, S, V, max_angle=180.):
    '''
    让 hi = (new_hue / angle_radius).int()
    当hi是3,4时， 最大颜色值是B。 当hi==3时，R < G，最小颜色是R； 当hi==4时，R > G，最小颜色是G
    当hi是1,2时， 最大颜色值是G。 当hi==1时，B < R，最小颜色是B； 当hi==2时，B > R，最小颜色是R
    当hi是0,5时， 最大颜色值是R。 当hi==0时，G > B，最小颜色是B； 当hi==5时，G < B，最小颜色是G
    '''
    angle_center = max_angle / 3.
    angle_radius = max_angle / 6.
    hi = (H / angle_radius).int()
    hi = torch.clamp(hi, min=0, max=5)
    V = V.to(torch.float32)
    max_BGR = V
    min_BGR = max_BGR - S * max_BGR / 255.

    B = torch.zeros_like(V)
    G = torch.zeros_like(V)
    R = torch.zeros_like(V)

    B = torch.where((hi == 3) | (hi == 4), max_BGR, B)
    R = torch.where(hi == 3, min_BGR, R)
    G = torch.where(hi == 3, min_BGR - (H - angle_center * 2) / angle_radius * (max_BGR - min_BGR + 1e-9), G)
    G = torch.where(hi == 4, min_BGR, G)
    R = torch.where(hi == 4, (H - angle_center * 2) / angle_radius * (max_BGR - min_BGR + 1e-9) + min_BGR, R)

    G = torch.where((hi == 1) | (hi == 2), max_BGR, G)
    B = torch.where(hi == 1, min_BGR, B)
    R = torch.where(hi == 1, min_BGR - (H - angle_center) / angle_radius * (max_BGR - min_BGR + 1e-9), R)
    R = torch.where(hi == 2, min_BGR, R)
    B = torch.where(hi == 2, (H - angle_center) / angle_radius * (max_BGR - min_BGR + 1e-9) + min_BGR, B)

    R = torch.where((hi == 0) | (hi == 5), max_BGR, R)
    B = torch.where(hi == 0, min_BGR, B)
    G = torch.where(hi == 0, H / angle_radius * (max_BGR - min_BGR + 1e-9) + min_BGR, G)
    G = torch.where(hi == 5, min_BGR, G)
    B = torch.where(hi == 5, min_BGR - (H - max_angle) / angle_radius * (max_BGR - min_BGR + 1e-9), B)
    return B, G, R


def torch_augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4, max_angle=180.):
    N, ch, H, W = img.shape
    device = img.device
    dtype = img.dtype
    img = img.to(torch.int32)
    img = img.to(dtype)

    r = torch.rand([N, 3], device=device) * 2. - 1.
    gain = torch.Tensor([hgain, sgain, vgain]).to(device).unsqueeze(0).repeat([N, 1])
    r = r * gain + 1.

    # BGR2HSV
    hue, sat, val = torch_BGR2HSV(img, max_angle)

    # 增强
    x = torch.arange(256, dtype=torch.int16, device=device).unsqueeze(0).repeat([N, 1])  # [N, 256]
    lut_hue = ((x * r[:, 0:1]) % int(max_angle)).int()
    lut_sat = torch.clamp(x * r[:, 1:2], min=0., max=255.).int()
    lut_val = torch.clamp(x * r[:, 2:3], min=0., max=255.).int()
    new_hue = torch_LUT(hue, lut_hue)
    new_sat = torch_LUT(sat, lut_sat)
    new_val = torch_LUT(val, lut_val)

    # HSV2BGR , new_B new_G new_R are  float32
    new_B, new_G, new_R = torch_HSV2BGR(new_hue, new_sat, new_val, max_angle)
    aug_imgs = torch.cat([new_B, new_G, new_R], 1)
    aug_imgs = aug_imgs.to(dtype)
    return aug_imgs


def data_aug(imgs):

    # ---------------------- TrainTransform ----------------------
    device = imgs.device
    dtype = imgs.dtype
    N, ch, H, W = imgs.shape
    # - - - - - - - - hsv aug - - - - - - - -
    imgs = torch_augment_hsv(imgs)

    # - - - - - - - - flip aug - - - - - - - -
    flip_prob = 0.5
    flip = (torch.rand([N], device=device) < flip_prob).float()  # 1 means flip

    key = (N, "eye3")
    eye3 = _constant_cache.get(key, None)
    if eye3 is None:
        logger.info('cache eye3...')
        eye3 = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat([N, 1, 1])
        _constant_cache[key] = eye3
    # 水平翻转矩阵。先翻转再右移
    horizonflip_matrix = eye3.clone()
    horizonflip_matrix[:, 0, 0] = 1. - 2. * flip
    # 翻转了才会向右平移W
    horizonflip_matrix[:, 0, 2] = flip * W
    # 水平翻转矩阵逆矩阵, 对应着逆变换。
    horizonflip_inverse_matrix = horizonflip_matrix.clone()

    transform_imgs = torch_warpAffine(imgs, horizonflip_inverse_matrix, dsize=(H, W), borderValue=(0, 0, 0))

    transform_imgs.requires_grad_(False)
    return transform_imgs


