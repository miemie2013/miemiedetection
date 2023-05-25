#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date:
#   Description :
#
# ================================================================

import math
import random

import torch
import torch.nn.functional as F
import cv2
import copy
import numpy as np
from loguru import logger
from numbers import Number, Integral


def torch_box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2, area_thr2=0.5):
    '''
    box1 is ori gt,    shape=[N, n, 4]
    box2 is trans gt,  shape=[N, n, 4]
    '''
    # Compute candidate boxes which include follwing 5 things:
    # box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[:, :, 2] - box1[:, :, 0], box1[:, :, 3] - box1[:, :, 1]
    w2, h2 = box2[:, :, 2] - box2[:, :, 0], box2[:, :, 3] - box2[:, :, 1]
    ar = torch.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (
        (w2 > wh_thr)
        & (h2 > wh_thr)
        & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr)
        & (w2 * h2 > area_thr2)
        & (ar < ar_thr)
    )  # candidates


def torch_warpAffine(img, transform_inverse_matrix, dsize, borderValue):
    N, ch, h, w = img.shape
    out_h = dsize[0]
    out_w = dsize[1]
    device = img.device
    dtype = img.dtype

    yv, xv = torch.meshgrid([torch.arange(out_h, dtype=dtype, device=device), torch.arange(out_w, dtype=dtype, device=device)])
    grid = torch.stack((xv, yv), 0).view(1, 2, out_h, out_w).repeat([N, 1, 1, 1])   # [N, 2, out_h, out_w]
    xy = torch.ones((N, 3, out_h, out_w), dtype=dtype, device=device)   # [N, 3, out_h, out_w]
    xy[:, :2, :, :] = grid
    xy = xy.reshape((1, N*3, out_h, out_w))   # [1, N*3, out_h, out_w]

    weight = transform_inverse_matrix.reshape((N*3, 3, 1, 1))   # [N*3, 3, 1, 1]
    ori_xy = F.conv2d(xy, weight, groups=N)   # [1, N*3, out_h, out_w]    matmul, 变换后的坐标和逆矩阵运算，得到变换之前的坐标
    ori_xy = ori_xy.reshape((N, 3, out_h, out_w))   # [N, 3, out_h, out_w]
    ori_xy = ori_xy[:, :2, :, :]              # [N, 2, out_h, out_w]
    ori_xy = ori_xy.permute((0, 2, 3, 1))     # [N, out_h, out_w, 2]

    # ori_xy_2 = F.affine_grid(theta=transform_inverse_matrix[:, :2, :], size=(N, ch, out_h, out_w), align_corners=False)
    # 映射到-1到1之间，迎合 F.grid_sample() 双线性插值
    ori_xy[:, :, :, 0] = ori_xy[:, :, :, 0] / (w - 1) * 2.0 - 1.0
    ori_xy[:, :, :, 1] = ori_xy[:, :, :, 1] / (h - 1) * 2.0 - 1.0

    transform_img = F.grid_sample(img, ori_xy, mode='bilinear', padding_mode='zeros', align_corners=True)  # [N, in_C, out_h, out_w]
    return transform_img



def get_mosaic_coordinate2(mosaic_image, mosaic_index, xc, yc, w, h, input_h, input_w):
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


def torch_random_perspective(
    img,
    targets=(),
    degrees=10,
    translate=0.1,
    scale=(0.1, 2),
    shear=10,
    perspective=0.0,
    border=(0, 0),
):
    '''
    degrees:        如果是10, 代表随机旋转-10度到10度。degrees单位是角度而不是弧度。
    translate:      最后的平移变换的范围
    scale:          (0.1, 2)   表示随机放缩0.1到2倍
    shear:          Shear变换角度最大值
    perspective:    0.0
    '''
    # targets = [cls, xyxy]
    N, ch, H, W = img.shape
    height = H + border[0] * 2  # shape(h,w,c)
    width = W + border[1] * 2
    device = img.device
    dtype = img.dtype

    # 方案一：用for循环
    '''
    transform_inverse_matrixes = []
    transform_matrixes = []
    scales = []
    for batch_idx in range(N):
        # Center
        # translation_inverse_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # 平移矩阵，往x轴正方向平移 -W / 2, 往y轴正方向平移 -H / 2, 即平移后图片中心位于坐标系原点O
        x_translation = -W / 2
        y_translation = -H / 2
        translation_matrix = torch.Tensor([[1, 0, x_translation], [0, 1, y_translation], [0, 0, 1]]).to(device)
        # 平移矩阵逆矩阵, 对应着逆变换
        translation_inverse_matrix = torch.Tensor([[1, 0, -x_translation], [0, 1, -y_translation], [0, 0, 1]]).to(device)

        # Rotation and Scale
        a = random.uniform(-degrees, degrees)
        s = random.uniform(scale[0], scale[1])
        # 旋转矩阵，x轴正方向指向右，y轴正方向指向下时，代表着以坐标系原点O为中心，顺时针旋转theta角
        theta = -a * math.pi / 180
        rotation_matrix = torch.Tensor([[math.cos(theta), math.sin(-theta), 0], [math.sin(theta), math.cos(theta), 0], [0, 0, 1]]).to(device)
        # 旋转矩阵逆矩阵, 对应着逆变换
        rotation_inverse_matrix = torch.Tensor([[math.cos(theta), math.sin(theta), 0], [math.sin(-theta), math.cos(theta), 0], [0, 0, 1]]).to(device)
        # 放缩矩阵
        scale_matrix = torch.Tensor([[s, 0, 0], [0, s, 0], [0, 0, 1]]).to(device)
        # 放缩矩阵逆矩阵, 对应着逆变换
        scale_inverse_matrix = torch.Tensor([[1./s, 0, 0], [0, 1./s, 0], [0, 0, 1]]).to(device)

        # Shear
        Shear1 = random.uniform(-shear, shear)
        Shear2 = random.uniform(-shear, shear)


        # 切变矩阵，x轴正方向指向右，y轴正方向指向下时，代表着以坐标系原点O为中心，顺时针旋转theta角
        shear_matrix = torch.Tensor([[1, math.tan(Shear1 * math.pi / 180), 0], [math.tan(Shear2 * math.pi / 180), 1, 0], [0, 0, 1]]).to(device)
        # 切变矩阵逆矩阵, 对应着逆变换
        fenmu = 1. - math.tan(Shear1 * math.pi / 180) * math.tan(Shear2 * math.pi / 180)
        shear_inverse_matrix = torch.Tensor([[1./fenmu, -math.tan(Shear1 * math.pi / 180)/fenmu, 0], [-math.tan(Shear2 * math.pi / 180)/fenmu, 1./fenmu, 0], [0, 0, 1]]).to(device)


        # Translation
        x_trans = random.uniform(0.5 - translate, 0.5 + translate) * width
        y_trans = random.uniform(0.5 - translate, 0.5 + translate) * height
        # 平移矩阵，往x轴正方向平移 x_trans, 往y轴正方向平移 y_trans
        translation2_matrix = torch.Tensor([[1, 0, x_trans], [0, 1, y_trans], [0, 0, 1]]).to(device)
        # 平移矩阵逆矩阵, 对应着逆变换
        translation2_inverse_matrix = torch.Tensor([[1, 0, -x_trans], [0, 1, -y_trans], [0, 0, 1]]).to(device)


        # 通过变换后的坐标寻找变换之前的坐标，由果溯因，使用逆矩阵求解初始坐标。
        transform_inverse_matrix = translation_inverse_matrix @ rotation_inverse_matrix @ scale_inverse_matrix @ shear_inverse_matrix @ translation2_inverse_matrix
        transform_matrix = translation2_matrix @ shear_matrix @ scale_matrix @ rotation_matrix @ translation_matrix
        transform_inverse_matrixes.append(transform_inverse_matrix)
        transform_matrixes.append(transform_matrix)
        scales.append(s)
    transform_inverse_matrixes = torch.stack(transform_inverse_matrixes, 0)
    transform_matrixes = torch.stack(transform_matrixes, 0)
    scales = torch.Tensor(scales).to(device)
    scales = scales.reshape((N, 1, 1))
    '''

    eye3 = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat([N, 1, 1])
    # 方案二：向量化实现
    # Center
    # translation_inverse_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # 平移矩阵，往x轴正方向平移 -W / 2, 往y轴正方向平移 -H / 2, 即平移后图片中心位于坐标系原点O
    x_translation = -W / 2
    y_translation = -H / 2
    translation_matrix = eye3.clone()
    translation_matrix[:, 0, 2] = x_translation
    translation_matrix[:, 1, 2] = y_translation
    # 平移矩阵逆矩阵, 对应着逆变换
    translation_inverse_matrix = eye3.clone()
    translation_inverse_matrix[:, 0, 2] = -x_translation
    translation_inverse_matrix[:, 1, 2] = -y_translation

    # Rotation and Scale
    a = torch.rand([N], device=device) * 2 * degrees - degrees
    scales = torch.rand([N], device=device) * (scale[1] - scale[0]) + scale[0]
    # 旋转矩阵，x轴正方向指向右，y轴正方向指向下时，代表着以坐标系原点O为中心，顺时针旋转theta角
    theta = -a * math.pi / 180
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    rotation_matrix = eye3.clone()
    rotation_matrix[:, 0, 0] = cos_theta
    rotation_matrix[:, 0, 1] = -sin_theta
    rotation_matrix[:, 1, 0] = sin_theta
    rotation_matrix[:, 1, 1] = cos_theta
    # 旋转矩阵逆矩阵, 对应着逆变换
    rotation_inverse_matrix = eye3.clone()
    rotation_inverse_matrix[:, 0, 0] = cos_theta
    rotation_inverse_matrix[:, 0, 1] = sin_theta
    rotation_inverse_matrix[:, 1, 0] = -sin_theta
    rotation_inverse_matrix[:, 1, 1] = cos_theta
    # 放缩矩阵
    scale_matrix = eye3.clone()
    scale_matrix[:, 0, 0] = scales
    scale_matrix[:, 1, 1] = scales
    # 放缩矩阵逆矩阵, 对应着逆变换
    scale_inverse_matrix = eye3.clone()
    scale_inverse_matrix[:, 0, 0] = 1./scales
    scale_inverse_matrix[:, 1, 1] = 1./scales

    # Shear
    shear1 = torch.rand([N], device=device) * 2 * shear - shear
    shear2 = torch.rand([N], device=device) * 2 * shear - shear
    tan_shear1 = torch.tan(shear1 * math.pi / 180)
    tan_shear2 = torch.tan(shear2 * math.pi / 180)

    # 切变矩阵，x轴正方向指向右，y轴正方向指向下时，代表着以坐标系原点O为中心，顺时针旋转theta角
    shear_matrix = eye3.clone()
    shear_matrix[:, 0, 1] = tan_shear1
    shear_matrix[:, 1, 0] = tan_shear2
    # 切变矩阵逆矩阵, 对应着逆变换
    mul_ = 1. / (1. - tan_shear1 * tan_shear2)
    shear_inverse_matrix = eye3.clone()
    shear_inverse_matrix[:, 0, 0] = mul_
    shear_inverse_matrix[:, 0, 1] = -mul_ * tan_shear1
    shear_inverse_matrix[:, 1, 0] = -mul_ * tan_shear2
    shear_inverse_matrix[:, 1, 1] = mul_

    # Translation
    x_trans = torch.rand([N], device=device) * 2 * translate - translate + 0.5
    y_trans = torch.rand([N], device=device) * 2 * translate - translate + 0.5
    x_trans *= width
    y_trans *= height
    # 平移矩阵，往x轴正方向平移 x_trans, 往y轴正方向平移 y_trans
    translation2_matrix = eye3.clone()
    translation2_matrix[:, 0, 2] = x_trans
    translation2_matrix[:, 1, 2] = y_trans
    # 平移矩阵逆矩阵, 对应着逆变换
    translation2_inverse_matrix = eye3.clone()
    translation2_inverse_matrix[:, 0, 2] = -x_trans
    translation2_inverse_matrix[:, 1, 2] = -y_trans

    # 与for实现有小偏差
    # transform_inverse_matrixes = translation_inverse_matrix @ rotation_inverse_matrix @ scale_inverse_matrix @ shear_inverse_matrix @ translation2_inverse_matrix
    # transform_matrixes = translation2_matrix @ shear_matrix @ scale_matrix @ rotation_matrix @ translation_matrix
    transform_inverse_matrixes = torch.zeros_like(translation2_inverse_matrix)
    transform_matrixes = torch.zeros_like(translation2_inverse_matrix)
    for bi in range(N):
        # 通过变换后的坐标寻找变换之前的坐标，由果溯因，使用逆矩阵求解初始坐标。
        transform_inverse_matrixes[bi] = translation_inverse_matrix[bi] @ rotation_inverse_matrix[bi] @ scale_inverse_matrix[bi] @ shear_inverse_matrix[bi] @ translation2_inverse_matrix[bi]
        transform_matrixes[bi] = translation2_matrix[bi] @ shear_matrix[bi] @ scale_matrix[bi] @ rotation_matrix[bi] @ translation_matrix[bi]
    scales = scales.reshape((N, 1, 1))


    transform_imgs = torch_warpAffine(img, transform_inverse_matrixes, dsize=(height, width), borderValue=(0, 0, 0))


    # 变换gt坐标
    n = targets.shape[1]
    bboxes = targets[:, :, 1:5]
    bboxes = bboxes.reshape((-1, 4))
    # warp points
    xy = torch.ones((N * n * 4, 3), device=device, dtype=dtype)
    xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
        -1, 2
    )  # x1y1, x2y2, x1y2, x2y1   gt4个顶点的坐标

    xy = xy.reshape((N, n * 4, 3))   # [N, n * 4, 3]
    xy = xy.permute((0, 2, 1))       # [N, 3, n * 4]
    xy = xy.reshape((1, N*3, n, 4))       # [1, N*3, n, 4]
    weight = transform_matrixes.reshape((N*3, 3, 1, 1))   # [N*3, 3, 1, 1]
    xy = F.conv2d(xy, weight, groups=N)   # [1, N*3, n, 4]    matmul
    xy = xy.reshape((N, 3, n, 4))   # [N, 3, n, 4]

    x = xy[:, 0, :, :]  # [N, n, 4]
    y = xy[:, 1, :, :]  # [N, n, 4]

    x1, _ = x.min(2)
    x2, _ = x.max(2)
    y1, _ = y.min(2)
    y2, _ = y.max(2)
    # clip boxes
    x1 = torch.clamp(x1, min=0., max=width)
    x2 = torch.clamp(x2, min=0., max=width)
    y1 = torch.clamp(y1, min=0., max=height)
    y2 = torch.clamp(y2, min=0., max=height)
    xy = torch.stack((x1, y1, x2, y2), 2)   # [N, n, 4]

    # filter candidates
    keep = torch_box_candidates(box1=bboxes.reshape((N, n, 4)) * scales, box2=xy)
    num_gts = keep.sum(1)
    G = num_gts.max()
    for batch_idx in range(N):
        targets[batch_idx, :num_gts[batch_idx], :1] = targets[batch_idx][keep[batch_idx]][:, :1]
        targets[batch_idx, :num_gts[batch_idx], 1:5] = xy[batch_idx][keep[batch_idx]]
        targets[batch_idx, num_gts[batch_idx]:, :] = 0
    if G < n:
        targets = targets[:, :G, :]

    # visual and debug
    # logger.info('rrrrrrrrrrtttttttttttttttttttttttttttttttttttttt')
    # logger.info(targets[:, :10, :])
    # for batch_idx in range(N):
    #     imgggg = transform_imgs[batch_idx].cpu().detach().numpy()
    #     imgggg = imgggg.transpose((1, 2, 0))
    #     cv2.imwrite("%d.jpg"%batch_idx, imgggg)

    return transform_imgs, targets


def torch_mixup(origin_img, origin_labels, cp_img, cp_labels, mixup_scale):
    N, ch, H, W = origin_img.shape
    device = origin_img.device
    # jit_factor = torch.rand([N], device=device) * (mixup_scale[1] - mixup_scale[0]) + mixup_scale[0]
    # 暂定所有sample使用相同的jit_factor, 以及 FLIP
    jit_factor = random.uniform(*mixup_scale)
    FLIP = random.uniform(0, 1) < 0.5
    # jit_factor = 0.78
    # FLIP = True

    cp_img = F.interpolate(cp_img, scale_factor=jit_factor, align_corners=False, mode='bilinear')

    cp_scale_ratio = jit_factor

    if FLIP:
        cp_img = cp_img[:, :, :, torch.arange(cp_img.shape[3] - 1, -1, -1, device=device).long()]

    origin_h, origin_w = cp_img.shape[2:4]
    target_h, target_w = origin_img.shape[2:4]
    if origin_h > target_h:
        # mixup的图片被放大时
        y_offset = random.randint(0, origin_h - target_h - 1)
        x_offset = random.randint(0, origin_w - target_w - 1)
        padded_cropped_img = cp_img[:, :, y_offset: y_offset + target_h, x_offset: x_offset + target_w]
    elif origin_h == target_h:
        x_offset, y_offset = 0, 0
        padded_cropped_img = cp_img
    else:
        # mixup的图片被缩小时
        padded_cropped_img = F.pad(cp_img, [0, target_w - origin_w, 0, target_h - origin_h])
        # aaaaaaaaaa2 = cp_img[1].cpu().detach().numpy()
        # aaaaaaaaaa2 = aaaaaaaaaa2.transpose((1, 2, 0))
        # cv2.imwrite("aaa2.jpg", aaaaaaaaaa2)
        x_offset, y_offset = 0, 0

    cp_labels[:, :, 1] = torch.clamp(cp_labels[:, :, 1] * cp_scale_ratio, min=0., max=origin_w)
    cp_labels[:, :, 2] = torch.clamp(cp_labels[:, :, 2] * cp_scale_ratio, min=0., max=origin_h)
    cp_labels[:, :, 3] = torch.clamp(cp_labels[:, :, 3] * cp_scale_ratio, min=0., max=origin_w)
    cp_labels[:, :, 4] = torch.clamp(cp_labels[:, :, 4] * cp_scale_ratio, min=0., max=origin_h)

    if FLIP:
        ori_x1 = cp_labels[:, :, 1].clone()
        ori_x2 = cp_labels[:, :, 3]
        cp_labels[:, :, 1] = origin_w - ori_x2
        cp_labels[:, :, 3] = origin_w - ori_x1
    old_bbox = cp_labels[:, :, 1:5].clone()
    cp_labels[:, :, 1] = torch.clamp(cp_labels[:, :, 1] - x_offset, min=0., max=target_w)
    cp_labels[:, :, 2] = torch.clamp(cp_labels[:, :, 2] - y_offset, min=0., max=target_h)
    cp_labels[:, :, 3] = torch.clamp(cp_labels[:, :, 3] - x_offset, min=0., max=target_w)
    cp_labels[:, :, 4] = torch.clamp(cp_labels[:, :, 4] - y_offset, min=0., max=target_h)


    keep = torch_box_candidates(box1=old_bbox, box2=cp_labels[:, :, 1:5], wh_thr=5)
    num_gts = keep.sum(1)
    G = num_gts.max()
    n = cp_labels.shape[1]
    for batch_idx in range(N):
        cp_labels[batch_idx, :num_gts[batch_idx], :] = cp_labels[batch_idx][keep[batch_idx]][:, :]
        cp_labels[batch_idx, num_gts[batch_idx]:, :] = 0
    if G < n:
        cp_labels = cp_labels[:, :G, :]
    origin_labels = torch.cat([origin_labels, cp_labels], 1)
    origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img
    return origin_img, origin_labels

def torch_LUT(index, value):
    assert index.dtype == torch.int64
    assert index.ndim == 4   # [N, 1, H, W]
    assert value.ndim == 2   # [N, 256]
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
    val = max_BGR.clone()
    val = val.to(torch.int64)
    sat = torch.where(max_BGR > 0., 255. * (max_BGR - min_BGR) / max_BGR, torch.zeros_like(max_BGR))
    sat = (sat + 0.5).to(torch.int64)
    '''
    把 max_angle 分成3份，BGR每种颜色占用1/3的角度，
    当最大颜色值是B时，hue的取值范围是[angle_center*2 - angle_radius, angle_center*2 + angle_radius]
    当最大颜色值是G时，hue的取值范围是[angle_center   - angle_radius, angle_center   + angle_radius]
    当最大颜色值是R时，hue的取值范围是[0, angle_radius] U [max_angle - angle_radius, max_angle]
    '''
    hue = torch.where(arg_max == 0, (R - G) / (max_BGR - min_BGR + 1e-9) * angle_radius + angle_center * 2., torch.zeros_like(max_BGR))  # B
    hue = torch.where(arg_max == 1, (B - R) / (max_BGR - min_BGR + 1e-9) * angle_radius + angle_center, hue)  # G
    hue = torch.where((arg_max == 2) & (G >= B), (G - B) / (max_BGR - min_BGR + 1e-9) * angle_radius, hue)  # R
    hue = torch.where((arg_max == 2) & (G < B), (G - B) / (max_BGR - min_BGR + 1e-9) * angle_radius + max_angle, hue)  # R
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
    x = torch.arange(256, dtype=torch.int16, device=device).unsqueeze(0).repeat([N, 1])   # [N, 256]
    lut_hue = ((x * r[:, 0:1]) % int(max_angle)).int()
    lut_sat = torch.clamp(x * r[:, 1:2], min=0., max=255.).int()
    lut_val = torch.clamp(x * r[:, 2:3], min=0., max=255.).int()
    new_hue = torch_LUT(hue, lut_hue)
    new_sat = torch_LUT(sat, lut_sat)
    new_val = torch_LUT(val, lut_val)

    # HSV2BGR , new_B new_G new_R are  float32
    new_B, new_G, new_R = torch_HSV2BGR(new_hue, new_sat, new_val, max_angle)
    aug_imgs = torch.cat([new_B, new_G, new_R], 1)
    aug_imgs = aug_imgs.to(dtype)  # to float16
    return aug_imgs

def filter_gt(imgs, targets, mossalse):
    aaaaa = 1

def yolox_torch_aug(imgs, targets, mosaic_cache, mixup_cache,
                    mosaic_max_cached_images, mixup_max_cached_images, random_pop, exp, use_mosaic=False):
    if use_mosaic:
        sample0 = dict(img=imgs, labels=targets)
        mosaic_cache.append(sample0)
        mixup_cache.append(copy.deepcopy(sample0))
        if len(mosaic_cache) > mosaic_max_cached_images:
            if random_pop:
                index = random.randint(0, len(mosaic_cache) - 2)  # 原版是-1，小改动，肯定不会丢弃最后一张图片
            else:
                index = 0
            mosaic_cache.pop(index)
        if len(mixup_cache) > mixup_max_cached_images:
            if random_pop:
                index = random.randint(0, len(mixup_cache) - 2)  # 原版是-1，小改动，肯定不会丢弃最后一张图片
            else:
                index = 0
            mixup_cache.pop(index)

        if len(mosaic_cache) <= 4:
            mosaic_samples = [copy.deepcopy(sample0) for _ in range(4)]
            mixup_samples = [copy.deepcopy(sample0) for _ in range(1)]
        else:
            # get index of three other images
            indexes = [np.random.randint(0, len(mosaic_cache)) for _ in range(3)]
            mosaic_samples = [copy.deepcopy(mosaic_cache[i]) for i in indexes]
            mosaic_samples = [copy.deepcopy(sample0)] + mosaic_samples
            # get index of one other images
            indexes = [np.random.randint(0, len(mixup_cache)) for _ in range(1)]
            mixup_samples = [copy.deepcopy(mixup_cache[i]) for i in indexes]

        # imgs_ = mosaic_samples[1]['img']
        # targets_ = mosaic_samples[1]['labels']
        # dic = {}
        # dic['mosaic_samples_0_img'] = mosaic_samples[0]['img'].cpu().detach().numpy()
        # dic['mosaic_samples_0_labels'] = mosaic_samples[0]['labels'].cpu().detach().numpy()
        # dic['mosaic_samples_1_img'] = mosaic_samples[1]['img'].cpu().detach().numpy()
        # dic['mosaic_samples_1_labels'] = mosaic_samples[1]['labels'].cpu().detach().numpy()
        # dic['mosaic_samples_2_img'] = mosaic_samples[2]['img'].cpu().detach().numpy()
        # dic['mosaic_samples_2_labels'] = mosaic_samples[2]['labels'].cpu().detach().numpy()
        # dic['mosaic_samples_3_img'] = mosaic_samples[3]['img'].cpu().detach().numpy()
        # dic['mosaic_samples_3_labels'] = mosaic_samples[3]['labels'].cpu().detach().numpy()
        # dic['mixup_samples_0_img'] = mixup_samples[0]['img'].cpu().detach().numpy()
        # dic['mixup_samples_0_labels'] = mixup_samples[0]['labels'].cpu().detach().numpy()
        # np.savez('data', **dic)

        # ---------------------- Mosaic ----------------------
        N, C, input_h, input_w = imgs.shape

        # yc, xc = s, s  # mosaic center x, y
        yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
        xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

        mosaic_img = torch.ones((N, C, input_h * 2, input_w * 2), dtype=imgs.dtype, device=imgs.device) * 114

        all_mosaic_labels = []
        for i_mosaic, sample in enumerate(mosaic_samples):
            # suffix l means large image, while s means small image in mosaic aug.
            (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = \
                get_mosaic_coordinate2(None, i_mosaic, xc, yc, input_w, input_h, input_h, input_w)

            img = sample['img']
            labels = sample['labels']
            mosaic_img[:, :, l_y1:l_y2, l_x1:l_x2] = img[:, :, s_y1:s_y2, s_x1:s_x2]
            padw, padh = l_x1 - s_x1, l_y1 - s_y1
            nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # [N, ]  每张图片gt数
            G = nlabel.max()
            labels = labels[:, :G, :]  # [N, G, 5]   gt的cid、cxcywh, 单位是像素
            # 转xyxy格式
            labels[:, :, 1] = labels[:, :, 1] - labels[:, :, 3] * 0.5
            labels[:, :, 2] = labels[:, :, 2] - labels[:, :, 4] * 0.5
            labels[:, :, 3] = labels[:, :, 3] + labels[:, :, 1]
            labels[:, :, 4] = labels[:, :, 4] + labels[:, :, 2]
            labels[:, :, 1] += padw
            labels[:, :, 2] += padh
            labels[:, :, 3] += padw
            labels[:, :, 4] += padh
            all_mosaic_labels.append(labels)

        all_mosaic_labels = torch.cat(all_mosaic_labels, 1)

        # 如果有gt超出图片范围，面积会是0
        all_mosaic_labels[:, :, 1] = torch.clamp(all_mosaic_labels[:, :, 1], min=0., max=2 * input_w - 1)
        all_mosaic_labels[:, :, 2] = torch.clamp(all_mosaic_labels[:, :, 2], min=0., max=2 * input_h - 1)
        all_mosaic_labels[:, :, 3] = torch.clamp(all_mosaic_labels[:, :, 3], min=0., max=2 * input_w - 1)
        all_mosaic_labels[:, :, 4] = torch.clamp(all_mosaic_labels[:, :, 4], min=0., max=2 * input_h - 1)

        # mosaic_scale = (0.1, 2)
        # mosaic_prob = 1.0
        # mixup_prob = 1.0
        # hsv_prob = 1.0
        # flip_prob = 0.5
        # degrees = 10.0
        # translate = 0.1
        # mixup_scale = (0.5, 1.5)
        # shear = 2.0

        mosaic_imgs, all_mosaic_labels = torch_random_perspective(
            mosaic_img,
            all_mosaic_labels,
            degrees=exp.degrees,
            translate=exp.translate,
            scale=exp.mosaic_scale,
            shear=exp.shear,
            perspective=0.,
            border=[-input_h // 2, -input_w // 2],
        )  # border to remove

        # dic = {}
        # dic['mosaic_imgs'] = mosaic_imgs.cpu().detach().numpy()
        # dic['all_mosaic_labels'] = all_mosaic_labels.cpu().detach().numpy()
        # dic['mixup_samples_0_img'] = mixup_samples[0]['img'].cpu().detach().numpy()
        # dic['mixup_samples_0_labels'] = mixup_samples[0]['labels'].cpu().detach().numpy()
        # np.savez('data2', **dic)

        # ---------------------- Mixup ----------------------
        mixup_img = mixup_samples[0]['img']
        mixup_label = mixup_samples[0]['labels']
        # cxcywh2xyxy
        mixup_label[:, :, 1:3] = mixup_label[:, :, 1:3] - mixup_label[:, :, 3:5] * 0.5
        mixup_label[:, :, 3:5] = mixup_label[:, :, 1:3] + mixup_label[:, :, 3:5]
        mosaic_imgs, all_mosaic_labels = torch_mixup(mosaic_imgs, all_mosaic_labels, mixup_img, mixup_label, exp.mixup_scale)
    else:
        mosaic_imgs = imgs
        all_mosaic_labels = targets
        # cxcywh2xyxy
        all_mosaic_labels[:, :, 1:3] = all_mosaic_labels[:, :, 1:3] - all_mosaic_labels[:, :, 3:5] * 0.5
        all_mosaic_labels[:, :, 3:5] = all_mosaic_labels[:, :, 1:3] + all_mosaic_labels[:, :, 3:5]

    # ---------------------- TrainTransform ----------------------
    device = mosaic_imgs.device
    dtype = mosaic_imgs.dtype
    N, ch, H, W = mosaic_imgs.shape
    # - - - - - - - - hsv aug - - - - - - - -
    mosaic_imgs = torch_augment_hsv(mosaic_imgs)

    # - - - - - - - - flip aug - - - - - - - -
    flip_prob = 0.5
    flip = (torch.rand([N], device=device) < flip_prob).float()   # 1 means flip
    eye3 = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat([N, 1, 1])
    # 水平翻转矩阵
    horizonflip_matrix = eye3.clone()
    horizonflip_matrix[:, 0, 0] = 1. - 2. * flip
    # 水平翻转矩阵逆矩阵, 对应着逆变换
    horizonflip_inverse_matrix = horizonflip_matrix.clone()
    # 平移矩阵。翻转了才会向右平移W
    x_trans = flip * W
    translation_matrix = eye3.clone()
    translation_matrix[:, 0, 2] = x_trans
    # 平移矩阵逆矩阵, 对应着逆变换
    translation_inverse_matrix = eye3.clone()
    translation_inverse_matrix[:, 0, 2] = -x_trans

    # 与for实现有小偏差
    # transform_inverse_matrixes = horizonflip_inverse_matrix @ translation_inverse_matrix
    # transform_matrixes = translation_matrix @ horizonflip_matrix
    transform_inverse_matrixes = torch.zeros_like(translation_inverse_matrix)
    transform_matrixes = torch.zeros_like(translation_inverse_matrix)
    for bi in range(N):
        # 通过变换后的坐标寻找变换之前的坐标，由果溯因，使用逆矩阵求解初始坐标。
        transform_inverse_matrixes[bi] = horizonflip_inverse_matrix[bi] @ translation_inverse_matrix[bi]
        transform_matrixes[bi] = translation_matrix[bi] @ horizonflip_matrix[bi]
    transform_imgs = torch_warpAffine(mosaic_imgs, transform_inverse_matrixes, dsize=(H, W), borderValue=(0, 0, 0))


    # 变换gt坐标
    n = all_mosaic_labels.shape[1]
    bboxes = all_mosaic_labels[:, :, 1:5]
    bboxes = bboxes.reshape((-1, 4))
    # warp points
    xy = torch.ones((N * n * 4, 3), device=device, dtype=dtype)
    xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
        -1, 2
    )  # x1y1, x2y2, x1y2, x2y1   gt4个顶点的坐标

    xy = xy.reshape((N, n * 4, 3))   # [N, n * 4, 3]
    xy = xy.permute((0, 2, 1))       # [N, 3, n * 4]
    xy = xy.reshape((1, N*3, n, 4))       # [1, N*3, n, 4]
    weight = transform_matrixes.reshape((N*3, 3, 1, 1))   # [N*3, 3, 1, 1]
    xy = F.conv2d(xy, weight, groups=N)   # [1, N*3, n, 4]    matmul
    xy = xy.reshape((N, 3, n, 4))   # [N, 3, n, 4]

    x = xy[:, 0, :, :]  # [N, n, 4]
    y = xy[:, 1, :, :]  # [N, n, 4]

    x1, _ = x.min(2)
    x2, _ = x.max(2)
    y1, _ = y.min(2)
    y2, _ = y.max(2)
    # clip boxes
    x1 = torch.clamp(x1, min=0., max=W)
    x2 = torch.clamp(x2, min=0., max=W)
    y1 = torch.clamp(y1, min=0., max=H)
    y2 = torch.clamp(y2, min=0., max=H)
    xy = torch.stack((x1, y1, x2, y2), 2)   # [N, n, 4]

    # filter candidates
    area_thr2 = 8.
    keep = torch_box_candidates(box1=bboxes.reshape((N, n, 4)), box2=xy, area_thr2=area_thr2)
    num_gts = keep.sum(1)
    G = num_gts.max()
    for batch_idx in range(N):
        all_mosaic_labels[batch_idx, :num_gts[batch_idx], :1] = all_mosaic_labels[batch_idx][keep[batch_idx]][:, :1]
        all_mosaic_labels[batch_idx, :num_gts[batch_idx], 1:5] = xy[batch_idx][keep[batch_idx]]
        all_mosaic_labels[batch_idx, num_gts[batch_idx]:, :] = 0
    if G < n:
        all_mosaic_labels = all_mosaic_labels[:, :G, :]

    # visual and debug
    # logger.info('last transform...')
    # logger.info(flip)
    # logger.info(all_mosaic_labels[:, :10, :])
    # for batch_idx in range(N):
    #     imgggg = transform_imgs[batch_idx].cpu().detach().numpy()
    #     imgggg = imgggg.transpose((1, 2, 0))
    #     cv2.imwrite("%d.jpg"%batch_idx, imgggg)

    # xyxy2cxcywh
    all_mosaic_labels[:, :, 3:5] = all_mosaic_labels[:, :, 3:5] - all_mosaic_labels[:, :, 1:3]
    all_mosaic_labels[:, :, 1:3] = all_mosaic_labels[:, :, 1:3] + all_mosaic_labels[:, :, 3:5] * 0.5
    transform_imgs.requires_grad_(False)
    all_mosaic_labels.requires_grad_(False)
    return transform_imgs, all_mosaic_labels



