import colorsys
import random

import numpy as np
import cv2
import math
import torch
import torch.nn.functional as F

from mmdet.data import torch_box_candidates
from mmdet.models.ops import gather_1d
from mmdet.utils import get_classes

class_names = get_classes('class_names/voc_classes.txt')

batch_size = 4

dic2 = np.load('data2.npz')
mosaic_imgs = dic2['mosaic_imgs']
all_mosaic_labels = dic2['all_mosaic_labels']
mixup_img = dic2['mixup_samples_0_img']
mixup_label = dic2['mixup_samples_0_labels']


# mosaic_samples_0_labels = mosaic_samples_0_labels[:, :3, :]
# mosaic_samples_1_labels = mosaic_samples_1_labels[:, :3, :]
# mosaic_samples_2_labels = mosaic_samples_2_labels[:, :3, :]
# mosaic_samples_3_labels = mosaic_samples_3_labels[:, :3, :]
# mixup_samples_0_labels = mixup_samples_0_labels[:, :3, :]


def save2(name, img, label=None):
    print(label)
    cv2.imwrite(name, img)


def save(name, img, label=None):
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach().numpy()
    if isinstance(label, torch.Tensor):
        label = label.cpu().detach().numpy()
    print(label)
    image = img.transpose((1, 2, 0))
    cv2.imwrite(name, image)

batch_idx = 0
save("mosaic_imgs0.jpg", mosaic_imgs[0])
save("mosaic_imgs1.jpg", mosaic_imgs[1])
save("mosaic_imgs2.jpg", mosaic_imgs[2])
save("mosaic_imgs3.jpg", mosaic_imgs[3])
save("mixup_img0.jpg", mixup_img[0])
save("mixup_img1.jpg", mixup_img[1])


mosaic_imgs = torch.Tensor(mosaic_imgs).to(torch.float32).cuda()
all_mosaic_labels = torch.Tensor(all_mosaic_labels).to(torch.float32).cuda()
mixup_img = torch.Tensor(mixup_img).to(torch.float32).cuda()
mixup_label = torch.Tensor(mixup_label).to(torch.float32).cuda()

# cxcywh2xyxy
mixup_label[:, :, 1:3] = mixup_label[:, :, 1:3] - mixup_label[:, :, 3:5] * 0.5
mixup_label[:, :, 3:5] = mixup_label[:, :, 1:3] + mixup_label[:, :, 3:5]


mixup_scale = (0.5, 1.5)


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

def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox


def mixup(origin_img, origin_labels, cp_img, cp_labels, input_dim, mixup_scale):
    # jit_factor = random.uniform(*mixup_scale)
    jit_factor = 0.78
    # jit_factor = 1.15
    # jit_factor = 1.0
    # FLIP = random.uniform(0, 1) > 0.5
    FLIP = True

    cp_scale_ratio = 1.0
    cp_img = cv2.resize(
        cp_img,
        (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
    )
    cp_scale_ratio *= jit_factor

    if FLIP:
        cp_img = cp_img[:, ::-1, :]

    origin_h, origin_w = cp_img.shape[:2]
    target_h, target_w = origin_img.shape[:2]
    if origin_h > target_h:
        # mixup的图片被放大时
        y_offset = random.randint(0, origin_h - target_h - 1)
        x_offset = random.randint(0, origin_w - target_w - 1)
        padded_cropped_img = cp_img[y_offset: y_offset + target_h, x_offset: x_offset + target_w]
    elif origin_h == target_h:
        x_offset, y_offset = 0, 0
        padded_cropped_img = cp_img
    else:
        # mixup的图片被缩小时
        padded_cropped_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3), dtype=np.uint8
        )
        padded_cropped_img[:origin_h, :origin_w] = cp_img
        cv2.imwrite("aaa.jpg", padded_cropped_img)
        x_offset, y_offset = 0, 0

    cp_bboxes_origin_np = adjust_box_anns(cp_labels[:, 1:5].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h)
    if FLIP:
        cp_bboxes_origin_np[:, 0::2] = origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1]
    cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
    cp_bboxes_transformed_np[:, 0::2] = np.clip(
        cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w
    )
    cp_bboxes_transformed_np[:, 1::2] = np.clip(
        cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h
    )
    keep_list = box_candidates(cp_bboxes_origin_np.T, cp_bboxes_transformed_np.T, 5)

    if keep_list.sum() >= 1.0:
        cls_labels = cp_labels[keep_list, 0:1].copy()
        box_labels = cp_bboxes_transformed_np[keep_list]
        labels = np.hstack((cls_labels, box_labels))
        origin_labels = np.vstack((origin_labels, labels))
        origin_img = origin_img.astype(np.float32)
        origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)

    return origin_img.astype(np.uint8), origin_labels

def torch_mixup(origin_img, origin_labels, cp_img, cp_labels, mixup_scale):
    N, ch, H, W = origin_img.shape
    device = origin_img.device
    # jit_factor = torch.rand([N], device=device) * (mixup_scale[1] - mixup_scale[0]) + mixup_scale[0]
    # 暂定所有sample使用相同的jit_factor, 以及 FLIP
    # jit_factor = torch.rand([1], device=device) * (mixup_scale[1] - mixup_scale[0]) + mixup_scale[0]
    jit_factor = 0.78
    # jit_factor = 1.78
    FLIP = True
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


bid = 1
origin_img22 = mosaic_imgs[bid].cpu().detach().numpy()
origin_img22 = origin_img22.transpose((1, 2, 0))
all_mosaic_labels222 = all_mosaic_labels[bid].cpu().detach().numpy()

mixup_img22 = mixup_img[bid].cpu().detach().numpy()
mixup_img22 = mixup_img22.transpose((1, 2, 0))
mixup_label222 = mixup_label[bid].cpu().detach().numpy()


input_dim = (640, 640)


mm_img22, mm_labels22 = mixup(origin_img22, all_mosaic_labels222, mixup_img22, mixup_label222, input_dim, mixup_scale)


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
    img = img.to(torch.int32)
    img = img.to(torch.float32)

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

    # HSV2BGR
    new_B, new_G, new_R = torch_HSV2BGR(new_hue, new_sat, new_val, max_angle)
    aug_imgs = torch.cat([new_B, new_G, new_R], 1)
    save("aug_imgs0.jpg", aug_imgs[0])
    save("aug_imgs1.jpg", aug_imgs[1])
    save("aug_imgs2.jpg", aug_imgs[2])
    save("aug_imgs3.jpg", aug_imgs[3])
    return aug_imgs


def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    aaaaaaa = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(aaaaaaa)
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge(
        (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
    ).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

cv2.imwrite("mm_img22.jpg", mm_img22)

augment_hsv(mm_img22)
cv2.imwrite("mm_img222.jpg", mm_img22)


mm_img, mm_labels = torch_mixup(mosaic_imgs, all_mosaic_labels, mixup_img, mixup_label, mixup_scale)

aaaaaaa = mm_labels[1].cpu().detach().numpy()
save("mm_img.jpg", mm_img[1], mm_labels)

# ---------------------- TrainTransform ----------------------

mm_img = torch_augment_hsv(mm_img)


aa = 1

