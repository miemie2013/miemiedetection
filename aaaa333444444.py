import colorsys
import random

import numpy as np
import cv2
import math
import torch
import torch.nn.functional as F


def my_LUT(index, value):
    out = np.copy(index)
    h, w = index.shape
    for j in range(h):
        for i in range(w):
            out[j, i] = value[index[j, i]]
    return out


def torch_LUT(index, value):
    assert index.dtype == torch.int64
    out = value[index]
    return out


def rgb2hsv(img):
    h = img.shape[0]
    w = img.shape[1]
    H = np.zeros((h,w),np.float32)
    S = np.zeros((h, w), np.float32)
    V = np.zeros((h, w), np.float32)
    r,g,b = cv2.split(img)
    r, g, b = r/255.0, g/255.0, b/255.0
    for i in range(0, h):
        for j in range(0, w):
            mx = max((b[i, j], g[i, j], r[i, j]))
            mn = min((b[i, j], g[i, j], r[i, j]))
            dt=mx-mn

            if mx == mn:
                H[i, j] = 0
            elif mx == r[i, j]:
                if g[i, j] >= b[i, j]:
                    H[i, j] = (60 * ((g[i, j]) - b[i, j]) / dt)
                else:
                    H[i, j] = (60 * ((g[i, j]) - b[i, j]) / dt)+360
            elif mx == g[i, j]:
                H[i, j] = 60 * ((b[i, j]) - r[i, j]) / dt + 120
            elif mx == b[i, j]:
                H[i, j] = 60 * ((r[i, j]) - g[i, j]) / dt+ 240
            H[i,j] =int( H[i,j] / 2)

            #S
            if mx == 0:
                S[i, j] = 0
            else:
                S[i, j] =int( dt/mx*255)
            #V
            V[i, j] =int( mx*255)

    return H, S, V


def torch_augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    N, ch, H, W = img.shape
    device = img.device
    img = img.to(torch.int32)   # 要先转成整型
    img = img.to(torch.float32)   # 要先转成整型
    r = torch.rand([3], device=device) * 2 - 1.
    gain = torch.Tensor([hgain, sgain, vgain]).to(device)
    r = r * gain + 1.
    r[0] = 0.98
    r[1] = 1.4
    r[2] = 0.76

    r1111 = r.cpu().detach().numpy()
    img_111 = img[0].cpu().detach().numpy()

    # 对于 np.uint8 和 np.float32 来说，cv2.cvtColor算出来的hsv是不一样的。这里对齐 np.uint8 的
    img_111 = img_111.transpose((1, 2, 0)).astype(np.uint8)
    # img_111 = img_111.transpose((1, 2, 0))

    # hue222, sat222, val222 = rgb2hsv(img_111)
    aaaaaaa = cv2.cvtColor(img_111, cv2.COLOR_BGR2HSV)
    hue22, sat22, val22 = cv2.split(aaaaaaa)


    # https://blog.csdn.net/u010251191/article/details/30113385
    # BGR2HSV
    B = img[:, 0:1, :, :]
    G = img[:, 1:2, :, :]
    R = img[:, 2:3, :, :]
    max_BGR, arg_max = torch.max(img, dim=1, keepdim=True)
    min_BGR, _ = torch.min(img, dim=1, keepdim=True)
    val = max_BGR.clone()
    val = val.to(torch.int64)
    sat = torch.where(max_BGR > 0., 255. * (max_BGR - min_BGR) / max_BGR, torch.zeros_like(max_BGR))
    sat = (sat + 0.5).to(torch.int64)
    max_angle = 180.
    angle_center = max_angle / 3.
    angle_radius = max_angle / 6.
    '''
    把 max_angle 分成3份，BGR每种颜色占用1/3的角度，
    当最大颜色值是B时，hue的取值范围是[angle_center*2 - angle_radius, angle_center*2 + angle_radius]
    当最大颜色值是G时，hue的取值范围是[angle_center   - angle_radius, angle_center   + angle_radius]
    当最大颜色值是R时，hue的取值范围是[0, angle_radius] U [max_angle - angle_radius, max_angle]
    '''
    hue = torch.where(arg_max == 0, (R - G) / (max_BGR - min_BGR + 1e-9) * angle_radius + angle_center*2., torch.zeros_like(max_BGR))   # B
    hue = torch.where(arg_max == 1, (B - R) / (max_BGR - min_BGR + 1e-9) * angle_radius + angle_center, hue)   # G
    hue = torch.where((arg_max == 2)&(G>=B), (G - B) / (max_BGR - min_BGR + 1e-9) * angle_radius + 0., hue)   # R
    hue = torch.where((arg_max == 2)&(G< B), (G - B) / (max_BGR - min_BGR + 1e-9) * angle_radius + max_angle, hue)   # R
    hue = torch.where(torch.abs(max_BGR - min_BGR) < 0.001, torch.zeros_like(max_BGR), hue)
    hue = (hue + 0.5).to(torch.int64)

    x = torch.arange(256, dtype=torch.int16, device=device)
    lut_hue = ((x * r[0]) % int(max_angle)).int()
    lut_sat = torch.clamp(x * r[1], min=0., max=255.).int()
    lut_val = torch.clamp(x * r[2], min=0., max=255.).int()
    new_h = torch_LUT(hue, lut_hue)
    new_s = torch_LUT(sat, lut_sat)
    new_v = torch_LUT(val, lut_val)

    # HSV2BGR

    aaa = 1

    img_hsv = cv2.merge(
        (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
    ).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed



img = cv2.imread("assets/000000000019.jpg")
img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)

# cv2.imwrite("0000.jpg", img)

img = img.transpose((2, 0, 1))
img = torch.Tensor(img).to(torch.float32).cuda()
img = img.unsqueeze(0)

mm_img = torch_augment_hsv(img)



aaaa = 1



