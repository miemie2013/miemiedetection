import colorsys
import random

import numpy as np
import cv2
import math
import torch
import torch.nn.functional as F



def torch_LUT(index, value):
    assert index.dtype == torch.int64
    out = value[index]
    return out

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
    hue = torch.where(arg_max == 0, (R - G) / (max_BGR - min_BGR + 1e-9) * angle_radius + angle_center*2., torch.zeros_like(max_BGR))   # B
    hue = torch.where(arg_max == 1, (B - R) / (max_BGR - min_BGR + 1e-9) * angle_radius + angle_center, hue)   # G
    hue = torch.where((arg_max == 2)&(G>=B), (G - B) / (max_BGR - min_BGR + 1e-9) * angle_radius, hue)   # R
    hue = torch.where((arg_max == 2)&(G< B), (G - B) / (max_BGR - min_BGR + 1e-9) * angle_radius + max_angle, hue)   # R
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
    G = torch.where(hi == 3, min_BGR - (H - angle_center*2) / angle_radius * (max_BGR - min_BGR + 1e-9), G)
    G = torch.where(hi == 4, min_BGR, G)
    R = torch.where(hi == 4, (H - angle_center*2) / angle_radius * (max_BGR - min_BGR + 1e-9) + min_BGR, R)

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
    # img = 76 92 105
    # img[:, 0, :, :] = 62
    # img[:, 1, :, :] = 92
    # img[:, 2, :, :] = 105
    # img[:, 0, :, :] = 97
    # img[:, 1, :, :] = 129
    # img[:, 2, :, :] = 138

    # img[:, 0, :, :] = 139
    # img[:, 1, :, :] = 117
    # img[:, 2, :, :] = 157

    # img[:, 0, :, :] = 117
    # img[:, 1, :, :] = 139
    # img[:, 2, :, :] = 157

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
    hue, sat, val = torch_BGR2HSV(img, max_angle)

    val33 = val[0, 0].cpu().detach().numpy()
    ddd = np.mean((val33 - val22)**2)
    print('ddd val=%.6f' % ddd)
    sat33 = sat[0, 0].cpu().detach().numpy()
    ddd = np.mean((sat33 - sat22.astype(np.float32))**2)
    print('ddd sat=%.6f' % ddd)
    hue33 = hue[0, 0].cpu().detach().numpy()
    ddd = np.mean((hue33 - hue22)**2)
    print('ddd hue=%.6f' % ddd)

    # 增强
    x = torch.arange(256, dtype=torch.int16, device=device)
    lut_hue = ((x * r[0]) % int(max_angle)).int()
    lut_sat = torch.clamp(x * r[1], min=0., max=255.).int()
    lut_val = torch.clamp(x * r[2], min=0., max=255.).int()
    new_hue = torch_LUT(hue, lut_hue)
    new_sat = torch_LUT(sat, lut_sat)
    new_val = torch_LUT(val, lut_val)

    # HSV2BGR
    B333, G333, R333 = torch_HSV2BGR(new_hue, new_sat, new_val, max_angle)
    new_max_BGR = new_val.clone()
    new_min_BGR = new_max_BGR - new_sat * new_max_BGR / 255.
    new_hue_numpy = new_hue.cpu().detach().numpy()
    new_sat_numpy = new_sat.cpu().detach().numpy()
    new_val_numpy = new_val.cpu().detach().numpy()
    new_max_BGR_numpy = new_max_BGR.cpu().detach().numpy()
    new_min_BGR_numpy = new_min_BGR.cpu().detach().numpy()

    '''
    out22 = np.zeros((hi_numpy.shape[0], 3, hi_numpy.shape[2], hi_numpy.shape[3]))
    hue = torch.where(arg_max == 0, (R - G) / (max_BGR - min_BGR + 1e-9) * angle_radius + angle_center*2., torch.zeros_like(max_BGR))   # B
    hue = torch.where(arg_max == 1, (B - R) / (max_BGR - min_BGR + 1e-9) * angle_radius + angle_center, hue)   # G
    hue = torch.where((arg_max == 2)&(G>=B), (G - B) / (max_BGR - min_BGR + 1e-9) * angle_radius, hue)   # R
    hue = torch.where((arg_max == 2)&(G< B), (G - B) / (max_BGR - min_BGR + 1e-9) * angle_radius + max_angle, hue)   # R
    
    当hi是3,4时， 最大颜色值是B。 当hi==3时，R < G，最小颜色是R； 当hi==4时，R > G，最小颜色是G
    当hi是1,2时， 最大颜色值是G。 当hi==1时，B < R，最小颜色是B； 当hi==2时，B > R，最小颜色是R
    当hi是0,5时， 最大颜色值是R。 当hi==0时，G > B，最小颜色是B； 当hi==5时，G < B，最小颜色是G
    '''
    '''
    for bi in range(hi_numpy.shape[0]):
        for ci in range(hi_numpy.shape[1]):
            for h_i in range(hi_numpy.shape[2]):
                for w_i in range(hi_numpy.shape[3]):
                    idx = hi_numpy[bi, ci, h_i, w_i]
                    if idx == 0:
                        out22[bi, 2, h_i, w_i] = new_max_BGR_numpy[bi, 0, h_i, w_i]
                        out22[bi, 0, h_i, w_i] = new_min_BGR_numpy[bi, 0, h_i, w_i]
                        out22[bi, 1, h_i, w_i] = (new_hue_numpy[bi, 0, h_i, w_i] - 0.) / angle_radius * (new_max_BGR_numpy[bi, 0, h_i, w_i] - new_min_BGR_numpy[bi, 0, h_i, w_i] + 1e-9) + new_min_BGR_numpy[bi, 0, h_i, w_i]
                    if idx == 5:
                        out22[bi, 2, h_i, w_i] = new_max_BGR_numpy[bi, 0, h_i, w_i]
                        out22[bi, 1, h_i, w_i] = new_min_BGR_numpy[bi, 0, h_i, w_i]
                        out22[bi, 0, h_i, w_i] = (new_hue_numpy[bi, 0, h_i, w_i] - max_angle) / angle_radius * (new_max_BGR_numpy[bi, 0, h_i, w_i] - new_min_BGR_numpy[bi, 0, h_i, w_i] + 1e-9) - new_min_BGR_numpy[bi, 0, h_i, w_i]
                        out22[bi, 0, h_i, w_i] *= -1
                    if idx == 3:
                        out22[bi, 0, h_i, w_i] = new_max_BGR_numpy[bi, 0, h_i, w_i]
                        out22[bi, 2, h_i, w_i] = new_min_BGR_numpy[bi, 0, h_i, w_i]
                        out22[bi, 1, h_i, w_i] = (new_hue_numpy[bi, 0, h_i, w_i] - angle_center*2) / angle_radius * (new_max_BGR_numpy[bi, 0, h_i, w_i] - new_min_BGR_numpy[bi, 0, h_i, w_i] + 1e-9) - new_min_BGR_numpy[bi, 0, h_i, w_i]
                        out22[bi, 1, h_i, w_i] *= -1
                    if idx == 4:
                        out22[bi, 0, h_i, w_i] = new_max_BGR_numpy[bi, 0, h_i, w_i]
                        out22[bi, 1, h_i, w_i] = new_min_BGR_numpy[bi, 0, h_i, w_i]
                        out22[bi, 2, h_i, w_i] = (new_hue_numpy[bi, 0, h_i, w_i] - angle_center*2) / angle_radius * (new_max_BGR_numpy[bi, 0, h_i, w_i] - new_min_BGR_numpy[bi, 0, h_i, w_i] + 1e-9) + new_min_BGR_numpy[bi, 0, h_i, w_i]
                    if idx == 1:
                        out22[bi, 1, h_i, w_i] = new_max_BGR_numpy[bi, 0, h_i, w_i]
                        out22[bi, 0, h_i, w_i] = new_min_BGR_numpy[bi, 0, h_i, w_i]
                        out22[bi, 2, h_i, w_i] = (new_hue_numpy[bi, 0, h_i, w_i] - angle_center) / angle_radius * (new_max_BGR_numpy[bi, 0, h_i, w_i] - new_min_BGR_numpy[bi, 0, h_i, w_i] + 1e-9) - new_min_BGR_numpy[bi, 0, h_i, w_i]
                        out22[bi, 2, h_i, w_i] *= -1
                    if idx == 2:
                        out22[bi, 1, h_i, w_i] = new_max_BGR_numpy[bi, 0, h_i, w_i]
                        out22[bi, 2, h_i, w_i] = new_min_BGR_numpy[bi, 0, h_i, w_i]
                        out22[bi, 0, h_i, w_i] = (new_hue_numpy[bi, 0, h_i, w_i] - angle_center) / angle_radius * (new_max_BGR_numpy[bi, 0, h_i, w_i] - new_min_BGR_numpy[bi, 0, h_i, w_i] + 1e-9) + new_min_BGR_numpy[bi, 0, h_i, w_i]


    # out22 = np.clip(out22, 0, 255).astype(np.uint8)

    '''

    img_hsv222 = cv2.merge((new_hue_numpy[0][0].astype(np.uint8), new_sat_numpy[0][0].astype(np.uint8), new_val_numpy[0][0].astype(np.uint8))).astype(np.uint8)
    img_222 = np.copy(img_hsv222)
    cv2.cvtColor(img_hsv222, cv2.COLOR_HSV2BGR, dst=img_222)  # no return needed
    aaaaaaaaaaaa1 = img_222
    # B333, G333, R333 = torch_HSV2BGR(new_hue, new_sat, new_val)
    out22 = torch.cat([B333, G333, R333], 1)
    asds = 1
    aaaaaaaaaaaa2 = out22[0].cpu().detach().numpy()
    aaaaaaaaaaaa2 = aaaaaaaaaaaa2.transpose((1, 2, 0))
    aaaaaaaaaaaa2 = aaaaaaaaaaaa2.astype(np.uint8)


    ddd = np.mean((aaaaaaaaaaaa2 - aaaaaaaaaaaa1)**2)
    print('dddddMMM=%.6f' % ddd)
    cv2.imwrite("aaaaaaaaaaaa1.jpg", aaaaaaaaaaaa1)
    cv2.imwrite("aaaaaaaaaaaa2.jpg", aaaaaaaaaaaa2)
    return 1



img = cv2.imread("assets/000000000019.jpg")
img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)

# cv2.imwrite("0000.jpg", img)

img = img.transpose((2, 0, 1))
img = torch.Tensor(img).to(torch.float32).cuda()
img = img.unsqueeze(0)


zzzzz = 1
# mm_img = torch_augment_hsv(img[:, :, 309:309+zzzzz, 219:219+zzzzz])
mm_img = torch_augment_hsv(img)



aaaa = 1



