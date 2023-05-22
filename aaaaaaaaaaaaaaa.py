import colorsys
import random

import numpy as np
import cv2
import math
import torch
import torch.nn.functional as F

from mmdet.data.datasets.mosaicdetection import get_mosaic_coordinate
from mmdet.models.ops import gather_1d
from mmdet.utils import get_classes

class_names = get_classes('class_names/voc_classes.txt')

batch_size = 2



dic2 = np.load('data.npz')
mosaic_samples_0_img = dic2['mosaic_samples_0_img']
mosaic_samples_0_labels = dic2['mosaic_samples_0_labels']
mosaic_samples_1_img = dic2['mosaic_samples_1_img']
mosaic_samples_1_labels = dic2['mosaic_samples_1_labels']
mosaic_samples_2_img = dic2['mosaic_samples_2_img']
mosaic_samples_2_labels = dic2['mosaic_samples_2_labels']
mosaic_samples_3_img = dic2['mosaic_samples_3_img']
mosaic_samples_3_labels = dic2['mosaic_samples_3_labels']
mixup_samples_0_img = dic2['mixup_samples_0_img']
mixup_samples_0_labels = dic2['mixup_samples_0_labels']

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
save("mosaic0.jpg", mosaic_samples_0_img[batch_idx], mosaic_samples_0_labels[batch_idx])
save("mosaic1.jpg", mosaic_samples_1_img[batch_idx])
save("mosaic2.jpg", mosaic_samples_2_img[batch_idx])
save("mosaic3.jpg", mosaic_samples_3_img[batch_idx])
save("mixup.jpg", mixup_samples_0_img[batch_idx])

aa = 1


mosaic_samples_0_img = torch.Tensor(mosaic_samples_0_img).to(torch.float32).cuda()
mosaic_samples_1_img = torch.Tensor(mosaic_samples_1_img).to(torch.float32).cuda()
mosaic_samples_2_img = torch.Tensor(mosaic_samples_2_img).to(torch.float32).cuda()
mosaic_samples_3_img = torch.Tensor(mosaic_samples_3_img).to(torch.float32).cuda()
mixup_samples_0_img = torch.Tensor(mixup_samples_0_img).to(torch.float32).cuda()


mosaic_samples_0_labels = torch.Tensor(mosaic_samples_0_labels).to(torch.float32).cuda()
mosaic_samples_1_labels = torch.Tensor(mosaic_samples_1_labels).to(torch.float32).cuda()
mosaic_samples_2_labels = torch.Tensor(mosaic_samples_2_labels).to(torch.float32).cuda()
mosaic_samples_3_labels = torch.Tensor(mosaic_samples_3_labels).to(torch.float32).cuda()
mixup_samples_0_labels = torch.Tensor(mixup_samples_0_labels).to(torch.float32).cuda()


imgs = mosaic_samples_0_img

# ---------------------- Mosaic ----------------------
N, C, input_h, input_w = imgs.shape

# yc, xc = s, s  # mosaic center x, y
# yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
# xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))
yc = int(640*1.2)
xc = int(640*1.5)

mosaic_labels = [mosaic_samples_0_labels, mosaic_samples_1_labels, mosaic_samples_2_labels, mosaic_samples_3_labels]
mosaic_img = torch.ones((N, C, input_h * 2, input_w * 2), dtype=imgs.dtype, device=imgs.device) * 114


max_labels = mosaic_samples_0_labels.shape[1]
all_mosaic_labels = []
for i_mosaic, img in enumerate([mosaic_samples_0_img, mosaic_samples_1_img, mosaic_samples_2_img, mosaic_samples_3_img]):
    # suffix l means large image, while s means small image in mosaic aug.
    (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = \
        get_mosaic_coordinate(None, i_mosaic, xc, yc, input_w, input_h, input_h, input_w)

    labels = mosaic_labels[i_mosaic]
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
all_mosaic_labels[:, :, 1] = torch.clamp(all_mosaic_labels[:, :, 1], min=0., max=2*input_w-1)
all_mosaic_labels[:, :, 2] = torch.clamp(all_mosaic_labels[:, :, 2], min=0., max=2*input_h-1)
all_mosaic_labels[:, :, 3] = torch.clamp(all_mosaic_labels[:, :, 3], min=0., max=2*input_w-1)
all_mosaic_labels[:, :, 4] = torch.clamp(all_mosaic_labels[:, :, 4], min=0., max=2*input_h-1)

is_real_gt = all_mosaic_labels[:, :, 0] >= 0.
is_area_valid = (all_mosaic_labels[:, :, 4] - all_mosaic_labels[:, :, 2]) * (all_mosaic_labels[:, :, 3] - all_mosaic_labels[:, :, 1]) > 8.
keep = is_real_gt & is_area_valid

save("all_mosaic0.jpg", mosaic_img[0], all_mosaic_labels[0])
save("all_mosaic1.jpg", mosaic_img[1], all_mosaic_labels[1])



scale = (0.1, 2)
mosaic_prob = 1.0
mixup_prob = 1.0
hsv_prob = 1.0
flip_prob = 0.5
degrees = 10.0
translate = 0.1
mixup_scale = (0.5, 1.5)
shear = 2.0
perspective = 0.0

mosaic_img_numpy = mosaic_img[1].cpu().detach().numpy()
mosaic_labels_numpy = all_mosaic_labels[1].cpu().detach().numpy()
mosaic_img_numpy = mosaic_img_numpy.transpose((1, 2, 0))




def box_candidates2(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2):
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



_constant_cache = dict()

def constant(value, shape=None, dtype=None, device=None, memory_format=None):
    value = np.asarray(value)
    if shape is not None:
        shape = tuple(shape)
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device('cpu')
    if memory_format is None:
        memory_format = torch.contiguous_format

    key = (value.shape, value.dtype, value.tobytes(), shape, dtype, device, memory_format)
    tensor = _constant_cache.get(key, None)
    if tensor is None:
        tensor = torch.as_tensor(value.copy(), dtype=dtype, device=device)
        if shape is not None:
            tensor, _ = torch.broadcast_tensors(tensor, torch.empty(shape))
        tensor = tensor.contiguous(memory_format=memory_format)
        _constant_cache[key] = tensor
    return tensor

def matrix(*rows, device=None):
    assert all(len(row) == len(rows[0]) for row in rows)
    elems = [x for row in rows for x in row]
    ref = [x for x in elems if isinstance(x, torch.Tensor)]
    if len(ref) == 0:
        return constant(np.asarray(rows), device=device)
    assert device is None or device == ref[0].device
    elems = [x if isinstance(x, torch.Tensor) else constant(x, shape=ref[0].shape, device=ref[0].device) for x in elems]
    return torch.stack(elems, dim=-1).reshape(ref[0].shape + (len(rows), -1))


def translation2d_matrix(offset_x, offset_y, **kwargs):
    return matrix(
        [1, 0, offset_x],
        [0, 1, offset_y],
        [0, 0, 1],
        **kwargs)

def scale2d_matrix(scale_x, scale_y, **kwargs):
    return matrix(
        [scale_x, 0, 0],
        [0, scale_y, 0],
        [0, 0, 1],
        **kwargs)

def rotate2d_matrix(theta, scale=1., **kwargs):
    return matrix(
        [scale*torch.cos(theta), scale*torch.sin(-theta), 0],
        [scale*torch.sin(theta), scale*torch.cos(theta),  0],
        [0,                0,                 1],
        **kwargs)

def shear_matrix(theta1, theta2, **kwargs):
    shear_matrix_ = matrix([1, torch.tan(-theta1), 0],
                           [torch.tan(-theta2), 1, 0],
                           [0,                  0, 1],
                           **kwargs)
    # shear_matrix_[:, 0, :] *= (torch.cos(theta2) ** 2)
    # shear_matrix_[:, 1, :] *= (torch.cos(theta1) ** 2)
    return shear_matrix_


def my_warpAffine(img, trans_matrix, dsize, borderValue):
    h = img.shape[0]
    w = img.shape[1]
    out_h = dsize[0]
    out_w = dsize[1]
    out = np.zeros(dsize + (3,)).astype(np.float32)
    zero = np.zeros((3,)).astype(np.float32)
    for y in range(out_h):
        for x in range(out_w):
            ori_xy = np.array([x, y, 1.]).astype(np.float32)
            new_xy = trans_matrix @ ori_xy
            w_im = new_xy[0]
            h_im = new_xy[1]
            cond = h_im > -1 and w_im > -1 and h_im < h and w_im < w
            if cond:
                h_low = int(h_im)
                w_low = int(w_im)
                h_high = h_low + 1
                w_high = w_low + 1
                lh = h_im - h_low
                lw = w_im - w_low
                hh = 1 - lh
                hw = 1 - lw

                v1_cond = (h_low >= 0 and w_low >= 0)
                v2_cond = (h_low >= 0 and w_high <= w - 1)
                v3_cond = (h_high <= h - 1 and w_low >= 0)
                v4_cond = (h_high <= h - 1 and w_high <= w - 1)

                w1 = hh * hw
                w2 = hh * lw
                w3 = lh * hw
                w4 = lh * lw
                v1 = img[h_low, w_low, :] if v1_cond else np.copy(zero)
                v2 = img[h_low, w_high, :] if v2_cond else np.copy(zero)
                v3 = img[h_high, w_low, :] if v3_cond else np.copy(zero)
                v4 = img[h_high, w_high, :] if v4_cond else np.copy(zero)
                val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
                out[y, x, :] = val
            else:
                out[y, x, :] = 0.
    return out


def random_perspective2(
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
    translate:      aaaaaaaa
    scale:          (0.1, 2)   表示随机放缩0.1到2倍
    shear:          aaaaaaaa
    perspective:    aaaaaaaa
    border:         aaaaaaaa
    '''
    # targets = [cls, xyxy]
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)
    C2 = np.eye(3)
    C2[0, 2] = img.shape[1] / 2  # x translation (pixels)
    C2[1, 2] = img.shape[0] / 2  # y translation (pixels)

    # Rotation and Scale
    R = np.eye(3)
    # a = random.uniform(-degrees, degrees)
    # a = -45.0
    # a = 30.0
    # a = 15.0
    a = 0.0
    # s = random.uniform(scale[0], scale[1])
    # s = 0.7
    s = 1.0

    # torch impl
    img_torch = torch.Tensor(img).cuda()
    img_torch = img_torch.permute((2, 0, 1))
    img_torch = img_torch.unsqueeze(0)
    N, ch, H, W = img_torch.shape
    device = img_torch.device
    I_3 = torch.eye(3, device=device)
    G_inv = I_3
    # batch_size = 8
    # i = torch.floor(torch.rand([batch_size], device=device) * 4)
    translation2d_matrix_ = translation2d_matrix(offset_x=torch.Tensor([1., ]).to(device).to(torch.float32), offset_y=torch.Tensor([1., ]).to(device).to(torch.float32))
    rotationMatrix2D_torch = rotate2d_matrix(torch.Tensor([a, ]).to(device).to(torch.float32) * math.pi / 180, scale=1./s)
    # G_inv = rotationMatrix2D_torch @ translation2d_matrix_

    theta = a * math.pi / 180
    R2 = np.array([[math.cos(theta), math.sin(-theta), 0], [math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
    rotationMatrix2D = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
    R[:2] = rotationMatrix2D

    # Shear
    S = np.eye(3)
    # S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    # S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)
    Shear1 = 63.43498823  # arctan2
    # Shear1 = 15.
    # Shear1 = 0.
    Shear2 = 63.43498823
    # Shear2 = -15.
    # Shear2 = 0.
    # shear = 45.0
    # Shear1 = random.uniform(-shear, shear)
    # Shear2 = random.uniform(-shear, shear)
    S[0, 1] = math.tan(Shear1 * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(Shear2 * math.pi / 180)  # y shear (deg)


    S2 = np.array([[1., -math.tan(Shear1 * math.pi / 180), 0], [-math.tan(Shear2 * math.pi / 180), 1., 0], [0, 0, 1]])

    shear_matrix_ = shear_matrix(theta1=torch.Tensor([Shear1, ]).to(device).to(torch.float32) * math.pi / 180, theta2=torch.Tensor([Shear2, ]).to(device).to(torch.float32) * math.pi / 180)
    shear_matrix_1 = shear_matrix(theta1=torch.Tensor([Shear1, ]).to(device).to(torch.float32) * math.pi / 180, theta2=torch.Tensor([0., ]).to(device).to(torch.float32) * math.pi / 180)
    shear_matrix_2 = shear_matrix(theta1=torch.Tensor([0., ]).to(device).to(torch.float32) * math.pi / 180, theta2=torch.Tensor([Shear2, ]).to(device).to(torch.float32) * math.pi / 180)
    # G_inv = shear_matrix_ @ rotationMatrix2D_torch @ translation2d_matrix_
    # G_inv = shear_matrix_1 @ shear_matrix_2 @ rotationMatrix2D_torch @ translation2d_matrix_

    fffff = scale2d_matrix(scale_x=torch.Tensor([math.cos(Shear1 * math.pi / 180)**2, ]).to(device).to(torch.float32), scale_y=torch.Tensor([math.cos(Shear2 * math.pi / 180)**2, ]).to(device).to(torch.float32))
    G_inv = fffff @ shear_matrix_ @ rotationMatrix2D_torch @ translation2d_matrix_
    # ggg = 5

    # Translation
    # T = np.eye(3)
    # T[0, 2] = (
    #     random.uniform(0.5 - translate, 0.5 + translate) * width
    # )  # x translation (pixels)
    # T[1, 2] = (
    #     random.uniform(0.5 - translate, 0.5 + translate) * height
    # )  # y translation (pixels)

    # Combined rotation matrix
    # M = T @ S @ R @ C  # order of operations (right to left) is IMPORTANT
    # M = R
    M = S @ R @ C
    # M = R @ C
    # M2 = R2 @ C2
    M2 = C2 @ R2 @ S2

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
            # img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
            aaaaaaa1 = M[:2]
            img2222 = my_warpAffine(img, M2, dsize=(int(2*height), int(2*width)), borderValue=(0, 0, 0))
            # img2222 = my_warpAffine(img, M2, dsize=(int(1.5*height), int(0.75*width)), borderValue=(0, 0, 0))
            cv2.imwrite("warpAffine2.jpg", img2222)
            # img = cv2.warpAffine(img, M[:2], dsize=(int(0.75*width), int(1.5*height)), borderValue=(0, 0, 0))
            img = cv2.warpAffine(img, M[:2], dsize=(int(2*width), int(2*height)), borderValue=(0, 0, 0))
            cv2.imwrite("warpAffine.jpg", img)
            ddd = np.mean((img2222 - img)**2)
            print('dddddddddddd=%.6f' % ddd)

            # shape = [N, ch, height, width]
            shape = [N, ch, H, W]
            G_inv_ = G_inv[:, :2, :]
            aaaaaaa2 = G_inv_[0].cpu().detach().numpy()
            grid = F.affine_grid(theta=G_inv_, size=shape, align_corners=False)
            # grid = grid[:, H//2:, W//2:, :]   # 只取右下角部分的图片，相当于cpu实现时的translation(平移)操作
            # grid = grid[:, :H//2, :W//2, :]
            images = F.grid_sample(img_torch, grid, mode="bilinear", align_corners=False)

            images = images[0].cpu().detach().numpy()
            images = images.transpose((1, 2, 0))
            # ddd = np.mean((images - img)**2)
            # print('ddd=%.6f' % ddd)
            # cv2.imwrite("warpAffine3333.jpg", images)

            aaaaa = 11

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
        i = box_candidates2(box1=targets[:, :4].T * s, box2=xy.T)
        targets = targets[i]
        targets[:, :4] = xy[i]

    return img, targets



mosaic_img_numpy, mosaic_labels_numpy = random_perspective2(
    mosaic_img_numpy,
    mosaic_labels_numpy,
    degrees=degrees,
    translate=translate,
    scale=scale,
    shear=shear,
    perspective=perspective,
    border=[-input_h // 2, -input_w // 2],
)  # border to remove


save2("mosaic_img_numpy.jpg", mosaic_img_numpy, mosaic_labels_numpy)


print()



