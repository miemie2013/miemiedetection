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

mosaic_img_numpy = mosaic_img[0].cpu().detach().numpy()
mosaic_labels_numpy = all_mosaic_labels[0].cpu().detach().numpy()
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

def torch_box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2):
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
        & (ar < ar_thr)
    )  # candidates



def my_warpAffine(img, transform_inverse_matrix, dsize, borderValue):
    h = img.shape[0]
    w = img.shape[1]
    out_h = dsize[0]
    out_w = dsize[1]
    out = np.zeros(dsize + (3,)).astype(np.float32)
    zero = np.zeros((3,)).astype(np.float32)
    for y in range(out_h):
        for x in range(out_w):
            out_xy = np.array([x, y, 1.]).astype(np.float32)
            ori_xy = transform_inverse_matrix @ out_xy   # 通过逆变换矩阵求变换前的坐标
            w_im = ori_xy[0]
            h_im = ori_xy[1]
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

def my_torch_warpAffine(img, transform_inverse_matrix, dsize, borderValue):
    N, _, h, w = img.shape
    out_h = dsize[0]
    out_w = dsize[1]

    device = img.device

    yv, xv = torch.meshgrid([torch.arange(out_h, dtype=torch.float32, device=device), torch.arange(out_w, dtype=torch.float32, device=device)])
    grid = torch.stack((xv, yv), 0).view(1, 2, out_h, out_w).repeat([N, 1, 1, 1])   # [N, 2, out_h, out_w]
    xy = torch.ones((N, 3, out_h, out_w), dtype=torch.float32, device=device)   # [N, 3, out_h, out_w]
    xy[:, :2, :, :] = grid
    xy = xy.reshape((1, N*3, out_h, out_w))   # [1, N*3, out_h, out_w]


    weight = transform_inverse_matrix.reshape((N*3, 3, 1, 1))   # [N*3, 3, 1, 1]
    ori_xy = F.conv2d(xy, weight, groups=N)   # [1, N*3, out_h, out_w]    matmul, 变换后的坐标和逆矩阵运算，得到变换之前的坐标
    ori_xy = ori_xy.reshape((N, 3, out_h, out_w))   # [N, 3, out_h, out_w]
    ori_xy = ori_xy[:, :2, :, :]              # [N, 2, out_h, out_w]
    ori_xy = ori_xy.permute((0, 2, 3, 1))     # [N, out_h, out_w, 2]

    # 映射到-1到1之间，迎合 F.grid_sample() 双线性插值
    ori_xy[:, :, :, 0] = ori_xy[:, :, :, 0] / (w - 1) * 2.0 - 1.0
    ori_xy[:, :, :, 1] = ori_xy[:, :, :, 1] / (h - 1) * 2.0 - 1.0
    transform_img = F.grid_sample(img, ori_xy, mode='bilinear', padding_mode='zeros', align_corners=True)  # [N, in_C, out_h, out_w]

    out = transform_img[0].cpu().detach().numpy()
    out = out.transpose((1, 2, 0))
    cv2.imwrite("warpAffine_out.jpg", out)
    out = transform_img[1].cpu().detach().numpy()
    out = out.transpose((1, 2, 0))
    cv2.imwrite("warpAffine_out2.jpg", out)

    return transform_img


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

    # translation_inverse_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # 平移矩阵，往x轴正方向平移 -img.shape[1] / 2, 往y轴正方向平移 -img.shape[0] / 2, 即平移后图片中心位于坐标系原点O
    x_translation = -img.shape[1] / 2
    y_translation = -img.shape[0] / 2
    translation_matrix = np.array([[1, 0, x_translation], [0, 1, y_translation], [0, 0, 1]])
    # 平移矩阵逆矩阵, 对应着逆变换
    translation_inverse_matrix = np.array([[1, 0, -x_translation], [0, 1, -y_translation], [0, 0, 1]])

    # Rotation and Scale
    R = np.eye(3)
    # a = random.uniform(-degrees, degrees)
    # a = -45.0
    # a = 30.0
    # a = 15.0
    a = 8.0
    # a = 0.0
    # s = random.uniform(scale[0], scale[1])
    s = 0.7
    # s = 1.0
    rotationMatrix2D = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
    R[:2] = rotationMatrix2D


    # 旋转矩阵，x轴正方向指向右，y轴正方向指向下时，代表着以坐标系原点O为中心，顺时针旋转theta角
    theta = -a * math.pi / 180
    rotation_matrix = np.array([[math.cos(theta), math.sin(-theta), 0], [math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
    # 旋转矩阵逆矩阵, 对应着逆变换
    rotation_inverse_matrix = np.array([[math.cos(theta), math.sin(theta), 0], [math.sin(-theta), math.cos(theta), 0], [0, 0, 1]])

    # 放缩矩阵
    scale_matrix = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]])
    # 放缩矩阵逆矩阵, 对应着逆变换
    scale_inverse_matrix = np.array([[1./s, 0, 0], [0, 1./s, 0], [0, 0, 1]])




    # Shear
    S = np.eye(3)
    # S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    # S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)
    # Shear1 = 63.43498823  # arctan2
    # Shear1 = 45.
    Shear1 = 10.
    # Shear1 = 0.
    # Shear2 = 63.43498823
    # Shear2 = 45.
    # Shear2 = 10.
    Shear2 = -7.
    # Shear2 = 0.
    # shear = 45.0
    # Shear1 = random.uniform(-shear, shear)
    # Shear2 = random.uniform(-shear, shear)
    S[0, 1] = math.tan(Shear1 * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(Shear2 * math.pi / 180)  # y shear (deg)


    # 切变矩阵，x轴正方向指向右，y轴正方向指向下时，代表着以坐标系原点O为中心，顺时针旋转theta角
    shear_matrix = np.array([[1, math.tan(Shear1 * math.pi / 180), 0], [math.tan(Shear2 * math.pi / 180), 1, 0], [0, 0, 1]])
    # 切变矩阵逆矩阵, 对应着逆变换
    fenmu = 1. - math.tan(Shear1 * math.pi / 180) * math.tan(Shear2 * math.pi / 180)
    shear_inverse_matrix = np.array([[1./fenmu, -math.tan(Shear1 * math.pi / 180)/fenmu, 0], [-math.tan(Shear2 * math.pi / 180)/fenmu, 1./fenmu, 0], [0, 0, 1]])

    # ggg = 5

    # Translation
    T = np.eye(3)
    # x_trans = random.uniform(0.5 - translate, 0.5 + translate) * width
    # y_trans = random.uniform(0.5 - translate, 0.5 + translate) * height
    x_trans = 264.2
    y_trans = 303.7
    T[0, 2] = x_trans
    T[1, 2] = y_trans

    # 平移矩阵，往x轴正方向平移 x_trans, 往y轴正方向平移 y_trans
    translation2_matrix = np.array([[1, 0, x_trans], [0, 1, y_trans], [0, 0, 1]])
    # 平移矩阵逆矩阵, 对应着逆变换
    translation2_inverse_matrix = np.array([[1, 0, -x_trans], [0, 1, -y_trans], [0, 0, 1]])



    # Combined rotation matrix
    M = T @ S @ R @ C  # order of operations (right to left) is IMPORTANT

    # 通过变换后的坐标寻找变换之前的坐标，由果溯因，使用逆矩阵求解初始坐标。
    transform_inverse_matrix = translation_inverse_matrix @ rotation_inverse_matrix @ scale_inverse_matrix @ shear_inverse_matrix @ translation2_inverse_matrix
    transform_matrix = translation2_matrix @ shear_matrix @ scale_matrix @ rotation_matrix @ translation_matrix
    M222 = translation2_matrix @ shear_matrix @ scale_matrix @ rotation_matrix @ translation_matrix
    ddd = np.mean((M222 - M)**2)
    print('dddddMMM=%.6f' % ddd)

    ###########################
    # For Aug out of Mosaic
    # s = 1.
    # M = np.eye(3)
    ###########################

    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            pass
        else:  # affine
            # img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))


            # img2222 = my_warpAffine(img, M2, dsize=(int(2*height), int(2*width)), borderValue=(0, 0, 0))
            img2222 = my_warpAffine(img, transform_inverse_matrix, dsize=(height, width), borderValue=(0, 0, 0))
            cv2.imwrite("warpAffine2.jpg", img2222)
            print('dddddddddddd=%.6f' % ddd)
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(0, 0, 0))

            # img = cv2.warpAffine(img, M[:2], dsize=(int(2*width), int(2*height)), borderValue=(0, 0, 0))
            cv2.imwrite("warpAffine.jpg", img)
            ddd = np.mean((img2222 - img)**2)
            print('dddddddddddd=%.6f' % ddd)

            aaaaa = 11

    # Transform label coordinates
    targets222 = np.copy(targets)
    n = len(targets)
    if n:
        bboxes = targets[:, 1:5]
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
            n * 4, 2
        )  # x1y1, x2y2, x1y2, x2y1   gt4个顶点的坐标
        xy = xy @ transform_matrix.T  # transform, gt经过变换后4个顶点的坐标
        if perspective:
            pass
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
        i = box_candidates2(box1=bboxes.T * s, box2=xy.T)
        targets = targets[i]
        targets[:, 1:5] = xy[i]
        print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        print(xy)
    return img, targets



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
    translate:      aaaaaaaa
    scale:          (0.1, 2)   表示随机放缩0.1到2倍
    shear:          aaaaaaaa
    perspective:    aaaaaaaa
    border:         aaaaaaaa
    '''
    # targets = [cls, xyxy]
    N, ch, H, W = img.shape
    height = H + border[0] * 2  # shape(h,w,c)
    width = W + border[1] * 2
    device = img.device

    # Center

    # translation_inverse_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # 平移矩阵，往x轴正方向平移 -W / 2, 往y轴正方向平移 -H / 2, 即平移后图片中心位于坐标系原点O
    x_translation = -W / 2
    y_translation = -H / 2
    translation_matrix = torch.Tensor([[1, 0, x_translation], [0, 1, y_translation], [0, 0, 1]]).to(device)
    # 平移矩阵逆矩阵, 对应着逆变换
    translation_inverse_matrix = torch.Tensor([[1, 0, -x_translation], [0, 1, -y_translation], [0, 0, 1]]).to(device)

    # Rotation and Scale
    # a = random.uniform(-degrees, degrees)
    # a = -45.0
    # a = 30.0
    # a = 15.0
    a = 8.0
    # a = 0.0
    # s = random.uniform(scale[0], scale[1])
    s = 0.7
    # s = 1.0


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
    # S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    # S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)
    # Shear1 = 63.43498823  # arctan2
    # Shear1 = 45.
    Shear1 = 10.
    # Shear1 = 0.
    # Shear2 = 63.43498823
    # Shear2 = 45.
    # Shear2 = 10.
    Shear2 = -7.
    # Shear2 = 0.
    # shear = 45.0
    # Shear1 = random.uniform(-shear, shear)
    # Shear2 = random.uniform(-shear, shear)


    # 切变矩阵，x轴正方向指向右，y轴正方向指向下时，代表着以坐标系原点O为中心，顺时针旋转theta角
    shear_matrix = torch.Tensor([[1, math.tan(Shear1 * math.pi / 180), 0], [math.tan(Shear2 * math.pi / 180), 1, 0], [0, 0, 1]]).to(device)
    # 切变矩阵逆矩阵, 对应着逆变换
    fenmu = 1. - math.tan(Shear1 * math.pi / 180) * math.tan(Shear2 * math.pi / 180)
    shear_inverse_matrix = torch.Tensor([[1./fenmu, -math.tan(Shear1 * math.pi / 180)/fenmu, 0], [-math.tan(Shear2 * math.pi / 180)/fenmu, 1./fenmu, 0], [0, 0, 1]]).to(device)


    # Translation
    # x_trans = random.uniform(0.5 - translate, 0.5 + translate) * width
    # y_trans = random.uniform(0.5 - translate, 0.5 + translate) * height
    x_trans = 264.2
    y_trans = 303.7

    # 平移矩阵，往x轴正方向平移 x_trans, 往y轴正方向平移 y_trans
    translation2_matrix = torch.Tensor([[1, 0, x_trans], [0, 1, y_trans], [0, 0, 1]]).to(device)
    # 平移矩阵逆矩阵, 对应着逆变换
    translation2_inverse_matrix = torch.Tensor([[1, 0, -x_trans], [0, 1, -y_trans], [0, 0, 1]]).to(device)


    # 通过变换后的坐标寻找变换之前的坐标，由果溯因，使用逆矩阵求解初始坐标。
    transform_inverse_matrix = translation_inverse_matrix @ rotation_inverse_matrix @ scale_inverse_matrix @ shear_inverse_matrix @ translation2_inverse_matrix
    transform_matrix = translation2_matrix @ shear_matrix @ scale_matrix @ rotation_matrix @ translation_matrix

    transform_inverse_matrix = transform_inverse_matrix.unsqueeze(0).repeat([N, 1, 1])
    transform_matrix = transform_matrix.unsqueeze(0).repeat([N, 1, 1])
    transform_imgs = my_torch_warpAffine(img, transform_inverse_matrix, dsize=(height, width), borderValue=(0, 0, 0))


    # 变换gt坐标
    n = targets.shape[1]
    bboxes = targets[:, :, 1:5]
    bboxes = bboxes.reshape((-1, 4))
    # warp points
    xy = torch.ones((N * n * 4, 3), device=device)
    xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
        -1, 2
    )  # x1y1, x2y2, x1y2, x2y1   gt4个顶点的坐标

    xy = xy.reshape((N, n * 4, 3))   # [N, n * 4, 3]
    xy = xy.permute((0, 2, 1))       # [N, 3, n * 4]
    xy = xy.reshape((1, N*3, n, 4))       # [1, N*3, n, 4]
    weight = transform_matrix.reshape((N*3, 3, 1, 1))   # [N*3, 3, 1, 1]
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
    print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
    print(xy)

    # filter candidates
    keep = torch_box_candidates(box1=bboxes.reshape((N, n, 4)) * s, box2=xy)
    num_gts = keep.sum(1)
    G = num_gts.max()
    for batch_idx in range(N):
        targets[batch_idx, :num_gts[batch_idx], :1] = targets[batch_idx][keep[batch_idx]][:, :1]
        targets[batch_idx, :num_gts[batch_idx], 1:5] = xy[batch_idx][keep[batch_idx]]
        targets[batch_idx, num_gts[batch_idx]:, :] = 0
    if G < n:
        targets = targets[:, :G, :]
    print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa2222222222222222222222')
    print(targets)

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



mosaic_img222, all_mosaic_labels222 = torch_random_perspective(
    mosaic_img,
    all_mosaic_labels,
    degrees=degrees,
    translate=translate,
    scale=scale,
    shear=shear,
    perspective=perspective,
    border=[-input_h // 2, -input_w // 2],
)  # border to remove


save2("mosaic_img_numpy.jpg", mosaic_img_numpy, mosaic_labels_numpy)


print()



