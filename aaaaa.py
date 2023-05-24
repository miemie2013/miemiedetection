

import numpy as np
import cv2
import math
import random
import torch
import torch.nn.functional as F


border = (-320, -320)

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


img = torch.ones((2, 3, 1280, 1280)).to(torch.float32).cuda()



# targets = [cls, xyxy]
N, ch, H, W = img.shape
height = H + border[0] * 2  # shape(h,w,c)
width = W + border[1] * 2
device = img.device

# 方案一：用for循环

transform_inverse_matrixes33 = []
transform_matrixes33 = []
scales33 = []
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
    if batch_idx == 0:
        a = 7.18
        s = 0.75
        # a = 0.
        # s = 1.
    elif batch_idx == 1:
        a = -9.34
        s = 1.12
        # a = 0.
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
    Shear1 = random.uniform(-shear, shear)
    Shear2 = random.uniform(-shear, shear)
    if batch_idx == 0:
        Shear1 = 35.4
        Shear2 = -13.5
        # Shear1 = 0.
        # Shear2 = 0.
    elif batch_idx == 1:
        Shear1 = -29.34
        Shear2 = 18.12
        # Shear1 = 0.
        # Shear2 = 0.


    # 切变矩阵，x轴正方向指向右，y轴正方向指向下时，代表着以坐标系原点O为中心，顺时针旋转theta角
    shear_matrix = torch.Tensor([[1, math.tan(Shear1 * math.pi / 180), 0], [math.tan(Shear2 * math.pi / 180), 1, 0], [0, 0, 1]]).to(device)
    # 切变矩阵逆矩阵, 对应着逆变换
    fenmu = 1. - math.tan(Shear1 * math.pi / 180) * math.tan(Shear2 * math.pi / 180)
    shear_inverse_matrix = torch.Tensor([[1./fenmu, -math.tan(Shear1 * math.pi / 180)/fenmu, 0], [-math.tan(Shear2 * math.pi / 180)/fenmu, 1./fenmu, 0], [0, 0, 1]]).to(device)
    print(shear_matrix)
    print(shear_inverse_matrix)


    # Translation
    x_trans = random.uniform(0.5 - translate, 0.5 + translate) * width
    y_trans = random.uniform(0.5 - translate, 0.5 + translate) * height
    if batch_idx == 0:
        x_trans = 319.5
        y_trans = 275.3
    elif batch_idx == 1:
        x_trans = 289.7
        y_trans = 307.9
    # 平移矩阵，往x轴正方向平移 x_trans, 往y轴正方向平移 y_trans
    translation2_matrix = torch.Tensor([[1, 0, x_trans], [0, 1, y_trans], [0, 0, 1]]).to(device)
    # 平移矩阵逆矩阵, 对应着逆变换
    translation2_inverse_matrix = torch.Tensor([[1, 0, -x_trans], [0, 1, -y_trans], [0, 0, 1]]).to(device)


    # 通过变换后的坐标寻找变换之前的坐标，由果溯因，使用逆矩阵求解初始坐标。
    transform_inverse_matrix = translation_inverse_matrix @ rotation_inverse_matrix @ scale_inverse_matrix @ shear_inverse_matrix @ translation2_inverse_matrix
    transform_matrix = translation2_matrix @ shear_matrix @ scale_matrix @ rotation_matrix @ translation_matrix
    transform_inverse_matrixes33.append(transform_inverse_matrix)
    transform_matrixes33.append(transform_matrix)
    scales33.append(s)
transform_inverse_matrixes33 = torch.stack(transform_inverse_matrixes33, 0)
transform_matrixes33 = torch.stack(transform_matrixes33, 0)
scales33 = torch.Tensor(scales33).to(device)
scales33 = scales33.reshape((N, 1, 1))


# 方案二：向量化实现
# Center
# translation_inverse_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# 平移矩阵，往x轴正方向平移 -W / 2, 往y轴正方向平移 -H / 2, 即平移后图片中心位于坐标系原点O
x_translation = -W / 2
y_translation = -H / 2
translation_matrix = torch.Tensor([[1, 0, x_translation], [0, 1, y_translation], [0, 0, 1]]).to(device).unsqueeze(0).repeat([N, 1, 1])
# 平移矩阵逆矩阵, 对应着逆变换
translation_inverse_matrix = torch.Tensor([[1, 0, -x_translation], [0, 1, -y_translation], [0, 0, 1]]).to(device).unsqueeze(0).repeat([N, 1, 1])

# Rotation and Scale
# a = torch.rand([N], device=device) * 2 * degrees - degrees
# scales = torch.rand([N], device=device) * (scale[1] - scale[0]) + scale[0]
a = torch.Tensor([7.18, -9.34]).to(device)
scales = torch.Tensor([0.75, 1.12]).to(device)
# 旋转矩阵，x轴正方向指向右，y轴正方向指向下时，代表着以坐标系原点O为中心，顺时针旋转theta角
theta = -a * math.pi / 180
cos_theta = torch.cos(theta)
sin_theta = torch.sin(theta)
rotation_matrix = torch.eye(3, device=device).unsqueeze(0).repeat([N, 1, 1])
rotation_matrix[:, 0, 0] = cos_theta
rotation_matrix[:, 0, 1] = -sin_theta
rotation_matrix[:, 1, 0] = sin_theta
rotation_matrix[:, 1, 1] = cos_theta
# 旋转矩阵逆矩阵, 对应着逆变换
rotation_inverse_matrix = torch.eye(3, device=device).unsqueeze(0).repeat([N, 1, 1])
rotation_inverse_matrix[:, 0, 0] = cos_theta
rotation_inverse_matrix[:, 0, 1] = sin_theta
rotation_inverse_matrix[:, 1, 0] = -sin_theta
rotation_inverse_matrix[:, 1, 1] = cos_theta
# 放缩矩阵
scale_matrix = torch.eye(3, device=device).unsqueeze(0).repeat([N, 1, 1])
scale_matrix[:, 0, 0] = scales
scale_matrix[:, 1, 1] = scales
# 放缩矩阵逆矩阵, 对应着逆变换
scale_inverse_matrix = torch.eye(3, device=device).unsqueeze(0).repeat([N, 1, 1])
scale_inverse_matrix[:, 0, 0] = 1. / scales
scale_inverse_matrix[:, 1, 1] = 1. / scales

# Shear
# shear1 = torch.rand([N], device=device) * 2 * shear - shear
# shear2 = torch.rand([N], device=device) * 2 * shear - shear
shear1 = torch.Tensor([35.4, -29.34]).to(device)
shear2 = torch.Tensor([-13.5, 18.12]).to(device)
tan_shear1 = torch.tan(shear1 * math.pi / 180)
tan_shear2 = torch.tan(shear2 * math.pi / 180)

# 切变矩阵，x轴正方向指向右，y轴正方向指向下时，代表着以坐标系原点O为中心，顺时针旋转theta角
shear_matrix = torch.eye(3, device=device).unsqueeze(0).repeat([N, 1, 1])
shear_matrix[:, 0, 1] = tan_shear1
shear_matrix[:, 1, 0] = tan_shear2
# 切变矩阵逆矩阵, 对应着逆变换
mul_ = 1. / (1. - tan_shear1 * tan_shear2)
shear_inverse_matrix = torch.eye(3, device=device).unsqueeze(0).repeat([N, 1, 1])
shear_inverse_matrix[:, 0, 0] = mul_
shear_inverse_matrix[:, 0, 1] = -mul_ * tan_shear1
shear_inverse_matrix[:, 1, 0] = -mul_ * tan_shear2
shear_inverse_matrix[:, 1, 1] = mul_

# Translation
# x_trans = torch.rand([N], device=device) * 2 * translate - translate + 0.5
# y_trans = torch.rand([N], device=device) * 2 * translate - translate + 0.5
# x_trans *= width
# y_trans *= height
x_trans = torch.Tensor([319.5, 289.7]).to(device)
y_trans = torch.Tensor([275.3, 307.9]).to(device)
# 平移矩阵，往x轴正方向平移 x_trans, 往y轴正方向平移 y_trans
translation2_matrix = torch.eye(3, device=device).unsqueeze(0).repeat([N, 1, 1])
translation2_matrix[:, 0, 2] = x_trans
translation2_matrix[:, 1, 2] = y_trans
# 平移矩阵逆矩阵, 对应着逆变换
translation2_inverse_matrix = torch.eye(3, device=device).unsqueeze(0).repeat([N, 1, 1])
translation2_inverse_matrix[:, 0, 2] = -x_trans
translation2_inverse_matrix[:, 1, 2] = -y_trans

# 与for实现有小偏差
transform_inverse_matrixes2 = torch.zeros_like(translation2_inverse_matrix)
transform_matrixes2 = torch.zeros_like(translation2_inverse_matrix)
for bi in range(N):
    # 通过变换后的坐标寻找变换之前的坐标，由果溯因，使用逆矩阵求解初始坐标。
    transform_inverse_matrixes2[bi] = translation_inverse_matrix[bi] @ rotation_inverse_matrix[bi] @ scale_inverse_matrix[bi] @ shear_inverse_matrix[bi] @ translation2_inverse_matrix[bi]
    transform_matrixes2[bi] = translation2_matrix[bi] @ shear_matrix[bi] @ scale_matrix[bi] @ rotation_matrix[bi] @ translation_matrix[bi]
scales = scales.reshape((N, 1, 1))

# 通过变换后的坐标寻找变换之前的坐标，由果溯因，使用逆矩阵求解初始坐标。
# transform_inverse_matrixes2 = translation_inverse_matrix @ rotation_inverse_matrix @ scale_inverse_matrix @ shear_inverse_matrix @ translation2_inverse_matrix
# transform_matrixes2 = translation2_matrix @ shear_matrix @ scale_matrix @ rotation_matrix @ translation_matrix




qqqq1 = transform_inverse_matrixes2.cpu().detach().numpy()
qqqq2 = transform_inverse_matrixes33.cpu().detach().numpy()
ddd = np.sum((qqqq1 - qqqq2) ** 2)
print('dddddMMM=%.6f' % ddd)

qqqq3 = transform_matrixes2.cpu().detach().numpy()
qqqq4 = transform_matrixes33.cpu().detach().numpy()
ddd2 = np.sum((qqqq3 - qqqq4) ** 2)
print('dddddMMM=%.6f' % ddd2)


aaaaa = 1

