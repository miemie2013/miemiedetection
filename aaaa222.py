import colorsys
import random

import numpy as np
import cv2
import math
import torch
import torch.nn.functional as F

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


aa = 1

