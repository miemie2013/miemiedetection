import colorsys
import random

import numpy as np
import cv2
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

def save(name, img, label=None):
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach().numpy()
    if isinstance(label, torch.Tensor):
        label = label.cpu().detach().numpy()
    print(label)
    if label is None:
        boxes = []
        classes = []
        scores = []
    else:
        boxes = label[:, 1:5]
        classes = label[:, 0].astype(np.int64)
        scores = label[:, 0] * 0. + 1.

    image = img.transpose((1, 2, 0)).astype(np.uint8)


    # 定义颜色
    num_classes = 20
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))


    for box, score, cl in zip(boxes, scores, classes):
        if cl < 0:
            continue
        x0, y0, x1, y1 = box
        left = max(0, np.floor(x0 + 0.5).astype(int))
        top = max(0, np.floor(y0 + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x1 + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y1 + 0.5).astype(int))
        bbox_color = colors[cl]
        # bbox_thick = 1 if min(image_h, image_w) < 400 else 2
        bbox_thick = 1
        # cv2.rectangle(image, (left, top), (right, bottom), bbox_color, bbox_thick)
        bbox_mess = '%s: %.2f' % (class_names[cl], score)
        t_size = cv2.getTextSize(bbox_mess, 0, 0.5, thickness=1)[0]
        # cv2.rectangle(image, (left, top), (left + t_size[0], top - t_size[1] - 3), bbox_color, -1)
        # cv2.putText(image, bbox_mess, (left, top - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
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
yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

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



save("all_mosaic0.jpg", mosaic_img[0], all_mosaic_labels[0])
save("all_mosaic1.jpg", mosaic_img[1], all_mosaic_labels[1])


print()



