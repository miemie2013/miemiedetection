#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from .backbones.darknet import CSPDarknet, Darknet
from .backbones.resnet_vd import Resnet18Vd, Resnet50Vd


from .losses.yolov3_loss import YOLOv3Loss
from .losses.losses import IOUloss
from .losses.iou_losses import MyIOUloss, IouLoss, IouAwareLoss


from .heads.yolov3_head import YOLOv3Head
from .heads.yolox_head import YOLOXHead


from .necks.yolo_pafpn import YOLOPAFPN
from .necks.yolo_fpn import YOLOFPN


from .architectures.yolo import PPYOLO
from .architectures.yolox import YOLOX
