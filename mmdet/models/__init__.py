#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from .backbones.darknet import CSPDarknet, Darknet


from .losses.losses import IOUloss


from .heads.yolo_head import YOLOXHead


from .necks.yolo_pafpn import YOLOPAFPN
from .necks.yolo_fpn import YOLOFPN


from .architectures.yolox import YOLOX
