#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from .base_exp import BaseExp
from .build import get_exp

from .yolox.yolox_base import YOLOXExp

from .ppyolo.ppyolo_r18vd_base import PPYOLO_R18VD_Exp
from .ppyolo.ppyolo_r50vd_2x_base import PPYOLO_R50VD_2x_Exp
from .ppyolo.ppyolov2_r50vd_365e_base import PPYOLOv2_R50VD_365e_Exp
from .ppyolo.ppyolov2_r101vd_365e_base import PPYOLOv2_R101VD_365e_Exp

from .fcos.fcos_rt_r50_fpn_4x_base import FCOS_RT_R50_FPN_4x_Exp
from .fcos.fcos_r50_fpn_2x_base import FCOS_R50_FPN_2x_Exp


