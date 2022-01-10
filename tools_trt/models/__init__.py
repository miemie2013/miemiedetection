#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date:
#   Description :
#
# ================================================================

from .backbones.resnet_vd import Resnet18Vd, Resnet50Vd, Resnet101Vd



from .heads.yolov3_head import YOLOv3Head


from .necks.yolo_fpn import YOLOFPN


from .architectures.yolo import PPYOLO
