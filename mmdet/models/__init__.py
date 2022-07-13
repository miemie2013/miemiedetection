#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from .backbones.darknet import CSPDarknet, Darknet
from .backbones.resnet_vd import Resnet18Vd, Resnet50Vd, Resnet101Vd
from .backbones.resnet_vb import Resnet50Vb
from .backbones.resnet import ResNet, ConvNormLayer, SELayer, BasicBlock, BottleNeck, Blocks, Res5Head
from .backbones.cspresnet import CSPResNet


from .losses.yolov3_loss import YOLOv3Loss
from .losses.losses import IOUloss
from .losses.iou_losses import MyIOUloss, IouLoss, IouAwareLoss
from .losses.fcos_loss import FCOSLoss


from .heads.yolov3_head import YOLOv3Head
from .heads.solov2_head import SOLOv2MaskHead, SOLOv2Head
from .heads.yolox_head import YOLOXHead
from .heads.fcos_head import FCOSHead
from .heads.ppyoloe_head import PPYOLOEHead


from .necks.yolo_pafpn import YOLOPAFPN
from .necks.yolo_fpn import YOLOFPN
from .necks.fpn import FPN
from .necks.custom_pan import CustomCSPPAN


from .architectures.yolo import PPYOLO
from .architectures.ppyoloe import PPYOLOE
from .architectures.yolox import YOLOX
from .architectures.fcos import FCOS
from .architectures.solo import SOLO

from .assigners.atss_assigner import ATSSAssigner
from .assigners.task_aligned_assigner import TaskAlignedAssigner

