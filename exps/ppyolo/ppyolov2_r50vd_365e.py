#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @miemie2013

import os

from mmdet.exp import PPYOLOv2_R50VD_365e_Exp


class Exp(PPYOLOv2_R50VD_365e_Exp):
    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
