#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @miemie2013

import os

from mmdet.exp import FCOS_RT_R50_FPN_4x_Exp


class Exp(FCOS_RT_R50_FPN_4x_Exp):
    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
