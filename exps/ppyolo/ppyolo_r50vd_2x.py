#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @miemie2013

import os

from mmdet.exp import PPYOLO_R50VD_2x_Exp


class Exp(PPYOLO_R50VD_2x_Exp):
    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
