#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @miemie2013

import os
import sys

from mmdet.exp.fcos.fcos_method_base import FCOS_Method_Exp


class FCOS_RT_R50_FPN_4x_Exp(FCOS_Method_Exp):
    def __init__(self):
        super().__init__()
        # 判断是否是调试状态
        isDebug = True if sys.gettrace() else False
        if isDebug:
            print('Debug Mode.')
            self.data_dir = '../' + self.data_dir
            self.cls_names = '../' + self.cls_names
            self.output_dir = '../' + self.output_dir
