#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @miemie2013

import os
import sys

from mmdet.exp import SOLO_Method_Exp


class Exp(SOLO_Method_Exp):
    def __init__(self):
        super(Exp, self).__init__()
        # COCO2017 dataset。用来调试。
        # self.num_classes = 80
        # self.data_dir = '../COCO'
        # self.cls_names = 'class_names/coco_classes.txt'
        # self.ann_folder = "annotations"
        # self.train_ann = "instances_val2017.json"
        # self.val_ann = "instances_val2017.json"
        # self.train_image_folder = "val2017"
        # self.val_image_folder = "val2017"

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # ---------------- dataloader config ---------------- #
        # 默认是4。如果报错“OSError: [WinError 1455] 页面文件太小,无法完成操作”，设置为2或0解决。
        # 如果报错“self = reduction.pickle.load(from_parent) EOFError: Ran out of input”，设置为0解决。
        self.data_num_workers = 0
        self.eval_data_num_workers = 0

        # 判断是否是调试状态
        isDebug = True if sys.gettrace() else False
        if isDebug:
            print('Debug Mode.')
            self.data_dir = '../' + self.data_dir
            self.cls_names = '../' + self.cls_names
            self.output_dir = '../' + self.output_dir
