#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# miemie2013: 在 __init__.py 里执行 configure_module() 会让YOLOX加速很多，
# 原理是 让每个worker都会执行 configure_module() 。
from .utils import configure_module
configure_module()

__version__ = "0.1.0"
