简体中文 | [English](README_en.md)

# miemiedetection

## 概述
miemiedetection是女装大佬[咩酱](https://github.com/miemie2013)基于[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)进行二次开发的个人检测库（使用的深度学习框架为pytorch），以女装大佬咩酱的名字命名。miemiedetection是一个不需要安装的检测库，用户可以直接更改其代码改变执行逻辑，所见即所得！所以往miemiedetection里加入新的算法是一件很容易的事情（可以参考PPYOLO的写法往miemiedetection里加入新的算法）。得益于YOLOX的优秀架构，miemiedetection里的算法训练速度都非常快，数据读取不再是训练速度的瓶颈！目前miemiedetection支持YOLOX、PPYOLO、PPYOLOv2等算法，预计未来会加入更多算法，所以请大家点个star吧！

## 安装依赖

```
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
pytorch版本建议1.9.1或者更高。

## 支持的算法

- [YOLOX](docs/README_YOLOX.md)
- [PPYOLO](docs/README_PPYOLO.md)
- [PPYOLOv2](docs/README_PPYOLO.md)

## 传送门
cv算法交流q群：645796480
但是关于仓库的疑问尽量在Issues上提，避免重复解答。

B站不定时女装: [_糖蜜](https://space.bilibili.com/646843384)

本人微信公众号：miemie_2013

技术博客：https://blog.csdn.net/qq_27311165

AIStudio主页：[asasasaaawws](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/165135)

欢迎在GitHub或AIStudio上关注我（求粉）~

## 引用

```latex
 @article{yolox2021,
  title={YOLOX: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```
