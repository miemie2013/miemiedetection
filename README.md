简体中文 | [English](README_en.md)

# miemiedetection

## 概述
miemiedetection是[咩酱](https://github.com/miemie2013)基于[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)进行二次开发的个人检测库（使用的深度学习框架为pytorch），实现了可变形卷积DCNv2、Matrix NMS等高难度算子，支持单机单卡、单机多卡、多机多卡训练模式（多卡训练模式建议使用Linux系统），支持Windows、Linux系统，以咩酱的名字命名。miemiedetection是一个不需要安装的检测库用户可以直接更改其代码改变执行逻辑，所见即所得！所以往miemiedetection里加入新的算法是一件很容易的事情（可以参考PPYOLO的写法往miemiedetection里加入新的算法）。得益于YOLOX的优秀架构，miemiedetection里的算法训练速度都非常快，数据读取不再是训练速度的瓶颈！目前miemiedetection支持YOLOX、PPYOLO、PPYOLOv2、PPYOLOE、SOLOv2等算法，预计未来会加入更多算法，所以请大家点个star吧！

## 安装依赖

```
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch==1.10.1+cu102 torchvision==0.11.2+cu102 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
```
torch版本建议1.10.1+cu102或者更高；torchvision版本建议0.11.2+cu102或者更高。

## 支持的算法

- [YOLOX](docs/README_YOLOX.md)
- [PPYOLO](docs/README_PPYOLO.md)
- [PPYOLOv2](docs/README_PPYOLO.md)
- [PPYOLOE](docs/README_PPYOLO.md)
- [PicoDet](readme_picodet.txt)

## Updates!!
* 【2023/04/17】 加入了PicoDet算法(基于PaddleDetection-release-2.6版本的代码复现)！实测可以复现官方的精度！转换权重、复现COCO上的精度相关命令见readme_picodet.txt，训练日志见train_coco/picodet_s_416_coco_lcnet_4gpu.txt
* 【2022/08/03】 PPYOLOv2、PPYOLO算法支持导出到NCNN！详情请参考[PPYOLO](docs/README_PPYOLO.md#NCNN) 文档的“NCNN”小节。
* 【2022/06/22】 PPYOLOE算法支持导出到NCNN！详情请参考[PPYOLOE](docs/README_PPYOLO.md#NCNN) 文档的“NCNN”小节。
* 【2022/05/15】 加入了PPYOLOE算法(基于PaddleDetection-release-2.4版本的代码复现)！


## 友情链接

- [miemieGAN](https://github.com/miemie2013/miemieGAN) miemieGAN是咩酱个人开发与维护的图像生成库，以咩酱的名字命名，实现了stylegan2ada等算法，目前文档完善中，欢迎大家试玩。


## 传送门

算法1群：645796480（人已满） 

算法2群：894642886 

粉丝群：704991252

关于仓库的疑问尽量在Issues上提，避免重复解答。

B站不定时女装: [_糖蜜](https://space.bilibili.com/646843384)

知乎不定时谢邀、写文章: [咩咩2013](https://www.zhihu.com/people/mie-mie-2013)

西瓜视频: [咩咩2013](https://www.ixigua.com/home/2088721227199148/?list_entrance=search)

微信：wer186259

本人微信公众号：miemie_2013

技术博客：https://blog.csdn.net/qq_27311165

AIStudio主页：[asasasaaawws](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/165135)

欢迎在GitHub或上面的平台关注我（求粉）~


## 打赏

如果你觉得这个仓库对你很有帮助，可以给我打钱↓

![Example 0](weixin/sk.png)

咩酱爱你哟！


## 引用

[miemiedetection](https://github.com/miemie2013/miemiedetection)

[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)

[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)

[PPYOLOE paper](https://arxiv.org/pdf/2203.16250.pdf)

```
 @article{yolox2021,
  title={YOLOX: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
@article{huang2021pp,
  title={PP-YOLOv2: A Practical Object Detector},
  author={Huang, Xin and Wang, Xinxin and Lv, Wenyu and Bai, Xiaying and Long, Xiang and Deng, Kaipeng and Dang, Qingqing and Han, Shumin and Liu, Qiwen and Hu, Xiaoguang and others},
  journal={arXiv preprint arXiv:2104.10419},
  year={2021}
}
@misc{long2020ppyolo,
title={PP-YOLO: An Effective and Efficient Implementation of Object Detector},
author={Xiang Long and Kaipeng Deng and Guanzhong Wang and Yang Zhang and Qingqing Dang and Yuan Gao and Hui Shen and Jianguo Ren and Shumin Han and Errui Ding and Shilei Wen},
year={2020},
eprint={2007.12099},
archivePrefix={arXiv},
primaryClass={cs.CV}
}
@misc{ppdet2019,
title={PaddleDetection, Object detection and instance segmentation toolkit based on PaddlePaddle.},
author={PaddlePaddle Authors},
howpublished = {\url{https://github.com/PaddlePaddle/PaddleDetection}},
year={2019}
}
```


