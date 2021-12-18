## 快速开始

(1)环境搭建

需要安装cuda10，Pytorch1.x。以及随意版本的Paddle来转换权重：
```
pip install paddlepaddle-gpu==1.8.4.post107 -i https://mirror.baidu.com/pypi/simple
```

(2)下载预训练模型

下载PaddleDetection的ppyolo.pdparams。如果你使用Linux，请使用以下命令：
```
wget https://paddlemodels.bj.bcebos.com/object_detection/ppyolo.pdparams
```

如果你使用Windows，请复制以下网址到浏览器或迅雷下载：
```
https://paddlemodels.bj.bcebos.com/object_detection/ppyolo.pdparams
```
下载好后将它放在项目根目录下。然后运行1_ppyolo_2x_2pytorch.py得到一个ppyolo_2x.pt，它也位于根目录下。


下载PaddleDetection的ppyolo_r18vd.pdparams。如果你使用Linux，请使用以下命令：
```
wget https://paddlemodels.bj.bcebos.com/object_detection/ppyolo_r18vd.pdparams
```

如果你使用Windows，请复制以下网址到浏览器或迅雷下载：
```
https://paddlemodels.bj.bcebos.com/object_detection/ppyolo_r18vd.pdparams
```
下载好后将它放在项目根目录下。然后运行1_ppyolo_r18vd_2pytorch.py得到一个ppyolo_r18vd.pt，它也位于根目录下。

(3)预测图片、获取FPS（预测images/test/里的图片，结果保存在images/res/）

(如果使用ppyolo_2x.py配置文件)
```
python demo.py --config=0
```

(如果使用ppyolo_r18vd.py配置文件)
```
python demo.py --config=2
```

## 数据集的放置位置
数据集应该和本项目位于同一级目录。一个示例：
```
D://GitHub
     |------COCO
     |        |------annotations
     |        |------test2017
     |        |------train2017
     |        |------val2017
     |
     |------VOCdevkit
     |        |------VOC2007
     |        |        |------Annotations
     |        |        |------ImageSets
     |        |        |------JPEGImages
     |        |        |------SegmentationClass
     |        |        |------SegmentationObject
     |        |
     |        |------VOC2012
     |                 |------Annotations
     |                 |------ImageSets
     |                 |------JPEGImages
     |                 |------SegmentationClass
     |                 |------SegmentationObject
     |
     |------Pytorch-PPYOLO-master
              |------annotation
              |------config
              |------data
              |------model
              |------...
```


## 训练
(如果使用ppyolo_2x.py配置文件)
```
python train.py --config=0
```

通过修改config/xxxxxxx.py的代码来进行更换数据集、更改超参数以及训练参数。

训练时如果发现mAP很稳定了，就停掉，修改学习率为原来的十分之一，接着继续训练，mAP还会再上升。暂时是这样手动操作。

## 训练自定义数据集
自带的voc2012数据集是一个很好的例子。

将自己数据集的txt注解文件放到annotation目录下，txt注解文件的格式如下：
```
xxx.jpg 18.19,6.32,424.13,421.83,20 323.86,2.65,640.0,421.94,20
xxx.jpg 48,240,195,371,11 8,12,352,498,14
# 图片名 物体1左上角x坐标,物体1左上角y坐标,物体1右下角x坐标,物体1右下角y坐标,物体1类别id 物体2左上角x坐标,物体2左上角y坐标,物体2右下角x坐标,物体2右下角y坐标,物体2类别id ...
```
注意：xxx.jpg仅仅是文件名而不是文件的路径！xxx.jpg仅仅是文件名而不是文件的路径！xxx.jpg仅仅是文件名而不是文件的路径！

运行1_txt2json.py会在annotation_json目录下生成两个coco注解风格的json注解文件，这是train.py支持的注解文件格式。
在config/ppyolo_2x.py里修改train_path、val_path、classes_path、train_pre_path、val_pre_path、num_classes这6个变量（自带的voc2012数据集直接解除注释就ok了）,就可以开始训练自己的数据集了。
而且，直接加载ppyolo_2x.pt的权重（即配置文件里修改train_cfg的model_path为'ppyolo_2x.pt'）训练也是可以的，这时候也仅仅不加载3个输出卷积层的6个权重（因为类别数不同导致了输出通道数不同）。
如果需要跑demo.py、eval.py，与数据集有关的变量也需要修改一下，应该很容易看懂。


## 评估
(如果使用ppyolo_2x.py配置文件)
```
python eval.py --config=0
```


## test-dev
(如果使用ppyolo_2x.py配置文件)
```
python test_dev.py --config=0
```


运行完之后，进入results目录，把bbox_detections.json压缩成bbox_detections.zip，提交到
https://competitions.codalab.org/competitions/20794#participate
获得bbox mAP。该mAP是test集的结果，也就是大部分检测算法论文的标准指标。


## 预测
(如果使用ppyolo_2x.py配置文件)
```
python demo.py --config=0
```
