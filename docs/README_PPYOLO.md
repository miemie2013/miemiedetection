简体中文 | [English](README_PPYOLO_en.md)

# PPYOLO & PPYOLOv2

（PPYOLOv2实现中...)


## 获取预训练模型(转换权重)
第一步，需要先安装paddlepaddle来方便读取权重：
```
pip install paddlepaddle-gpu==2.2.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

第二步，下载权重文件，项目根目录下执行（即下载文件，Windows用户可以用迅雷或浏览器下载wget后面的链接）：

```
wget https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_2x_coco.pdparams
wget https://paddledet.bj.bcebos.com/models/ppyolo_r18vd_coco.pdparams
wget https://paddledet.bj.bcebos.com/models/ppyolov2_r50vd_dcn_365e_coco.pdparams
```

第二步，转换权重，项目根目录下执行：

```
python tools/convert_weights.py -f exps/ppyolo/ppyolo_r50vd_2x.py -c ppyolo_r50vd_dcn_2x_coco.pdparams -oc ppyolo_2x.pth -nc 80
python tools/convert_weights.py -f exps/ppyolo/ppyolo_r18vd.py -c ppyolo_r18vd_coco.pdparams -oc ppyolo_r18vd.pth -nc 80
```

**参数解释:**
- -f表示的是使用的配置文件；
- -c表示的是读取的源权重文件；
- -oc表示的是输出（保存）的pytorch权重文件；
- -nc表示的是数据集的类别数；

执行完毕后就会在项目根目录下获得转换好的*.pth权重文件。


## 配置文件详解

在下面的命令中，大部分都会使用模型的配置文件，所以一开始就有必要先详细解释配置文件。

（1）mmdet.exp.base_exp.BaseExp为配置文件基类，是一个抽象类，声明了一堆抽象方法，如get_model()表示如何获取模型，get_data_loader()表示如何获取训练的dataloader，get_optimizer()表示如何获取优化器等等。


（2）mmdet.exp.datasets.coco_base.COCOBaseExp是数据集的配置，继承了BaseExp，它只给出数据集的配置。本仓库只支持COCO标注格式的数据集的训练！其它标注格式的数据集，需要先转换成COCO标注格式，才能训练（支持太多标注格式的话，工作量太大）。如何把自定义数据集转换成COCO标注格式，可以看[miemieLabels](https://github.com/miemie2013/miemieLabels) 。所有的检测算法配置类都会继承COCOBaseExp，表示所有的检测算法共用同样的数据集的配置。

COCOBaseExp的配置项有：
```
        self.num_classes = 80
        self.data_dir = '../COCO'
        self.cls_names = 'class_names/coco_classes.txt'
        self.ann_folder = "annotations"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.train_image_folder = "train2017"
        self.val_image_folder = "val2017"
```

- self.num_classes表示的是数据集的类别数；
- self.data_dir表示的是数据集的根目录；
- self.cls_names表示的是数据集的类别名文件路径，是一个txt文件，一行表示一个类别名。如果是自定义数据集，需要新建一个txt文件并编辑好类别名，再修改self.cls_names指向它；
- self.ann_folder表示的是数据集的注解文件根目录，需要位于self.data_dir目录下；
- self.train_ann表示的是数据集的训练集的注解文件名，需要位于self.ann_folder目录下；
- self.val_ann表示的是数据集的验证集的注解文件名，需要位于self.ann_folder目录下；
- self.train_image_folder表示的是数据集的训练集的图片文件夹名，需要位于self.data_dir目录下；
- self.val_image_folder表示的是数据集的验证集的图片文件夹名，需要位于self.data_dir目录下；


另外，自带有一个VOC2012数据集的配置，把
```
        # self.num_classes = 20
        # self.data_dir = '../VOCdevkit/VOC2012'
        # self.cls_names = 'class_names/voc_classes.txt'
        # self.ann_folder = "annotations2"
        # self.train_ann = "voc2012_train.json"
        # self.val_ann = "voc2012_val.json"
        # self.train_image_folder = "JPEGImages"
        # self.val_image_folder = "JPEGImages"
```
解除注释，注释掉COCO数据集的配置，就是使用VOC2012数据集了。

voc2012_train.json、voc2012_val.json是我个人转换好的COCO标注格式的注解文件，可以到这个链接下载：
链接：https://pan.baidu.com/s/1ehEqnNYKb9Nz0XNeqAcwDw 
提取码：qe3i

下载好后，在VOC2012数据集的self.data_dir目录下新建一个文件夹annotations2，把voc2012_train.json、voc2012_val.json放进这个文件夹。


所以，COCO数据集、VOC2012数据集、本项目的放置位置应该是这样：
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
     |                 |------annotations2
     |                 |         |----------voc2012_train.json
     |                 |         |----------voc2012_val.json
     |                 |------ImageSets
     |                 |------JPEGImages
     |                 |------SegmentationClass
     |                 |------SegmentationObject
     |
     |------miemiedetection-master
              |------assets
              |------class_names
              |------mmdet
              |------tools
              |------...
```

数据集根目录和miemiedetection-master是同一级目录。我个人非常不建议把数据集放在miemiedetection-master里，那样的话PyCharm打开会巨卡无比；而且，多个项目（如mmdetection、PaddleDetection、AdelaiDet）共用数据集时，可以做到数据集路径和项目名无关。



## 预测

（1）预测一张图片，项目根目录下执行：
```
python tools/demo.py image -f exps/ppyolo/ppyolo_r50vd_2x.py -c ppyolo_2x.pth --path assets/000000000019.jpg --conf 0.15 --tsize 608 --save_result --device gpu
```

```
python tools/demo.py image -f exps/ppyolo/ppyolo_r18vd.py -c ppyolo_r18vd.pth --path assets/000000000019.jpg --conf 0.15 --tsize 416 --save_result --device gpu
```

**参数解释:**
- -f表示的是使用的配置文件；
- -c表示的是读取的权重文件；
- --path表示的是图片的路径；
- --conf表示的是分数阈值，只会画出高于这个阈值的预测框；
- --tsize表示的是预测时将图片Resize成--tsize的分辨率；

预测完成后控制台会打印结果图片的保存路径，用户可打开查看。

**其它可选的参数:**
- --fp16，自动混合精度预测，使得预测速度更快；
- --fuse，把模型的卷积层与其之后的bn层合并成一个卷积层，使得预测速度更快（实现中）；

如果是使用训练自定义数据集保存的模型进行预测，修改-c为你的模型的路径即可。


（2）预测图片文件夹，项目根目录下执行：
```
python tools/demo.py image -f exps/ppyolo/ppyolo_r50vd_2x.py -c ppyolo_2x.pth --path assets --conf 0.15 --tsize 608 --save_result --device gpu
```

```
python tools/demo.py image -f exps/ppyolo/ppyolo_r18vd.py -c ppyolo_r18vd.pth --path assets --conf 0.15 --tsize 416 --save_result --device gpu
```

很简单，--path改成图片文件夹的路径即可。


## 评估

项目根目录下执行：
```
python tools/eval.py -f exps/ppyolo/ppyolo_r50vd_2x.py -d 1 -b 4 -c ppyolo_2x.pth --conf 0.01 --tsize 608
```

结果是
```
Average forward time: 36.18 ms, Average NMS time: 0.00 ms, Average inference time: 36.18 ms
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.453
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.654
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.498
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.300
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.485
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.593
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.345
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.578
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.631
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.450
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.666
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.780
```

项目根目录下执行：
```
python tools/eval.py -f exps/ppyolo/ppyolo_r50vd_2x.py -d 1 -b 8 -c ppyolo_2x.pth --conf 0.01 --tsize 320
```

结果是
```
Average forward time: 10.69 ms, Average NMS time: 0.00 ms, Average inference time: 10.69 ms
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.395
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.593
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.428
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.173
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.432
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.590
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.314
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.515
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.559
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.305
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.614
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.761
```

项目根目录下执行：
```
python tools/eval.py -f exps/ppyolo/ppyolo_r18vd.py -d 1 -b 8 -c ppyolo_r18vd.pth --conf 0.01 --tsize 416
```

结果是
```
Average forward time: 5.40 ms, Average NMS time: 0.00 ms, Average inference time: 5.40 ms
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.286
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.470
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.303
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.125
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.307
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.428
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.255
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.421
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.449
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.222
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.482
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.649
```

需要注意的是，PPYOLO使用的是matrix_nms，为SOLOv2中提出的新的后处理算法，已经包含在head里面，所以评估时的代码捕捉不到NMS的时间，所以显示"Average NMS time: 0.00 ms"。

**参数解释:**
- -f表示的是使用的配置文件；
- -d表示的是显卡数量；
- -b表示的是评估时的批大小；
- -c表示的是读取的权重文件；
- --conf表示的是分数阈值；
- --tsize表示的是评估时将图片Resize成--tsize的分辨率；

**其它可选的参数:**
- --fp16，自动混合精度评估，使得预测速度更快；
- --fuse，把模型的卷积层与其之后的bn层合并成一个卷积层，使得预测速度更快（实现中）；

如果是使用训练自定义数据集保存的模型进行评估（评估的是自定义数据集的验证集），修改-c为你的模型的路径即可。


## 预测

```

```



## 预测

```

```



## 预测

```

```



## 预测

```

```



## 预测

```

```



## 预测

```

```


