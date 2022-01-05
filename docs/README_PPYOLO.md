简体中文 | [English](README_PPYOLO_en.md)

# PPYOLO & PPYOLOv2

|          模型            | GPU个数 | 每GPU图片个数 |  骨干网络  | 输入尺寸 | Box AP<sup>val</sup> | Box AP<sup>test</sup> | V100 FP32(FPS) | V100 TensorRT FP16(FPS) | 模型下载 | 配置文件 |
|:------------------------:|:-------:|:-------------:|:----------:| :-------:| :------------------: | :-------------------: | :------------: | :---------------------: | :------: | :------: |
| PP-YOLO                  |     8      |     24     | ResNet50vd |     608     |         44.8         |         45.2          |      72.9      |          155.6          | [model](https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml)                   |
| PP-YOLO                  |     8      |     24     | ResNet50vd |     512     |         43.9         |         44.4          |      89.9      |          188.4          | [model](https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml)                   |
| PP-YOLO                  |     8      |     24     | ResNet50vd |     416     |         42.1         |         42.5          |      109.1      |          215.4          | [model](https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml)                   |
| PP-YOLO                  |     8      |     24     | ResNet50vd |     320     |         38.9         |         39.3          |      132.2      |          242.2          | [model](https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml)                   |
| PP-YOLO_2x               |     8      |     24     | ResNet50vd |     608     |         45.3         |         45.9          |      72.9      |          155.6          | [model](https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_2x_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolo_r50vd_dcn_2x_coco.yml)                   |
| PP-YOLO_2x               |     8      |     24     | ResNet50vd |     512     |         44.4         |         45.0          |      89.9      |          188.4          | [model](https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_2x_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolo_r50vd_dcn_2x_coco.yml)                   |
| PP-YOLO_2x               |     8      |     24     | ResNet50vd |     416     |         42.7         |         43.2          |      109.1      |          215.4          | [model](https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_2x_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolo_r50vd_dcn_2x_coco.yml)                   |
| PP-YOLO_2x               |     8      |     24     | ResNet50vd |     320     |         39.5         |         40.1          |      132.2      |          242.2          | [model](https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_2x_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolo_r50vd_dcn_2x_coco.yml)                   |
| PP-YOLO               |     4      |     32     | ResNet18vd |     512     |         29.2         |         29.5          |      357.1      |          657.9          | [model](https://paddledet.bj.bcebos.com/models/ppyolo_r18vd_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolo_r18vd_coco.yml)                   |
| PP-YOLO               |     4      |     32     | ResNet18vd |     416     |         28.6         |         28.9          |      409.8      |          719.4          | [model](https://paddledet.bj.bcebos.com/models/ppyolo_r18vd_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolo_r18vd_coco.yml)                   |
| PP-YOLO               |     4      |     32     | ResNet18vd |     320     |         26.2         |         26.4          |      480.7      |          763.4          | [model](https://paddledet.bj.bcebos.com/models/ppyolo_r18vd_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolo_r18vd_coco.yml)                   |
| PP-YOLOv2               |     8      |     12     | ResNet50vd |     640     |         49.1         |         49.5          |      68.9      |          106.5          | [model](https://paddledet.bj.bcebos.com/models/ppyolov2_r50vd_dcn_365e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml)                   |
| PP-YOLOv2               |     8      |     12     | ResNet101vd |     640     |         49.7         |         50.3          |     49.5     |         87.0         | [model](https://paddledet.bj.bcebos.com/models/ppyolov2_r101vd_dcn_365e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolov2_r101vd_dcn_365e_coco.yml)                   |

**注意:**

- PP-YOLO模型使用COCO数据集中train2017作为训练集，使用val2017和test-dev2017作为测试集，Box AP<sup>test</sup>为`mAP(IoU=0.5:0.95)`评估结果。
- PP-YOLO模型训练过程中使用8 GPUs，每GPU batch size为24进行训练，如训练GPU数和batch size不使用上述配置，须参考[FAQ](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.3/docs/tutorials/FAQ.md)调整学习率和迭代次数。
- PP-YOLO模型推理速度测试采用单卡V100，batch size=1进行测试，使用CUDA 10.2, CUDNN 7.5.1，TensorRT推理速度测试使用TensorRT 5.1.2.2。
- PP-YOLO模型FP32的推理速度测试数据为使用`tools/export_model.py`脚本导出模型后，使用`deploy/python/infer.py`脚本中的`--run_benchnark`参数使用Paddle预测库进行推理速度benchmark测试结果, 且测试的均为不包含数据预处理和模型输出后处理(NMS)的数据(与[YOLOv4(AlexyAB)](https://github.com/AlexeyAB/darknet)测试方法一致)。
- TensorRT FP16的速度测试相比于FP32去除了`yolo_box`(bbox解码)部分耗时，即不包含数据预处理，bbox解码和NMS(与[YOLOv4(AlexyAB)](https://github.com/AlexeyAB/darknet)测试方法一致)。


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
wget https://paddledet.bj.bcebos.com/models/ppyolov2_r101vd_dcn_365e_coco.pdparams
wget https://paddledet.bj.bcebos.com/models/pretrained/ResNet50_vd_ssld_pretrained.pdparams
wget https://paddledet.bj.bcebos.com/models/pretrained/ResNet18_vd_pretrained.pdparams
wget https://paddledet.bj.bcebos.com/models/pretrained/ResNet101_vd_ssld_pretrained.pdparams
```

注意，带有pretrained字样的模型是在ImageNet上预训练的骨干网路，PPYOLO和PPYOLOv2加载这些权重以训练COCO数据集。其余为COCO上的预训练模型。


第二步，转换权重，项目根目录下执行：

```
python tools/convert_weights.py -f exps/ppyolo/ppyolo_r50vd_2x.py -c ppyolo_r50vd_dcn_2x_coco.pdparams -oc ppyolo_2x.pth -nc 80
python tools/convert_weights.py -f exps/ppyolo/ppyolo_r18vd.py -c ppyolo_r18vd_coco.pdparams -oc ppyolo_r18vd.pth -nc 80
python tools/convert_weights.py -f exps/ppyolo/ppyolov2_r50vd_365e.py -c ppyolov2_r50vd_dcn_365e_coco.pdparams -oc ppyolov2_r50vd_365e.pth -nc 80
python tools/convert_weights.py -f exps/ppyolo/ppyolov2_r101vd_365e.py -c ppyolov2_r101vd_dcn_365e_coco.pdparams -oc ppyolov2_r101vd_365e.pth -nc 80
python tools/convert_weights.py -f exps/ppyolo/ppyolo_r18vd.py -c ResNet18_vd_pretrained.pdparams -oc ResNet18_vd_pretrained.pth -nc 80 --only_backbone True
python tools/convert_weights.py -f exps/ppyolo/ppyolov2_r50vd_365e.py -c ResNet50_vd_ssld_pretrained.pdparams -oc ResNet50_vd_ssld_pretrained.pth -nc 80 --only_backbone True
python tools/convert_weights.py -f exps/ppyolo/ppyolov2_r101vd_365e.py -c ResNet101_vd_ssld_pretrained.pdparams -oc ResNet101_vd_ssld_pretrained.pth -nc 80 --only_backbone True
```

**参数解释:**
- -f表示的是使用的配置文件；
- -c表示的是读取的源权重文件；
- -oc表示的是输出（保存）的pytorch权重文件；
- -nc表示的是数据集的类别数；
- --only_backbone为True时表示只转换骨干网络的权重；

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


（3）mmdet.exp.ppyolo.ppyolo_method_base.PPYOLO_Method_Exp是实现具体算法所有抽象方法的类，继承了COCOBaseExp，它实现了所有抽象方法。


（4）mmdet.exp.ppyolo.ppyolo_r50vd_2x_base.PPYOLO_R50VD_2x_Exp是PPYOLO算法的Resnet50Vd模型的配置类，继承了PPYOLO_Method_Exp，它给出了ppyolo_r50vd_2x具体的所有配置（包括训练轮数、学习率、ema、网络结构配置、NMS配置、预处理配置等）；


mmdet.exp.ppyolo.ppyolo_r18vd_base.PPYOLO_R18VD_Exp是PPYOLO算法的Resnet18Vd模型的配置类，继承了PPYOLO_Method_Exp，它给出了ppyolo_r18vd具体的所有配置（包括训练轮数、学习率、ema、网络结构配置、NMS配置、预处理配置等）；


注意，xxx_base_coco.py和xxx_base_custom.py仅为了方便复制粘贴而存在，实际配置文件是xxx_base.py。如果是训练自定义数据集，复制xxx_base_custom.py里的全部内容，粘贴到xxx_base.py，再根据自己的需求更改相关配置项。如果是训练COCO数据集，复制xxx_base_coco.py里的全部内容，粘贴到xxx_base.py。最初xxx_base_coco.py里的内容和xxx_base.py里的内容是完全一样的。


（5）exp.ppyolo_r50vd_2x.Exp是PPYOLO算法的Resnet50Vd模型的最终配置类，继承了PPYOLO_R50VD_2x_Exp，除了self.exp_name什么都没有修改；


exp.ppyolo_r18vd.Exp是PPYOLO算法的Resnet50Vd模型的最终配置类，继承了PPYOLO_R18VD_Exp，除了self.exp_name什么都没有修改；



## 预测

（1）预测一张图片，项目根目录下执行：
```
python tools/demo.py image -f exps/ppyolo/ppyolo_r50vd_2x.py -c ppyolo_2x.pth --path assets/000000000019.jpg --conf 0.15 --tsize 608 --save_result --device gpu
```

```
python tools/demo.py image -f exps/ppyolo/ppyolo_r18vd.py -c ppyolo_r18vd.pth --path assets/000000000019.jpg --conf 0.15 --tsize 416 --save_result --device gpu
```

```
python tools/demo.py image -f exps/ppyolo/ppyolov2_r50vd_365e.py -c ppyolov2_r50vd_365e.pth --path assets/000000000019.jpg --conf 0.15 --tsize 640 --save_result --device gpu
```

```
python tools/demo.py image -f exps/ppyolo/ppyolov2_r101vd_365e.py -c ppyolov2_r101vd_365e.pth --path assets/000000000019.jpg --conf 0.15 --tsize 640 --save_result --device gpu
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

项目根目录下执行：
```
python tools/eval.py -f exps/ppyolo/ppyolov2_r50vd_365e.py -d 1 -b 4 -c ppyolov2_r50vd_365e.pth --conf 0.01 --tsize 640
```

结果是
```
Average forward time: 42.58 ms, Average NMS time: 0.00 ms, Average inference time: 42.58 ms
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.491
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.677
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.538
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.315
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.534
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.622
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.363
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.612
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.665
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.464
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.711
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.801
```

项目根目录下执行：
```
python tools/eval.py -f exps/ppyolo/ppyolov2_r50vd_365e.py -d 1 -b 8 -c ppyolov2_r50vd_365e.pth --conf 0.01 --tsize 320
```

结果是
```
Average forward time: 12.62 ms, Average NMS time: 0.00 ms, Average inference time: 12.62 ms
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.424
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.608
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.457
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.207
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.469
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.631
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.330
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.542
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.588
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.331
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.655
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.800
```

项目根目录下执行：
```
python tools/eval.py -f exps/ppyolo/ppyolov2_r101vd_365e.py -d 1 -b 4 -c ppyolov2_r101vd_365e.pth --conf 0.01 --tsize 640
```

结果是
```
Average forward time: 56.81 ms, Average NMS time: 0.00 ms, Average inference time: 56.81 ms
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.497
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.683
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.545
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.336
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.543
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.633
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.366
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.616
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.668
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.490
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.713
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.812
```

项目根目录下执行：
```
python tools/eval.py -f exps/ppyolo/ppyolov2_r101vd_365e.py -d 1 -b 8 -c ppyolov2_r101vd_365e.pth --conf 0.01 --tsize 320
```

结果是
```
Average forward time: 16.42 ms, Average NMS time: 0.00 ms, Average inference time: 16.42 ms
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.431
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.614
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.466
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.214
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.477
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.640
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.333
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.545
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.589
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.338
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.653
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.807
```

需要注意的是，PPYOLO和PPYOLOv2使用的是matrix_nms，为SOLOv2中提出的新的后处理算法，已经包含在head里面，所以评估时的代码捕捉不到NMS的时间，所以显示"Average NMS time: 0.00 ms"。

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


## 训练COCO数据集

如果从头训练COCO数据集，项目根目录下执行：
```
python tools/train.py -f exps/ppyolo/ppyolo_r50vd_2x.py -d 8 -b 24 -eb 8
```

或者
```
python tools/train.py -f exps/ppyolo/ppyolo_r18vd.py -d 4 -b 32 -eb 8
```

或者
```
python tools/train.py -f exps/ppyolo/ppyolov2_r50vd_365e.py -d 8 -b 12 -eb 8
```

或者
```
python tools/train.py -f exps/ppyolo/ppyolov2_r101vd_365e.py -d 8 -b 12 -eb 8
```


这些是高端玩家才能输入的命令，使用8卡训练，每卡的批大小是24，需要每张卡的显存为32GB或以上。建议8张Tesla V100。咩酱没有试过多卡训练，如果有报错或你发现什么错误，请提出，让咩酱修正多卡部分的代码。


有一个细节是miemiedetection的PPYOLO把RandomShape、NormalizeImage、Permute、Gt2YoloTarget这4个预处理步骤放到了sample_transforms中，不像PaddleDetection放到batch_transforms中(配合DataLoader的collate_fn使用)，虽然这样写不美观，但是可以提速n倍。因为用collate_fn实现batch_transforms太耗时了！能不使用batch_transforms尽量不使用batch_transforms！唯一的缺点是对于随机种子玩家，可能需要写额外代码初始化随机种子，决定每个epoch怎么打乱所有图片，以及每个iter怎么选随机尺度。


**参数解释:**
- -f表示的是使用的配置文件；
- -d表示的是显卡数量；
- -b表示的是训练时的批大小（单张卡）；
- -eb表示的是评估时的批大小（单张卡）；

**其它可选的参数:**
- --fp16，自动混合精度训练；
- --num_machines，机器数量，建议单机多卡训练；
- -c表示的是读取的权重文件；
- --resume表示的是是否是恢复训练；


还没有转换骨干网络ImageNet预训练权重，目前正在实现中。


## 训练自定义数据集

建议读取COCO预训练权重进行训练，因为收敛快。
以上述的VOC2012数据集为例，

一、ppyolo_r50vd模型

只需修改2个配置文件：

（1）mmdet.exp.datasets.coco_base.COCOBaseExp，修改数据集的配置项。把COCO数据集的配置注释掉，把VOC2012数据集的配置项解除注释即可。如果是其它的自定义数据集，需要手动写一下配置项。

（2）mmdet.exp.ppyolo.ppyolo_r50vd_2x_base.PPYOLO_R50VD_2x_Exp，修改该模型的配置项。复制ppyolo_r50vd_2x_base_custom.py里的全部内容，粘贴到ppyolo_r50vd_2x_base.py，再根据自己的需求更改相关配置项（或不改，使用咩酱预设配置）。我个人建议修改的配置项有：

- self.max_epoch，训练轮数；
- self.aug_epochs，前几轮进行mixup或cutmix或mosaic数据增强；
- self.eval_interval，每训练几轮评估一次模型；
- self.warmup_epochs，前几轮学习率warm up；
- self.milestones_epoch，第几个epoch学习率衰减一次；
- self.backbone['freeze_at']，冻结骨干网络的前多少个stage，5表示冻结前5个（全部）stage，0表示不冻结骨干网络；

不建议修改基础学习率self.basic_lr_per_img，这是算法作者默认的配置，是每张图片的学习率。本仓库会根据用户输入的训练时的批大小动态调节真正的basic_lr，即basic_lr = self.basic_lr_per_img * batch_size

输入命令开始训练：
```
python tools/train.py -f exps/ppyolo/ppyolo_r50vd_2x.py -d 1 -b 8 -eb 4 -c ppyolo_2x.pth
```

如果训练因为某些原因中断，想要读取之前保存的模型恢复训练，只要修改-c，加上--resume，输入：
```
python tools/train.py -f exps/ppyolo/ppyolo_r50vd_2x.py -d 1 -b 8 -eb 4 -c PPYOLO_outputs/ppyolo_r50vd_2x/13.pth --resume
```
把13.pth替换成你要读取的模型的名字。

迁移学习VOC2012数据集，实测ppyolo_r50vd_2x的AP(0.50:0.95)可以到达0.59+、AP(0.50)可以到达0.82+、AP(small)可以到达0.18+。


二、ppyolo_r18vd模型

只需修改2个配置文件：

（1）mmdet.exp.datasets.coco_base.COCOBaseExp，修改数据集的配置项。把COCO数据集的配置注释掉，把VOC2012数据集的配置项解除注释即可。如果是其它的自定义数据集，需要手动写一下配置项。

（2）mmdet.exp.ppyolo.ppyolo_r18vd_base.PPYOLO_R50VD_2x_Exp，修改该模型的配置项。复制ppyolo_r18vd_base_custom.py里的全部内容，粘贴到ppyolo_r18vd_base.py，再根据自己的需求更改相关配置项（或不改，使用咩酱预设配置）。

输入命令开始训练：
```
python tools/train.py -f exps/ppyolo/ppyolo_r18vd.py -d 1 -b 8 -eb 4 -c ppyolo_r18vd.pth
```

迁移学习VOC2012数据集，实测ppyolo_r18vd的AP(0.50:0.95)可以到达0.39+、AP(0.50)可以到达0.65+、AP(small)可以到达0.06+。


三、ppyolov2_r50vd模型

只需修改2个配置文件：

（1）mmdet.exp.datasets.coco_base.COCOBaseExp，修改数据集的配置项。把COCO数据集的配置注释掉，把VOC2012数据集的配置项解除注释即可。如果是其它的自定义数据集，需要手动写一下配置项。

（2）mmdet.exp.ppyolo.ppyolov2_r50vd_365e_base.PPYOLOv2_R50VD_365e_Exp，修改该模型的配置项。复制ppyolov2_r50vd_365e_base_custom.py里的全部内容，粘贴到ppyolov2_r50vd_365e_base.py，再根据自己的需求更改相关配置项（或不改，使用咩酱预设配置）。

输入命令开始训练：
```
python tools/train.py -f exps/ppyolo/ppyolov2_r50vd_365e.py -d 1 -b 6 -eb 2 -c ppyolov2_r50vd_365e.pth
```

迁移学习VOC2012数据集，实测ppyolo_r18vd的AP(0.50:0.95)可以到达0.63+、AP(0.50)可以到达0.84+、AP(small)可以到达0.25+。



## 导出为ONNX

目前只支持ppyolo_r18vd导出。其它模型实现中..

导出：
```
python tools/export_onnx.py --output-name ppyolo_r18vd.onnx -f exps/ppyolo/ppyolo_r18vd.py -c ppyolo_r18vd.pth
```


ONNX预测：
```
python tools/onnx_inference.py -an PPYOLO -acn ppyolo_r18vd -m ppyolo_r18vd.onnx -i assets/dog.jpg -o ONNX_PPYOLO_R18VD_outputs -s 0.15 --input_shape 416,416 -cn class_names/coco_classes.txt
```


用onnx模型进行验证：
```
python tools/onnx_eval.py -an PPYOLO -acn ppyolo_r18vd -m ppyolo_r18vd.onnx -i ../COCO/val2017 -a ../COCO/annotations/instances_val2017.json -s 0.01 --input_shape 416,416 --eval_type eval
```



## NCNN

实现中...
```

```



## TensorRT

实现中...
```

```



## 预测

```

```


