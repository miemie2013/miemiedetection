简体中文 | [English](README_PPYOLO_en.md)

# PPYOLO & PPYOLOv2 & PPYOLOE

首先，这3个PPYOLO模型均经过了loss对齐、梯度对齐的实验，你也可以在源码中看到我注释掉的读写*.npz 的代码，都是做对齐实验遗留的代码。所以，说它们是如假包换的PPYOLO算法一点都不过分，因为它们拥有和原版仓库一样的损失、一样的梯度。另外，我也试着用原版仓库和miemiedetection迁移学习voc2012数据集，获得了一样的精度（使用了相同的超参数），训练日志在train_ppyolo_in_voc2012文件夹里，相关命令可以查看“训练自定义数据集”小节。证明了复现PPYOLO系列算法的正确性！


|          模型            | GPU个数 | 每GPU图片个数 |  骨干网络  | 输入尺寸 | Box AP<sup>val</sup> | Box AP<sup>test</sup> | inference time(ms) |
|:------------------------:|:-------:|:-------------:|:----------:| :-------:| :------------------: | :-------------------: |:------------------:|
| PP-YOLO_2x               |     8      |     24     | ResNet50vd |     608     |         45.3         |         45.9          |       34.49        |
| PP-YOLO_2x               |     8      |     24     | ResNet50vd |     320     |         39.5         |         40.1          |       10.69        |
| PP-YOLO               |     4      |     32     | ResNet18vd |     416     |         28.6         |         28.9          |        5.40        |
| PP-YOLOv2               |     8      |     12     | ResNet50vd |     640     |         49.1         |         49.5          |       42.58        |
| PP-YOLOv2               |     8      |     12     | ResNet101vd |     640     |         49.7         |         50.3          |       56.81        |
| PP-YOLOE-s               |     8      |     32     | cspresnet-s |     640     |         42.7         |         43.1          |       -        |
| PP-YOLOE-m               |     8      |     28     | cspresnet-m |     640     |         48.6         |         48.9          |       -        |
| PP-YOLOE-l               |     8      |     20     | cspresnet-l |     640     |         50.9         |         51.4          |       -        |
| PP-YOLOE-x               |     8      |     16     | cspresnet-x |     640     |         51.9         |         52.2          |       -        |

**注意:**

- inference time通过tools/eval.py脚本测出，相关命令见下文“评估”小节。测速环境为Ubuntu20.04、GTX 1660TI、torch==1.9.1+cu102。


## 获取预训练模型(转换权重)

读者可以到这个链接下载转换好的*.pth权重文件：

链接：https://pan.baidu.com/s/1ehEqnNYKb9Nz0XNeqAcwDw 
提取码：qe3i

或者按照下面的步骤获取：

第一步，下载权重文件，项目根目录下执行（即下载文件，Windows用户可以用迅雷或浏览器下载wget后面的链接）：

```
wget https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_2x_coco.pdparams
wget https://paddledet.bj.bcebos.com/models/ppyolo_r18vd_coco.pdparams
wget https://paddledet.bj.bcebos.com/models/ppyolov2_r50vd_dcn_365e_coco.pdparams
wget https://paddledet.bj.bcebos.com/models/ppyolov2_r101vd_dcn_365e_coco.pdparams
wget https://paddledet.bj.bcebos.com/models/pretrained/ResNet50_vd_ssld_pretrained.pdparams
wget https://paddledet.bj.bcebos.com/models/pretrained/ResNet18_vd_pretrained.pdparams
wget https://paddledet.bj.bcebos.com/models/pretrained/ResNet101_vd_ssld_pretrained.pdparams
wget https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_300e_coco.pdparams
wget https://paddledet.bj.bcebos.com/models/ppyoloe_crn_m_300e_coco.pdparams
wget https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams
wget https://paddledet.bj.bcebos.com/models/ppyoloe_crn_x_300e_coco.pdparams
wget https://paddledet.bj.bcebos.com/models/pretrained/CSPResNetb_s_pretrained.pdparams
wget https://paddledet.bj.bcebos.com/models/pretrained/CSPResNetb_m_pretrained.pdparams
wget https://paddledet.bj.bcebos.com/models/pretrained/CSPResNetb_l_pretrained.pdparams
wget https://paddledet.bj.bcebos.com/models/pretrained/CSPResNetb_x_pretrained.pdparams
```

注意，带有pretrained字样的模型是在ImageNet上预训练的骨干网路，PPYOLO、PPYOLOv2、PPYOLOE加载这些权重以训练COCO数据集。其余为COCO上的预训练模型。


第二步，转换权重，项目根目录下执行：

```
python tools/convert_weights.py -f exps/ppyolo/ppyolo_r50vd_2x.py -c ppyolo_r50vd_dcn_2x_coco.pdparams -oc ppyolo_r50vd_2x.pth -nc 80
python tools/convert_weights.py -f exps/ppyolo/ppyolo_r18vd.py -c ppyolo_r18vd_coco.pdparams -oc ppyolo_r18vd.pth -nc 80
python tools/convert_weights.py -f exps/ppyolo/ppyolov2_r50vd_365e.py -c ppyolov2_r50vd_dcn_365e_coco.pdparams -oc ppyolov2_r50vd_365e.pth -nc 80
python tools/convert_weights.py -f exps/ppyolo/ppyolov2_r101vd_365e.py -c ppyolov2_r101vd_dcn_365e_coco.pdparams -oc ppyolov2_r101vd_365e.pth -nc 80
python tools/convert_weights.py -f exps/ppyolo/ppyolo_r18vd.py -c ResNet18_vd_pretrained.pdparams -oc ResNet18_vd_pretrained.pth -nc 80 --only_backbone True
python tools/convert_weights.py -f exps/ppyolo/ppyolov2_r50vd_365e.py -c ResNet50_vd_ssld_pretrained.pdparams -oc ResNet50_vd_ssld_pretrained.pth -nc 80 --only_backbone True
python tools/convert_weights.py -f exps/ppyolo/ppyolov2_r101vd_365e.py -c ResNet101_vd_ssld_pretrained.pdparams -oc ResNet101_vd_ssld_pretrained.pth -nc 80 --only_backbone True
python tools/convert_weights.py -f exps/ppyoloe/ppyoloe_crn_s_300e_coco.py -c ppyoloe_crn_s_300e_coco.pdparams -oc ppyoloe_crn_s_300e_coco.pth -nc 80
python tools/convert_weights.py -f exps/ppyoloe/ppyoloe_crn_m_300e_coco.py -c ppyoloe_crn_m_300e_coco.pdparams -oc ppyoloe_crn_m_300e_coco.pth -nc 80
python tools/convert_weights.py -f exps/ppyoloe/ppyoloe_crn_l_300e_coco.py -c ppyoloe_crn_l_300e_coco.pdparams -oc ppyoloe_crn_l_300e_coco.pth -nc 80
python tools/convert_weights.py -f exps/ppyoloe/ppyoloe_crn_x_300e_coco.py -c ppyoloe_crn_x_300e_coco.pdparams -oc ppyoloe_crn_x_300e_coco.pth -nc 80
python tools/convert_weights.py -f exps/ppyoloe/ppyoloe_crn_s_300e_coco.py -c CSPResNetb_s_pretrained.pdparams -oc CSPResNetb_s_pretrained.pth -nc 80 --only_backbone True
python tools/convert_weights.py -f exps/ppyoloe/ppyoloe_crn_m_300e_coco.py -c CSPResNetb_m_pretrained.pdparams -oc CSPResNetb_m_pretrained.pth -nc 80 --only_backbone True
python tools/convert_weights.py -f exps/ppyoloe/ppyoloe_crn_l_300e_coco.py -c CSPResNetb_l_pretrained.pdparams -oc CSPResNetb_l_pretrained.pth -nc 80 --only_backbone True
python tools/convert_weights.py -f exps/ppyoloe/ppyoloe_crn_x_300e_coco.py -c CSPResNetb_x_pretrained.pdparams -oc CSPResNetb_x_pretrained.pth -nc 80 --only_backbone True
```

第三步，可以选择删除paddle的模型：

```
rm -rf *.pdparams
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

另外，你也可以像exps/ppyoloe/ppyoloe_crn_l_voc2012.py中一样，在子类中修改self.num_classes、self.data_dir这些数据集的配置，这样COCOBaseExp的配置就被覆盖掉（无效）了。


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


（4）exp.ppyolo.ppyolo_r50vd_2x.Exp是PPYOLO算法的Resnet50Vd模型的最终配置类，继承了PPYOLO_Method_Exp；

PPYOLOE的配置文件也是类似这样的结构。


## 预测

（1）预测一张图片，项目根目录下执行：
```
python tools/demo.py image -f exps/ppyolo/ppyolo_r50vd_2x.py -c ppyolo_r50vd_2x.pth --path assets/000000000019.jpg --conf 0.15 --tsize 608 --save_result --device gpu
```

```
python tools/demo.py image -f exps/ppyolo/ppyolo_r18vd.py -c ppyolo_r18vd.pth --path assets/000000000019.jpg --conf 0.15 --tsize 416 --save_result --device gpu
```

```
python tools/demo.py image -f exps/ppyolo/ppyolov2_r50vd_365e.py -c ppyolov2_r50vd_365e.pth --path assets/000000000019.jpg --conf 0.15 --tsize 640 --save_result --device gpu
```

```
python tools/demo.py image -f exps/ppyoloe/ppyoloe_crn_s_300e_coco.py -c ppyoloe_crn_s_300e_coco.pth --path assets/000000000019.jpg --conf 0.15 --tsize 640 --save_result --device gpu
```

```
python tools/demo.py image -f exps/ppyoloe/ppyoloe_crn_l_300e_coco.py -c ppyoloe_crn_l_300e_coco.pth --path assets/000000000019.jpg --conf 0.15 --tsize 640 --save_result --device gpu
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
python tools/demo.py image -f exps/ppyolo/ppyolo_r50vd_2x.py -c ppyolo_r50vd_2x.pth --path assets --conf 0.15 --tsize 608 --save_result --device gpu
```

```
python tools/demo.py image -f exps/ppyolo/ppyolo_r18vd.py -c ppyolo_r18vd.pth --path assets --conf 0.15 --tsize 416 --save_result --device gpu
```

很简单，--path改成图片文件夹的路径即可。


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


## 训练COCO数据集

如果读取ImageNet预训练骨干网络训练COCO数据集，项目根目录下执行：

(1)ppyolo_r50vd_2x
```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tools/train.py -f exps/ppyolo/ppyolo_r50vd_2x.py -d 8 -b 192 -eb 16 -c ResNet50_vd_ssld_pretrained.pth
```

(2)ppyolo_r18vd
```
export CUDA_VISIBLE_DEVICES=0,1,2,3
python tools/train.py -f exps/ppyolo/ppyolo_r18vd.py -d 4 -b 128 -eb 16 -c ResNet18_vd_pretrained.pth
```

(3)ppyolov2_r50vd_365e
```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tools/train.py -f exps/ppyolo/ppyolov2_r50vd_365e.py -d 8 -b 96 -eb 16 -c ResNet50_vd_ssld_pretrained.pth
```

(4)ppyolov2_r101vd_365e
```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tools/train.py -f exps/ppyolo/ppyolov2_r101vd_365e.py -d 8 -b 96 -eb 16 -c ResNet101_vd_ssld_pretrained.pth
```

(5)ppyoloe_crn_s_300e_coco
```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tools/train.py -f exps/ppyoloe/ppyoloe_crn_s_300e_coco.py -d 8 -b 256 -eb 16 -c CSPResNetb_s_pretrained.pth --fp16
```

(6)ppyoloe_crn_m_300e_coco
```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tools/train.py -f exps/ppyoloe/ppyoloe_crn_m_300e_coco.py -d 8 -b 224 -eb 16 -c CSPResNetb_m_pretrained.pth --fp16
```

(7)ppyoloe_crn_l_300e_coco
```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tools/train.py -f exps/ppyoloe/ppyoloe_crn_l_300e_coco.py -d 8 -b 160 -eb 16 -c CSPResNetb_l_pretrained.pth --fp16
```

(8)ppyoloe_crn_x_300e_coco
```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tools/train.py -f exps/ppyoloe/ppyoloe_crn_x_300e_coco.py -d 8 -b 128 -eb 16 -c CSPResNetb_x_pretrained.pth --fp16
```

你需要有一台单机8卡的超算，Money is all you need!

有一个细节是miemiedetection的PPYOLO把RandomShape、NormalizeImage、Permute、Gt2YoloTarget这4个预处理步骤放到了sample_transforms中，不像PaddleDetection放到batch_transforms中(配合DataLoader的collate_fn使用)，虽然这样写不美观，但是可以提速n倍。因为用collate_fn实现batch_transforms太耗时了！能不使用batch_transforms尽量不使用batch_transforms！


**参数解释:**

- -f表示的是使用的配置文件；
- -d表示的是显卡数量；
- -b表示的是训练时的批大小（所有卡的）；
- -eb表示的是评估时的批大小（所有卡的）；
- -c表示的是读取的权重文件；

**其它可选的参数:**

- --fp16，自动混合精度训练；
- --num_machines，机器数量，建议单机多卡训练；
- --resume表示的是是否是恢复训练；




## 训练自定义数据集

建议读取COCO预训练权重进行训练，因为收敛快。
以上述的VOC2012数据集为例，

一、ppyolo_r50vd模型

（1）如果是1机1卡，输入命令开始训练：
```
python tools/train.py -f exps/ppyolo/ppyolo_r50vd_voc2012.py -d 1 -b 8 -eb 4 -c ppyolo_r50vd_2x.pth
```

如果训练因为某些原因中断，想要读取之前保存的模型恢复训练，只要修改-c，加上--resume，输入：
```
python tools/train.py -f exps/ppyolo/ppyolo_r50vd_voc2012.py -d 1 -b 8 -eb 4 -c PPYOLO_outputs/ppyolo_r50vd_voc2012/13.pth --resume
```
修改-c成你要读取的模型的路径即可。


（2）如果是2机2卡（每台机上1张卡），
0号机输入以下命令：
```
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -f exps/ppyolo/ppyolo_r50vd_voc2012.py --dist-url tcp://192.168.0.107:12312 --num_machines 2 --machine_rank 0 -b 8 -eb 4 -c ppyolo_r50vd_2x.pth
```

1号机输入以下命令：
```
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -f exps/ppyolo/ppyolo_r50vd_voc2012.py --dist-url tcp://192.168.0.107:12312 --num_machines 2 --machine_rank 1 -b 8 -eb 4 -c ppyolo_r50vd_2x.pth
```

你需要把上面2条命令的192.168.0.107改成0号机的局域网ip。


（3）如果是1机2卡，输入命令开始训练：
```
export CUDA_VISIBLE_DEVICES=0,1
nohup python tools/train.py -f exps/ppyolo/ppyolo_r50vd_voc2012.py -d 2 -b 8 -eb 2 -c ppyolo_r50vd_2x.pth     > ppyolo.log 2>&1 &
```

迁移学习VOC2012数据集，实测ppyolo_r50vd_2x的AP(0.50:0.95)可以到达0.59+、AP(0.50)可以到达0.82+、AP(small)可以到达0.18+。不管是单卡还是多卡，都能得到这个结果。迁移学习时和PaddleDetection获得了一样的精度、一样的收敛速度，二者的训练日志位于train_ppyolo_in_voc2012文件夹下。


在这里我给出了多机多卡、单机多卡的命令示例，其余模型按着改就是，下文我只展示单机单卡的命令，如果你想复制粘贴多卡的命令，可以查看readme_yolo.txt



二、ppyolo_r18vd模型

输入命令开始训练：
```
python tools/train.py -f exps/ppyolo/ppyolo_r18vd_voc2012.py -d 1 -b 8 -eb 4 -c ppyolo_r18vd.pth
```

迁移学习VOC2012数据集，实测ppyolo_r18vd的AP(0.50:0.95)可以到达0.39+、AP(0.50)可以到达0.65+、AP(small)可以到达0.06+。


三、ppyolov2_r50vd模型

输入命令开始训练：
```
python tools/train.py -f exps/ppyolo/ppyolov2_r50vd_voc2012.py -d 1 -b 8 -eb 2 -c ppyolov2_r50vd_365e.pth
```

迁移学习VOC2012数据集，实测ppyolov2_r50vd的AP(0.50:0.95)可以到达0.63+、AP(0.50)可以到达0.84+、AP(small)可以到达0.25+。


四、ppyoloe_s模型

输入命令开始训练（不冻结骨干网络）：
```
python tools/train.py -f exps/ppyoloe/ppyoloe_crn_s_voc2012.py -d 1 -b 4 -eb 2 -c ppyoloe_crn_s_300e_coco.pth --fp16
```

迁移学习VOC2012数据集，实测ppyoloe_s的AP(0.50:0.95)可以到达0.48+、AP(0.50)可以到达0.68+、AP(small)可以到达0.15+。


五、ppyoloe_l模型

（1）如果是1机1卡，输入命令开始训练（冻结了骨干网络）：
```
python tools/train.py -f exps/ppyoloe/ppyoloe_crn_l_voc2012.py -d 1 -b 8 -eb 2 -c ppyoloe_crn_l_300e_coco.pth --fp16
```

（2）如果是1机2卡，输入命令开始训练（冻结了骨干网络）：
```
export CUDA_VISIBLE_DEVICES=0,1
nohup python tools/train.py -f exps/ppyoloe/ppyoloe_crn_l_voc2012.py -d 2 -b 8 -eb 2 -c ppyoloe_crn_l_300e_coco.pth --fp16     > ppyoloe_l.log 2>&1 &
```


迁移学习VOC2012数据集，实测ppyoloe_l的AP(0.50:0.95)可以到达0.66+、AP(0.50)可以到达0.85+、AP(small)可以到达0.28+。不管是单卡还是多卡，都能得到这个结果。迁移学习时和PaddleDetection获得了一样的精度、一样的收敛速度，二者的训练日志位于train_ppyolo_in_voc2012文件夹下。


## 评估

总体的命令格式是
```
python tools/eval.py -f {exp配置文件的路径} -d 1 -b 4 -c {读取的.pth权重文件的路径} --conf {分数过滤阈值} --tsize {输入网络的图片分辨率}
```


比如：
项目根目录下执行：
```
python tools/eval.py -f exps/ppyolo/ppyolo_r50vd_2x.py -d 1 -b 4 -c ppyolo_r50vd_2x.pth --conf 0.01 --tsize 608
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
python tools/eval.py -f exps/ppyolo/ppyolo_r50vd_2x.py -d 1 -b 8 -c ppyolo_r50vd_2x.pth --conf 0.01 --tsize 320
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

项目根目录下执行：
```
python tools/eval.py -f exps/ppyoloe/ppyoloe_crn_s_300e_coco.py -d 1 -b 8 -c ppyoloe_crn_s_300e_coco.pth --conf 0.01 --tsize 640
```

结果是
```
 Average forward time: 21.56 ms, Average NMS time: 0.00 ms, Average inference time: 21.56 ms
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.423
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.593
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.458
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.237
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.464
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.582
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.336
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.546
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.583
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.356
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.640
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.756
```

项目根目录下执行：
```
python tools/eval.py -f exps/ppyoloe/ppyoloe_crn_m_300e_coco.py -d 1 -b 8 -c ppyoloe_crn_m_300e_coco.pth --conf 0.01 --tsize 640
```

结果是
```
Average forward time: 35.90 ms, Average NMS time: 0.00 ms, Average inference time: 35.90 ms
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.482
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.654
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.523
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.294
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.527
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.645
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.365
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.597
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.635
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.427
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.687
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.803
```

项目根目录下执行：
```
python tools/eval.py -f exps/ppyoloe/ppyoloe_crn_l_300e_coco.py -d 1 -b 4 -c ppyoloe_crn_l_300e_coco.pth --conf 0.01 --tsize 640
```

结果是
```
Average forward time: 49.45 ms, Average NMS time: 0.00 ms, Average inference time: 49.45 ms
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.505
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.681
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.547
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.335
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.552
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.671
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.377
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.617
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.657
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.461
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.709
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.819
```

项目根目录下执行：
```
python tools/eval.py -f exps/ppyoloe/ppyoloe_crn_x_300e_coco.py -d 1 -b 2 -c ppyoloe_crn_x_300e_coco.pth --conf 0.01 --tsize 640
```

结果是
```
Average forward time: 79.69 ms, Average NMS time: 0.00 ms, Average inference time: 79.70 ms
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.515
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.690
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.555
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.338
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.562
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.682
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.383
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.628
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.669
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.484
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.718
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.823
```

如果是使用训练自定义数据集保存的模型进行评估（评估的是自定义数据集的验证集），修改-c为你的模型的路径即可。


需要注意的是，PPYOLO和PPYOLOv2使用的是matrix_nms，为SOLOv2中提出的新的后处理算法，已经包含在head里面，所以评估时的代码捕捉不到NMS的时间，所以显示"Average NMS time: 0.00 ms"。
PPYOLOE使用的是multiclass_nms，也包含在head里面，所以评估时的代码捕捉不到NMS的时间，所以显示"Average NMS time: 0.00 ms"。



## 导出为ONNX

雨露均沾意味着平庸，导出ONNX意义不大。
```

```


## NCNN

### PPYOLOE

(1)第一步，在miemiedetection根目录下输入这些命令下载paddle模型：

```
wget https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_300e_coco.pdparams
wget https://paddledet.bj.bcebos.com/models/ppyoloe_crn_m_300e_coco.pdparams
wget https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams
wget https://paddledet.bj.bcebos.com/models/ppyoloe_crn_x_300e_coco.pdparams
```

(2)第二步，在miemiedetection根目录下输入这些命令将paddle模型转pytorch模型：

```
python tools/convert_weights.py -f exps/ppyoloe/ppyoloe_crn_s_300e_coco.py -c ppyoloe_crn_s_300e_coco.pdparams -oc ppyoloe_crn_s_300e_coco.pth -nc 80
python tools/convert_weights.py -f exps/ppyoloe/ppyoloe_crn_m_300e_coco.py -c ppyoloe_crn_m_300e_coco.pdparams -oc ppyoloe_crn_m_300e_coco.pth -nc 80
python tools/convert_weights.py -f exps/ppyoloe/ppyoloe_crn_l_300e_coco.py -c ppyoloe_crn_l_300e_coco.pdparams -oc ppyoloe_crn_l_300e_coco.pth -nc 80
python tools/convert_weights.py -f exps/ppyoloe/ppyoloe_crn_x_300e_coco.py -c ppyoloe_crn_x_300e_coco.pdparams -oc ppyoloe_crn_x_300e_coco.pth -nc 80
```

(3)第三步，在miemiedetection根目录下输入这些命令将pytorch模型转ncnn模型：

```
python tools/demo.py ncnn -f exps/ppyoloe/ppyoloe_crn_s_300e_coco.py -c ppyoloe_crn_s_300e_coco.pth --ncnn_output_path ppyoloe_crn_s_300e_coco
python tools/demo.py ncnn -f exps/ppyoloe/ppyoloe_crn_m_300e_coco.py -c ppyoloe_crn_m_300e_coco.pth --ncnn_output_path ppyoloe_crn_m_300e_coco
python tools/demo.py ncnn -f exps/ppyoloe/ppyoloe_crn_l_300e_coco.py -c ppyoloe_crn_l_300e_coco.pth --ncnn_output_path ppyoloe_crn_l_300e_coco
python tools/demo.py ncnn -f exps/ppyoloe/ppyoloe_crn_x_300e_coco.py -c ppyoloe_crn_x_300e_coco.pth --ncnn_output_path ppyoloe_crn_x_300e_coco
python tools/demo.py ncnn -f exps/ppyoloe/ppyoloe_crn_l_voc2012.py -c PPYOLOE_outputs/ppyoloe_crn_l_voc2012/6.pth --ncnn_output_path ppyoloe_crn_l_voc2012_epoch_6
```

-c代表读取的权重，--ncnn_output_path表示的是保存为NCNN所用的*.param和*.bin文件的文件名。
运行完这些命令后可在miemiedetection根目录下看到ppyoloe_crn_s_300e_coco.param、ppyoloe_crn_s_300e_coco.bin、...这些文件。

然后，下载[ncnn_ppyolov2](https://github.com/miemie2013/ncnn_ppyolov2) 这个仓库（它自带了glslang和实现了ppyoloe推理），按照官方[how-to-build](https://github.com/Tencent/ncnn/wiki/how-to-build) 文档进行编译ncnn。
编译完成后，
将上文得到的ppyoloe_crn_s_300e_coco.param、ppyoloe_crn_s_300e_coco.bin、...这些文件复制到ncnn_ppyolov2的build/examples/目录下，最后在ncnn_ppyolov2根目录下运行以下命令进行ppyoloe的预测：
```
cd build/examples
./test2_06_ppyoloe_ncnn ../../my_tests/000000000019.jpg ppyoloe_crn_s_300e_coco.param ppyoloe_crn_s_300e_coco.bin

cd build/examples
./test2_06_ppyoloe_ncnn ../../my_tests/000000000019.jpg ppyoloe_crn_m_300e_coco.param ppyoloe_crn_m_300e_coco.bin

cd build/examples
./test2_06_ppyoloe_ncnn ../../my_tests/000000000019.jpg ppyoloe_crn_l_300e_coco.param ppyoloe_crn_l_300e_coco.bin

cd build/examples
./test2_06_ppyoloe_ncnn ../../my_tests/000000000019.jpg ppyoloe_crn_x_300e_coco.param ppyoloe_crn_x_300e_coco.bin

cd build/examples
./test2_06_ppyoloe_ncnn ../../my_tests/000000000019.jpg ppyoloe_crn_l_voc2012_epoch_6.param ppyoloe_crn_l_voc2012_epoch_6.bin

```

test2_06_ppyoloe_ncnn的源码位于examples/test2_06_ppyoloe_ncnn.cpp，参考了yolox.cpp。PPYOLOE算法目前在Linux和Windows平台均已成功预测。

### PPYOLOv2 + PPYOLO

(1)第一步，在miemiedetection根目录下输入这些命令下载paddle模型：

```
wget https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_2x_coco.pdparams
wget https://paddledet.bj.bcebos.com/models/ppyolo_r18vd_coco.pdparams
wget https://paddledet.bj.bcebos.com/models/ppyolov2_r50vd_dcn_365e_coco.pdparams
wget https://paddledet.bj.bcebos.com/models/ppyolov2_r101vd_dcn_365e_coco.pdparams
```

(2)第二步，在miemiedetection根目录下输入这些命令将paddle模型转pytorch模型：

```
python tools/convert_weights.py -f exps/ppyolo/ppyolo_r50vd_2x.py -c ppyolo_r50vd_dcn_2x_coco.pdparams -oc ppyolo_r50vd_2x.pth -nc 80
python tools/convert_weights.py -f exps/ppyolo/ppyolo_r18vd.py -c ppyolo_r18vd_coco.pdparams -oc ppyolo_r18vd.pth -nc 80
python tools/convert_weights.py -f exps/ppyolo/ppyolov2_r50vd_365e.py -c ppyolov2_r50vd_dcn_365e_coco.pdparams -oc ppyolov2_r50vd_365e.pth -nc 80
python tools/convert_weights.py -f exps/ppyolo/ppyolov2_r101vd_365e.py -c ppyolov2_r101vd_dcn_365e_coco.pdparams -oc ppyolov2_r101vd_365e.pth -nc 80
```

(3)第三步，在miemiedetection根目录下输入这些命令将pytorch模型转ncnn模型：

```
python tools/demo.py ncnn -f exps/ppyolo/ppyolo_r18vd.py -c ppyolo_r18vd.pth --ncnn_output_path ppyolo_r18vd --conf 0.15
python tools/demo.py ncnn -f exps/ppyolo/ppyolo_r50vd_2x.py -c ppyolo_r50vd_2x.pth --ncnn_output_path ppyolo_r50vd_2x --conf 0.15
python tools/demo.py ncnn -f exps/ppyolo/ppyolov2_r50vd_365e.py -c ppyolov2_r50vd_365e.pth --ncnn_output_path ppyolov2_r50vd_365e --conf 0.15
python tools/demo.py ncnn -f exps/ppyolo/ppyolov2_r101vd_365e.py -c ppyolov2_r101vd_365e.pth --ncnn_output_path ppyolov2_r101vd_365e --conf 0.15
```

-c代表读取的权重，--ncnn_output_path表示的是保存为NCNN所用的*.param和.bin文件的文件名，--conf 0.15表示的是在PPYOLODecodeMatrixNMS层中将score_threshold和post_threshold设置为0.15，你可以在导出的*.param中修改score_threshold和post_threshold，分别是PPYOLODecodeMatrixNMS层的5=xxx 7=xxx属性。

然后，下载[ncnn_ppyolov2](https://github.com/miemie2013/ncnn_ppyolov2) 这个仓库（它自带了glslang和实现了ppyolov2推理），按照官方[how-to-build](https://github.com/Tencent/ncnn/wiki/how-to-build) 文档进行编译ncnn。
编译完成后，
将上文得到的ppyolov2_r50vd_365e.param、ppyolov2_r50vd_365e.bin、...这些文件复制到ncnn_ppyolov2的build/examples/目录下，最后在ncnn_ppyolov2根目录下运行以下命令进行ppyolov2的预测：

```
cd build/examples
./test2_06_ppyolo_ncnn ../../my_tests/000000013659.jpg ppyolo_r18vd.param ppyolo_r18vd.bin 416
./test2_06_ppyolo_ncnn ../../my_tests/000000013659.jpg ppyolo_r50vd_2x.param ppyolo_r50vd_2x.bin 608
./test2_06_ppyolo_ncnn ../../my_tests/000000013659.jpg ppyolov2_r50vd_365e.param ppyolov2_r50vd_365e.bin 640
./test2_06_ppyolo_ncnn ../../my_tests/000000013659.jpg ppyolov2_r101vd_365e.param ppyolov2_r101vd_365e.bin 640
```

每条命令最后1个参数416、608、640表示的是将图片resize到416、608、640进行推理，即target_size参数。

test2_06_ppyolo_ncnn的源码位于ncnn_ppyolov2仓库的examples/test2_06_ppyolo_ncnn.cpp。PPYOLOv2和PPYOLO算法目前在Linux和Windows平台均已成功预测。


## TensorRT

实现中...
```

```


## 引用

[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)

[PPYOLOE paper](https://arxiv.org/pdf/2203.16250.pdf)

```
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

