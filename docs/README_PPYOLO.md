简体中文 | [English](README_PPYOLO_en.md)

# miemiedetection

详细命令见根目录下readme_yolo.txt，如果命令不能成功执行，说明咩酱实现中，，，（and，把PPYOLOv2完全实现再补全文档)


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

-f表示的是使用的配置文件；-c表示的是读取的源权重文件；-oc表示的是输出（保存）的pytorch权重文件；-nc表示的是数据集的类别数。
执行完毕后就会在项目根目录下获得转换好的*.pth权重文件。


## 预测

（1）预测一张图片，项目根目录下执行：
```
python tools/demo.py image -f exps/ppyolo/ppyolo_r50vd_2x.py -c ppyolo_2x.pth --path assets/000000000019.jpg --conf 0.15 --tsize 608 --save_result --device gpu
```

```
python tools/demo.py image -f exps/ppyolo/ppyolo_r18vd.py -c ppyolo_r18vd.pth --path assets/000000000019.jpg --conf 0.15 --tsize 416 --save_result --device gpu
```

-f表示的是使用的配置文件；-c表示的是读取的权重文件；--path表示的是图片的路径；--conf表示的是分数阈值，只会画出高于这个阈值的预测框；--tsize表示的是预测时将图片Resize成--tsize的分辨率。预测完成后控制台会打印结果图片的保存路径，用户可打开查看。

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



## 预测

```

```


