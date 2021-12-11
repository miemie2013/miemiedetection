



https://zhuanlan.zhihu.com/p/392221567


pip install thop -i https://mirror.baidu.com/pypi/simple
pip install tabulate -i https://mirror.baidu.com/pypi/simple


yolox_base.py里
self.data_dir = '../COCO'
设置数据集路径



----------------------- 转换权重 -----------------------
python tools/convert_weights.py -f exps/ppyolo/ppyolo_r50vd_2x.py -c ppyolo.pdparams -oc ppyolo_2x.pth -nc 80


python tools/convert_weights.py -f ../exps/ppyolo/ppyolo_r50vd_2x.py -c ../ppyolo.pdparams -oc ../ppyolo_2x.pth -nc 80





----------------------- 预测 -----------------------
python tools/demo.py image -f exps/ppyolo/ppyolo_r50vd_2x.py -c ppyolo_2x.pth --path assets/dog.jpg --conf 0.15 --tsize 608 --save_result --device gpu


(调试，配置文件中self.data_dir、self.cls_names、self.output_dir的前面已经自动加上'../')
python tools/demo.py image -f ../exps/ppyolo/ppyolo_r50vd_2x.py -c ../ppyolo_2x.pth --path ../assets/dog.jpg --conf 0.15 --tsize 608 --save_result --device gpu



python tools/demo.py image -f exps/ppyolo/ppyolo_r50vd_2x.py -c PPYOLO_outputs/yolox_m/1.pth --path D://PycharmProjects/Paddle-PPYOLO-master/images/test --conf 0.15 --tsize 640 --save_result --device gpu





预测（逐行调试）
python tools/demo.py image -f exps/ppyolo/ppyolo_r50vd_2x.py -c ppyolo_2x.pth --path assets/dog.jpg --conf 0.15 --tsize 640 --save_result --device gpu

python demo2.py image -f exps/ppyolo/ppyolo_r50vd_2x.py -c yolox_s.pth --path assets/dog.jpg --conf 0.15 --tsize 640 --save_result --device gpu



----------------------- 评估 -----------------------
python tools/eval.py -f exps/ppyolo/ppyolo_r50vd_2x.py -d 1 -b 8 -c ppyolo_2x.pth --conf 0.01 [--fp16] [--fuse]

python tools/eval.py -f exps/ppyolo/ppyolo_r50vd_2x.py -d 1 -b 8 -c ppyolo_2x.pth --conf 0.01


(调试，配置文件中self.data_dir和self.cls_names前面也要加上../)
python tools/eval.py -f ../exps/ppyolo/ppyolo_r50vd_2x.py -d 1 -b 8 -c ../ppyolo_2x.pth --conf 0.01



----------------------- 训练 -----------------------
修改配置文件
        self.train_ann = "instances_train2017.json"
        self.train_ann = "instances_val2017.json"
        self.val_ann = "instances_val2017.json"
修改coco.py
        name = 'val2017'
        self.data_dir = '../COCO'
        self.json_file = json_file

python tools/train.py -f exps/ppyolo/ppyolo_r50vd_2x.py -d 1 -b 8 --fp16


python train2.py -f exps/ppyolo/ppyolo_r50vd_2x.py -d 1 -b 2 --fp16 -o




----------------------- 迁移学习，带上-c（--ckpt）参数读取预训练模型。 -----------------------
复现paddle版yolox迁移学习:（可以加--fp16， -eb表示验证时的批大小）
python tools/train.py -f exps/ppyolo/ppyolo_r50vd_2x.py -d 1 -b 8 -eb 2 --fp16 -c ppyolo_2x.pth


实测yolox_m的AP(0.50:0.95)可以到达0.62+、AP(small)可以到达0.25+。



复现paddle版yolox迁移学习:（用来调试。另外，yolox_base.py的self.data_dir、self.cls_names前面也要加上../）
python tools/train.py -f ../exps/ppyolo/ppyolo_r50vd_2x.py -d 1 -b 2 -eb 2 -c ../ppyolo_2x.pth
python tools/train.py -f ../exps/ppyolo/ppyolo_r50vd_2x.py -d 1 -b 2 -eb 2




----------------------- 恢复训练（加上参数--resume） -----------------------
python tools/train.py -f exps/ppyolo/ppyolo_r50vd_2x.py -d 1 -b 8 -eb 2 --fp16 -c PPYOLO_outputs/ppyolo_r50vd_2x/3.pth --resume





----------------------- 导出为ONNX -----------------------
见demo/ONNXRuntime/README.md

会设置model.head.decode_in_inference = False，此时只对置信位和各类别概率进行sigmoid()激活。xywh没有进行解码，更没有进行nms。
python tools/export_onnx.py --output-name yolox_s.onnx -f exps/ppyolo/yolox_s.py -c ppyolo_2x.pth



ONNX预测，命令改动为（用numpy对xywh进行解码，进行nms。）
python tools/onnx_inference.py -m yolox_s.onnx -i assets/dog.jpg -o ONNX_YOLOX_outputs -s 0.3 --input_shape 640,640 -cn ./class_names/coco_classes.txt


ONNX预测（调试）
python tools/onnx_inference.py -m ../yolox_s.onnx -i ../assets/dog.jpg -o ../ONNX_YOLOX_outputs -s 0.3 --input_shape 640,640 -cn ../class_names/coco_classes.txt



