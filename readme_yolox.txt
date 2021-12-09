



https://zhuanlan.zhihu.com/p/392221567


pip install thop -i https://mirror.baidu.com/pypi/simple
pip install tabulate -i https://mirror.baidu.com/pypi/simple


yolox_base.py里
self.data_dir = '../COCO'
设置数据集路径



预测
python tools/demo.py image -f exps/yolox/yolox_s.py -c yolox_s.pth --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu



预测（逐行调试）
python tools/demo.py image -f exps/yolox/yolox_s.py -c yolox_s.pth --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu

python demo2.py image -f exps/yolox/yolox_s.py -c yolox_s.pth --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu


评估
python tools/eval.py -f exps/yolox/yolox_s.py -d 1 -b 8 -c yolox_s.pth --conf 0.001 [--fp16] [--fuse]

python tools/eval.py -f exps/yolox/yolox_s.py -d 1 -b 8 -c yolox_s.pth --conf 0.001


python tools/eval.py -f exps/yolox/yolox_m.py -d 1 -b 8 -c yolox_m.pth --conf 0.001



训练
修改配置文件
        self.train_ann = "instances_train2017.json"
        self.train_ann = "instances_val2017.json"
        self.val_ann = "instances_val2017.json"
修改coco.py
        name = 'val2017'
        self.data_dir = '../COCO'
        self.json_file = json_file

python tools/train.py -f exps/yolox/yolox_s.py -d 8 -b 64 --fp16 -o [--cache]


python tools/train.py -f exps/yolox/yolox_s.py -d 1 -b 8 --fp16 -o --cache


python tools/train.py -f exps/yolox/yolox_s.py -d 1 -b 8 --fp16


python train2.py -f exps/yolox/yolox_s.py -d 1 -b 2 --fp16 -o



恢复训练




迁移学习，带上-c（--ckpt）参数读取预训练模型。
python tools/train.py -f exps/yolox/yolox_m.py -d 1 -b 8 --fp16 -c yolox_m.pth



复现paddle版yolox迁移学习:（不加--fp16）
python tools/train.py -f exps/yolox/yolox_m.py -d 1 -b 8 -c yolox_m.pth

实测yolox_m的AP(0.50:0.95)可以到达0.62+、AP(small)可以到达0.25+。






cd demo/ONNXRuntime
python onnx_inference.py -m ../../yolox_s.onnx -i ../../assets/dog.jpg -o aaa -s 0.3 --input_shape 640,640




