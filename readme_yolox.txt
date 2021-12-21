
nvidia-smi


pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


----------------------- 预测 -----------------------
python tools/demo.py image -f exps/yolox/yolox_s.py -c yolox_s.pth --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu


python tools/demo.py image -f exps/yolox/yolox_m.py -c YOLOX_outputs/yolox_m/1.pth --path D://PycharmProjects/Paddle-PPYOLO-master/images/test --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu






----------------------- 评估 -----------------------
python tools/eval.py -f exps/yolox/yolox_s.py -d 1 -b 8 -c yolox_s.pth --conf 0.001


Average forward time: 10.76 ms, Average NMS time: 2.75 ms, Average inference time: 13.51 ms
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.405
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.593
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.438
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.233
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.448
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.541
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.326
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.531
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.574
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.366
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.635
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.724



python tools/eval.py -f exps/yolox/yolox_m.py -d 1 -b 8 -c yolox_m.pth --conf 0.001



----------------------- 训练 -----------------------
python tools/train.py -f exps/yolox/yolox_s.py -d 8 -b 64 --fp16 -o [--cache]


python tools/train.py -f exps/yolox/yolox_s.py -d 1 -b 8 --fp16 -o --cache


python tools/train.py -f exps/yolox/yolox_s.py -d 1 -b 8 --fp16


python train2.py -f exps/yolox/yolox_s.py -d 1 -b 2 --fp16 -o




----------------------- 迁移学习，带上-c（--ckpt）参数读取预训练模型。 -----------------------
复现paddle版yolox_m迁移学习:（可以加--fp16， -eb表示验证时的批大小）
python tools/train.py -f exps/yolox/yolox_m.py -d 1 -b 8 -eb 2 --fp16 -c yolox_m.pth


实测yolox_m的AP(0.50:0.95)可以到达0.62+、AP(small)可以到达0.25+。




复现ppdet版yolox_s迁移学习:（可以加--fp16， -eb表示验证时的批大小）
python tools/train.py -f exps/yolox/yolox_s.py -d 1 -b 8 -eb 2 --fp16 -c yolox_s.pth


实测yolox_s的AP(0.50:0.95)可以到达0.53+、AP(small)可以到达0.22+。






----------------------- 恢复训练（加上参数--resume） -----------------------
python tools/train.py -f exps/yolox/yolox_m.py -d 1 -b 8 -eb 2 --fp16 -c YOLOX_outputs/yolox_m/3.pth --resume





----------------------- 导出为ONNX -----------------------
见demo/ONNXRuntime/README.md

会设置model.head.decode_in_inference = False，此时只对置信位和各类别概率进行sigmoid()激活。xywh没有进行解码，更没有进行nms。
python tools/export_onnx.py --output-name yolox_s.onnx -f exps/yolox/yolox_s.py -c yolox_s.pth



ONNX预测，命令改动为（用numpy对xywh进行解码，进行nms。）
python tools/onnx_inference.py -m yolox_s.onnx -i assets/dog.jpg -o ONNX_YOLOX_outputs -s 0.3 --input_shape 640,640 -cn ./class_names/coco_classes.txt



用onnx模型进行验证，

python tools/onnx_eval.py -an YOLOX -m yolox_s.onnx -i ../COCO/val2017 -a ../COCO/annotations/instances_val2017.json -s 0.001 --input_shape 640,640 --eval_type eval

(不是因为少预测了没有gt的图片)
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.394
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.578
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.426
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.217
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.437
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.525
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.311
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.497
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.535
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.327
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.588
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.670

