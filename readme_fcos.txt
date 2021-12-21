
nvidia-smi


pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


----------------------- 转换权重 -----------------------
python tools/convert_weights.py -f exps/ppyolo/ppyolo_r50vd_2x.py -c ppyolo.pdparams -oc ppyolo_2x.pth -nc 80


python tools/convert_weights.py -f exps/ppyolo/ppyolo_r18vd.py -c ppyolo_r18vd.pdparams -oc ppyolo_r18vd.pth -nc 80




----------------------- 预测 -----------------------
python tools/demo.py image -f exps/fcos/fcos_rt_r50_fpn_4x.py -c ppyolo_2x.pth --path assets/dog.jpg --conf 0.15 --tsize 512 --save_result --device gpu


python tools/demo.py image -f exps/ppyolo/ppyolo_r18vd.py -c ppyolo_r18vd.pth --path assets/dog.jpg --conf 0.15 --tsize 416 --save_result --device gpu




python tools/demo.py image -f exps/ppyolo/ppyolo_r50vd_2x.py -c ppyolo_2x.pth --path D://GitHub/Pytorch-YOLO/images/test --conf 0.15 --tsize 608 --save_result --device gpu


python tools/demo.py image -f exps/ppyolo/ppyolo_r18vd.py -c ppyolo_r18vd.pth --path D://GitHub/Pytorch-YOLO/images/test --conf 0.15 --tsize 416 --save_result --device gpu







python tools/demo.py image -f exps/ppyolo/ppyolo_r50vd_2x.py -c PPYOLO_outputs/yolox_m/1.pth --path D://PycharmProjects/Paddle-PPYOLO-master/images/test --conf 0.15 --tsize 640 --save_result --device gpu





预测（逐行调试）
python tools/demo.py image -f exps/ppyolo/ppyolo_r50vd_2x.py -c ppyolo_2x.pth --path assets/dog.jpg --conf 0.15 --tsize 640 --save_result --device gpu

python demo2.py image -f exps/ppyolo/ppyolo_r50vd_2x.py -c yolox_s.pth --path assets/dog.jpg --conf 0.15 --tsize 640 --save_result --device gpu



----------------------- 评估 -----------------------
python tools/eval.py -f exps/ppyolo/ppyolo_r50vd_2x.py -d 1 -b 2 -c ppyolo_2x.pth --conf 0.01 --tsize 608


Average forward time: 43.35 ms, Average NMS time: 0.01 ms, Average inference time: 43.36 ms
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.448
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.649
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.492
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.265
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.483
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.593
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.337
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.571
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.624
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.420
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.665
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.773



python tools/eval.py -f exps/ppyolo/ppyolo_r50vd_2x.py -d 1 -b 2 -c ppyolo_2x.pth --conf 0.01 --tsize 320


Average forward time: 17.06 ms, Average NMS time: 0.00 ms, Average inference time: 17.06 ms
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.389
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.585
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.420
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.173
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.420
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.592
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.309
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.508
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.553
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.300
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.604
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.766


python tools/eval.py -f exps/ppyolo/ppyolo_r18vd.py -d 1 -b 8 -c ppyolo_r18vd.pth --conf 0.01 --tsize 416


Average forward time: 8.06 ms, Average NMS time: 0.00 ms, Average inference time: 8.06 ms
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




----------------------- 训练 -----------------------
python tools/train.py -f exps/ppyolo/ppyolo_r50vd_2x.py -d 1 -b 8 --fp16





----------------------- 迁移学习，带上-c（--ckpt）参数读取预训练模型。 -----------------------
复现paddle版ppyolo2x迁移学习:（可以加--fp16， -eb表示验证时的批大小）
python tools/train.py -f exps/ppyolo/ppyolo_r50vd_2x.py -d 1 -b 8 -eb 4 -c ppyolo_2x.pth


实测ppyolo_r50vd_2x的AP(0.50:0.95)可以到达0.59+、AP(0.50)可以到达0.82+、AP(small)可以到达0.18+。


复现paddle版ppyolo_r18vd迁移学习:（可以加--fp16， -eb表示验证时的批大小）
python tools/train.py -f exps/ppyolo/ppyolo_r18vd.py -d 1 -b 16 -eb 8 -c ppyolo_r18vd.pth


实测ppyolo_r50vd_2x的AP(0.50:0.95)可以到达0.xx+、AP(0.50)可以到达0.xx+、AP(small)可以到达0.xx+。





----------------------- 恢复训练（加上参数--resume） -----------------------
python tools/train.py -f exps/ppyolo/ppyolo_r50vd_2x.py -d 1 -b 8 -eb 4 -c PPYOLO_outputs/ppyolo_r50vd_2x/13.pth --resume


python tools/train.py -f exps/ppyolo/ppyolo_r18vd.py -d 1 -b 16 -eb 8 -c PPYOLO_outputs/ppyolo_r18vd/7.pth --resume





----------------------- 导出为ONNX -----------------------
见demo/ONNXRuntime/README.md

会设置model.head.decode_in_inference = False，此时只对置信位和各类别概率进行sigmoid()激活。xywh没有进行解码，更没有进行nms。
python tools/export_onnx.py --output-name ppyolo_2x.onnx -f exps/ppyolo/ppyolo_r50vd_2x.py -c ppyolo_2x.pth


python tools/export_onnx.py --output-name ppyolo_r18vd.onnx -f exps/ppyolo/ppyolo_r18vd.py -c ppyolo_r18vd.pth



ONNX预测，命令改动为（用numpy对xywh进行解码，进行nms。）


python tools/onnx_inference.py -an PPYOLO -acn ppyolo_r18vd -m ppyolo_r18vd.onnx -i assets/dog.jpg -o ONNX_PPYOLO_R18VD_outputs -s 0.15 --input_shape 416,416 -cn class_names/coco_classes.txt


python tools/onnx_inference.py -an PPYOLO -acn ppyolo_r18vd -m ppyolo_r18vd.onnx -i D://GitHub/Pytorch-YOLO/images/test/000000052996.jpg -o ONNX_PPYOLO_R18VD_outputs -s 0.15 --input_shape 416,416 -cn class_names/coco_classes.txt






用onnx模型进行验证，

python tools/onnx_eval.py -an PPYOLO -acn ppyolo_r18vd -m ppyolo_r18vd.onnx -i ../COCO/val2017 -a ../COCO/annotations/instances_val2017.json -s 0.01 --input_shape 416,416 --eval_type eval


python tools/onnx_eval.py -an PPYOLO -acn ppyolo_r18vd -m ppyolo_r18vd.onnx -i ../COCO/val2017 -a ../COCO/annotations/instances_val2017.json -s 0.01 --input_shape 416,416 --eval_type test_dev



