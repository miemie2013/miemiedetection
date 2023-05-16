
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


----------------------- 转换权重 -----------------------
wget https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_2x_coco.pdparams
wget https://paddledet.bj.bcebos.com/models/ppyolo_r18vd_coco.pdparams
wget https://paddledet.bj.bcebos.com/models/ppyolov2_r50vd_dcn_365e_coco.pdparams
wget https://paddledet.bj.bcebos.com/models/ppyolov2_r101vd_dcn_365e_coco.pdparams
wget https://paddledet.bj.bcebos.com/models/pretrained/ResNet50_vd_ssld_pretrained.pdparams
wget https://paddledet.bj.bcebos.com/models/pretrained/ResNet18_vd_pretrained.pdparams
wget https://paddledet.bj.bcebos.com/models/pretrained/ResNet101_vd_ssld_pretrained.pdparams



python tools/convert_weights.py -f exps/ppyolo/ppyolo_r50vd_2x.py -c ppyolo_r50vd_dcn_2x_coco.pdparams -oc ppyolo_r50vd_2x.pth -nc 80
python tools/convert_weights.py -f exps/ppyolo/ppyolo_r18vd.py -c ppyolo_r18vd_coco.pdparams -oc ppyolo_r18vd.pth -nc 80
python tools/convert_weights.py -f exps/ppyolo/ppyolov2_r50vd_365e.py -c ppyolov2_r50vd_dcn_365e_coco.pdparams -oc ppyolov2_r50vd_365e.pth -nc 80
python tools/convert_weights.py -f exps/ppyolo/ppyolov2_r101vd_365e.py -c ppyolov2_r101vd_dcn_365e_coco.pdparams -oc ppyolov2_r101vd_365e.pth -nc 80

python tools/convert_weights.py -f exps/ppyolo/ppyolo_r18vd.py -c ResNet18_vd_pretrained.pdparams -oc ResNet18_vd_pretrained.pth -nc 80 --only_backbone True
python tools/convert_weights.py -f exps/ppyolo/ppyolov2_r50vd_365e.py -c ResNet50_vd_ssld_pretrained.pdparams -oc ResNet50_vd_ssld_pretrained.pth -nc 80 --only_backbone True
python tools/convert_weights.py -f exps/ppyolo/ppyolov2_r101vd_365e.py -c ResNet101_vd_ssld_pretrained.pdparams -oc ResNet101_vd_ssld_pretrained.pth -nc 80 --only_backbone True


----------------------- 预测 -----------------------
python tools/demo.py image -f exps/ppyolo/ppyolo_r50vd_2x.py -c ppyolo_r50vd_2x.pth --path assets/000000000019.jpg --conf 0.15 --tsize 608 --save_result --device gpu


python tools/demo.py image -f exps/ppyolo/ppyolo_r18vd.py -c ppyolo_r18vd.pth --path assets/000000000019.jpg --conf 0.15 --tsize 416 --save_result --device gpu

python tools/demo.py image -f exps/ppyolo/ppyolo_r18vd.py -c ppyolo_r18vd.pth --path assets/000000013659.jpg --conf 0.15 --tsize 416 --save_result --device gpu



python tools/demo.py image -f exps/ppyolo/ppyolov2_r50vd_365e.py -c ppyolov2_r50vd_365e.pth --path assets/000000000019.jpg --conf 0.15 --tsize 640 --save_result --device gpu


python tools/demo.py image -f exps/ppyolo/ppyolov2_r101vd_365e.py -c ppyolov2_r101vd_365e.pth --path assets/000000000019.jpg --conf 0.15 --tsize 640 --save_result --device gpu




python tools/demo.py image -f exps/ppyolo/ppyolo_r50vd_2x.py -c ppyolo_r50vd_2x.pth --path D://GitHub/Pytorch-YOLO/images/test --conf 0.15 --tsize 608 --save_result --device gpu


python tools/demo.py image -f exps/ppyolo/ppyolo_r18vd.py -c ppyolo_r18vd.pth --path D://GitHub/Pytorch-YOLO/images/test --conf 0.15 --tsize 416 --save_result --device gpu







python tools/demo.py image -f exps/ppyolo/ppyolo_r50vd_2x.py -c 1.pth --path D://PycharmProjects/Paddle-PPYOLO-master/images/test --conf 0.15 --tsize 640 --save_result --device gpu



----------------------- 导出为ncnn -----------------------
python tools/demo.py ncnn -f exps/ppyolo/ppyolo_r18vd.py -c ppyolo_r18vd.pth --ncnn_output_path ppyolo_r18vd --conf 0.15

python tools/demo.py ncnn -f exps/ppyolo/ppyolo_r50vd_2x.py -c ppyolo_r50vd_2x.pth --ncnn_output_path ppyolo_r50vd_2x --conf 0.15

python tools/demo.py ncnn -f exps/ppyolo/ppyolov2_r50vd_365e.py -c ppyolov2_r50vd_365e.pth --ncnn_output_path ppyolov2_r50vd_365e --conf 0.15

python tools/demo.py ncnn -f exps/ppyolo/ppyolov2_r101vd_365e.py -c ppyolov2_r101vd_365e.pth --ncnn_output_path ppyolov2_r101vd_365e --conf 0.15

(导出半精度的ncnn模型)
* 【2022/08/07】 支持导出半精度的NCNN模型！详情请参考[PPYOLO](docs/README_PPYOLO.md#NCNN) 文档的“NCNN”小节。
python tools/demo.py ncnn -f exps/ppyolo/ppyolo_r18vd.py -c ppyolo_r18vd.pth --ncnn_output_path ppyolo_r18vd_fp16 --conf 0.15 --fp16

python tools/demo.py ncnn -f exps/ppyolo/ppyolo_r50vd_2x.py -c ppyolo_r50vd_2x.pth --ncnn_output_path ppyolo_r50vd_2x_fp16 --conf 0.15 --fp16

python tools/demo.py ncnn -f exps/ppyolo/ppyolov2_r50vd_365e.py -c ppyolov2_r50vd_365e.pth --ncnn_output_path ppyolov2_r50vd_365e_fp16 --conf 0.15 --fp16

python tools/demo.py ncnn -f exps/ppyolo/ppyolov2_r101vd_365e.py -c ppyolov2_r101vd_365e.pth --ncnn_output_path ppyolov2_r101vd_365e_fp16 --conf 0.15 --fp16


cd build/examples
./test2_06_ppyolo_ncnn ../../my_tests/000000000019.jpg ppyolo_r18vd.param ppyolo_r18vd.bin 416

./test2_06_ppyolo_ncnn ../../my_tests/000000013659.jpg ppyolo_r18vd.param ppyolo_r18vd.bin 416

./test2_06_ppyolo_ncnn ../../my_tests/000000013659.jpg ppyolo_r50vd_2x.param ppyolo_r50vd_2x.bin 608

./test2_06_ppyolo_ncnn ../../my_tests/000000013659.jpg ppyolov2_r50vd_365e.param ppyolov2_r50vd_365e.bin 640

./test2_06_ppyolo_ncnn ../../my_tests/000000013659.jpg ppyolov2_r101vd_365e.param ppyolov2_r101vd_365e.bin 640


./test2_06_ppyolo_ncnn ../../my_tests/000000013659.jpg ppyolo_r18vd_fp16.param ppyolo_r18vd_fp16.bin 416

./test2_06_ppyolo_ncnn ../../my_tests/000000013659.jpg ppyolo_r50vd_2x_fp16.param ppyolo_r50vd_2x_fp16.bin 608

./test2_06_ppyolo_ncnn ../../my_tests/000000013659.jpg ppyolov2_r50vd_365e_fp16.param ppyolov2_r50vd_365e_fp16.bin 640

./test2_06_ppyolo_ncnn ../../my_tests/000000013659.jpg ppyolov2_r101vd_365e_fp16.param ppyolov2_r101vd_365e_fp16.bin 640


(和mmdet比较结果)
python tools/demo.py image -f exps/ppyolo/ppyolo_r18vd.py -c ppyolo_r18vd.pth --path assets/000000013659.jpg --conf 0.15 --tsize 416 --save_result --device gpu

python tools/demo.py image -f exps/ppyolo/ppyolo_r50vd_2x.py -c ppyolo_r50vd_2x.pth --path assets/000000013659.jpg --conf 0.15 --tsize 608 --save_result --device gpu

python tools/demo.py image -f exps/ppyolo/ppyolov2_r50vd_365e.py -c ppyolov2_r50vd_365e.pth --path assets/000000013659.jpg --conf 0.15 --tsize 640 --save_result --device gpu

python tools/demo.py image -f exps/ppyolo/ppyolov2_r101vd_365e.py -c ppyolov2_r101vd_365e.pth --path assets/000000013659.jpg --conf 0.15 --tsize 640 --save_result --device gpu



(用pnnx导出)
D://GitHub/ncnn2/tools/pnnx/build/install/bin/pnnx ppyolov2_r50vd_365e.pt inputshape=[1,3,640,640]





----------------------- 训练 -----------------------
python tools/train.py -f exps/ppyolo/ppyolo_r50vd_2x.py -d 1 -b 8 -eb 4





----------------------- 迁移学习，带上-c（--ckpt）参数读取预训练模型。 -----------------------
后台启动：
nohup xxx     > ppyolo.log 2>&1 &


PPYOLO把RandomShape、NormalizeImage、Permute、Gt2YoloTarget这4个预处理步骤放到了sample_transforms中，
而不是放到batch_transforms中，虽然这样写不美观，但是可以提速n倍。因为用collate_fn实现batch_transforms太耗时了！能不使用batch_transforms尽量不使用batch_transforms！

复现paddle版ppyolo2x迁移学习:（可以加--fp16， -eb表示验证时的批大小）
python tools/train.py -f exps/ppyolo/ppyolo_r50vd_voc2012.py -d 1 -b 8 -eb 4 -c ppyolo_r50vd_2x.pth

python tools/eval.py -f exps/ppyolo/ppyolo_r50vd_voc2012.py -d 1 -b 4 -c 16.pth --conf 0.01 --tsize 608


2机2卡训练：(发现一个隐藏知识点：获得损失（训练）、推理 都要放在模型的forward()中进行，否则DDP会计算错误结果。)
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -f exps/ppyolo/ppyolo_r50vd_voc2012.py --dist-url tcp://192.168.0.107:12312 --num_machines 2 --machine_rank 0 -b 8 -eb 4 -c ppyolo_r50vd_2x.pth

python tools/eval.py -f exps/ppyolo/ppyolo_r50vd_voc2012.py -d 1 -b 8 -c 16.pth --conf 0.01 --tsize 608


1机2卡训练：(发现一个隐藏知识点：获得损失（训练）、推理 都要放在模型的forward()中进行，否则DDP会计算错误结果。)
export CUDA_VISIBLE_DEVICES=0,1
nohup python tools/train.py -f exps/ppyolo/ppyolo_r50vd_voc2012.py -d 2 -b 8 -eb 2 -c ppyolo_r50vd_2x.pth     > ppyolo.log 2>&1 &

tail -n 20 ppyolo.log


实测ppyolo_r50vd_2x的AP(0.50:0.95)可以到达0.59+、AP(0.50)可以到达0.82+、AP(small)可以到达0.18+。
- - - - - - - - - - - - - - - - - - - - - -


复现paddle版ppyolo_r18vd迁移学习:（可以加--fp16， -eb表示验证时的批大小）
python tools/train.py -f exps/ppyolo/ppyolo_r18vd_voc2012.py -d 1 -b 8 -eb 4 -c ppyolo_r18vd.pth


2机2卡训练：
after_epoch(self):里会调用
            all_reduce_norm(self.model)
多卡训练时会报错，所以设置配置文件里的self.eval_interval = 99999999
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -f exps/ppyolo/ppyolo_r18vd_voc2012.py --dist-url tcp://192.168.0.107:12312 --num_machines 2 --machine_rank 0 -b 8 -eb 4 -c ppyolo_r18vd.pth




1机2卡训练：
export CUDA_VISIBLE_DEVICES=0,1
python tools/train.py -f exps/ppyolo/ppyolo_r18vd_voc2012.py -d 2 -b 8 -eb 4 -c ppyolo_r18vd.pth


实测ppyolo_r18vd的AP(0.50:0.95)可以到达0.39+、AP(0.50)可以到达0.65+、AP(small)可以到达0.06+。
- - - - - - - - - - - - - - - - - - - - - -

复现paddle版ppyolov2迁移学习:（可以加--fp16， -eb表示验证时的批大小）
python tools/train.py -f exps/ppyolo/ppyolov2_r50vd_voc2012.py -d 1 -b 8 -eb 2 -c ppyolov2_r50vd_365e.pth


2机2卡训练：(发现一个隐藏知识点：获得损失（训练）、推理 都要放在模型的forward()中进行，否则DDP会计算错误结果。)
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -f exps/ppyolo/ppyolov2_r50vd_voc2012.py --dist-url tcp://192.168.0.107:12312 --num_machines 2 --machine_rank 0 -b 8 -eb 2 -c ppyolov2_r50vd_365e.pth


1机2卡训练：(发现一个隐藏知识点：获得损失（训练）、推理 都要放在模型的forward()中进行，否则DDP会计算错误结果。)
export CUDA_VISIBLE_DEVICES=0,1
nohup python tools/train.py -f exps/ppyolo/ppyolov2_r50vd_voc2012.py -d 2 -b 8 -eb 2 -c ppyolov2_r50vd_365e.pth     > ppyolov2.log 2>&1 &

tail -n 20 ppyolov2.log



实测ppyolov2_r50vd_365e的AP(0.50:0.95)可以到达0.63+、AP(0.50)可以到达0.84+、AP(small)可以到达0.25+。


- - - - - - - - - - - - - - - - - - - - - -


----------------------- 恢复训练（加上参数--resume） -----------------------
python tools/train.py -f exps/ppyolo/ppyolo_r50vd_2x.py -d 1 -b 8 -eb 4 -c 13.pth --resume


python tools/train.py -f exps/ppyolo/ppyolo_r18vd.py -d 1 -b 16 -eb 8 -c 7.pth --resume





----------------------- 评估 -----------------------
python tools/eval.py -f exps/ppyolo/ppyolo_r50vd_2x.py -d 1 -b 4 -c ppyolo_r50vd_2x.pth --conf 0.01 --tsize 608


Average forward time: 34.49 ms, Average NMS time: 0.00 ms, Average inference time: 34.49 ms
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



python tools/eval.py -f exps/ppyolo/ppyolo_r50vd_2x.py -d 1 -b 8 -c ppyolo_r50vd_2x.pth --conf 0.01 --tsize 320


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


python tools/eval.py -f exps/ppyolo/ppyolo_r18vd.py -d 1 -b 8 -c ppyolo_r18vd.pth --conf 0.01 --tsize 416


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



python tools/eval.py -f exps/ppyolo/ppyolov2_r50vd_365e.py -d 1 -b 4 -c ppyolov2_r50vd_365e.pth --conf 0.01 --tsize 640


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


python tools/eval.py -f exps/ppyolo/ppyolov2_r50vd_365e.py -d 1 -b 8 -c ppyolov2_r50vd_365e.pth --conf 0.01 --tsize 320


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



python tools/eval.py -f exps/ppyolo/ppyolov2_r101vd_365e.py -d 1 -b 4 -c ppyolov2_r101vd_365e.pth --conf 0.01 --tsize 640


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


python tools/eval.py -f exps/ppyolo/ppyolov2_r101vd_365e.py -d 1 -b 8 -c ppyolov2_r101vd_365e.pth --conf 0.01 --tsize 320


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


----------------------- 导出为ONNX -----------------------
见demo/ONNXRuntime/README.md

会设置model.head.decode_in_inference = False，此时只对置信位和各类别概率进行sigmoid()激活。xywh没有进行解码，更没有进行nms。
python tools/export_onnx.py --output-name ppyolo_r50vd_2x.onnx -f exps/ppyolo/ppyolo_r50vd_2x.py -c ppyolo_r50vd_2x.pth


python tools/export_onnx.py --output-name ppyolov2_r50vd_365e.onnx -f exps/ppyolo/ppyolov2_r50vd_365e.py -c ppyolov2_r50vd_365e.pth


python tools/export_onnx.py --output-name ppyolo_r18vd.onnx -f exps/ppyolo/ppyolo_r18vd.py -c ppyolo_r18vd.pth



ONNX预测，命令改动为（用numpy对xywh进行解码，进行nms。）


python tools/onnx_inference.py -an PPYOLO -acn ppyolo_r18vd -m ppyolo_r18vd.onnx -i assets/dog.jpg -o ONNX_PPYOLO_R18VD_outputs -s 0.15 --input_shape 416,416 -cn class_names/coco_classes.txt


python tools/onnx_inference.py -an PPYOLO -acn ppyolo_r50vd_2x -m ppyolo_r50vd_2x.onnx -i assets/dog.jpg -o ONNX_PPYOLO_R50VD_outputs -s 0.15 --input_shape 608,608 -cn class_names/coco_classes.txt


python tools/onnx_inference.py -an PPYOLO -acn ppyolov2_r50vd_365e -m ppyolov2_r50vd_365e.onnx -i assets/dog.jpg -o ONNX_PPYOLOv2_R50VD_outputs -s 0.15 --input_shape 640,640 -cn class_names/coco_classes.txt





python tools/onnx_inference.py -an PPYOLO -acn ppyolo_r18vd -m ppyolo_r18vd.onnx -i D://GitHub/Pytorch-YOLO/images/test/000000052996.jpg -o ONNX_PPYOLO_R18VD_outputs -s 0.15 --input_shape 416,416 -cn class_names/coco_classes.txt






用onnx模型进行验证，

python tools/onnx_eval.py -an PPYOLO -acn ppyolo_r18vd -m ppyolo_r18vd.onnx -i ../COCO/val2017 -a ../COCO/annotations/instances_val2017.json -s 0.01 --input_shape 416,416 --eval_type eval





----------------------- 导出为TensorRT -----------------------
python tools_trt/export_trt.py -f exps/ppyolo/ppyolo_r18vd.py -c ppyolo_r18vd.pth --conf 0.15 --tsize 416





