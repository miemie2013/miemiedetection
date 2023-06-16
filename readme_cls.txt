

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


----------------------- 转换权重 -----------------------
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_75_pretrained.pdparams
wget https://paddledet.bj.bcebos.com/models/pretrained/LCNet_x1_5_pretrained.pdparams
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x2_0_pretrained.pdparams

python tools/convert_weights.py -f exps/picodet/picodet_l_640_coco_lcnet.py -c PPLCNet_x2_0_pretrained.pdparams -oc PPLCNet_x2_0_pretrained.pth -nc 80 --only_backbone True



----------------------- 预测 -----------------------
python tools/demo.py image -f exps/picodet/picodet_s_416_coco_lcnet.py -c picodet_s_416_coco_lcnet.pth --path assets/000000000019.jpg --conf 0.15 --tsize 416 --save_result --device gpu






----------------------- 训练 -----------------------
python tools/train_cls.py -f exps/cls/mcnet_s_afhq.py -d 1 -b 128 -eb 128 -w 0 -ew 0


python tools/train_cls.py -f exps/cls/cspdarknet_s_afhq.py -d 1 -b 128 -eb 128 -w 0 -ew 0


python tools/train_cls.py -f exps/cls/lcnet_x2_0_afhq.py -d 1 -b 128 -eb 128 -w 0 -ew 0



export CUDA_VISIBLE_DEVICES=0,1
nohup python tools/train_cls.py -f exps/cls/lcnet_x2_0_afhq.py -d 2 -b 512 -eb 512 -w 4 -ew 4     > lcnet_x2_0_afhq.log 2>&1 &


export CUDA_VISIBLE_DEVICES=0,1
nohup python tools/train_cls.py -f exps/cls/cspdarknet_s_ImageNet1k.py -d 2 -b 512 -eb 512 -w 4 -ew 4     > cspdarknet_s_ImageNet1k.log 2>&1 &

export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup python tools/train_cls.py -f exps/cls/cspdarknet_s_ImageNet1k.py -d 4 -b 1024 -eb 1024 -w 4 -ew 4     > cspdarknet_s_ImageNet1k.log 2>&1 &



----------------------- 迁移学习，带上-c（--ckpt）参数读取预训练模型。 -----------------------
后台启动：
nohup xxx     > xxx.log 2>&1 &

python tools/train_cls.py -f exps/cls/lcnet_x2_0_afhq.py -d 1 -b 64 -eb 64 -w 0 -ew 0 -lrs 1.0 -c PPLCNet_x2_0_pretrained.pth


读取某个模型跑验证集（修改-c， 加上 --resume -oe）：
python tools/train_cls.py -f exps/cls/lcnet_x2_0_afhq.py -d 1 -b 64 -eb 64 -w 0 -ew 0 -lrs 1.0 -c 3.pth --resume -oe


----------------------- 恢复训练（加上参数--resume） -----------------------



----------------------- 评估 -----------------------
python tools/train_cls.py -f exps/cls/lcnet_x2_0_ImageNet1k.py -d 1 -b 64 -eb 64 -w 0 -ew 0 -lrs 1.0 -c 300.pth --resume -oe





