

pip install --upgrade pip
pip install -U pip setuptools
pip install pyautogui -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pypiwin32 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install easyocr -i https://pypi.tuna.tsinghua.edu.cn/simple






----------------------- 迁移学习，带上-c（--ckpt）参数读取预训练模型。 -----------------------
ppyoloe_l迁移学习（冻结了骨干网络）:（可以加--fp16， -eb表示验证时的批大小）
python tools/train.py -f exps/ppyoloe/ppyoloe_crn_l_bh3letu.py -d 1 -b 8 -eb 2 -c ppyoloe_crn_l_300e_coco.pth --fp16

python tools/eval.py -f exps/ppyoloe/ppyoloe_crn_l_bh3letu.py -d 1 -b 4 -c PPYOLOE_outputs/ppyoloe_crn_l_bh3letu/60.pth --conf 0.01 --tsize 640

python tools/eval.py -f exps/ppyoloe/ppyoloe_crn_l_bh3letu.py -d 1 -b 4 -c PPYOLOE_outputs/ppyoloe_crn_l_bh3letu/60.pth --conf 0.01 --tsize 416

python tools/demo.py image -f exps/ppyoloe/ppyoloe_crn_l_bh3letu.py -c PPYOLOE_outputs/ppyoloe_crn_l_bh3letu/60.pth --path ../bh3_letu_dataset/images --conf 0.15 --tsize 640 --save_result --device gpu

python tools/demo.py image -f exps/ppyoloe/ppyoloe_crn_l_bh3letu.py -c PPYOLOE_outputs/ppyoloe_crn_l_bh3letu/60.pth --path ../bh3_letu_dataset/images --conf 0.15 --tsize 416 --save_result --device gpu


启动脚本刷乐土（游戏分辨率调整为1280x720）：
python tools_game/bh3_letu.py -f exps/ppyoloe/ppyoloe_crn_l_bh3letu.py -c PPYOLOE_outputs/ppyoloe_crn_l_bh3letu/60.pth --path ../bh3_letu_dataset/images --conf 0.15 --tsize 416 --save_result --device gpu



----------------------- 恢复训练（加上参数--resume） -----------------------
python tools/train.py -f exps/ppyoloe/ppyoloe_crn_l_bh3letu.py -d 1 -b 8 -eb 2 -c PPYOLOE_outputs/ppyoloe_crn_l_bh3letu/60.pth --resume





----------------------- 评估 -----------------------






