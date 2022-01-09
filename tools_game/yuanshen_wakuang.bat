d:
cd Github/miemiedetection/tools_game
python yuanshen_wakuang.py image -f ../exps/yolox/yolox_m.py -c ../YOLOX_outputs/yolox_m/166.pth --conf 0.25 --nms 0.45 --tsize 640 --save_result --device cpu
start