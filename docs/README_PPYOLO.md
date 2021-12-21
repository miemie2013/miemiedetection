简体中文 | [English](README_PPYOLO_en.md)

# miemiedetection

## 转换权重

```
python tools/convert_weights.py -f exps/ppyolo/ppyolo_r50vd_2x.py -c ppyolo.pdparams -oc ppyolo_2x.pth -nc 80

python tools/convert_weights.py -f exps/ppyolo/ppyolo_r18vd.py -c ppyolo_r18vd.pdparams -oc ppyolo_r18vd.pth -nc 80

```


## 预测

```
python tools/demo.py image -f exps/ppyolo/ppyolo_r50vd_2x.py -c ppyolo_2x.pth --path assets/dog.jpg --conf 0.15 --tsize 608 --save_result --device gpu

python tools/demo.py image -f exps/ppyolo/ppyolo_r18vd.py -c ppyolo_r18vd.pth --path assets/dog.jpg --conf 0.15 --tsize 416 --save_result --device gpu

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



## 预测

```

```


