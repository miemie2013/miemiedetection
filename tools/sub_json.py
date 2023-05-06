import argparse
import os
import json
import copy

# add python path of this repo to sys.path
import sys
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)


def make_parser():
    parser = argparse.ArgumentParser("MieMieDetection json tools")
    parser.add_argument("-j", "--json_path", default=None, type=str, help="json_path")
    parser.add_argument("-i", "--image_name", default=None, type=str, help="image_name")
    parser.add_argument("-sj", "--save_json_path", default=None, type=str, help="save_json_path")
    return parser


def main(exp, args):
    print("Args: {}".format(args))
    target_json = None
    with open(args.json_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            json_file = json.loads(line)
            target_json = copy.deepcopy(json_file)
            images = json_file['images']
            annotations = json_file['annotations']
            new_images = []
            ids = []
            new_annotations = []
            for img in images:
                fn = img['file_name']
                if fn == args.image_name:
                    new_images.append(img)
                    ids.append(img['id'])
                    break
            for anno in annotations:
                image_id = anno['image_id']
                if image_id in ids:
                    new_annotations.append(anno)
            target_json['images'] = new_images
            target_json['annotations'] = new_annotations
    with open(args.save_json_path, 'w') as f2:
        json.dump(target_json, f2)
    print("Done.")

'''
将json注解文件减少为只有1张图片。方便测试对比输出。
'000000403385.jpg'有2个注解：
[[411.1  237.7  504.11 480.    61.  ], [  9.21 313.62 151.92 391.88  71.  ]]
分别是 马桶(id==61) 和 洗手池(id==71)

python tools/sub_json.py -j ../COCO/annotations/instances_val2017.json -i 000000403385.jpg -sj instances_val2017_000000403385.json

python tools/sub_json.py -j ../VOCdevkit/VOC2012/annotations2/voc2012_val.json -i 2008_000073.jpg -sj voc2012_val_2008_000073.json


'''
if __name__ == "__main__":
    args = make_parser().parse_args()
    # 判断是否是调试状态
    isDebug = True if sys.gettrace() else False
    if isDebug:
        print('Debug Mode.')
        args.json_path = '../' + args.json_path
    main(None, args)
