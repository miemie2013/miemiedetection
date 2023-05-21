#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
from loguru import logger

import cv2
import copy
import json
import torch
import numpy as np
from pycocotools.coco import COCO

from .. import RandomShapeSingle, YOLOXResizeImage
from ..dataloading import get_yolox_datadir
from .datasets_wrapper import Dataset

def xyxy2cxcywh_(bboxes):
    bboxes_ = np.copy(bboxes)
    bboxes_[2] = bboxes_[2] - bboxes_[0]
    bboxes_[3] = bboxes_[3] - bboxes_[1]
    bboxes_[0] = bboxes_[0] + bboxes_[2] * 0.5
    bboxes_[1] = bboxes_[1] + bboxes_[3] * 0.5
    return bboxes_


class COCODataset(Dataset):
    """
    COCO dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        json_file="instances_train2017.json",
        ann_folder="annotations",
        name="train2017",
        img_size=(416, 416),
        preproc=None,
        cache=False,
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            ann_folder (str): COCO annotations folder name (e.g. 'annotations')
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__(img_size)
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "COCO")
        self.data_dir = data_dir
        self.json_file = json_file
        self.ann_folder = ann_folder

        self.coco = COCO(os.path.join(self.data_dir, self.ann_folder, self.json_file))
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in cats])
        self.imgs = None
        self.name = name
        self.img_size = img_size
        self.preproc = preproc
        self.annotations = self._load_coco_annotations()
        if cache:
            self._cache_images()

    def __len__(self):
        return len(self.ids)

    def __del__(self):
        del self.imgs

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def _cache_images(self):
        logger.warning(
            "\n********************************************************************************\n"
            "You are using cached images in RAM to accelerate training.\n"
            "This requires large system RAM.\n"
            "Make sure you have 200G+ RAM and 136G available disk space for training COCO.\n"
            "********************************************************************************\n"
        )
        max_h = self.img_size[0]
        max_w = self.img_size[1]
        cache_file = self.data_dir + "/img_resized_cache_" + self.name + ".array"
        if not os.path.exists(cache_file):
            logger.info(
                "Caching images for the first time. This might take about 20 minutes for COCO"
            )
            self.imgs = np.memmap(
                cache_file,
                shape=(len(self.ids), max_h, max_w, 3),
                dtype=np.uint8,
                mode="w+",
            )
            from tqdm import tqdm
            from multiprocessing.pool import ThreadPool

            NUM_THREADs = min(8, os.cpu_count())
            loaded_images = ThreadPool(NUM_THREADs).imap(
                lambda x: self.load_resized_img(x),
                range(len(self.annotations)),
            )
            pbar = tqdm(enumerate(loaded_images), total=len(self.annotations))
            for k, out in pbar:
                self.imgs[k][: out.shape[0], : out.shape[1], :] = out.copy()
            self.imgs.flush()
            pbar.close()
        else:
            logger.warning(
                "You are using cached imgs! Make sure your dataset is not changed!!\n"
                "Everytime the self.input_size is changed in your exp file, you need to delete\n"
                "the cached data and re-generate them.\n"
            )

        logger.info("Loading cached imgs...")
        self.imgs = np.memmap(
            cache_file,
            shape=(len(self.ids), max_h, max_w, 3),
            dtype=np.uint8,
            mode="r+",
        )

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 5))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )

        return (res, img_info, resized_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        file_name = self.annotations[index][3]

        img_file = os.path.join(self.data_dir, self.name, file_name)

        img = cv2.imread(img_file)
        assert img is not None

        return img

    def pull_item(self, index):
        id_ = self.ids[index]

        res, img_info, resized_info, _ = self.annotations[index]
        if self.imgs is not None:
            pad_img = self.imgs[index]
            img = pad_img[: resized_info[0], : resized_info[1], :].copy()
        else:
            img = self.load_resized_img(index)

        return img, res.copy(), img_info, np.array([id_])

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w.
                h, w (int): original shape of the image
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, img_id


class SimpleCOCODataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir=None,
        json_file="instances_train2017.json",
        ann_folder="annotations",
        name="train2017",
        img_size=(416, 416),
        max_labels=120,
    ):
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "COCO")
        self.data_dir = data_dir
        self.json_file = json_file
        self.ann_folder = ann_folder

        self.coco = COCO(os.path.join(self.data_dir, self.ann_folder, self.json_file))
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in cats])
        self.name = name
        self.img_size = img_size
        self.max_labels = max_labels
        self.annotations = self._load_coco_annotations()
        self.init_bbox = [-1.0, 0.0, 0.0, 0.0, 0.0]

    def __len__(self):
        return len(self.ids)

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 5))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 1:5] = xyxy2cxcywh_(obj["clean_bbox"])
            # res[ix, 1:5] = obj["clean_bbox"]
            res[ix, 0] = cls

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, 1:5] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )

        return (res, img_info, resized_info, file_name)

    def pull_item(self, index):
        id_ = self.ids[index]
        # target.shape = [?, 5]   [cid cxcywh] format.
        target, img_info, resized_info, _ = self.annotations[index]

        file_name = self.annotations[index][3]
        img_file = os.path.join(self.data_dir, self.name, file_name)
        img = cv2.imread(img_file)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)

        if len(resized_img.shape) == 3:
            padded_img = np.ones((self.img_size[0], self.img_size[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(self.img_size, dtype=np.uint8) * 114
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
        padded_img = padded_img.transpose((2, 0, 1))
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)

        num_gt = len(target)
        # 批量mosaic时，填充的假gt 利用 cid = -1 过滤掉
        padded_labels = np.ones((self.max_labels, 5), dtype=np.float32) * self.init_bbox
        padded_labels[:num_gt, :] = target
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        return padded_img, padded_labels, img_info, np.array([id_])

    def __getitem__(self, index):
        img, target, img_info, img_id = self.pull_item(index)
        return img, target, img_info, img_id


# 数据清洗
def data_clean(coco, img_ids, catid2clsid, image_dir, type, xy_plus_1=False):
    records = []
    ct = 0
    for img_id in img_ids:
        img_anno = coco.loadImgs(img_id)[0]
        im_fname = img_anno['file_name']
        im_w = float(img_anno['width'])
        im_h = float(img_anno['height'])

        ins_anno_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)   # 读取这张图片所有标注anno的id
        instances = coco.loadAnns(ins_anno_ids)   # 这张图片所有标注anno。每个标注有'segmentation'、'bbox'、...

        bboxes = []
        anno_id = []    # 注解id
        for inst in instances:
            x, y, box_w, box_h = inst['bbox']   # 读取物体的包围框
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(im_w - 1, x1 + max(0, box_w - 1))
            y2 = min(im_h - 1, y1 + max(0, box_h - 1))
            if inst['area'] > 0 and x2 >= x1 and y2 >= y1:
                inst['clean_bbox'] = [x1, y1, x2, y2]   # inst增加一个键值对
                bboxes.append(inst)   # 这张图片的这个物体标注保留
                anno_id.append(inst['id'])
            else:
                logger.warning(
                    'Found an invalid bbox in annotations: im_id: {}, '
                    'area: {} x1: {}, y1: {}, x2: {}, y2: {}.'.format(
                        img_id, float(inst['area']), x1, y1, x2, y2))
        num_bbox = len(bboxes)   # 这张图片的物体数

        # 用于调试bug，获得所有没有gt的图片
        # if num_bbox > 0:
        #     continue

        # 左上角坐标+右下角坐标+类别id
        gt_bbox = np.zeros((num_bbox, 4), dtype=np.float32)
        gt_class = np.zeros((num_bbox, 1), dtype=np.int32)
        gt_score = np.ones((num_bbox, 1), dtype=np.float32)   # 得分的标注都是1
        is_crowd = np.zeros((num_bbox, 1), dtype=np.int32)
        difficult = np.zeros((num_bbox, 1), dtype=np.int32)
        gt_poly = [None] * num_bbox

        for i, box in enumerate(bboxes):
            catid = box['category_id']
            gt_class[i][0] = catid2clsid[catid]
            gt_bbox[i, :] = box['clean_bbox']
            if xy_plus_1:
                gt_bbox[i, 2] += 1.
                gt_bbox[i, 3] += 1.
            is_crowd[i][0] = box['iscrowd']
            if 'segmentation' in box:
                gt_poly[i] = box['segmentation']

        im_fname = os.path.join(image_dir,
                                im_fname) if image_dir else im_fname
        coco_rec = {
            'im_file': im_fname,
            'im_id': np.array([img_id]),
            'h': im_h,
            'w': im_w,
            'is_crowd': is_crowd,
            'gt_class': gt_class,
            'anno_id': anno_id,
            'gt_bbox': gt_bbox,
            'gt_score': gt_score,
            'gt_poly': gt_poly,
        }

        # logger.debug('Load file: {}, im_id: {}, h: {}, w: {}.'.format(im_fname, img_id, im_h, im_w))
        records.append(coco_rec)   # 注解文件。
        ct += 1
    logger.info('{} samples in {} set.'.format(ct, type))
    return records


def get_class_msg(anno_path):
    _catid2clsid = {}
    _clsid2catid = {}
    _clsid2cname = {}
    with open(anno_path, 'r', encoding='utf-8') as f2:
        dataset_text = ''
        for line in f2:
            line = line.strip()
            dataset_text += line
        eval_dataset = json.loads(dataset_text)
        categories = eval_dataset['categories']
        for clsid, cate_dic in enumerate(categories):
            catid = cate_dic['id']
            cname = cate_dic['name']
            _catid2clsid[catid] = clsid
            _clsid2catid[clsid] = catid
            _clsid2cname[clsid] = cname
    class_names = []
    num_classes = len(_clsid2cname.keys())
    for clsid in range(num_classes):
        class_names.append(_clsid2cname[clsid])
    return _catid2clsid, _clsid2catid, _clsid2cname, class_names


class PPYOLO_COCOEvalDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, json_file, ann_folder, name, cfg, transforms):
        self.data_dir = data_dir
        self.json_file = json_file
        self.ann_folder = ann_folder
        self.name = name

        # 验证集
        val_path = os.path.join(self.data_dir, self.ann_folder, self.json_file)
        val_pre_path = os.path.join(self.data_dir, self.name)

        # 种类id
        _catid2clsid, _clsid2catid, _clsid2cname, class_names = get_class_msg(val_path)

        val_dataset = COCO(val_path)
        val_img_ids = val_dataset.getImgIds()

        keep_img_ids = []  # 只跑有gt的图片，跟随PaddleDetection
        for img_id in val_img_ids:
            ins_anno_ids = val_dataset.getAnnIds(imgIds=img_id, iscrowd=False)  # 读取这张图片所有标注anno的id
            if len(ins_anno_ids) == 0:
                continue
            keep_img_ids.append(img_id)
        val_img_ids = keep_img_ids

        val_records = data_clean(val_dataset, val_img_ids, _catid2clsid, val_pre_path, 'val')

        self.coco = val_dataset
        self.records = val_records
        self.context = cfg.context
        self.transforms = transforms
        self.catid2clsid = _catid2clsid
        self.clsid2catid = _clsid2catid
        self.num_record = len(val_records)
        self.indexes = [i for i in range(self.num_record)]

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        img_idx = self.indexes[idx]
        sample = copy.deepcopy(self.records[img_idx])

        # transforms
        for transform in self.transforms:
            sample = transform(sample, self.context)

        # 取出感兴趣的项
        pimage = sample['image']
        im_size = np.array([sample['h'], sample['w']]).astype(np.float32)
        id = sample['im_id']
        return pimage, im_size, id


class SOLO_COCOEvalDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, json_file, ann_folder, name, cfg, transforms):
        self.data_dir = data_dir
        self.json_file = json_file
        self.ann_folder = ann_folder
        self.name = name

        # 验证集
        val_path = os.path.join(self.data_dir, self.ann_folder, self.json_file)
        val_pre_path = os.path.join(self.data_dir, self.name)

        # 种类id
        _catid2clsid, _clsid2catid, _clsid2cname, class_names = get_class_msg(val_path)

        val_dataset = COCO(val_path)
        val_img_ids = val_dataset.getImgIds()

        keep_img_ids = []  # 只跑有gt的图片，跟随PaddleDetection
        for img_id in val_img_ids:
            ins_anno_ids = val_dataset.getAnnIds(imgIds=img_id, iscrowd=False)  # 读取这张图片所有标注anno的id
            if len(ins_anno_ids) == 0:
                continue
            keep_img_ids.append(img_id)
        val_img_ids = keep_img_ids

        val_records = data_clean(val_dataset, val_img_ids, _catid2clsid, val_pre_path, 'val')

        self.coco = val_dataset
        self.records = val_records
        self.context = cfg.context
        self.transforms = transforms
        self.catid2clsid = _catid2clsid
        self.clsid2catid = _clsid2catid
        self.num_record = len(val_records)
        self.indexes = [i for i in range(self.num_record)]

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        img_idx = self.indexes[idx]
        sample = copy.deepcopy(self.records[img_idx])

        # transforms
        for transform in self.transforms:
            sample = transform(sample, self.context)


        # 方案1，DataLoader里使用collate_fn参数，慢
        # 取出感兴趣的项
        pimage = sample['image']
        im_info = sample['im_info']
        im_id = sample['im_id']
        h = sample['h']
        w = sample['w']
        return pimage, im_info, im_id, h, w

        # 方案2，用SOLOv2Pad
        # 取出感兴趣的项
        # pimage = sample['image']
        # im_size = np.array([sample['im_info'][0], sample['im_info'][1]]).astype(np.int32)
        # ori_shape = np.array([sample['h'], sample['w']]).astype(np.int32)
        # id = sample['im_id']
        # return pimage, im_size, ori_shape, id


class PPYOLOE_COCOEvalDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, json_file, ann_folder, name, cfg, transforms):
        self.data_dir = data_dir
        self.json_file = json_file
        self.ann_folder = ann_folder
        self.name = name

        # 验证集
        val_path = os.path.join(self.data_dir, self.ann_folder, self.json_file)
        val_pre_path = os.path.join(self.data_dir, self.name)

        # 种类id
        _catid2clsid, _clsid2catid, _clsid2cname, class_names = get_class_msg(val_path)

        val_dataset = COCO(val_path)
        val_img_ids = val_dataset.getImgIds()

        keep_img_ids = []  # 只跑有gt的图片，跟随PaddleDetection
        for img_id in val_img_ids:
            ins_anno_ids = val_dataset.getAnnIds(imgIds=img_id, iscrowd=False)  # 读取这张图片所有标注anno的id
            if len(ins_anno_ids) == 0:
                continue
            keep_img_ids.append(img_id)
        val_img_ids = keep_img_ids

        val_records = data_clean(val_dataset, val_img_ids, _catid2clsid, val_pre_path, 'val')

        self.coco = val_dataset
        self.records = val_records
        self.context = cfg.context
        self.transforms = transforms
        self.catid2clsid = _catid2clsid
        self.clsid2catid = _clsid2catid
        self.num_record = len(val_records)
        self.indexes = [i for i in range(self.num_record)]

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        img_idx = self.indexes[idx]
        sample = copy.deepcopy(self.records[img_idx])

        # transforms
        for transform in self.transforms:
            sample = transform(sample, self.context)

        # 取出感兴趣的项
        pimage = sample['image']
        scale_factor = np.array([sample['scale_factor'][1], sample['scale_factor'][0]]).astype(np.float32)
        id = sample['im_id']
        return pimage, scale_factor, id


class FCOS_COCOEvalDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, json_file, ann_folder, name, cfg, sample_transforms):
        self.data_dir = data_dir
        self.json_file = json_file
        self.ann_folder = ann_folder
        self.name = name

        # 验证集
        val_path = os.path.join(self.data_dir, self.ann_folder, self.json_file)
        val_pre_path = os.path.join(self.data_dir, self.name)

        # 种类id
        _catid2clsid, _clsid2catid, _clsid2cname, class_names = get_class_msg(val_path)

        val_dataset = COCO(val_path)
        val_img_ids = val_dataset.getImgIds()

        keep_img_ids = []  # 只跑有gt的图片，跟随PaddleDetection
        for img_id in val_img_ids:
            ins_anno_ids = val_dataset.getAnnIds(imgIds=img_id, iscrowd=False)  # 读取这张图片所有标注anno的id
            if len(ins_anno_ids) == 0:
                continue
            keep_img_ids.append(img_id)
        val_img_ids = keep_img_ids

        val_records = data_clean(val_dataset, val_img_ids, _catid2clsid, val_pre_path, 'val')

        self.coco = val_dataset
        self.records = val_records
        self.context = cfg.context
        self.sample_transforms = sample_transforms
        self.catid2clsid = _catid2clsid
        self.clsid2catid = _clsid2catid
        self.num_record = len(val_records)
        self.indexes = [i for i in range(self.num_record)]

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        img_idx = self.indexes[idx]
        sample = copy.deepcopy(self.records[img_idx])

        # sample_transforms
        for sample_transform in self.sample_transforms:
            sample = sample_transform(sample, self.context)

        # 取出感兴趣的项
        pimage = sample['image']
        im_info = sample['im_info']
        im_id = sample['im_id']
        return pimage, im_info, im_id


class PPYOLO_COCOTrainDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, json_file, ann_folder, name, max_epoch, num_gpus, cfg, sample_transforms, batch_size):
        self.data_dir = data_dir
        self.json_file = json_file
        self.ann_folder = ann_folder
        self.name = name
        self.max_epoch = max_epoch

        # 训练集
        train_path = os.path.join(self.data_dir, self.ann_folder, self.json_file)
        train_pre_path = os.path.join(self.data_dir, self.name)

        # 种类id
        _catid2clsid, _clsid2catid, _clsid2cname, class_names = get_class_msg(train_path)

        train_dataset = COCO(train_path)
        train_img_ids = train_dataset.getImgIds()
        train_records = data_clean(train_dataset, train_img_ids, _catid2clsid, train_pre_path, 'train')

        self.coco = train_dataset
        self.records = train_records
        self.context = cfg.context
        self.sample_transforms = sample_transforms
        self.catid2clsid = _catid2clsid
        self.clsid2catid = _clsid2catid
        self.num_record = len(train_records)
        self.with_mixup = cfg.decodeImage.get('with_mixup', False)
        self.with_cutmix = cfg.decodeImage.get('with_cutmix', False)
        self.with_mosaic = cfg.decodeImage.get('with_mosaic', False)
        self.batch_size = batch_size
        self.batch_gpu = batch_size // num_gpus


        # 一轮的步数。丢弃最后几个样本。
        self.train_steps = self.num_record // batch_size

        # mixup、cutmix、mosaic数据增强的轮数
        self.aug_epochs = cfg.aug_epochs

        # 多尺度训练
        self.sizes = cfg.randomShape['sizes']
        self.random_shapes = []
        self.random_shape_i = 0
        while len(self.random_shapes) < (self.num_record * (self.max_epoch + 1)):
            shape = np.random.choice(self.sizes)
            for _ in range(self.batch_gpu):
                self.random_shapes.append(shape)

        # 输出特征图数量
        self.n_layers = len(cfg.head['downsample'])
        self._epoch = 0


    def __len__(self):
        return self.num_record

    def __getitem__(self, idx):
        iter_id = self.random_shape_i // self.batch_size
        img_idx = idx
        random_shape = self.random_shapes[self.random_shape_i]
        self.random_shape_i += 1
        sample = copy.deepcopy(self.records[img_idx])
        sample["curr_iter"] = iter_id

        # 为mixup数据增强做准备
        if self.with_mixup and self._epoch <= self.aug_epochs:
            num = len(self.records)
            mix_idx = np.random.randint(0, num)
            while mix_idx == img_idx:   # 为了不选到自己
                mix_idx = np.random.randint(0, num)
            sample['mixup'] = copy.deepcopy(self.records[mix_idx])
            sample['mixup']["curr_iter"] = iter_id

        # 为cutmix数据增强做准备
        if self.with_cutmix and self._epoch <= self.aug_epochs:
            num = len(self.records)
            mix_idx = np.random.randint(0, num)
            while mix_idx == img_idx:   # 为了不选到自己
                mix_idx = np.random.randint(0, num)
            sample['cutmix'] = copy.deepcopy(self.records[mix_idx])
            sample['cutmix']["curr_iter"] = iter_id

        # 为mosaic数据增强做准备
        if self.with_mosaic and self._epoch <= self.aug_epochs:
            num = len(self.records)
            mix_idx = np.random.randint(0, num)
            while mix_idx == img_idx:   # 为了不选到自己
                mix_idx = np.random.randint(0, num)
            sample['mosaic1'] = copy.deepcopy(self.records[mix_idx])
            sample['mosaic1']["curr_iter"] = iter_id

            mix_idx2 = np.random.randint(0, num)
            while mix_idx2 in [img_idx, mix_idx]:   # 为了不重复
                mix_idx2 = np.random.randint(0, num)
            sample['mosaic2'] = copy.deepcopy(self.records[mix_idx2])
            sample['mosaic2']["curr_iter"] = iter_id

            mix_idx3 = np.random.randint(0, num)
            while mix_idx3 in [img_idx, mix_idx, mix_idx2]:   # 为了不重复
                mix_idx3 = np.random.randint(0, num)
            sample['mosaic3'] = copy.deepcopy(self.records[mix_idx3])
            sample['mosaic3']["curr_iter"] = iter_id

        # sample_transforms
        for sample_transform in self.sample_transforms:
            if isinstance(sample_transform, RandomShapeSingle):
                sample = sample_transform(random_shape, sample, self.context)
            else:
                sample = sample_transform(sample, self.context)

        # 取出感兴趣的项
        # pimage = sample['image']
        # im_info = sample['im_info']
        # im_id = sample['im_id']
        # h = sample['h']
        # w = sample['w']
        # is_crowd = sample['is_crowd']
        # gt_class = sample['gt_class']
        # gt_bbox = sample['gt_bbox']
        # gt_score = sample['gt_score']
        # curr_iter = sample['curr_iter']
        # return pimage, im_info, im_id, h, w, is_crowd, gt_class, gt_bbox, gt_score, curr_iter

        # 取出感兴趣的项
        image = sample['image'].astype(np.float32)
        gt_bbox = sample['gt_bbox'].astype(np.float32)
        target0 = sample['target0'].astype(np.float32)
        target1 = sample['target1'].astype(np.float32)
        im_id = sample['im_id'].astype(np.int32)
        if self.n_layers == 3:
            target2 = sample['target2'].astype(np.float32)
            return image, gt_bbox, target0, target1, target2, im_id
        return image, gt_bbox, target0, target1, im_id


class SOLO_COCOTrainDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, json_file, ann_folder, name, max_epoch, num_gpus, cfg, sample_transforms, batch_size):
        self.data_dir = data_dir
        self.json_file = json_file
        self.ann_folder = ann_folder
        self.name = name
        self.max_epoch = max_epoch

        # 训练集
        train_path = os.path.join(self.data_dir, self.ann_folder, self.json_file)
        train_pre_path = os.path.join(self.data_dir, self.name)

        # 种类id
        _catid2clsid, _clsid2catid, _clsid2cname, class_names = get_class_msg(train_path)

        train_dataset = COCO(train_path)
        train_img_ids = train_dataset.getImgIds()
        # PPYOLOE（最新的ppdet）右下角坐标要+1
        train_records = data_clean(train_dataset, train_img_ids, _catid2clsid, train_pre_path, 'train', xy_plus_1=True)

        self.coco = train_dataset
        self.records = train_records
        self.context = cfg.context
        self.sample_transforms = sample_transforms
        self.catid2clsid = _catid2clsid
        self.clsid2catid = _clsid2catid
        self.num_record = len(train_records)
        self.batch_size = batch_size
        self.batch_gpu = batch_size // num_gpus


        # 一轮的步数。丢弃最后几个样本。
        self.train_steps = self.num_record // batch_size

        # 输出特征图数量
        self.n_layers = len(cfg.head['num_grids'])
        self._epoch = 0


    def __len__(self):
        return self.num_record

    def __getitem__(self, idx):
        img_idx = idx
        sample = copy.deepcopy(self.records[img_idx])

        # sample_transforms
        for sample_transform in self.sample_transforms:
            sample = sample_transform(sample, self.context)

        # 取出感兴趣的项
        pimage = sample['image']
        im_shape = sample['im_shape']
        scale_factor = sample['scale_factor']
        im_id = sample['im_id']
        h = sample['h']
        w = sample['w']
        is_crowd = sample['is_crowd']
        gt_class = sample['gt_class']
        gt_bbox = sample['gt_bbox']
        gt_segm = sample['gt_segm']
        gt_poly = sample['gt_poly']
        gt_score = sample['gt_score']
        return pimage, im_shape, scale_factor, im_id, h, w, is_crowd, gt_class, gt_bbox, gt_segm, gt_poly, gt_score

        # 取出感兴趣的项
        # image = sample['image'].astype(np.float32)
        # gt_bbox = sample['gt_bbox'].astype(np.float32)
        # target0 = sample['target0'].astype(np.float32)
        # target1 = sample['target1'].astype(np.float32)
        # im_id = sample['im_id'].astype(np.int32)
        # return image, gt_bbox, target0, target1, im_id


class PPYOLOE_COCOTrainDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, json_file, ann_folder, name, max_epoch, num_gpus, cfg, sample_transforms, batch_size):
        self.data_dir = data_dir
        self.json_file = json_file
        self.ann_folder = ann_folder
        self.name = name
        self.max_epoch = max_epoch

        # 训练集
        train_path = os.path.join(self.data_dir, self.ann_folder, self.json_file)
        train_pre_path = os.path.join(self.data_dir, self.name)

        # 种类id
        _catid2clsid, _clsid2catid, _clsid2cname, class_names = get_class_msg(train_path)

        train_dataset = COCO(train_path)
        train_img_ids = train_dataset.getImgIds()
        # PPYOLOE（最新的ppdet）右下角坐标要+1
        train_records = data_clean(train_dataset, train_img_ids, _catid2clsid, train_pre_path, 'train', xy_plus_1=True)

        self.coco = train_dataset
        self.records = train_records
        self.context = cfg.context
        self.sample_transforms = sample_transforms
        self.catid2clsid = _catid2clsid
        self.clsid2catid = _clsid2catid
        self.num_record = len(train_records)
        self.with_mixup = cfg.decodeImage.get('with_mixup', False)
        self.with_cutmix = cfg.decodeImage.get('with_cutmix', False)
        self.with_mosaic = cfg.decodeImage.get('with_mosaic', False)
        self.batch_size = batch_size
        self.batch_gpu = batch_size // num_gpus


        # 一轮的步数。丢弃最后几个样本。
        self.train_steps = self.num_record // batch_size

        # mixup、cutmix、mosaic数据增强的轮数
        self.aug_epochs = -1

        # 多尺度训练
        self.sizes = cfg.randomShape['sizes']
        self.random_shapes = []
        self.random_shape_i = 0
        while len(self.random_shapes) < (self.num_record * (self.max_epoch + 1)):
            shape = np.random.choice(self.sizes)
            for _ in range(self.batch_gpu):
                self.random_shapes.append(shape)

        # 输出特征图数量
        fpn_strides = cfg.head.get('fpn_strides', None)
        if fpn_strides is None:
            fpn_strides = cfg.head.get('fpn_stride', None)
        self.n_layers = len(fpn_strides)
        self._epoch = 0


    def __len__(self):
        return self.num_record

    def __getitem__(self, idx):
        iter_id = self.random_shape_i // self.batch_size
        img_idx = idx
        random_shape = self.random_shapes[self.random_shape_i]
        self.random_shape_i += 1
        sample = copy.deepcopy(self.records[img_idx])
        sample["curr_iter"] = iter_id

        # 为mixup数据增强做准备
        if self.with_mixup and self._epoch <= self.aug_epochs:
            num = len(self.records)
            mix_idx = np.random.randint(0, num)
            while mix_idx == img_idx:   # 为了不选到自己
                mix_idx = np.random.randint(0, num)
            sample['mixup'] = copy.deepcopy(self.records[mix_idx])
            sample['mixup']["curr_iter"] = iter_id

        # 为cutmix数据增强做准备
        if self.with_cutmix and self._epoch <= self.aug_epochs:
            num = len(self.records)
            mix_idx = np.random.randint(0, num)
            while mix_idx == img_idx:   # 为了不选到自己
                mix_idx = np.random.randint(0, num)
            sample['cutmix'] = copy.deepcopy(self.records[mix_idx])
            sample['cutmix']["curr_iter"] = iter_id

        # 为mosaic数据增强做准备
        if self.with_mosaic and self._epoch <= self.aug_epochs:
            num = len(self.records)
            mix_idx = np.random.randint(0, num)
            while mix_idx == img_idx:   # 为了不选到自己
                mix_idx = np.random.randint(0, num)
            sample['mosaic1'] = copy.deepcopy(self.records[mix_idx])
            sample['mosaic1']["curr_iter"] = iter_id

            mix_idx2 = np.random.randint(0, num)
            while mix_idx2 in [img_idx, mix_idx]:   # 为了不重复
                mix_idx2 = np.random.randint(0, num)
            sample['mosaic2'] = copy.deepcopy(self.records[mix_idx2])
            sample['mosaic2']["curr_iter"] = iter_id

            mix_idx3 = np.random.randint(0, num)
            while mix_idx3 in [img_idx, mix_idx, mix_idx2]:   # 为了不重复
                mix_idx3 = np.random.randint(0, num)
            sample['mosaic3'] = copy.deepcopy(self.records[mix_idx3])
            sample['mosaic3']["curr_iter"] = iter_id

        # sample_transforms
        for sample_transform in self.sample_transforms:
            if isinstance(sample_transform, RandomShapeSingle):
                sample = sample_transform(random_shape, sample, self.context)
            else:
                sample = sample_transform(sample, self.context)

        # 取出感兴趣的项
        # pimage = sample['image']
        # im_info = sample['im_info']
        # im_id = sample['im_id']
        # h = sample['h']
        # w = sample['w']
        # is_crowd = sample['is_crowd']
        # gt_class = sample['gt_class']
        # gt_bbox = sample['gt_bbox']
        # gt_score = sample['gt_score']
        # curr_iter = sample['curr_iter']
        # return pimage, im_info, im_id, h, w, is_crowd, gt_class, gt_bbox, gt_score, curr_iter

        # 取出感兴趣的项
        image = sample['image'].astype(np.float32)
        gt_class = sample['gt_class'].astype(np.int32)
        gt_bbox = sample['gt_bbox'].astype(np.float32)
        pad_gt_mask = sample['pad_gt_mask'].astype(np.float32)
        im_id = sample['im_id'].astype(np.int32)
        return image, gt_class, gt_bbox, pad_gt_mask, im_id



class FCOS_COCOTrainDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, json_file, ann_folder, name, cfg, sample_transforms, batch_size):
        self.data_dir = data_dir
        self.json_file = json_file
        self.ann_folder = ann_folder
        self.name = name

        # 训练集
        train_path = os.path.join(self.data_dir, self.ann_folder, self.json_file)
        train_pre_path = os.path.join(self.data_dir, self.name)

        # 种类id
        _catid2clsid, _clsid2catid, _clsid2cname, class_names = get_class_msg(train_path)

        train_dataset = COCO(train_path)
        train_img_ids = train_dataset.getImgIds()
        train_records = data_clean(train_dataset, train_img_ids, _catid2clsid, train_pre_path, 'train')

        self.coco = train_dataset
        self.records = train_records
        self.context = cfg.context
        self.sample_transforms = sample_transforms
        self.catid2clsid = _catid2clsid
        self.clsid2catid = _clsid2catid
        self.num_record = len(train_records)
        self.with_mixup = cfg.decodeImage.get('with_mixup', False)
        self.with_cutmix = cfg.decodeImage.get('with_cutmix', False)
        self.with_mosaic = cfg.decodeImage.get('with_mosaic', False)
        self.batch_size = batch_size


        # 一轮的步数。丢弃最后几个样本。
        self.train_steps = self.num_record // batch_size

        # mixup、cutmix、mosaic数据增强的轮数
        self.aug_epochs = cfg.aug_epochs

        # 训练样本
        self.indexes_ori = [i for i in range(self.num_record)]
        self.indexes = copy.deepcopy(self.indexes_ori)
        # 每个epoch之前洗乱
        np.random.shuffle(self.indexes)
        self.indexes = self.indexes[:self.train_steps * self.batch_size]
        self._len = len(self.indexes)

        # 输出特征图数量
        self.n_layers = len(cfg.head['fpn_stride'])
        self._epoch = 0


    def __len__(self):
        return self._len

    def set_epoch(self, epoch_id):
        self._epoch = epoch_id

        self.indexes = copy.deepcopy(self.indexes_ori)
        # 每个epoch之前洗乱
        np.random.shuffle(self.indexes)
        self.indexes = self.indexes[:self._len]

    def __getitem__(self, idx):
        iter_id = idx // self.batch_size
        img_idx = self.indexes[idx]
        sample = copy.deepcopy(self.records[img_idx])
        sample["curr_iter"] = iter_id

        # 为mixup数据增强做准备
        if self.with_mixup and self._epoch <= self.aug_epochs:
            num = len(self.records)
            mix_idx = np.random.randint(0, num)
            while mix_idx == img_idx:   # 为了不选到自己
                mix_idx = np.random.randint(0, num)
            sample['mixup'] = copy.deepcopy(self.records[mix_idx])
            sample['mixup']["curr_iter"] = iter_id

        # 为cutmix数据增强做准备
        if self.with_cutmix and self._epoch <= self.aug_epochs:
            num = len(self.records)
            mix_idx = np.random.randint(0, num)
            while mix_idx == img_idx:   # 为了不选到自己
                mix_idx = np.random.randint(0, num)
            sample['cutmix'] = copy.deepcopy(self.records[mix_idx])
            sample['cutmix']["curr_iter"] = iter_id

        # 为mosaic数据增强做准备
        if self.with_mosaic and self._epoch <= self.aug_epochs:
            num = len(self.records)
            mix_idx = np.random.randint(0, num)
            while mix_idx == img_idx:   # 为了不选到自己
                mix_idx = np.random.randint(0, num)
            sample['mosaic1'] = copy.deepcopy(self.records[mix_idx])
            sample['mosaic1']["curr_iter"] = iter_id

            mix_idx2 = np.random.randint(0, num)
            while mix_idx2 in [img_idx, mix_idx]:   # 为了不重复
                mix_idx2 = np.random.randint(0, num)
            sample['mosaic2'] = copy.deepcopy(self.records[mix_idx2])
            sample['mosaic2']["curr_iter"] = iter_id

            mix_idx3 = np.random.randint(0, num)
            while mix_idx3 in [img_idx, mix_idx, mix_idx2]:   # 为了不重复
                mix_idx3 = np.random.randint(0, num)
            sample['mosaic3'] = copy.deepcopy(self.records[mix_idx3])
            sample['mosaic3']["curr_iter"] = iter_id

        # sample_transforms
        for sample_transform in self.sample_transforms:
            sample = sample_transform(sample, self.context)

        # 取出感兴趣的项
        pimage = sample['image']
        im_info = sample['im_info']
        im_id = sample['im_id']
        h = sample['h']
        w = sample['w']
        is_crowd = sample['is_crowd']
        gt_class = sample['gt_class']
        gt_bbox = sample['gt_bbox']
        gt_score = sample['gt_score']
        return pimage, im_info, im_id, h, w, is_crowd, gt_class, gt_bbox, gt_score


