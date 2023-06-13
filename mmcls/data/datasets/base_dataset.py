
import os
from loguru import logger

import cv2
import torch
import numpy as np


class BaseClsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        anno,
        img_size=(256, 256),
        type='Train',
    ):
        assert type in ['Train', 'Val']
        self.data_dir = data_dir
        self.img_size = img_size
        self.type = type
        anno_ = os.path.join(self.data_dir, anno)
        self.annotations = []
        with open(anno_, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                path_and_cid = line.split(' ')
                path = path_and_cid[0]
                cid = path_and_cid[1]
                sample = (path, int(cid))
                self.annotations.append(sample)
        self.num_record = len(self.annotations)
        logger.info("%s Image num = %d"%(anno, self.num_record))

    def __len__(self):
        return self.num_record

    def __getitem__(self, index):
        path, cid = self.annotations[index]
        img_file = os.path.join(self.data_dir, path)
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

        return padded_img, np.array([cid])

