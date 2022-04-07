from os import path as osp
from typing import Dict
from unicodedata import name
import os
import subprocess
import numpy as np
import torch
import torch.utils as utils
from numpy.linalg import inv
import cv2
from src.utils.dataset import read_scannet_gray
from src.utils.dataset import read_megadepth_gray,pad_bottom_right
# class Linemod2dDataset(utils.data.Dataset):
#     def __init__(self,
#                  root_dir,
#                  txt_path=None, # image id to train/test
#                  mode='train',
#                  augment_fn=None,
#                  **kwargs):
#         super().__init__()
#         self.root_dir = root_dir
#         self.mode = mode
#         if txt_path:
#             txt_path = os.path.join(txt_path, 'img_list.txt')
#         self.txt_path = txt_path
#         # prepare data_names
#         if txt_path:
#             self.data_names = np.loadtxt(txt_path, dtype=np.str_)
#         else:
#             # TODO: read all files in the root_dir
#             pass
#
#
#
#         self.augment_fn = augment_fn if mode == 'train' else None
#
#     def __len__(self):
#         return len(self.data_names)
#
#     def __getitem__(self, idx):
#         # TODO: Support augmentation
#         img_name = self.data_names[idx]
#         img_name1 = osp.join(self.root_dir, 'img', img_name)
#         img_name0 = osp.join(self.root_dir, 'template', img_name)
#         print(img_name0)
#         image0 = read_scannet_gray(img_name0, resize=(640, 480), augment_fn=None)
#         #    augment_fn=np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
#         image1 = read_scannet_gray(img_name1, resize=(640, 480), augment_fn=None)
#         #    augment_fn=np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
#
#         data = {
#             'image0': image0,  # (1, h, w)
#             'image1': image1,
#             'pair_id': idx,
#         }
#         return data

class Linemod2dDataset(utils.data.Dataset):
    def __init__(self,
                 root_dir,
                 txt_path=None, # image id to train/test
                 mode='train',
                 img_resize=512,
                 augment_fn=None,
                 **kwargs):
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.img_resize = img_resize
        if txt_path:
            txt_path = os.path.join(txt_path, 'img_list.txt')
        self.txt_path = txt_path
        # prepare data_names
        if txt_path:
            self.data_names = np.loadtxt(txt_path, dtype=np.str_)
        else:
            # TODO: read all files in the root_dir
            pass



        self.augment_fn = augment_fn if mode == 'train' else None

    def __len__(self):
        return len(self.data_names)

    def __getitem__(self, idx):
        # TODO: Support augmentation
        img_name = self.data_names[idx]
        # print(self.root_dir)
        img_name0 = osp.join(self.root_dir, img_name, 'template.jpg')
        img_name1 = osp.join(self.root_dir, img_name, 'localObjImg.jpg')
        bias = np.loadtxt(os.path.join(self.root_dir, img_name, 'bias.txt'))

        image0 = cv2.imread(img_name0, cv2.IMREAD_GRAYSCALE)


        image1, mask1, scale1 = read_megadepth_gray(
            img_name1, self.img_resize, None, True, None)

        image0 = cv2.resize(image0, dsize=(0, 0), fx=1 / float(scale1[0]), fy=1 / float(scale1[1]))

        image0, mask0 = pad_bottom_right(image0, self.img_resize, ret_mask=True)
        image0 = torch.from_numpy(image0).float()[None] / 255  # (h, w) -> (1, h, w) and normalized
        data = {
            'image0': image0,  # (1, h, w)
            'image1': image1,
            'pair_id': idx,
            'dataset_name': 'linemod_2d',
            'scale': scale1,
            'bias': bias,
            'pair_names': (img_name0,
                           img_name1)
        }
        if mask1 is not None:  # img_padding is True
            data.update({'mask1': mask1})
        return data