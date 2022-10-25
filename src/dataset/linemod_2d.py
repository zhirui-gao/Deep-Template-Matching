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
from PIL import Image
import torchvision.transforms as transforms
from src.utils.dataset import read_scannet_gray
from src.utils.dataset import read_megadepth_gray,pad_bottom_right

class SyntheticDataset(utils.data.Dataset):
    def __init__(self,
                 root_dir,
                 txt_path=None, # image id to train/test
                 mode='train',
                 img_resize=256,
                 augment_fn=None,
                 **kwargs):
        super().__init__()

        self.root_dir = root_dir
        self.mode = mode
        if mode=='train':
            self.mode = 'training'
        elif mode == 'val':
            self.mode = 'validation'

        self.img_resize = img_resize

        fold_num = len(os.listdir(osp.join(root_dir,'trans',self.mode)))//2
        self.data_names = [i for i in range(0, fold_num)]
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])

        self.augment_fn = augment_fn if mode == 'train' else None

    def __len__(self):
        return len(self.data_names)



    def __getitem__(self, idx):
        # TODO: Support augmentation
        img_name = self.data_names[idx]

        img_name0 = osp.join(self.root_dir,'images',self.mode, str(img_name)+'_template.png')
        img_name1 = osp.join(self.root_dir,'images',self.mode, str(img_name)+'_homo.png') # _homo

        trans = np.load(osp.join(self.root_dir,'trans',self.mode, str(img_name)+'_trans_homo.npy')) # _homo
        points_template = np.load(osp.join(self.root_dir, 'points', self.mode, str(img_name) + '_template.npy'))


        image0 = cv2.imread(img_name0, cv2.IMREAD_GRAYSCALE) #tamplate
        image1 = cv2.imread(img_name1, cv2.IMREAD_GRAYSCALE)


        image1_rgb = cv2.imread(img_name1)
        image1_rgb = cv2.cvtColor(image1_rgb, cv2.COLOR_BGR2RGB)


        if image0.shape[0]>image1.shape[0] or image0.shape[1]>image1.shape[1]:
            image0 = image0[0:min(image0.shape[0],image1.shape[0]), 0:min(image0.shape[1],image1.shape[1])]
        assert(image0.shape[0]<=image1.shape[0] and image0.shape[1]<=image1.shape[1])
        data = {
            'image0': image0,  # ( h, w)
            'image1': image1,
            # 'image0_raw': image0_raw,
            # 'image1_raw': image1_raw,
            'image1_rgb': image1_rgb, # [3,h,w]
            'pair_id': idx,
            'dataset_name': 'synthetic',
            'bias': [0,0],
            'trans': trans,
            'points_template': points_template,
            # 'points_homo': points_homo,
            'pair_names': (img_name0,
                           img_name1)
        }
        return data


