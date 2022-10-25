import os
import cv2
import numpy as np
import random
import imgaug.augmenters as iaa
random.seed(66)
#
# def is_save(white_mask, white_mask_visib, thr=0.2):
#     print((white_mask-white_mask_visib)/white_mask)
#     return np.abs(white_mask-white_mask_visib)/white_mask < thr
#
# path_base = '/home/gzr/下载/chorme_download/lm_test_bop19/test'
# scene_name = '000012'
#
# masks_path = os.path.join(path_base, scene_name, 'mask')
# masks_visib_path = os.path.join(path_base, scene_name, 'mask_visib')
#
# files = os.listdir(masks_path)
# np.random.shuffle(files)
#
# num_test = 50
# w, h = 640, 480
#
# cnt=0
# for file in files:
#     path_mask = os.path.join(masks_path, file)
#     path_mask_visib = os.path.join(masks_visib_path, file)
#
#     mask = cv2.imread(path_mask, 0)
#     mask_visib = cv2.imread(path_mask_visib, 0)
#
#     white_mask = np.sum(mask/255.0)
#     white_mask_visib = np.sum(mask_visib/255.0)
#
#     if is_save(white_mask, white_mask_visib, thr=0.2):
#         cnt+=1
# print(cnt/len(files))

# path = '/home/gzr/Data/hole_data/hole_dataset/1_hole/images/validation/31.png'
# image0 = cv2.imread(path,0)
# image = cv2.Canny(image0, 5, 10)
# cv2.imshow('1',image)
#
# image = cv2.Canny(image0, 4, 8)
# cv2.imshow('2', image)
# cv2.waitKey(0)