
# batch process
import os.path
import random

import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch.nn as nn
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import cv2
import os
from tqdm import tqdm
import math
from hole_dataset.deal_data import get_range_random,get_external_contours_points,\
    get_external_contours_points_sample,mkdir
np.random.seed(66)
random.seed(66)
# data_dir = "/home/gzr/Data/linemod_2d/DATA/steel"
save_dir = '/home/gzr/Data/generative_steel/steel_dataset_now'



def get_external_contours(image):
    """
    :param image: (H,W)
    :return: (N,2)
    """
    assert (len(image.shape)==2)
    ret, binary = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    return contours

def pad_middle(inp, pad_size=[480,640],random_pad=False):
    assert isinstance(pad_size[0], int) and pad_size[0] >= inp.shape[0] and pad_size[1] >= inp.shape[1]
    if random_pad:
        padded = np.zeros((pad_size[0], pad_size[1]), dtype=inp.dtype)
        pad_0 = np.random.randint(0, pad_size[0] - inp.shape[0]+1)
        pad_1 = np.random.randint(0, pad_size[1] - inp.shape[1]+1)
        padded[pad_0:inp.shape[0] + pad_0, pad_1:inp.shape[1] + pad_1] = inp
    else:
        padded = np.zeros((pad_size[0], pad_size[1]), dtype=inp.dtype)
        pad_0 = (pad_size[0] - inp.shape[0])//2
        pad_1 = (pad_size[1] - inp.shape[1])//2
        padded[pad_0:inp.shape[0]+pad_0, pad_1:inp.shape[1]+pad_1] = inp

    return padded,[pad_0,pad_1]

def process_template(mask,img,cnt):
    W,H = 640, 480
    NUM_EACH = 240
    print(cnt // NUM_EACH, ": ", cnt % NUM_EACH)
    image_save_dir =os.path.join(save_dir,str(cnt//NUM_EACH),'images/test')
    cnt = cnt%NUM_EACH
    point_save_dir = image_save_dir.replace('images', 'points')
    trans_save_dir = image_save_dir.replace('images', 'trans')
    mkdir(image_save_dir)
    mkdir(point_save_dir)
    mkdir(trans_save_dir)

    img = cv2.resize(img, dsize=(W,H ))
    mask = cv2.resize(mask, dsize=(W,H ))
    w, h = mask.shape[1], mask.shape[0]

    # template
    points = get_external_contours_points(mask)
    center = np.mean(points, axis=0)
    M = cv2.getRotationMatrix2D(center, angle=0, scale=1)
    raw_add = np.array([0, 0, 1])
    M = np.r_[M, [raw_add]]
    translate = [w / 2 - center[0], h / 2 - center[1]]
    M[0, 2] += translate[0]
    M[1, 2] += translate[1]
    M_trans = np.linalg.inv(M)
    template = cv2.warpAffine(src=mask, M=M[0:2], dsize=(w, h))

    center = (W//2,H//2 )
    random_scale = get_range_random(min_= 0.8, max_= 1.2)
    random_angel = get_range_random(min_= -15, max_= 15)
    M = cv2.getRotationMatrix2D(center, angle=random_angel, scale=random_scale)
    raw_add = np.array([0, 0, 1])
    M = np.r_[M, [raw_add]]
    translate = [w / 2 - center[0], h / 2 - center[1]]
    M[0, 2] += translate[0]
    M[1, 2] += translate[1]
    img = cv2.warpAffine(src=img, M=M[0:2], dsize=(w, h))

    cv2.imwrite(os.path.join(image_save_dir, str(cnt) + '_template.png'), template)
    np.save(os.path.join(trans_save_dir, str(cnt) + '_trans.npy'), np.matmul(M,M_trans))
    template_points = get_external_contours_points_sample(template, 20)
    np.save(os.path.join(point_save_dir, str(cnt) + '_template.npy'), template_points)
    warp_image_Homo(os.path.join(image_save_dir, str(cnt) + '.png'), img, H=img.shape[0], W=img.shape[1])




img_dir = '/home/gzr/Data/generative_steel/'
img_path = os.path.join(img_dir, '2400*640_image_10000')
mask_path = os.path.join(img_dir, '2400*640_mask_10000')
# img_save = os.path.join(img_dir, 'process')
cnt = 0
NUM_DATA = 10000 #10000
arr = np.array(range(0,NUM_DATA,1))
np.random.shuffle(arr)
arr = arr[0:2000]
for i in arr:
    str_id = '%05d' % i
    img = cv2.imread(os.path.join(img_path, str_id+'.jpg'), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(os.path.join(mask_path, str_id + '.png'), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, dsize=(2400,640))
    mask = cv2.resize(mask, dsize=(2400,640))

    H, W = mask.shape[0], mask.shape[1]

    contours = get_external_contours(mask)
    for countur in contours:
        if random.randint(0,10) > 3:
            continue
        # for countur in contours:
        countur_x = countur.reshape(-1, 2)[:, 0]
        countur_y = countur.reshape(-1, 2)[:, 1]
        max_x = countur_x[np.argmax(countur_x)] + 10  #
        max_y = countur_y[np.argmax(countur_y)] + 10
        min_x = countur_x[np.argmin(countur_x)] - 10
        min_y = countur_y[np.argmin(countur_y)] - 10
        min_x = max(min_x, 0)
        min_y = max(min_y, 0)
        max_x = min(max_x, W)
        max_y = min(max_y, H)
        cropped = mask[min_y:max_y, min_x:max_x]
        h, w = cropped.shape[0],cropped.shape[1]
        # rd1,rd2 = random.randint(0,10),random.randint(0,10)
        min_x_new = max(min_x-random.randint(50,100), 0)
        min_y_new = max(min_y-random.randint(50,100), 0)
        max_x_new = min(max_x+random.randint(50,100), W)
        max_y_new = min(max_y+random.randint(50,100), H)
        cropped_img = img[min_y_new:max_y_new, min_x_new:max_x_new]
        cropped_mask_raw = mask[min_y_new:max_y_new, min_x_new:max_x_new]
        cropped_mask = np.zeros_like(cropped_img)

        cropped_mask[min_y-min_y_new:min_y-min_y_new+h,min_x-min_x_new:min_x-min_x_new+w] = cropped
        num_raw = np.mean(cropped_mask_raw)
        num_new = np.mean(cropped_mask)
        if num_raw > 1.8* num_new:  # out the multi-target data
            continue

        assert cropped_img.shape[0] == cropped_mask.shape[0] and cropped_img.shape[1] == cropped_mask.shape[1]
        process_template(cropped_mask, cropped_img, cnt)
        cnt = cnt+1








