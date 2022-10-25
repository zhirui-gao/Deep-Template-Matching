# batch process
import os.path
import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np
import torch.nn as nn
import torchvision
from src.dataset.linemod_2d import read_scannet_gray
from src.utils.plotting import make_matching_figure
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import cv2
import os
from tqdm import tqdm
import math
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
import shutil
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# def crop_image(image, padding=20):
#     # adopt padding scheme instead of min/max
#     """
#     crop the black block
#     :param image:(H,W)
#     :param padding:int
#     :return:image:(H_,W_) bias [x,y]
#     """
#     assert (len(image.shape)==2)
#     H, W = image.shape[0],image.shape[1]
#     ind = list(np.nonzero(image))
#     max_y,min_y = int(np.max(ind[0])),int(np.min(ind[0]))
#     max_x,min_x = int(np.max(ind[1])),int(np.min(ind[1]))
#     cropped = image[min_y:max_y+1, min_x:max_x+1]
#     cropped = np.pad(cropped, ((padding, padding), (padding, padding)),'constant')
#     bias = np.array([min_x-padding, min_y-padding])
#     return cropped, bias

def crop_image(image, padding=50):
    """
    crop the black block
    :param image:(H,W)
    :param padding:int
    :return:image:(H_,W_) bias [x,y]
    """
    assert (len(image.shape)==2)
    H, W = image.shape[0],image.shape[1]
    ind = list(np.nonzero(image))
    max_y,min_y = int(np.max(ind[0])),int(np.min(ind[0]))
    max_x,min_x = int(np.max(ind[1])),int(np.min(ind[1]))
    min_x = max(min_x-padding,0)
    min_y = max(min_y-padding,0)
    max_x = min((max_x+padding),W)
    max_y = min((max_y+padding),H)
    cropped = image[min_y:max_y, min_x:max_x]
    bias = np.array([min_x,min_y])
    return cropped, bias

def process_template(data_dir,image_id):
    image_dir = os.path.join(data_dir, image_id)
    image_path = os.path.join(image_dir, 'matchedMask.jpg')
    img_mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    img_mask ,bias = crop_image(img_mask)
    img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2RGB)
    np.savetxt(os.path.join(image_dir, 'bias.txt'), bias,fmt='%i')
    # plt.imshow(img_mask)
    # plt.show()
    cv2.imwrite(os.path.join(image_dir, 'template.jpg'),img_mask)

def get_innder_square(raw_data_shape, img_aug, edge_aug, a):
    assert a > 0
    if a == 90 or a == 180 or a == 270 or a == 360:
        return img_aug, edge_aug
    while a > 90:
        a = a - 90
    min_wh = min(raw_data_shape[0], raw_data_shape[1])
    radian = math.radians(a)
    # print(radian)
    r = min_wh / (2 * math.cos(radian) * (1 + math.tan(radian)))
    center_h = img_aug.shape[0] / 2
    center_w = img_aug.shape[1] / 2
    h_min = math.ceil(center_h - r)
    h_max = math.floor(center_h + r)
    w_min = math.ceil(center_w - r)
    w_max = math.floor(center_w + r)
    img_aug_innder = img_aug[h_min: h_max, w_min: w_max]
    edge_aug_innder = edge_aug[h_min: h_max, w_min: w_max]
    return img_aug_innder, edge_aug_innder

# a bug in get_innder
def get_innder(raw_data_shape, img_aug, edge_aug, a):
    assert a > 0
    if a == 90 or a == 180 or a == 270 or a == 360:
        return img_aug, edge_aug
    while a > 90:
        a = a - 90
    radian = math.radians(a)
    # print(radian)
    r_h = raw_data_shape[0] / (2 * math.cos(radian) * (1 + math.tan(radian)))
    r_w = raw_data_shape[1] / (2 * math.cos(radian) * (1 + math.tan(radian)))
    center_h = img_aug.shape[0] / 2
    center_w = img_aug.shape[1] / 2
    h_min = math.ceil(center_h - r_h)
    h_max = math.floor(center_h + r_h)
    w_min = math.ceil(center_w - r_w)
    w_max = math.floor(center_w + r_w)
    img_aug_innder = img_aug[h_min: h_max, w_min: w_max]
    edge_aug_innder = edge_aug[h_min: h_max, w_min: w_max]
    return img_aug_innder, edge_aug_innder


def aug_single_img(img, edge_map, augmenter):
    images = img[np.newaxis, :, :]
    # edge_map[edge_map == 255] = 1
    segmaps = edge_map[np.newaxis, :, :, np.newaxis]
    # print("images and segmaps:", images.shape, segmaps.shape)   # (1, 384, 544, 3) (1, 384, 544, 1)
    images_aug, segmaps_aug = augmenter(images=images, segmentation_maps=segmaps)
    # print("each image and segmap:", images_aug[0].shape, segmaps_aug[0].shape)    # (384, 544, 3) (384, 544, 1)
    segmaps_aug = np.concatenate((segmaps_aug[0], segmaps_aug[0], segmaps_aug[0]), axis=-1)
    # print(segmaps_aug.shape)   # (384, 544, 3)
    return images_aug[0], segmaps_aug

def rotate_aug_dir(raw_img_dir, aug_imgs_dir, rot_interval=90):
    wait_aug_lists = os.listdir(raw_img_dir)
    print("waiting rotate list:", wait_aug_lists)
    rot_interval = rot_interval
    for angle in np.arange(0,rot_interval+1, rot_interval):
        if "rot" in raw_img_dir:
            continue
        if angle == 0:
            continue
        print("processing angle" + str(angle) + " ... ...")
        img_base = os.path.split(raw_img_dir)[-1]
        cnt = 0
        for aug_list_dir in  tqdm(wait_aug_lists):
            cnt = cnt + 1
            if cnt > 10:
                # break
                pass
            img = (os.path.join(raw_img_dir, aug_list_dir, 'localObjImg.jpg'))
            if os.path.isfile(img):
                img_data = cv2.imread(img)
                gt_path = os.path.join(raw_img_dir, aug_list_dir, 'matchedMask.jpg')
                gt_data = np.array(cv2.imread(gt_path, 0), dtype=np.int32)
                aug_rotate = iaa.Sequential([iaa.Affine(rotate=angle, fit_output=True,mode='edge')])
                img_aug, edge_aug = aug_single_img(img_data, gt_data, aug_rotate)
                img_aug_innder, edge_aug_innder = get_innder(img_data.shape, img_aug, edge_aug, angle)
                aug_img_path = os.path.join(aug_imgs_dir, img_base + "_rot" + str(angle), aug_list_dir)
                create_dir(aug_img_path)
                cv2.imwrite(os.path.join(aug_img_path, 'localObjImg.jpg'), img_aug_innder)
                cv2.imwrite(os.path.join(aug_img_path, 'matchedMask.jpg'), edge_aug_innder)
                process_template(os.path.join(aug_imgs_dir, img_base + "_rot")+ str(angle), aug_list_dir)
            else:
                print(img,'is not a img path')



def crop_aug_dir(raw_img_dir, aug_imgs_dir, cut_ratio_max=0.7):
    wait_aug_lists = os.listdir(raw_img_dir)
    print("waiting crop list:", wait_aug_lists)
    if "crop" in raw_img_dir:
        return
    img_base = os.path.split(raw_img_dir)[-1]
    print("processing " + img_base + " ... ...")
    cnt = 0
    for aug_list_dir in tqdm(wait_aug_lists):
        cnt = cnt + 1
        if cnt > 10:
            break
        img = (os.path.join(raw_img_dir, aug_list_dir, 'localObjImg.jpg'))
        if os.path.isfile(img):
            img_data = cv2.imread(img)
            gt_path = os.path.join(raw_img_dir, aug_list_dir, 'matchedMask.jpg')
            gt_data = np.array(cv2.imread(gt_path, 0), dtype=np.int32)
            height, width = img_data.shape[:2]
            if height <= width:
                cut_ratio = min(1 - height / width, cut_ratio_max)
                # 如果percent=是一个4个元素的tuple,那么4个元素分别代表(top, right, bottom, left)
                aug_crop_p1 = iaa.Sequential([iaa.Crop(percent=(0, cut_ratio, 0, 0), keep_size=False)])
                img_aug_p1, edge_aug_p1 = aug_single_img(img_data, gt_data, aug_crop_p1)
                aug_crop_p2 = iaa.Sequential([iaa.Crop(percent=(0, 0, 0, cut_ratio), keep_size=False)])
                img_aug_p2, edge_aug_p2 = aug_single_img(img_data, gt_data, aug_crop_p2)
            else:
                cut_ratio = min(1 - width / height, cut_ratio_max)
                aug_crop_p1 = iaa.Sequential([iaa.Crop(percent=(cut_ratio, 0, 0, 0), keep_size=False)])
                img_aug_p1, edge_aug_p1 = aug_single_img(img_data, gt_data, aug_crop_p1)
                aug_crop_p2 = iaa.Sequential([iaa.Crop(percent=(0, 0, cut_ratio, 0), keep_size=False)])
                img_aug_p2, edge_aug_p2 = aug_single_img(img_data, gt_data, aug_crop_p2)
            # ----------------------writing the crop1(left or top)----------------------
            aug_img_path = os.path.join(aug_imgs_dir, img_base + "_crop1", aug_list_dir)
            create_dir(aug_img_path)
            cv2.imwrite(os.path.join(aug_img_path, 'localObjImg.jpg'), img_aug_p1)
            cv2.imwrite(os.path.join(aug_img_path, 'matchedMask.jpg'), edge_aug_p1)
            process_template(os.path.join(aug_imgs_dir, img_base + "_crop1"), aug_list_dir)

            # ----------------------writing the crop1( right or bottom)----------------------
            aug_img_path = os.path.join(aug_imgs_dir, img_base + "_crop2", aug_list_dir)
            create_dir(aug_img_path)
            cv2.imwrite(os.path.join(aug_img_path, 'localObjImg.jpg'), img_aug_p2)
            cv2.imwrite(os.path.join(aug_img_path, 'matchedMask.jpg'), edge_aug_p2)
            process_template(os.path.join(aug_imgs_dir, img_base + "_crop2"), aug_list_dir)
        else:
            print(img, 'is not a img path')



def CoarseDropout_aug_dir(raw_img_dir, aug_imgs_dir):
    wait_aug_lists = os.listdir(raw_img_dir)
    print("waiting crop list:", wait_aug_lists)
    if "drop" in raw_img_dir:
        return
    img_base = os.path.split(raw_img_dir)[-1]
    print("processing " + img_base + " ... ...")
    cnt = 0
    for aug_list_dir in tqdm(wait_aug_lists):
        cnt = cnt + 1
        if cnt > 10:
            break
        img_base = os.path.split(raw_img_dir)[-1]
        img = (os.path.join(raw_img_dir, aug_list_dir, 'localObjImg.jpg'))
        if os.path.isfile(img):
            img_data = cv2.imread(img)
            gt_path = os.path.join(raw_img_dir, aug_list_dir, 'matchedMask.jpg')
            gt_data = np.array(cv2.imread(gt_path, 0), dtype=np.int32)

            aug_drop = iaa.Sequential([iaa.CoarseDropout(0.03, size_percent=0.5)])
            img_aug, edge_aug = aug_single_img(img_data, gt_data, aug_drop)

            aug_img_path = os.path.join(aug_imgs_dir, img_base + "_drop", aug_list_dir)
            create_dir(aug_img_path)
            cv2.imwrite(os.path.join(aug_img_path, 'localObjImg.jpg'), img_aug)
            cv2.imwrite(os.path.join(aug_img_path, 'matchedMask.jpg'), edge_aug)

            process_template(os.path.join(aug_imgs_dir, img_base + "_drop"), aug_list_dir)

        else:
            print(img,'is not a img path')



def data_augument_both():
    raw_img_dir = "/home/gzr/Data/linemod_2d/DATA/steel"
    aug_imgs_dir = "/home/gzr/Data/linemod_2d/DATA/augument_data"
    rotate_aug_dir(raw_img_dir, aug_imgs_dir, rot_interval=20)
    # rotate_aug_dir(raw_img_dir, aug_imgs_dir, rot_interval=160)
    # rotate_aug_dir(raw_img_dir, aug_imgs_dir, rot_interval=180)
    # crop_aug_dir(raw_img_dir, aug_imgs_dir, cut_ratio_max=0.3)
    # CoarseDropout_aug_dir(raw_img_dir, aug_imgs_dir)
    pass


# kps = KeypointsOnImage([
#     Keypoint(x=65, y=100),
#     Keypoint(x=75, y=200),
#     Keypoint(x=100, y=100),
#     Keypoint(x=200, y=80)
# ], shape=image.shape)
#
# seq = iaa.Sequential([
#     iaa.Multiply((1.2, 1.5)), # change brightness, doesn't affect keypoints
#     iaa.Affine(
#         rotate=10,
#         scale=(0.5, 0.7)
#     ) # rotate by exactly 10deg and scale to 50-70%, affects keypoints
# ])
#
# # Augment keypoints and images.
# image_aug, kps_aug = seq(image=image, keypoints=kps)
#
# # print coordinates before/after augmentation (see below)
# # use after.x_int and after.y_int to get rounded integer coordinates
# for i in range(len(kps.keypoints)):
#     before = kps.keypoints[i]
#     after = kps_aug.keypoints[i]
#     print("Keypoint %d: (%.8f, %.8f) -> (%.8f, %.8f)" % (
#         i, before.x, before.y, after.x, after.y)
#     )
#
# # image with keypoints before/after augmentation (shown below)
# image_before = kps.draw_on_image(image, size=7)
# image_after = kps_aug.draw_on_image(image_aug, size=7)

def data_augument_one():
    raw_img_dir = "/home/gzr/Data/linemod_2d/DATA/steel"
    aug_imgs_dir = "/home/gzr/Data/linemod_2d/DATA/augument_data"

    # scale invariant augment







data_dir = "/home/gzr/Data/linemod_2d/DATA/steel"
for i in tqdm(range(0,8166)):

    image_id = f'{i:05d}'
    print(image_id,'\t')
    process_template(data_dir,image_id)

# rotation
# data_augument_both()