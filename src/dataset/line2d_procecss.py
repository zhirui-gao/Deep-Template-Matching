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

image_id = "00000"

def crop_image(image, padding=20):
    # adopt padding scheme instead of min/max
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
    cropped = image[min_y:max_y, min_x:max_x]
    cropped = np.pad(cropped, ((padding, padding), (padding, padding)),'constant')
    bias = np.array([min_x-padding, min_y-padding])
    return cropped, bias


def process_template(data_dir,image_id):
    image_dir = os.path.join(data_dir, image_id)
    image_path = os.path.join(image_dir, 'matchedEdge.jpg')
    img_mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    img_mask ,bias = crop_image(img_mask, padding=16)
    img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2RGB)
    np.savetxt(os.path.join(image_dir, 'bias.txt'), bias,fmt='%i')
    # plt.imshow(img_mask)
    # plt.show()
    cv2.imwrite(os.path.join(image_dir, 'template_edge.jpg'),img_mask)

def data_augument_both(data_dir,image_id, image_save):
    image_dir = os.path.join(data_dir, image_id)
    edge_path = os.path.join(image_dir, 'matchedEdge.jpg')




data_dir = "/home/gzr/Data/linemod_2d/DATA/steel_val"
for i in range(8156,8166):
    image_id = f'{i:05d}'
    print(image_id,'\t')
    process_template(data_dir,image_id)

