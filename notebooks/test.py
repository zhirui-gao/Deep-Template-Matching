from PIL import Image
import numpy as np
import os
import os.path
import matplotlib.pyplot as plt
import cv2
from src.utils.draw import draw_keypoints,draw_matches
from src.utils.utils import toNumpy
def get_contours_points(image):
    """
    :param image: (H,W)
    :return: (N,2)
    """
    assert (len(image.shape)==2)
    ret, binary = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    xcnts = np.vstack((x.reshape(-1,2) for x in contours))
    return xcnts
#
# data_dir = "/home/gzr/Data/linemod_2d/DATA/steel"
# image_id = "00000"
# image_dir = os.path.join(data_dir,image_id)
# image_path = os.path.join(image_dir,'localObjImg.jpg')
#
#
# img_mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# img_mask = cv2.Canny(img_mask,30,150)
# img_mask3 = cv2.imread(image_path)
# img_mask3 = cv2.Canny(img_mask3,30,150)
# img_mask = cv2.resize(img_mask,(640,480))
# img_mask3 = cv2.resize(img_mask3,(640,480))
#
# print(img_mask.shape)
#
# contours_points = get_contours_points(img_mask)
# for index,point in enumerate(contours_points):
#     if index%5==0:
#         cv2.circle(img_mask3, (point[0],point[1]), radius=1, color=(0,0,255), thickness=1)
# cv2.namedWindow("image")
# cv2.imshow('image', img_mask3)
# cv2.waitKey(10000)
# cv2.destroyAllWindows()

def visualize_gt_pair(dataroot):
    bias = np.loadtxt(os.path.join(dataroot, 'bias.txt'))
    image_path = os.path.join(dataroot, 'localObjImg.jpg')
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img,(0,0),fx=1/8,fy=1/8)

    img3 = cv2.imread(image_path)

    template_path = os.path.join(dataroot, 'template.jpg')
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    contours_points = get_contours_points(template)
    contours_points = contours_points//8
    contours_points = np.array(list(set([tuple(t) for t in contours_points])))
    template = cv2.resize(template, (0, 0), fx=1 / 8, fy=1 / 8)
    template3 = cv2.imread(template_path)
    template = cv2.Canny(template,30,100)



    num = 100
    if num <= contours_points.shape[0]:
        gap = contours_points.shape[0]//num
        contours_points = contours_points[:num*gap:gap, :]
    else:
        num_pad = num - contours_points.shape[0]
        pad = np.random.choice(contours_points.shape[0], num_pad, replace=True)
        choice = np.concatenate([range(contours_points.shape[0]), pad])
        contours_points = contours_points[choice, :]


    match_pair = []
    for index,point in enumerate(contours_points):

        match_pair.append([point[0],point[1],point[0]+bias[0],point[1]+bias[1]])
    match_pair = np.array(match_pair)
    print(match_pair.shape)
    draw_matches(template, img,match_pairs=match_pair,show=True)

data_dir = "/home/gzr/Data/linemod_2d/DATA/steel"
image_id = "03000"
dataroot = os.path.join(data_dir, image_id)
visualize_gt_pair(dataroot)



