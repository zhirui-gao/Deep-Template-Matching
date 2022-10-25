import os
import cv2
import numpy as np
import random
import imgaug.augmenters as iaa
np.random.seed(66)
random.seed(66)

image_base_1 = '/home/gzr/Data/coco/raw_data/val_dataset/'
# image_base_2 = '/home/gzr/Data/coco/raw_data/train_dataset/'
W, H = 640, 480


def get_range_random(min_=0.9, max_=1.1):
    return random.random() * (max_ - min_) + min_


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def TwoPointRandInterP(p1,p2):
    v = (p2[0]-p1[0], p2[1]-p1[1])
    rr = random.random()
    v = (v[0]*rr, v[1]*rr)
    v = (int(v[0]+0.5), int(v[1]+0.5))
    return (p1[0]+v[0], p1[1]+v[1])


def positive_or_negative():
  if random.random() < 0.5:
    return 1
  else:
    return -1

def get_newpoint_between(p1, p2):
  v = (p2[0] - p1[0], p2[1] - p1[1])
  rr = random.random()
  v = (v[0] * rr, v[1] * rr)

  return ( round(p1[0]+positive_or_negative() * v[0]), round(p1[1]+ positive_or_negative() * v[1]))

def get_external_contours_points(image):
    """
    :param image: (H,W)
    :return: (N,2)
    """
    assert (len(image.shape) == 2)
    ret, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    xcnts = np.vstack([x.reshape(-1, 2) for x in contours])

    # gap = xcnts.shape[0] // num_point
    # xcnts = xcnts[:num_point * gap:gap, :]
    return xcnts

def RPTransform(w,h):
    ratio = 0.05
    c_c = ratio * w
    d_d = ratio * h

    A = (0,0)
    B = (0,h-1)
    C = (w-1,h-1)
    D = (w-1,0)

    a = (c_c,d_d)
    b = (c_c, h-1-d_d)
    c = (w-1-c_c, h-1-d_d)
    d = (w-1-c_c, d_d)

    At = get_newpoint_between(A, a)
    Bt = get_newpoint_between(B, b)
    Ct = get_newpoint_between(C, c)
    Dt = get_newpoint_between(D, d)

    Pts0 = np.array([(0, 0), (0, h), (w, h), (w, 0)]).astype('float32')
    Pts1 = np.array([At, Bt, Ct, Dt]).astype('float32')
    H = cv2.getPerspectiveTransform(Pts0, Pts1)
    return H

def warp_image_Homo(path,image, H=480, W=640):
    Homo = RPTransform(w=H, h=W)
    img = image # cv2.imread(path)

    img_homo = cv2.warpPerspective(img, Homo, (W, H))
    homo_path = path.replace('.png', '_homo.png')
    cv2.imwrite(homo_path, img_homo)

    path_trans = path.replace('images', 'trans')
    path_trans = path_trans.replace('.png', '_trans.npy')
    M = np.load(path_trans)
    Homo_M = np.matmul(Homo, M)
    Homo_M = Homo_M/Homo_M[2][2] # template -> img
    path_trans_homo = path_trans.replace('trans.npy', 'trans_homo.npy')
    np.save(path_trans_homo, Homo_M)
    return Homo_M

def get_external_contours_points_sample(image, num_point):
    """
    :param image: (H,W)
    :return: (N,2)
    """
    assert (len(image.shape) == 2)
    ret, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    xcnts = np.vstack([x.reshape(-1, 2) for x in contours])
    if xcnts.shape[0] < num_point * 5: # get out of the two small case
        return None
    gap = xcnts.shape[0] // num_point
    xcnts = xcnts[:num_point * gap:gap, :]
    return xcnts

def save_data(image_save_dir, image, image_mask, cnt, resize=True):
    point_save_dir = image_save_dir.replace('images', 'points')
    trans_save_dir = image_save_dir.replace('images', 'trans')
    mkdir(image_save_dir)
    mkdir(point_save_dir)
    mkdir(trans_save_dir)

    # 2. resize data
    if resize:
        scale = [W / image.shape[1], H / image.shape[0]]  # x,y
        image_mask = cv2.resize(image_mask, dsize=(W, H))
        image = cv2.resize(image, dsize=(W, H))
        w, h = image_mask.shape[1], image_mask.shape[0]
    else:
        assert image[:,:,0].shape == image_mask.shape
        w, h = image_mask.shape[1], image_mask.shape[0]


    points = get_external_contours_points(image_mask)
    center = np.mean(points, axis=0)
    random_scale = get_range_random(min_= 0.9, max_=1.1)
    random_angel = get_range_random(min_=-30, max_= 30)
    M = cv2.getRotationMatrix2D(center, angle=random_angel, scale=random_scale)
    raw_add = np.array([0, 0, 1])
    M = np.r_[M, [raw_add]]
    translate = [w / 2 - center[0], h / 2 - center[1]]
    M[0, 2] += translate[0]
    M[1, 2] += translate[1]
    M_trans = np.linalg.inv(M)  #

    template = cv2.warpAffine(src=image_mask, M=M[0:2], dsize=(w, h))

    template_points = get_external_contours_points_sample(template, 20)
    if template_points is None:
        return None
    # cv2.imwrite(os.path.join(image_save_dir, str(cnt) + '.png'), image)
    cv2.imwrite(os.path.join(image_save_dir, str(cnt) + '_template.png'), template)
    np.save(os.path.join(trans_save_dir, str(cnt) + '_trans.npy'), M_trans)


    np.save(os.path.join(point_save_dir, str(cnt) + '_template.npy'), template_points)
    warp_image_Homo(os.path.join(image_save_dir, str(cnt) + '.png'),image,H=image.shape[0],W = image.shape[1])
    return True

def deal_sample(path_mask, path_image, data_type, cnt):

    # raw_image
    image = cv2.imread(path_image, 0)
    image_mask = cv2.imread(path_mask, 0)
    image_save_dir = '/home/gzr/Data/coco/coco_dataset/' + data_type + '/images/test/' # training
    return save_data(image_save_dir, image, image_mask, cnt)



num_test = 800 #50010
num_type= 125
path_image_list = []
dir_images = image_base_1 + 'images/'
files = os.listdir(dir_images)
for file in files:
    path_image = os.path.join(dir_images, file)
    path_image_list.append(path_image)

print(len(path_image_list))
np.random.shuffle(path_image_list)
path_mask_list_test = path_image_list[0:num_test]
cnt = 0

for i in range(len(path_mask_list_test)):
    path_image = path_mask_list_test[i]
    path_mask = path_image.replace('/images', '/masks').replace('.jpg','.png')
    if deal_sample(path_mask, path_image, str(cnt//num_type), cnt%num_type) is not None:
        cnt += 1
        print(cnt)