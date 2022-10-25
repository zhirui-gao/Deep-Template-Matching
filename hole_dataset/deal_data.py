import os
import cv2
import numpy as np
import random
import imgaug.augmenters as iaa
np.random.seed(66)
random.seed(66)

image_base = '/home/gzr/Data/hole_data/hole_dataset/hole/images/test/'
image_base_1 = '/home/gzr/Data/hole_data/hole_dataset/hole/images/train/'
image_base_2 = '/home/gzr/Data/hole_data/hole_dataset/hole/images/val/'
# point_base = image_base.replace('images','points')
data_type = 'data_'
W, H = 640, 480


def get_range_random(min_=0.75, max_=1.25):
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
    return -1
  else:
    return -1

def get_newpoint_between(p1, p2):
  v = (p2[0] - p1[0], p2[1] - p1[1])
  rr = random.random()
  v = (v[0] * rr, v[1] * rr)

  return ( round(p1[0]+positive_or_negative() * v[0]), round(p1[1]+ positive_or_negative() * v[1]))

def get_external_contours_points_sample(image, num_point):
    """
    :param image: (H,W)
    :return: (N,2)
    """
    assert (len(image.shape) == 2)
    ret, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    xcnts = np.vstack([x.reshape(-1, 2) for x in contours])
    if xcnts.shape[0] < num_point:
        return None
    gap = xcnts.shape[0] // num_point
    xcnts = xcnts[:num_point * gap:gap, :]
    return xcnts

def get_external_contours_points(image):
    """
    :param image: (H,W)
    :return: (N,2)
    """
    assert (len(image.shape) == 2)
    ret, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return None
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

def save_data(image_save_dir, image, image_mask, is_ellipse, cnt, resize=True):
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

    if is_ellipse:
        ellipse = cv2.fitEllipse(points)

        # 3.translate + scaling
        random_scale = get_range_random()
        center_ellipse = ellipse[0]
        axis_ellipse = (ellipse[1][0] * random_scale, ellipse[1][1] * random_scale)

        M = cv2.getRotationMatrix2D(center_ellipse, angle=0, scale=random_scale)
        raw_add = np.array([0, 0, 1])
        M = np.r_[M, [raw_add]]
        translate = [w / 2 - center_ellipse[0], h / 2 - center_ellipse[1]]
        M[0, 2] += translate[0]
        M[1, 2] += translate[1]
        M_trans = np.linalg.inv(M)  #

        template = np.zeros_like(image_mask)
        ellipse = ((w / 2, h / 2), axis_ellipse, ellipse[2])

        cv2.ellipse(template, ellipse, (255, 255, 255), -1)
        # cv2.imwrite(os.path.join(image_save_dir, str(cnt) + '.png'), image)
        cv2.imwrite(os.path.join(image_save_dir, str(cnt) + '_template.png'), template)
        np.save(os.path.join(trans_save_dir, str(cnt) + '_trans.npy'), M_trans)
        template_points = get_external_contours_points_sample(template, 20)
        np.save(os.path.join(point_save_dir, str(cnt) + '_template.npy'), template_points)
        warp_image_Homo(os.path.join(image_save_dir, str(cnt) + '.png'),image,H=image.shape[0],W = image.shape[1])

    else:
        center = np.mean(points, axis=0)
        random_scale = get_range_random()
        M = cv2.getRotationMatrix2D(center, angle=0, scale=random_scale)
        raw_add = np.array([0, 0, 1])
        M = np.r_[M, [raw_add]]
        translate = [w / 2 - center[0], h / 2 - center[1]]
        M[0, 2] += translate[0]
        M[1, 2] += translate[1]
        M_trans = np.linalg.inv(M)  #

        template = cv2.warpAffine(src=image_mask, M=M[0:2], dsize=(w, h))
        # cv2.imwrite(os.path.join(image_save_dir, str(cnt) + '.png'), image)
        cv2.imwrite(os.path.join(image_save_dir, str(cnt) + '_template.png'), template)
        np.save(os.path.join(trans_save_dir, str(cnt) + '_trans.npy'), M_trans)
        template_points = get_external_contours_points_sample(template, 20)
        np.save(os.path.join(point_save_dir, str(cnt) + '_template.npy'), template_points)
        warp_image_Homo(os.path.join(image_save_dir, str(cnt) + '.png'),image,H=image.shape[0],W = image.shape[1])


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

def crop_augment(img_data, gt_data, cut_ratio_max=0.2):
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
    return img_aug_p1, edge_aug_p1, img_aug_p2,edge_aug_p2

def noise_augment(img_data, gt_data):

    aug_drop = iaa.Sequential([iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))])
    img_aug, edge_aug = aug_single_img(img_data, gt_data, aug_drop)

    return img_aug, edge_aug

def noise_crop_augment(img_data, gt_data, cut_ratio_max=0.2):
    height, width = img_data.shape[:2]
    if height <= width:
        cut_ratio = min(1 - height / width, cut_ratio_max)
        # 如果percent=是一个4个元素的tuple,那么4个元素分别代表(top, right, bottom, left)
        aug_crop_p1 = iaa.Sequential([iaa.Crop(percent=(0, cut_ratio, 0, 0), keep_size=False),iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))])
        img_aug_p1, edge_aug_p1 = aug_single_img(img_data, gt_data, aug_crop_p1)
        aug_crop_p2 = iaa.Sequential([iaa.Crop(percent=(0, 0, 0, cut_ratio), keep_size=False),iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))])
        img_aug_p2, edge_aug_p2 = aug_single_img(img_data, gt_data, aug_crop_p2)
    else:
        cut_ratio = min(1 - width / height, cut_ratio_max)
        aug_crop_p1 = iaa.Sequential([iaa.Crop(percent=(cut_ratio, 0, 0, 0), keep_size=False),iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))])
        img_aug_p1, edge_aug_p1 = aug_single_img(img_data, gt_data, aug_crop_p1)
        aug_crop_p2 = iaa.Sequential([iaa.Crop(percent=(0, 0, cut_ratio, 0), keep_size=False),iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))])
        img_aug_p2, edge_aug_p2 = aug_single_img(img_data, gt_data, aug_crop_p2)
    return img_aug_p1, edge_aug_p1, img_aug_p2, edge_aug_p2

def deal_sample(path_mask, path_image, is_ellipse, data_type, cnt, is_train,total_num):

    if is_train: # data augmentation

        # raw_image
        image = cv2.imread(path_image)
        image_mask = cv2.imread(path_mask, 0)
        image_save_dir = '/home/gzr/Data/hole_data/hole_dataset_new/' + data_type + '_hole/images/training/'
        save_data(image_save_dir, image, image_mask, is_ellipse, cnt)

        # GuassinNoise_augument
        if total_num < 5000:
            image = cv2.imread(path_image)
            image_mask = cv2.imread(path_mask, 0)
            img_aug_p1, edge_aug_p1 = noise_augment(image, image_mask)
            save_data(image_save_dir, img_aug_p1, edge_aug_p1[:, :, 0], is_ellipse, cnt + total_num * 1)


        # crop_augument
        if total_num < 2000:
            image = cv2.imread(path_image)
            image_mask = cv2.imread(path_mask, 0)
            img_aug_p1, edge_aug_p1, img_aug_p2, edge_aug_p2 = crop_augment(image, image_mask, cut_ratio_max=0.2)
            save_data(image_save_dir, img_aug_p1, edge_aug_p1[:,:,0], is_ellipse, cnt+total_num*2)
            save_data(image_save_dir, img_aug_p2, edge_aug_p2[:,:,0], is_ellipse, cnt+total_num*3)

        if total_num < 1000:
            image = cv2.imread(path_image)
            image_mask = cv2.imread(path_mask, 0)
            img_aug_p1, edge_aug_p1, img_aug_p2, edge_aug_p2 = crop_augment(image, image_mask, cut_ratio_max=0.2)
            save_data(image_save_dir, img_aug_p1, edge_aug_p1[:, :, 0], is_ellipse, cnt + total_num*4)
            save_data(image_save_dir, img_aug_p2, edge_aug_p2[:, :, 0], is_ellipse, cnt + total_num * 5)

        # crop + dropout

    else: # no data augmentation
        image = cv2.imread(path_image)
        image_mask = cv2.imread(path_mask, 0)
        image_save_dir = '/home/gzr/Data/hole_data/hole_dataset_new/' + data_type + '_hole/images/validation/'
        save_data(image_save_dir, image, image_mask, is_ellipse, cnt)

        image_save_dir = '/home/gzr/Data/hole_data/hole_dataset_new/' + data_type + '_hole/images/test/'
        save_data(image_save_dir, image, image_mask, is_ellipse, cnt, resize=False)


def main():
    for data_type in range(0, 8):
        num_test = 50
        path_mask_list = []

        dir_images = image_base + 'data_' + str(data_type)
        files = os.listdir(dir_images)
        for file in files:
            if 'mask' not in file:
                continue
            path_mask = os.path.join(dir_images, file)
            path_mask_list.append(path_mask)


        dir_images = image_base_1 + 'data_' + str(data_type)
        files = os.listdir(dir_images)
        for file in files:
            if 'mask' not in file:
                continue
            path_mask = os.path.join(dir_images, file)
            path_mask_list.append(path_mask)


        dir_images = image_base_2 + 'data_' + str(data_type)
        files = os.listdir(dir_images)
        for file in files:
            if 'mask' not in file:
                continue
            path_mask = os.path.join(dir_images, file)
            path_mask_list.append(path_mask)

        is_ellipse = data_type < 6

        print(len(path_mask_list))

        if len(path_mask_list) > 1000:
            num_test = 100

        np.random.shuffle(path_mask_list)

        path_mask_list_test = path_mask_list[0:num_test]
        path_mask_list_train = path_mask_list[num_test:]


        cnt = 0
        for i in range(len(path_mask_list_test)):
            path_mask = path_mask_list_test[i]
            path_image = path_mask.replace('_mask', '')
            deal_sample(path_mask, path_image, is_ellipse, str(data_type), cnt, is_train=False,total_num=len(path_mask_list_train))
            cnt += 1

        # cnt = 0
        # for i in range(len(path_mask_list_train)):
        #     path_mask = path_mask_list_train[i]
        #     path_image = path_mask.replace('_mask', '')
        #     deal_sample(path_mask, path_image, is_ellipse, str(data_type), cnt, is_train=True, total_num=len(path_mask_list_train))
        #     cnt += 1











