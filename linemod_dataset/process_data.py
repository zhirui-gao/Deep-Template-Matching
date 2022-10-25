import os
import cv2
import numpy as np
import random
import imgaug.augmenters as iaa
random.seed(66)
w, h = 640, 480
def is_save(white_mask, white_mask_visib, thr=0.2):
    return np.abs(white_mask-white_mask_visib)/white_mask < thr

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

    c_c = 0.05*w
    d_d = 0.05*h

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

def warp_image_Homo(path,img,H=480,W=640):
    Homo = RPTransform(w=H, h=W)
    # img = cv2.imread(path)

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


def get_range_random(min_=0.9, max_=1.1):
    return random.random() * (max_ - min_) + min_



def save_data(image_save_dir, image, image_mask, rk):
    point_save_dir = image_save_dir.replace('images', 'points')
    trans_save_dir = image_save_dir.replace('images', 'trans')
    mkdir(image_save_dir)
    mkdir(point_save_dir)
    mkdir(trans_save_dir)
    points = get_external_contours_points(image_mask)
    center = np.mean(points, axis=0)

    M = cv2.getRotationMatrix2D(center, angle=0, scale=get_range_random())
    raw_add = np.array([0, 0, 1])
    M = np.r_[M, [raw_add]]
    translate = [w / 2 - center[0], h / 2 - center[1]]
    M[0, 2] += translate[0]
    M[1, 2] += translate[1]
    M_trans = np.linalg.inv(M)  #

    template = cv2.warpAffine(src=image_mask, M=M[0:2], dsize=(w, h))
    # cv2.imwrite(os.path.join(image_save_dir, str(rk) + '.png'), image)
    cv2.imwrite(os.path.join(image_save_dir, str(rk) + '_template.png'), template)
    np.save(os.path.join(trans_save_dir, str(rk) + '_trans.npy'), M_trans)
    np.save(os.path.join(point_save_dir, str(rk) + '_template.npy'), np.array([[w / 2, h / 2]]))
    warp_image_Homo(os.path.join(image_save_dir, str(rk) + '.png'),image)



def deal_scene(scene_name):
    path_base = '/home/gzr/下载/chorme_download/lm_train_pbr/train_pbr'
    save_base = '/home/gzr/下载/chorme_download/lm_train_pbr/train_pbr_select'
    masks_path = os.path.join(path_base, scene_name, 'mask')
    masks_visib_path = os.path.join(path_base, scene_name, 'mask_visib')

    files = os.listdir(masks_path)
    np.random.shuffle(files)

    num_test = 50
    num_traning = 5000



    cnt=0
    for file in files:
        path_mask = os.path.join(masks_path, file)
        path_mask_visib = os.path.join(masks_visib_path, file)

        mask = cv2.imread(path_mask, 0)
        mask_visib = cv2.imread(path_mask_visib, 0)

        white_mask = np.sum(mask/255.0)
        white_mask_visib = np.sum(mask_visib/255.0)


        if is_save(white_mask, white_mask_visib, thr=0.2):
            object_name = file.split('_')[1].split('.')[0]
            image_name = file.split('_')[0]

            image_path = path_mask.replace('mask','rgb').replace(file,image_name+'.jpg')
            image = cv2.imread(image_path)

            if cnt >= num_test:
                image_save_dir = os.path.join(save_base,scene_name,'training/images')
                save_data(image_save_dir, image, mask, rk=cnt-num_test)
            else:
                image_save_dir = os.path.join(save_base, scene_name, 'validation/images')
                save_data(image_save_dir, image, mask, rk=cnt)

            cnt += 1
            if cnt > num_traning+num_test:
                break


for rk_scene in range(8):
    print(rk_scene)
    scene_name = '%06d' % rk_scene
    deal_scene(scene_name)
