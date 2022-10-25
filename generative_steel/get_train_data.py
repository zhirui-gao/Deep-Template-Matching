import os
import random
from pathlib import Path
import cv2
import numpy as np
random.seed(66)
np.random.seed(66)

def get_external_contours_points(image):
    """
    :param image: (H,W)
    :return: (N,2)
    """
    assert (len(image.shape)==2)
    ret, binary = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    return contours

def Imagetranslate(image,bias=[0,0]):
    height, width = image.shape[:2]  # 输入(H,W,C)，取 H，W 的值
    M = np.array([[1,0,bias[1]],[0,1,bias[0]]]).astype(np.float64)

    # 进行仿射变换，边界填充为255，即白色，默认为0，即黑色
    image_rotation = cv2.warpAffine(src=image, M=M, dsize=(width, height))

    #save transformation in 512*512 dimention
    return image_rotation, M



def get_trans_image():
    img_dir = '/home/gzr/Data/generative_steel/process/data'
    img_path = os.path.join(img_dir, 'images/synthesized_image')
    mask_path = os.path.join(img_dir, 'masks')
    img_save = os.path.join(img_dir, 'translate_images')
    rk_id =0
    for i in range(0,30000):
        str_id = '%05d' % i
        try:
            img = cv2.imread(os.path.join(img_path, str_id+'.png'), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(os.path.join(mask_path, str_id + '.png'), cv2.IMREAD_GRAYSCALE)
            # mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
            img = cv2.resize(img, (mask.shape[1], mask.shape[0]))
            contours = get_external_contours_points(mask)
            xcnts = np.vstack([x.reshape(-1, 2) for x in contours])
            # print(xcnts.min(0))
            if random.randint(0,1)==0:
                pad_0 = random.randint(0, xcnts.min(0)[0])//2 # x
                pad_1 = random.randint(0, xcnts.min(1)[1])//2 # y
                cropped_img,M = Imagetranslate(img,[pad_0,pad_1])
            else:
                pad_0 = -random.randint(0, mask.shape[1]-xcnts.max(0)[0]) // 2  # x
                pad_1 = -random.randint(0, mask.shape[0]-xcnts.max(1)[1]) // 2  # y
                cropped_img, M = Imagetranslate(img, [pad_0, pad_1])

            cv2.imwrite(os.path.join(img_save, '%05d' % rk_id + '.png'), cropped_img)
            ave_aligned = ((cropped_img * 0.6 + mask * 0.3)).round().astype(np.int32)
            cv2.imwrite(os.path.join(img_save, '%05d' % rk_id + '_mask.png'), ave_aligned)

            np.save(os.path.join(img_dir + '/translate_trans', '%05d' % rk_id + '.npy'), M)
            rk_id += 1
        except:
            print('wrong:', i)


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


def warp_image_Homo(path,img, H=480,W=640):
    Homo = RPTransform(w=H, h=W)
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

    gap = xcnts.shape[0] // num_point
    xcnts = xcnts[:num_point * gap:gap, :]
    return xcnts

def get_range_random(min_=2, max_= 3):
    return random.random() * (max_ - min_) + min_

def save_dataset(dataset_type, split,split_data):
    save_dir = "/home/gzr/Data/generative_steel/steel_dataset/" + dataset_type
    image_save = os.path.join(save_dir, 'images/'+split)
    points_save = os.path.join(save_dir, 'points/'+split)
    trans_save = os.path.join(save_dir, 'trans/'+split)
    if not os.path.exists(image_save):
        os.makedirs(image_save)
    if not os.path.exists(points_save):
        os.makedirs(points_save)
    if not os.path.exists(trans_save):
        os.makedirs(trans_save)
    for rk_id, item in enumerate(split_data):
        print(rk_id, item)
        item_mask = item
        if dataset_type=='translate':
            id = int(item.split('.')[0])
            if id>=91:
               id +=1
            item_mask = '%05d' % id+'.png'

        image_mask_path = os.path.join('/home/gzr/Data/generative_steel/process/data/masks', item_mask)
        img_mask = cv2.imread(image_mask_path, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(os.path.join(image_save, str(rk_id) + '_template.png'), img_mask)
        image_path = os.path.join("/home/gzr/Data/generative_steel/process/data/" + dataset_type + "_images", item)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_mask.shape[0], img_mask.shape[1]))
        # cv2.imwrite(os.path.join(image_save, str(rk_id) + '.png'), img)
        # read rotation/translate matrix
        trans_M = np.load(os.path.join("/home/gzr/Data/generative_steel/process/data/" + dataset_type + "_trans",
                                       item.replace('png', 'npy')))
        trans_M = np.row_stack((trans_M, [0, 0, 1]))

        center = (img.shape[1] / 2, img.shape[0]/ 2)
        M = cv2.getRotationMatrix2D(center, angle=0, scale=get_range_random())
        img = cv2.warpAffine(src=img, M=M[0:2], dsize=(img.shape[1], img.shape[0]))
        raw_add = np.array([0, 0, 1])
        M = np.r_[M, [raw_add]]
        trans_M = np.matmul(M, trans_M)
        np.save(os.path.join(trans_save, str(rk_id) + '_trans'), trans_M)
        tran_matrix_homo = warp_image_Homo(path=os.path.join(image_save, str(rk_id) + '.png'), img=img, H=img_mask.shape[0],
                                           W=img_mask.shape[1])

        # find contures in template img
        template_points = get_external_contours_points_sample(img_mask, 20)
        np.save(os.path.join(points_save, str(rk_id) + '_template'), template_points)
        x = template_points[:, 0] * tran_matrix_homo[0, 0] + template_points[:, 1] * tran_matrix_homo[0, 1] + \
            tran_matrix_homo[
                0, 2]
        y = template_points[:, 0] * tran_matrix_homo[1, 0] + template_points[:, 1] * tran_matrix_homo[1, 1] + \
            tran_matrix_homo[
                1, 2]
        z = template_points[:, 0] * tran_matrix_homo[2, 0] + template_points[:, 1] * tran_matrix_homo[2, 1] + \
            tran_matrix_homo[
                2, 2]
        points = np.concatenate(((x / z).reshape(-1, 1), (y / z).reshape(-1, 1)), axis=1)
        np.save(os.path.join(points_save, str(rk_id) + ''), points)



def organize_data(dataset_type):

    name_list = os.listdir('/home/gzr/Data/generative_steel/process/data/translate_images')
    random.shuffle(name_list)
    # name_list_1 = name_list[:12700]
    name_list_2 = name_list[12700:25400]


    name_list_test = name_list_2[:100]
    # name_list_validation = name_list_2[100:200]
    # name_list_training = name_list_2[200:]
    #
    #
    save_dataset(dataset_type,split='test', split_data=name_list_test)
    # save_dataset(dataset_type,split='validation', split_data=name_list_validation)
    # save_dataset(dataset_type,split='training', split_data=name_list_training)




# get_trans_image()
organize_data(dataset_type='translate')


