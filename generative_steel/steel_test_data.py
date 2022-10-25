# import os
# import random
#
# import cv2
# import numpy as np
# random.seed(66)
# np.random.seed(66)
# def get_external_contours_points(image):
#     """
#     :param image: (H,W)
#     :return: (N,2)
#     """
#     assert (len(image.shape)==2)
#     ret, binary = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
#     contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
#
#     return contours
#
# def pad_middle(inp, pad_size=[480,640]):
#
#     if inp.ndim == 2:
#         assert isinstance(pad_size[0], int) and pad_size[0] >= inp.shape[0] and pad_size[1] >= inp.shape[1]
#         padded = np.zeros((pad_size[0], pad_size[1]), dtype=inp.dtype)
#         pad_0 = (pad_size[0] - inp.shape[0])//2
#         pad_1 = (pad_size[1] - inp.shape[1])//2
#         padded[pad_0:inp.shape[0]+pad_0, pad_1:inp.shape[1]+pad_1] = inp
#
#     return padded,[pad_0,pad_1]
#
# def crop_random(inp, crop_border, pad_size=[480,640]):
#     max_y, min_y, max_x, min_x = crop_border[0],crop_border[1],crop_border[2],crop_border[3]
#     pad_0 = random.randint(0, pad_size[0] -(max_y - min_y))
#     pad_1 = random.randint(0, pad_size[1] -(max_x - min_x))
#     if pad_0>min_y:
#         pad_0 = random.randint(0,min_y)
#     if pad_1>min_x:
#         pad_1 = random.randint(0,min_x)
#     cropped_img = inp[min_y-pad_0:pad_size[0]+(min_y-pad_0), min_x-pad_1:pad_size[1]+(min_x-pad_1)]
#     print(pad_0,pad_1)
#
#     return cropped_img, [pad_0, pad_1]
#
#
# def ImageRotate(image):
#     height, width = image.shape[:2]  # 输入(H,W,C)，取 H，W 的值
#     center = (width / 2, height / 2)  # 绕图片中心进行旋转
#     angle = random.randint(-90, 90)  # 旋转方向取（-180，180）中的随机整数值，角度为负时是顺时针旋转，角度为正时是逆时针旋转。
#     # scale = 0.8  # 将图像缩放为80%
#     # 获得旋转矩阵
#     M = cv2.getRotationMatrix2D(center, angle, 1)
#
#     # 进行仿射变换，边界填充为255，即白色，默认为0，即黑色
#     image_rotation = cv2.warpAffine(src=image, M=M, dsize=(width, height))
#
#     return image_rotation, M
#
#
#
#
# img_dir = '/home/gzr/Data/generative_steel/real_dataset/1209-1231'
# img_path = os.path.join(img_dir, 'imgs')
# mask_path = os.path.join(img_dir, 'mask')
# img_save = os.path.join(img_dir, 'process')
# rk_id = 0
# H_save, W_save = 480, 640
# for i in range(10):
#     str_id = '%05d' % i
#     img = cv2.imread(os.path.join(img_path, str_id+'.jpg'), cv2.IMREAD_GRAYSCALE)
#     mask = cv2.imread(os.path.join(mask_path, str_id + '.jpg'), cv2.IMREAD_GRAYSCALE)
#     img = cv2.resize(img, dsize=(2400,600))
#     mask = cv2.resize(mask, dsize=(2400,600))
#
#     H, W = mask.shape[0], mask.shape[1]
#
#     contours = get_external_contours_points(mask)
#
#     for countur in contours:
#         countur_x = countur.reshape(-1, 2)[:, 0]
#         countur_y = countur.reshape(-1, 2)[:, 1]
#         max_x = countur_x[np.argmax(countur_x)]
#         max_y = countur_y[np.argmax(countur_y)]
#         min_x = countur_x[np.argmin(countur_x)]
#         min_y = countur_y[np.argmin(countur_y)]
#
#         min_x = max(min_x - 10, 0)//2 * 2
#         min_y = max(min_y - 10, 0)//2 * 2
#         max_x = min((max_x + 10), W)//2 * 2
#         max_y = min((max_y + 10), H)//2 * 2
#
#         h, w = cropped_img.shape[0], cropped_img.shape[1]
#         fx = fy = 1
#         if h > H_save:
#             fy = H_save / h
#         if w > W_save:
#             fx = W_save / w
#         cropped_mask = cv2.resize(cropped_mask, dsize=(0, 0), fx=fx, fy=fy)
#
#         cropped_mask = mask[min_y:max_y, min_x:max_x]
#
#         random1 = random.randint(-50, 50)
#         random2 = random.randint(-50, 50)
#         min_x = max(min_x + random1, 0)
#         min_y = max(min_y + random2, 0)
#         max_x = min((max_x - ), W)
#         max_y = min((max_y + 10), H)
#
#         cropped_img = img[min_y:max_y, min_x:max_x]
#
#
#
#
#         cropped_img = cv2.resize(cropped_img, dsize=(0, 0),fx=fx,fy=fy)
#
#
#         pad_mask,pad = pad_middle(cropped_mask, pad_size=[H_save, W_save])
#
#
#
#         crop_image, pad_bias = crop_random(img, [max_y, min_y, max_x, min_x], pad_size=[H_save, W_save])
#
#         M_translate = [[1,0,pad_bias[1]],[0,1,pad_bias[0]]]
#         # pad_image = pad_middle(cropped_img,pad_size=[H_save,W_save])
#         rotate_image, M_rotate = ImageRotate(pad_mask)
#
#         # cv2.imwrite(os.path.join(img_save+'/image', '%05d' % rk_id+'.png'), pad_mask)
#         # cv2.imwrite(os.path.join(img_save+'/rotate', '%05d' % rk_id+'.png'), rotate_image)
#         cv2.imwrite(os.path.join(img_save+'/translate', '%05d' % rk_id+'.png'), crop_image)
#         # np.save(os.path.join(img_save+'/trans_rotate', '%05d' % rk_id+'.npy'), M_rotate)
#         np.save(os.path.join(img_save+'/trans_translate', '%05d' % rk_id+'.npy'), M_translate)
#         # cv2.imwrite(os.path.join(img_save, str(rk_id)+'.png'), pad_image)
#
#         # pad_ave = pad_mask*0.2 + pad_image*0.8
#         # cv2.imwrite(os.path.join(img_save, str(rk_id) +'_align.png'), pad_ave)
#         rk_id += 1
#
#
