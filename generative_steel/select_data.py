# import os
# import numpy
# import numpy as np
# import cv2
#
# data_list = np.loadtxt('/home/gzr/Data/generative_steel/steel_dataset_now/data_list3.txt').astype(int)
# path_img = '/home/gzr/Data/generative_steel/steel_dataset_now/3/images/test_old'
# path_point = '/home/gzr/Data/generative_steel/steel_dataset_now/3/points/test_old'
# path_trans = '/home/gzr/Data/generative_steel/steel_dataset_now/3/trans/test_old'
#
# path_img2 = '/home/gzr/Data/generative_steel/steel_dataset_now/3/images/test'
# path_point2 = '/home/gzr/Data/generative_steel/steel_dataset_now/3/points/test'
# path_trans2 = '/home/gzr/Data/generative_steel/steel_dataset_now/3/trans/test'
#
# # os.makedirs('/home/gzr/Data/generative_steel/steel_dataset_now/0/images/test')
# # os.makedirs('/home/gzr/Data/generative_steel/steel_dataset_now/0/points/test')
# # os.makedirs('/home/gzr/Data/generative_steel/steel_dataset_now/0/trans/test')
#
# cnt = 0
# for i in range(240):
#     if i in data_list:
#         continue
#
#     img = cv2.imread(os.path.join(path_img,str(i)+'_homo.png'))
#     tem = cv2.imread(os.path.join(path_img,str(i)+'_template.png'))
#     cv2.imwrite(os.path.join(path_img2,str(cnt)+'_homo.png'),img)
#     cv2.imwrite(os.path.join(path_img2,str(cnt)+'_template.png'),tem)
#
#     points = np.load(os.path.join(path_point, str(i)+'_template.npy'))
#     np.save(os.path.join(path_point2, str(cnt)+'_template.npy'),points)
#
#     trans = np.load(os.path.join(path_trans, str(i) + '_trans.npy'))
#     trans_homo = np.load(os.path.join(path_trans, str(i) + '_trans_homo.npy'))
#     np.save(os.path.join(path_trans2, str(cnt) + '_trans.npy'),trans)
#     np.save(os.path.join(path_trans2, str(cnt) + '_trans_homo.npy'),trans_homo)
#
#     cnt = cnt + 1
#
