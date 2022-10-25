import random
import numpy as np
import cv2

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

    c_c = 0.1*w
    d_d = 0.1*h

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


def warp_image_Homo(path,H=480,W=640):
  Homo = RPTransform(w=H,h=W)
  img = cv2.imread(path)

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


warp_image_Homo('/home/gzr/Data/synthetic_dataset_new/draw_contours/images/test/2.png')