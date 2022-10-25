import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
image_path = '/home/gzr/workspace/pytorch-superpoint-master/datasets/draw_lines/images/test/0.png'
point_path = '/home/gzr/workspace/pytorch-superpoint-master/datasets/draw_lines/points/test/0.npy'
points = np.load(point_path)
print(points)
img = Image.open(image_path)
img.show()


# im_aug = transforms.Compose([
#     ,
#     transforms.RandomRotation(degrees=90, resample=False, expand=False, ,fill=int(np.mean(img)))
# ])
rotate_img = transforms.functional.rotate(img, angle=30, expand=True,center=[0,0], fill=int(np.mean(img)))
# img = im_aug(img)
rotate_img.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import torch
# import torchvision.transforms.functional as TF
# image_path = '/home/gzr/workspace/pytorch-superpoint-master/datasets/draw_lines/images/test/0.png'
# point_path = '/home/gzr/workspace/pytorch-superpoint-master/datasets/draw_lines/points/test/0.npy'
# points = np.load(point_path)
# print(points)
# img = Image.open(image_path)
# img.show()
# affine_imgs = TF.affine(img=img, angle=0, translate=[100, 30], scale=1,shear=0,fill=20)
# affine_imgs.show()

