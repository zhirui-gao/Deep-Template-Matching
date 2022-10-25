import os
import random
from pathlib import Path
import cv2
import numpy as np
random.seed(66)

path = '/home/gzr/Data/synthetic_dataset_transfer/draw_star/points/test/10.npy'
points = np.load(path)
print(points.shape)

