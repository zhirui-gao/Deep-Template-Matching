import os

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
import numpy as np
from PIL import Image, ImageDraw

class FeatureProcess(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self):
        pass