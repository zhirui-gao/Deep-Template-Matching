# -*- coding: utf-8 -*-
"""
Created on 2022年9月28日
@author: zwb
function: 得到中心点 -- 志锐匹配v5
"""

import os
import cv2
import math
import time
import logging
import numpy as np
import pandas as pd
from loguru import logger
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import copy
from copy import deepcopy
from utils import Path, get_contour_points

import torch
import torch.nn as nn
from torch.nn import Module, Dropout
import torch.nn.functional as F
import torchvision.transforms as transforms
from yacs.config import CfgNode as CN
from einops.einops import rearrange, repeat
import kornia
from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid
import pytorch_lightning as pl
from pytorch_lightning.profiler import PassThroughProfiler
import warnings

warnings.filterwarnings('ignore')
INF = 1e9


def get_cfg_defaults():
    from yacs.config import CfgNode as CN
    import torch
    _CN = CN()

    ##############  ↓  SuperPonit  ↓  ##############
    _CN.TM = CN()
    _CN.TM.SUPERPOINT = CN()
    _CN.TM.SUPERPOINT.NAME = 'superpoint_glue'  # 'SuperPointNet_gauss2'
    _CN.TM.SUPERPOINT.BLOCK_DIMS = [64, 64, 128, 128, 256, 256]  # c1, c2, c3, c4, c5, d1
    _CN.TM.SUPERPOINT.DET_DIM = 65
    _CN.TM.SUPERPOINT.OUT_NUM_POINTS = 128  # change num=128 in data.py
    _CN.TM.SUPERPOINT.PATCH_SIZE = 5
    _CN.TM.SUPERPOINT.NMS_DIST = 2
    _CN.TM.SUPERPOINT.CONF_THRESH = 0.004
    _CN.TM.SUPERPOINT.subpixel_channel = 1

    _CN.TM.FINE_CONCAT_COARSE_FEAT = True

    # 2. LoFTR-coarse module config

    _CN.TM.COARSE = CN()
    _CN.TM.COARSE.TWO_STAGE = False
    _CN.TM.COARSE.D_MODEL = 256
    _CN.TM.COARSE.D_FFN = 256
    _CN.TM.COARSE.NHEAD = 8
    _CN.TM.RESOLUTION = (8, 2)
    if _CN.TM.COARSE.TWO_STAGE:
        _CN.TM.COARSE.LAYER_NAMES = ['self', 'cross'] * 2
    else:
        _CN.TM.COARSE.LAYER_NAMES = ['self', 'cross'] * 4
    _CN.TM.COARSE.ATTENTION = 'linear'  # options: ['linear', 'full']
    _CN.TM.COARSE.POSITION = 'rotary'  # options: ['rotary', 'sin']
    _CN.TM.COARSE.TEMP_BUG_FIX = True

    # fine_global
    _CN.TM.FINE_GLOBAL = CN()
    _CN.TM.FINE_GLOBAL.D_MODEL = 256
    _CN.TM.FINE_GLOBAL.D_FFN = 256
    _CN.TM.FINE_GLOBAL.NHEAD = 8
    _CN.TM.FINE_GLOBAL.LAYER_NAMES = ['self', 'cross'] * 2
    _CN.TM.FINE_GLOBAL.ATTENTION = 'linear'
    _CN.TM.FINE_GLOBAL.POSITION = 'rotary'

    # 3. Coarse-Matching config
    _CN.TM.MATCH_COARSE = CN()
    _CN.TM.MATCH_COARSE.THR = 0.2
    _CN.TM.MATCH_COARSE.TRAIN_STAGE = 'whole'  # 'whole' or 'only_coarse'
    _CN.TM.MATCH_COARSE.USE_EDGE = False  # 'whole' or 'only_coarse'

    _CN.TM.MATCH_COARSE.OUT_NUM_POINTS = _CN.TM.SUPERPOINT.OUT_NUM_POINTS
    _CN.TM.MATCH_COARSE.BORDER_RM = 2
    _CN.TM.MATCH_COARSE.MATCH_TYPE = 'sinkhorn'  # options: ['dual_softmax, 'sinkhorn']
    _CN.TM.MATCH_COARSE.DSMAX_TEMPERATURE = 0.1
    _CN.TM.MATCH_COARSE.SKH_ITERS = 3
    _CN.TM.MATCH_COARSE.SKH_INIT_BIN_SCORE = 1.0
    _CN.TM.MATCH_COARSE.SKH_PREFILTER = False
    _CN.TM.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.2  # training tricks: save GPU memory
    _CN.TM.MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN = 400  # training tricks: avoid DDP deadlock
    _CN.TM.MATCH_COARSE.SPARSE_SPVS = True

    # 3.1 Fine module config
    _CN.TM.FINE = CN()
    _CN.TM.STN = False
    _CN.TM.FINE.PHOTOMETRIC = True
    _CN.TM.FINE.THR = 0.0
    _CN.TM.FINE.D_MODEL = 64
    _CN.TM.FINE.D_FFN = 64
    _CN.TM.FINE.NHEAD = 4
    _CN.TM.FINE.LAYER_NAMES = ['self', 'cross'] * 2
    _CN.TM.FINE.ATTENTION = 'linear'  # 'full' #TODO:change it to full
    _CN.TM.FINE.POSITION = 'rotary'  # options: ['rotary', 'sin']
    _CN.TM.FINE.RESOLUTION = _CN.TM.RESOLUTION
    _CN.TM.FINE.FINE_CONCAT_COARSE_FEAT = _CN.TM.FINE_CONCAT_COARSE_FEAT
    _CN.TM.FINE.FINE_WINDOW_SIZE = 5  # window_size in fine_level, must be odd
    _CN.TM.FINE.DSMAX_TEMPERATURE = 0.025
    # 3.2 transformation

    _CN.TM.TRANS = CN()
    _CN.TM.TRANS.SIGMA_D = 0.4  # 0.5
    _CN.TM.TRANS.SIGMA_A = 1.0  # 1.5
    _CN.TM.TRANS.GAMA = 0.5  # 0.5
    _CN.TM.TRANS.ANGEL_K = 3
    _CN.TM.TRANS.INLIER_THRESHOLD = 4  # 4

    _CN.TM.TRANS.NUM_ITERATIONS = 10

    # 4. Losses
    # -- # coarse-level
    _CN.TM.LOSS = CN()
    _CN.TM.LOSS.FINE_TYPE = 'l2_with_std'
    _CN.TM.LOSS.COARSE_TYPE = 'focal'  # ['focal', 'cross_entropy']
    _CN.TM.LOSS.COARSE_WEIGHT = 1.0
    _CN.TM.LOSS.COARSE_WEIGHT = 1.0

    # _CN.TM.LOSS.SPARSE_SPVS = False

    # -- # fine-level
    _CN.TM.LOSS.FINE_TYPE = 'l2_with_std'  # ['l2_with_std', 'l2']
    _CN.TM.LOSS.FINE_WEIGHT = 1.0
    _CN.TM.LOSS.FINE_CORRECT_THR = 1.0  # for filtering valid fine-leve

    # -- - -- # focal loss (coarse)
    _CN.TM.LOSS.FOCAL_ALPHA = 0.25
    _CN.TM.LOSS.FOCAL_GAMMA = 2.0
    _CN.TM.LOSS.POS_WEIGHT = 1.0
    _CN.TM.LOSS.NEG_WEIGHT = 1.0

    # 5.edge config
    _CN.TM.EDGE = CN()
    _CN.TM.EDGE.NAME = 'pidinet_tiny_converted'
    _CN.TM.EDGE.LOAD = True  # load

    ##############  Dataset  ##############
    _CN.DATASET = CN()
    # 1. data config
    # training and validating
    _CN.DATASET.TRAINVAL_DATA_SOURCE = None  # options: ['ScanNet', 'MegaDepth']
    _CN.DATASET.TRAIN_DATA_ROOT = None
    _CN.DATASET.TRAIN_POSE_ROOT = None  # (optional directory for poses)
    _CN.DATASET.TRAIN_NPZ_ROOT = None
    _CN.DATASET.TRAIN_LIST_PATH = None
    _CN.DATASET.TRAIN_INTRINSIC_PATH = None
    _CN.DATASET.VAL_DATA_ROOT = None
    _CN.DATASET.VAL_POSE_ROOT = None  # (optional directory for poses)
    _CN.DATASET.VAL_NPZ_ROOT = None
    _CN.DATASET.VAL_LIST_PATH = None  # None if val data from all scenes are bundled into a single npz file
    _CN.DATASET.VAL_INTRINSIC_PATH = None
    _CN.DATASET.IMG_RESIZE = 256

    # testing
    _CN.DATASET.TEST_DATA_SOURCE = None
    _CN.DATASET.TEST_DATA_ROOT = None
    _CN.DATASET.TEST_POSE_ROOT = None  # (optional directory for poses)
    _CN.DATASET.TEST_NPZ_ROOT = None
    _CN.DATASET.TEST_LIST_PATH = None  # None if test data from all scenes are bundled into a single npz file
    _CN.DATASET.TEST_INTRINSIC_PATH = None

    # 2. dataset config
    _CN.DATASET.AUGMENTATION_TYPE = None  # options: [None, 'dark', 'mobile']

    ##############  Trainer  ##############
    _CN.TRAINER = CN()
    if _CN.TM.MATCH_COARSE.TRAIN_STAGE == 'only_coarse':
        _CN.TRAINER.WORLD_SIZE = 1
        _CN.TRAINER.CANONICAL_BS = 64  #
        _CN.TRAINER.CANONICAL_LR = 6e-3
        _CN.TRAINER.SCALING = None  # this will be calculated automatically
        _CN.TRAINER.FIND_LR = False  # use learning rate finder from pytorch-lightning

        # optimizer
        _CN.TRAINER.OPTIMIZER = "adamw"  # [adam, adamw]
        _CN.TRAINER.TRUE_LR = None  # this will be calculated automatically at runtime
        _CN.TRAINER.ADAM_DECAY = 0.  # ADAM: for adam
        _CN.TRAINER.ADAMW_DECAY = 0.1

        # step-based warm-up
        _CN.TRAINER.WARMUP_TYPE = 'linear'  # [linear, constant]
        _CN.TRAINER.WARMUP_RATIO = 0.
        _CN.TRAINER.WARMUP_STEP = 4800
    else:
        _CN.TRAINER.WORLD_SIZE = 1
        _CN.TRAINER.CANONICAL_BS = 64  #
        _CN.TRAINER.CANONICAL_LR = 6e-4
        _CN.TRAINER.SCALING = None  # this will be calculated automatically
        _CN.TRAINER.FIND_LR = False  # use learning rate finder from pytorch-lightning
        # optimizer
        _CN.TRAINER.OPTIMIZER = "adamw"  # [adam, adamw]
        _CN.TRAINER.TRUE_LR = None  # this will be calculated automatically at runtime
        _CN.TRAINER.ADAM_DECAY = 0.  # ADAM: for adam
        _CN.TRAINER.ADAMW_DECAY = 0.1
        # step-based warm-up
        _CN.TRAINER.WARMUP_TYPE = 'linear'  # [linear, constant]
        _CN.TRAINER.WARMUP_RATIO = 0.
        _CN.TRAINER.WARMUP_STEP = 4800

    # learning rate scheduler
    _CN.TRAINER.SCHEDULER = 'MultiStepLR'  # [MultiStepLR, CosineAnnealing, ExponentialLR]
    _CN.TRAINER.SCHEDULER_INTERVAL = 'epoch'  # [epoch, step]
    _CN.TRAINER.MSLR_MILESTONES = [3, 6, 9, 12]  # MSLR: MultiStepLR
    _CN.TRAINER.MSLR_GAMMA = 0.5
    _CN.TRAINER.COSA_TMAX = 30  # COSA: CosineAnnealing
    _CN.TRAINER.ELR_GAMMA = 0.999992  # ELR: ExponentialLR, this value for 'step' interval

    # plotting related
    _CN.TRAINER.ENABLE_PLOTTING = True
    _CN.TRAINER.N_VAL_PAIRS_TO_PLOT = 12  # number of val/test paris for plotting
    _CN.TRAINER.PLOT_MODE = 'evaluation'  # ['evaluation', 'confidence']
    _CN.TRAINER.PLOT_MATCHES_ALPHA = 'dynamic'

    # geometric metrics and pose solver
    _CN.TRAINER.DIS_ERR_THR = 3
    _CN.TRAINER.EPI_ERR_THR = 5e-4  # recommendation: 5e-4 for ScanNet, 1e-4 for MegaDepth (from SuperGlue)
    _CN.TRAINER.POSE_GEO_MODEL = 'E'  # ['E', 'F', 'H']
    _CN.TRAINER.POSE_ESTIMATION_METHOD = 'RANSAC'  # [RANSAC, DEGENSAC, MAGSAC]
    _CN.TRAINER.RANSAC_PIXEL_THR = 0.5
    _CN.TRAINER.RANSAC_CONF = 0.99999
    _CN.TRAINER.RANSAC_MAX_ITERS = 10000
    _CN.TRAINER.USE_MAGSACPP = False

    # data sampler for train_dataloader
    _CN.TRAINER.DATA_SAMPLER = 'scene_balance'  # options: ['scene_balance', 'random', 'normal']
    # 'scene_balance' config
    _CN.TRAINER.N_SAMPLES_PER_SUBSET = 5000  # for hole dataset
    _CN.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT = True  # whether sample each scene with replacement or not
    _CN.TRAINER.SB_SUBSET_SHUFFLE = True  # after sampling from scenes, whether shuffle within the epoch or not
    _CN.TRAINER.SB_REPEAT = 1  # repeat N times for training the sampled data
    # 'random' config
    _CN.TRAINER.RDM_REPLACEMENT = True
    _CN.TRAINER.RDM_NUM_SAMPLES = None

    # gradient clipping
    _CN.TRAINER.GRADIENT_CLIPPING = 0.5

    # reproducibility
    # This seed affects the data sampling. With the same seed, the data sampling is promised
    # to be the same. When resume training from a checkpoint, it's better to use a different
    # seed, otherwise the sampled data will be exactly the same as before resuming, which will
    # cause less unique data items sampled during the entire training.
    # Use of different seed values might affect the final training result, since not all data items
    # are used during training on ScanNet. (60M pairs of images sampled during traing from 230M pairs in total.)
    _CN.TRAINER.SEED = 68
    return _CN.clone()


class Arg():
    def __init__(self, edge_ckpt=None):
        self.config = 'carv4'
        self.evaluate = edge_ckpt
        self.evaluate_converted = True
        self.dil = False
        self.sa = False
        self.savedir = './pidinet/savedir'  # no use


nets = {
    'baseline': {
        'layer0': 'cv',
        'layer1': 'cv',
        'layer2': 'cv',
        'layer3': 'cv',
        'layer4': 'cv',
        'layer5': 'cv',
        'layer6': 'cv',
        'layer7': 'cv',
        'layer8': 'cv',
        'layer9': 'cv',
        'layer10': 'cv',
        'layer11': 'cv',
        'layer12': 'cv',
        'layer13': 'cv',
        'layer14': 'cv',
        'layer15': 'cv',
    },
    'c-v15': {
        'layer0': 'cd',
        'layer1': 'cv',
        'layer2': 'cv',
        'layer3': 'cv',
        'layer4': 'cv',
        'layer5': 'cv',
        'layer6': 'cv',
        'layer7': 'cv',
        'layer8': 'cv',
        'layer9': 'cv',
        'layer10': 'cv',
        'layer11': 'cv',
        'layer12': 'cv',
        'layer13': 'cv',
        'layer14': 'cv',
        'layer15': 'cv',
    },
    'a-v15': {
        'layer0': 'ad',
        'layer1': 'cv',
        'layer2': 'cv',
        'layer3': 'cv',
        'layer4': 'cv',
        'layer5': 'cv',
        'layer6': 'cv',
        'layer7': 'cv',
        'layer8': 'cv',
        'layer9': 'cv',
        'layer10': 'cv',
        'layer11': 'cv',
        'layer12': 'cv',
        'layer13': 'cv',
        'layer14': 'cv',
        'layer15': 'cv',
    },
    'r-v15': {
        'layer0': 'rd',
        'layer1': 'cv',
        'layer2': 'cv',
        'layer3': 'cv',
        'layer4': 'cv',
        'layer5': 'cv',
        'layer6': 'cv',
        'layer7': 'cv',
        'layer8': 'cv',
        'layer9': 'cv',
        'layer10': 'cv',
        'layer11': 'cv',
        'layer12': 'cv',
        'layer13': 'cv',
        'layer14': 'cv',
        'layer15': 'cv',
    },
    'cvvv4': {
        'layer0': 'cd',
        'layer1': 'cv',
        'layer2': 'cv',
        'layer3': 'cv',
        'layer4': 'cd',
        'layer5': 'cv',
        'layer6': 'cv',
        'layer7': 'cv',
        'layer8': 'cd',
        'layer9': 'cv',
        'layer10': 'cv',
        'layer11': 'cv',
        'layer12': 'cd',
        'layer13': 'cv',
        'layer14': 'cv',
        'layer15': 'cv',
    },
    'avvv4': {
        'layer0': 'ad',
        'layer1': 'cv',
        'layer2': 'cv',
        'layer3': 'cv',
        'layer4': 'ad',
        'layer5': 'cv',
        'layer6': 'cv',
        'layer7': 'cv',
        'layer8': 'ad',
        'layer9': 'cv',
        'layer10': 'cv',
        'layer11': 'cv',
        'layer12': 'ad',
        'layer13': 'cv',
        'layer14': 'cv',
        'layer15': 'cv',
    },
    'rvvv4': {
        'layer0': 'rd',
        'layer1': 'cv',
        'layer2': 'cv',
        'layer3': 'cv',
        'layer4': 'rd',
        'layer5': 'cv',
        'layer6': 'cv',
        'layer7': 'cv',
        'layer8': 'rd',
        'layer9': 'cv',
        'layer10': 'cv',
        'layer11': 'cv',
        'layer12': 'rd',
        'layer13': 'cv',
        'layer14': 'cv',
        'layer15': 'cv',
    },
    'cccv4': {
        'layer0': 'cd',
        'layer1': 'cd',
        'layer2': 'cd',
        'layer3': 'cv',
        'layer4': 'cd',
        'layer5': 'cd',
        'layer6': 'cd',
        'layer7': 'cv',
        'layer8': 'cd',
        'layer9': 'cd',
        'layer10': 'cd',
        'layer11': 'cv',
        'layer12': 'cd',
        'layer13': 'cd',
        'layer14': 'cd',
        'layer15': 'cv',
    },
    'aaav4': {
        'layer0': 'ad',
        'layer1': 'ad',
        'layer2': 'ad',
        'layer3': 'cv',
        'layer4': 'ad',
        'layer5': 'ad',
        'layer6': 'ad',
        'layer7': 'cv',
        'layer8': 'ad',
        'layer9': 'ad',
        'layer10': 'ad',
        'layer11': 'cv',
        'layer12': 'ad',
        'layer13': 'ad',
        'layer14': 'ad',
        'layer15': 'cv',
    },
    'rrrv4': {
        'layer0': 'rd',
        'layer1': 'rd',
        'layer2': 'rd',
        'layer3': 'cv',
        'layer4': 'rd',
        'layer5': 'rd',
        'layer6': 'rd',
        'layer7': 'cv',
        'layer8': 'rd',
        'layer9': 'rd',
        'layer10': 'rd',
        'layer11': 'cv',
        'layer12': 'rd',
        'layer13': 'rd',
        'layer14': 'rd',
        'layer15': 'cv',
    },
    'c16': {
        'layer0': 'cd',
        'layer1': 'cd',
        'layer2': 'cd',
        'layer3': 'cd',
        'layer4': 'cd',
        'layer5': 'cd',
        'layer6': 'cd',
        'layer7': 'cd',
        'layer8': 'cd',
        'layer9': 'cd',
        'layer10': 'cd',
        'layer11': 'cd',
        'layer12': 'cd',
        'layer13': 'cd',
        'layer14': 'cd',
        'layer15': 'cd',
    },
    'a16': {
        'layer0': 'ad',
        'layer1': 'ad',
        'layer2': 'ad',
        'layer3': 'ad',
        'layer4': 'ad',
        'layer5': 'ad',
        'layer6': 'ad',
        'layer7': 'ad',
        'layer8': 'ad',
        'layer9': 'ad',
        'layer10': 'ad',
        'layer11': 'ad',
        'layer12': 'ad',
        'layer13': 'ad',
        'layer14': 'ad',
        'layer15': 'ad',
    },
    'r16': {
        'layer0': 'rd',
        'layer1': 'rd',
        'layer2': 'rd',
        'layer3': 'rd',
        'layer4': 'rd',
        'layer5': 'rd',
        'layer6': 'rd',
        'layer7': 'rd',
        'layer8': 'rd',
        'layer9': 'rd',
        'layer10': 'rd',
        'layer11': 'rd',
        'layer12': 'rd',
        'layer13': 'rd',
        'layer14': 'rd',
        'layer15': 'rd',
    },
    'carv4': {
        'layer0': 'cd',
        'layer1': 'ad',
        'layer2': 'rd',
        'layer3': 'cv',
        'layer4': 'cd',
        'layer5': 'ad',
        'layer6': 'rd',
        'layer7': 'cv',
        'layer8': 'cd',
        'layer9': 'ad',
        'layer10': 'rd',
        'layer11': 'cv',
        'layer12': 'cd',
        'layer13': 'ad',
        'layer14': 'rd',
        'layer15': 'cv',
    },
}


def config_model_converted(model):
    model_options = list(nets.keys())
    assert model in model_options, \
        'unrecognized model, please choose from %s' % str(model_options)

    print(str(nets[model]))

    pdcs = []
    for i in range(16):
        layer_name = 'layer%d' % i
        op = nets[model][layer_name]
        pdcs.append(op)

    return pdcs


class MapReduce(nn.Module):
    """
    Reduce feature maps into a single edge map
    """

    def __init__(self, channels):
        super(MapReduce, self).__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, padding=0)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


class PDCBlock_converted(nn.Module):
    """
    CPDC, APDC can be converted to vanilla 3x3 convolution
    RPDC can be converted to vanilla 5x5 convolution
    """

    def __init__(self, pdc, inplane, ouplane, stride=1):
        super(PDCBlock_converted, self).__init__()
        self.stride = stride

        if self.stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0)
        if pdc == 'rd':
            self.conv1 = nn.Conv2d(inplane, inplane, kernel_size=5, padding=2, groups=inplane, bias=False)
        else:
            self.conv1 = nn.Conv2d(inplane, inplane, kernel_size=3, padding=1, groups=inplane, bias=False)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        if self.stride > 1:
            x = self.pool(x)
        y = self.conv1(x)
        y = self.relu2(y)
        y = self.conv2(y)
        if self.stride > 1:
            x = self.shortcut(x)
        y = y + x
        return y


class PiDiNet(nn.Module):
    def __init__(self, inplane, pdcs, dil=None, sa=False, convert=False):
        super(PiDiNet, self).__init__()
        self.sa = sa
        if dil is not None:
            assert isinstance(dil, int), 'dil should be an int'
        self.dil = dil

        self.fuseplanes = []

        self.inplane = inplane
        if convert:
            if pdcs[0] == 'rd':
                init_kernel_size = 5
                init_padding = 2
            else:
                init_kernel_size = 3
                init_padding = 1
            self.init_block = nn.Conv2d(3, self.inplane,
                                        kernel_size=init_kernel_size, padding=init_padding, bias=False)
            block_class = PDCBlock_converted
        else:
            self.init_block = Conv2d(pdcs[0], 3, self.inplane, kernel_size=3, padding=1)
            block_class = PDCBlock

        self.block1_1 = block_class(pdcs[1], self.inplane, self.inplane)
        self.block1_2 = block_class(pdcs[2], self.inplane, self.inplane)
        self.block1_3 = block_class(pdcs[3], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)  # C

        inplane = self.inplane
        self.inplane = self.inplane * 2
        self.block2_1 = block_class(pdcs[4], inplane, self.inplane, stride=2)
        self.block2_2 = block_class(pdcs[5], self.inplane, self.inplane)
        self.block2_3 = block_class(pdcs[6], self.inplane, self.inplane)
        self.block2_4 = block_class(pdcs[7], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)  # 2C

        inplane = self.inplane
        self.inplane = self.inplane * 2
        self.block3_1 = block_class(pdcs[8], inplane, self.inplane, stride=2)
        self.block3_2 = block_class(pdcs[9], self.inplane, self.inplane)
        self.block3_3 = block_class(pdcs[10], self.inplane, self.inplane)
        self.block3_4 = block_class(pdcs[11], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)  # 4C

        self.block4_1 = block_class(pdcs[12], self.inplane, self.inplane, stride=2)
        self.block4_2 = block_class(pdcs[13], self.inplane, self.inplane)
        self.block4_3 = block_class(pdcs[14], self.inplane, self.inplane)
        self.block4_4 = block_class(pdcs[15], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)  # 4C

        self.conv_reduces = nn.ModuleList()
        if self.sa and self.dil is not None:
            self.attentions = nn.ModuleList()
            self.dilations = nn.ModuleList()
            for i in range(4):
                self.dilations.append(CDCM(self.fuseplanes[i], self.dil))
                self.attentions.append(CSAM(self.dil))
                self.conv_reduces.append(MapReduce(self.dil))
        elif self.sa:
            self.attentions = nn.ModuleList()
            for i in range(4):
                self.attentions.append(CSAM(self.fuseplanes[i]))
                self.conv_reduces.append(MapReduce(self.fuseplanes[i]))
        elif self.dil is not None:
            self.dilations = nn.ModuleList()
            for i in range(4):
                self.dilations.append(CDCM(self.fuseplanes[i], self.dil))
                self.conv_reduces.append(MapReduce(self.dil))
        else:
            for i in range(4):
                self.conv_reduces.append(MapReduce(self.fuseplanes[i]))

        self.classifier = nn.Conv2d(4, 1, kernel_size=1)  # has bias
        nn.init.constant_(self.classifier.weight, 0.25)
        nn.init.constant_(self.classifier.bias, 0)

        print('initialization done')

    def get_weights(self):
        conv_weights = []
        bn_weights = []
        relu_weights = []
        for pname, p in self.named_parameters():
            if 'bn' in pname:
                bn_weights.append(p)
            elif 'relu' in pname:
                relu_weights.append(p)
            else:
                conv_weights.append(p)

        return conv_weights, bn_weights, relu_weights

    def forward(self, x):
        H, W = x.size()[2:]

        x = self.init_block(x)

        x1 = self.block1_1(x)
        x1 = self.block1_2(x1)
        x1 = self.block1_3(x1)

        x2 = self.block2_1(x1)
        x2 = self.block2_2(x2)
        x2 = self.block2_3(x2)
        x2 = self.block2_4(x2)

        x3 = self.block3_1(x2)
        x3 = self.block3_2(x3)
        x3 = self.block3_3(x3)
        x3 = self.block3_4(x3)

        x4 = self.block4_1(x3)
        x4 = self.block4_2(x4)
        x4 = self.block4_3(x4)
        x4 = self.block4_4(x4)

        x_fuses = []
        if self.sa and self.dil is not None:
            for i, xi in enumerate([x1, x2, x3, x4]):
                x_fuses.append(self.attentions[i](self.dilations[i](xi)))
        elif self.sa:
            for i, xi in enumerate([x1, x2, x3, x4]):
                x_fuses.append(self.attentions[i](xi))
        elif self.dil is not None:
            for i, xi in enumerate([x1, x2, x3, x4]):
                x_fuses.append(self.dilations[i](xi))
        else:
            x_fuses = [x1, x2, x3, x4]

        e1 = self.conv_reduces[0](x_fuses[0])
        e1 = F.interpolate(e1, (H, W), mode="bilinear", align_corners=False)

        e2 = self.conv_reduces[1](x_fuses[1])
        e2 = F.interpolate(e2, (H, W), mode="bilinear", align_corners=False)

        e3 = self.conv_reduces[2](x_fuses[2])
        e3 = F.interpolate(e3, (H, W), mode="bilinear", align_corners=False)

        e4 = self.conv_reduces[3](x_fuses[3])
        e4 = F.interpolate(e4, (H, W), mode="bilinear", align_corners=False)

        outputs = [e1, e2, e3, e4]

        output = self.classifier(torch.cat(outputs, dim=1))
        # if not self.training:
        #    return torch.sigmoid(output)

        outputs.append(output)
        outputs = [torch.sigmoid(r) for r in outputs]
        return outputs


def pidinet_tiny_converted(args):
    pdcs = config_model_converted(args.config)
    dil = 8 if args.dil else None
    return PiDiNet(20, pdcs, dil=dil, sa=args.sa, convert=True)


def load_checkpoint(args, running_file=None):
    model_dir = os.path.join(args.savedir, 'save_models')
    latest_filename = os.path.join(model_dir, 'latest.txt')
    model_filename = ''

    if args.evaluate is not None:
        model_filename = args.evaluate
    else:
        if os.path.exists(latest_filename):
            with open(latest_filename, 'r') as fin:
                model_filename = fin.readlines()[0].strip()
    loadinfo = "=> loading checkpoint from '{}'".format(model_filename)
    print(loadinfo)

    state = None
    if os.path.exists(model_filename):
        state = torch.load(model_filename, map_location='cpu')
        loadinfo2 = "=> loaded checkpoint '{}' successfully".format(model_filename)
    else:
        loadinfo2 = "no checkpoint loaded"
    print(loadinfo2)
    # running_file.write('%s\n%s\n' % (loadinfo, loadinfo2))
    # running_file.flush()

    return state


def convert_pdc(op, weight):
    if op == 'cv':
        return weight
    elif op == 'cd':
        shape = weight.shape
        weight_c = weight.sum(dim=[2, 3])
        weight = weight.view(shape[0], shape[1], -1)
        weight[:, :, 4] = weight[:, :, 4] - weight_c
        weight = weight.view(shape)
        return weight
    elif op == 'ad':
        shape = weight.shape
        weight = weight.view(shape[0], shape[1], -1)
        weight_conv = (weight - weight[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view(shape)
        return weight_conv
    elif op == 'rd':
        shape = weight.shape
        buffer = torch.zeros(shape[0], shape[1], 5 * 5, device=weight.device)
        weight = weight.view(shape[0], shape[1], -1)
        buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weight[:, :, 1:]
        buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weight[:, :, 1:]
        buffer = buffer.view(shape[0], shape[1], 5, 5)
        return buffer
    raise ValueError("wrong op {}".format(str(op)))


def convert_pidinet(state_dict, config):
    pdcs = config_model_converted(config)
    new_dict = {}
    for pname, p in state_dict.items():
        if 'init_block.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[0], p)
        elif 'block1_1.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[1], p)
        elif 'block1_2.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[2], p)
        elif 'block1_3.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[3], p)
        elif 'block2_1.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[4], p)
        elif 'block2_2.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[5], p)
        elif 'block2_3.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[6], p)
        elif 'block2_4.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[7], p)
        elif 'block3_1.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[8], p)
        elif 'block3_2.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[9], p)
        elif 'block3_3.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[10], p)
        elif 'block3_4.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[11], p)
        elif 'block4_1.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[12], p)
        elif 'block4_2.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[13], p)
        elif 'block4_3.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[14], p)
        elif 'block4_4.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[15], p)
        else:
            new_dict[pname] = p

    return new_dict


class Edge_Net(nn.Module):
    def __init__(self, config, edge_ckpt=None):
        super().__init__()
        args = Arg(edge_ckpt)
        if config['name'] == 'pidinet_tiny_converted':
            self.model = pidinet_tiny_converted(args)
        checkpoint = load_checkpoint(args)
        if args.evaluate_converted:
            state_dict = convert_pidinet(checkpoint['state_dict'], args.config)
            model_dict = self.model.state_dict()
            for k, v in state_dict.items():
                if k[7:] in model_dict.keys() and v.shape == model_dict[k[7:]].shape:
                    model_dict[k[7:]] = v  # 7 is to get out of module
            self.model.load_state_dict(model_dict, strict=True)

    def forward(self, image):
        _, _, H, W = image.shape  # [bs,3,h,w]
        results = self.model(image)  # [bs,1,h,w] list len :5

        return results[-1]  # [bs.1,h,w]


class PositionEncodingSine_line(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 1-dimensional sequences
    """

    def __init__(self, d_model, temp_bug_fix=True):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()
        if temp_bug_fix:
            div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() * (-math.log(10000.0) / (d_model // 2)))
        else:  # a buggy implementation (for backward compatability only)
            div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() * (-math.log(10000.0) / d_model // 2))

        self.div_term = div_term[:, None, None]  # [C//4, 1]

    def forward(self, x, pts_int):
        """
        Args:
            x: [bs, C, L]
            pts_int:[bs,L,2]
        """
        device = x.device
        d_model = x.shape[1]
        x_position = pts_int[:, :, 0].unsqueeze(0)
        y_position = pts_int[:, :, 1].unsqueeze(0)
        self.div_term = self.div_term.to(device)
        pe = torch.zeros((x.shape[0], d_model, x.shape[2]), device=device)
        pe[:, 0::4, :] = torch.sin(x_position * self.div_term).permute((1, 0, 2))
        pe[:, 1::4, :] = torch.cos(x_position * self.div_term).permute((1, 0, 2))
        pe[:, 2::4, :] = torch.sin(y_position * self.div_term).permute((1, 0, 2))
        pe[:, 3::4, :] = torch.cos(y_position * self.div_term).permute((1, 0, 2))
        return x + pe


class RoFormerPositionEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    @staticmethod
    def embed_rotary(x, cos, sin):
        '''

        :param x: [bs, N, d]
        :param cos: [bs, N, d]
        :param sin: [bs, N, d]
        :return:
        '''
        x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x).contiguous()
        x = x * cos + x2 * sin

        # x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x).contiguous()
        # x = x * cos + x2 * sin

        return x

    @staticmethod
    def embed_pos(x, pe):
        """
        conbline feature and position code
        :param x: [bs, N, d]
        :param pe: [bs, N, d,2]
        :return:
        """
        # ... 表示省略前面所有的维度
        return RoFormerPositionEncoding.embed_rotaty(x, pe[..., 0], pe[..., 1])

    def forward(self, pts_int):
        '''
        @param XYZ: [B,N,2]
        @return:[B,N,dim,2]
        '''
        bsize, npoint, _ = pts_int.shape

        x_position, y_position = pts_int[..., 0:1], pts_int[..., 1:2]

        div_term = torch.exp(torch.arange(0, self.d_model // 2, 2, dtype=torch.float, device=pts_int.device) * (
                -math.log(10000.0) / (self.d_model // 2)))
        div_term = div_term.view(1, 1, -1)  # [1, 1, d//4]

        sinx = torch.sin(x_position * div_term)  # [B, N, d//4]
        cosx = torch.cos(x_position * div_term)
        siny = torch.sin(y_position * div_term)
        cosy = torch.cos(y_position * div_term)

        # sin/cos [θ0,θ1,θ2......θd/4-1] -> sin/cos [θ0,θ0,θ1,θ1,θ2,θ2......θd/4-1,θd/4-1]
        sinx, cosx, siny, cosy = map(
            lambda feat: torch.stack([feat, feat], dim=-1).view(bsize, npoint, -1),
            [sinx, cosx, siny, cosy])
        sin_pos = torch.cat([sinx, siny], dim=-1)
        cos_pos = torch.cat([cosx, cosy], dim=-1)
        position_code = torch.stack([cos_pos, sin_pos], dim=-1)

        if position_code.requires_grad:
            position_code = position_code.detach()

        return position_code


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class Rotary_LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(Rotary_LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead
        self.eps = 1e-6
        self.feature_map = elu_feature_map

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_pe, source_pe, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source
        qp, kvp = x_pe, source_pe
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        Q = self.feature_map(query)
        K = self.feature_map(key)

        if qp is not None:
            q_cos, q_sin = qp[..., 0], qp[..., 1]
            k_cos, k_sin = kvp[..., 0], kvp[..., 1]
            Q_pos = RoFormerPositionEncoding.embed_rotary(query, q_cos, q_sin)
            K_pos = RoFormerPositionEncoding.embed_rotary(key, k_cos, k_sin)

        # multi-head attention
        Q = Q.view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        K = K.view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]

        Q_pos = Q_pos.view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        K_pos = K_pos.view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]

        value = value.view(bs, -1, self.nhead, self.dim)
        # set padded position to zero
        q_mask, kv_mask = x_mask, source_mask
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
            Q_pos = Q_pos * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            K_pos = K_pos * kv_mask[:, :, None, None]
            value = value * kv_mask[:, :, None, None]

        v_length = value.size(1)
        values = value / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K_pos, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = (torch.einsum("nlhd,nhdv,nlh->nlhv", Q_pos, KV, Z) * v_length)

        message = queried_values.contiguous()
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(LocalFeatureTransformer, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        self.attention = config['attention']
        if self.attention == 'full':
            encoder_layer = LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'])
        else:
            encoder_layer = Rotary_LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'])
        # geo_encoder_layer = Geo_LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'])
        attention_module_list = []
        for i in range(len(self.layer_names)):
            if self.layer_names[i] == 'Geoself':
                pass
                # attention_module_list.append(copy.deepcopy(geo_encoder_layer))
            else:
                attention_module_list.append(copy.deepcopy(encoder_layer))
        self.layers = nn.ModuleList(attention_module_list)
        # self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, pos_encoding_0, pos_encoding_1, mask0=None, mask1=None, only_self=False):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                if only_self:
                    feat0 = layer(feat0, feat0, pos_encoding_0, pos_encoding_0, mask0, mask0)
                    pass
                feat0 = layer(feat0, feat0, pos_encoding_0, pos_encoding_0, mask0, mask0)
                feat1 = layer(feat1, feat1, pos_encoding_1, pos_encoding_1, mask1, mask1)
                pass
            elif name == 'cross':
                feat0 = layer(feat0, feat1, pos_encoding_0, pos_encoding_1, mask0, mask1)
                feat1 = layer(feat1, feat0, pos_encoding_1, pos_encoding_0, mask1, mask0)
            else:
                raise KeyError
        if only_self:
            return feat0
        return feat0, feat1


def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


class CoarseMatching(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # general config
        self.thr = config['thr']
        self.out_num_points = config['out_num_points']
        self.border_rm = config['border_rm']
        # -- # for trainig fine-level LoFTR
        self.train_coarse_percent = config['train_coarse_percent']
        self.train_pad_num_gt_min = config['train_pad_num_gt_min']
        self.training_stage = config['train_stage']
        # we provide 2 options for differentiable matching
        self.match_type = config['match_type']
        if self.match_type == 'dual_softmax':
            self.temperature = config['dsmax_temperature']
        elif self.match_type == 'sinkhorn':
            # try:
            #     from superglue.superglue import log_optimal_transport
            # except ImportError:
            #     raise ImportError("download superglue.py first!")
            self.log_optimal_transport = log_optimal_transport
            self.bin_score = nn.Parameter(
                torch.tensor(config['skh_init_bin_score'], requires_grad=True))
            self.skh_iters = config['skh_iters']
            self.skh_prefilter = config['skh_prefilter']
        else:
            raise NotImplementedError()

    def forward(self, feat_c0, feat_c1, data, mask_c0=None, mask_c1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            data (dict)
            mask_c0 (torch.Tensor): [N, L] (optional)
            mask_c1 (torch.Tensor): [N, S] (optional)
        Update:
            data (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
            NOTE: M' != M during training.
        """
        N, L, S, C = feat_c0.size(0), feat_c0.size(1), feat_c1.size(1), feat_c0.size(2)

        # normalize
        feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1] ** .5, [feat_c0, feat_c1])

        if self.match_type == 'dual_softmax':
            sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0,
                                      feat_c1) / self.temperature
            if mask_c1 is not None:
                sim_matrix.masked_fill_(
                    ~(mask_c0[..., None] * mask_c1[:, None]).bool(),
                    -INF)

            conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)  # dim is same


        elif self.match_type == 'sinkhorn':
            # sinkhorn, dustbin included
            sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1)
            if mask_c1 is not None:
                # mask = mask_c1[:, None].expand(mask_c1.shape[0], feat_c0.shape[1], mask_c1.shape[1])  # N,1,S -> N, L,S
                # sim_matrix[:, :L, :S].masked_fill_(
                #     ~(mask).bool(),
                #     -INF)
                sim_matrix[:, :L, :S].masked_fill_(
                    ~(mask_c0[..., None] * mask_c1[:, None]).bool(),
                    -INF)

            # build uniform prior & use sinkhorn
            log_assign_matrix = self.log_optimal_transport(
                sim_matrix, self.bin_score, self.skh_iters)
            assign_matrix = log_assign_matrix.exp()
            conf_matrix = assign_matrix[:, :-1, :-1]

            # filter prediction with dustbin score (only in evaluation mode)
            if not self.training and self.skh_prefilter:
                filter0 = (assign_matrix.max(dim=2)[1] == S)[:, :-1]  # [N, L]
                filter1 = (assign_matrix.max(dim=1)[1] == L)[:, :-1]  # [N, S]
                conf_matrix[filter0[..., None].repeat(1, 1, S)] = 0
                conf_matrix[filter1[:, None].repeat(1, L, 1)] = 0

            if self.config['sparse_spvs']:
                data.update({'conf_matrix_with_bin': assign_matrix.clone()})

        # predict coarse matches from conf_matrix

        data.update({'conf_matrix': conf_matrix})
        data.update(**self.get_coarse_match(conf_matrix, data, mask_c1))

    # @torch.no_grad()
    def get_coarse_match(self, conf_matrix, data, mask_c1=None):
        """
        Args:
            conf_matrix (torch.Tensor): [N, L, S]
            data (dict): with keys ['hw0_i', 'hw1_i', 'hw0_c', 'hw1_c']
        Returns:
            coarse_matches (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'm_bids' (torch.Tensor): [M],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
        """
        scale = data['resolution'][0]
        h0c = torch.div(data['hw0'][0], scale, rounding_mode="floor")
        w0c = torch.div(data['hw0'][1], scale, rounding_mode="floor")
        h1c = torch.div(data['hw1'][0], scale, rounding_mode="floor")
        w1c = torch.div(data['hw1'][1], scale, rounding_mode="floor")
        _device = conf_matrix.device
        # 1. confidence thresholding
        mask = conf_matrix > self.thr  # N,L,S
        # 2. mutual nearest
        mask = mask \
               * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0]) \
               * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])
        # 3. find all valid coarse matches
        # this only works when at most one `True` in each row
        mask_v, all_j_ids = mask.max(dim=2)  # mask_v: [N,L]
        num_matches_pred = torch.sum(mask_v, dim=1).int()
        # truncated_num = torch.min(num_matches_pred).long()
        # num_bs = num_matches_pred.tolist()
        b_ids, i_ids = torch.where(mask_v)  # i_ids : [N*L]
        j_ids = all_j_ids[b_ids, i_ids]  # [n]
        mconf = conf_matrix[b_ids, i_ids, j_ids]

        if self.training and self.training_stage == 'whole':
            # the more predict,the less padding
            # num_cal = torch.clip(self.out_num_points-num_matches_pred*0.2, min=4, max=self.out_num_points)
            # only pad 4 point (in the case no enough points to calculate)
            num_to_padding = (torch.zeros(num_matches_pred.shape[0], device=_device) + 4).long()
            # num_to_padding = torch.clip(self.out_num_points*0.8 - num_matches_pred, min=0)
            low = 0
            high = len(torch.where(data['spv_b_ids'] == 0)[0])
            mconf_gt = torch.zeros(len(data['spv_b_ids']), device=_device)  # set conf of gt paddings to all zero
            for i in range(len(num_to_padding)):
                if high > low:
                    gt_pad_indices = torch.randint(
                        low,
                        high,
                        (int(num_to_padding[i]),), device=_device)
                    b_ids = torch.cat([b_ids, data['spv_b_ids'][gt_pad_indices]], dim=0)
                    i_ids = torch.cat([i_ids, data['spv_i_ids'][gt_pad_indices]], dim=0)
                    j_ids = torch.cat([j_ids, data['spv_j_ids'][gt_pad_indices]], dim=0)
                    mconf = torch.cat([mconf, mconf_gt[gt_pad_indices]], dim=0)
                low = high
                high += len(torch.where(data['spv_b_ids'] == i + 1)[0])
            sorted, indices = torch.sort(b_ids)
            b_ids = b_ids[indices]
            i_ids = i_ids[indices]
            j_ids = j_ids[indices]
            mconf = mconf[indices]

        # These matches select patches that feed into fine-level network
        coarse_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids}

        # 4. Update with matches in to original coordinate
        # data['pts_0'] :[N,L,2]

        # scale = scale * data['scale'][b_ids] if 'scale' in data else scale  # image 0 and image 1 has the same scale (original scale)

        mkpts0_c = data['pts_0'][b_ids, i_ids] * scale  # [n,2]
        mkpts1_c = torch.stack(
            [j_ids % w1c, torch.div(j_ids, w1c, rounding_mode="floor")],  # j_ids // data['hw1_c'][1]
            dim=1) * scale  # [n,2]

        # for template,get it's real i_ids
        i_ids_fine = data['pts_0'][b_ids, i_ids][:, 1] * w0c + data['pts_0'][b_ids, i_ids][:, 0]

        # for fine stage
        # b_ids_fine, i_ids_fine = torch.where(data['mask_fine'])  # i_ids : [N*L]
        # i_ids_fine_offset = data['f_points'][b_ids_fine, i_ids_fine][:, 1]*data['hw0_f'][1] + data['f_points'][b_ids_fine, i_ids_fine][:,0]

        # These matches is the current prediction (for visualization)
        coarse_matches.update({
            'predict_mask': mconf != 0,  # mconf == 0 => gt matches
            # 'b_ids': b_ids[mconf != 0],  # if only train the coarse-stage ,use this instead !!!! but only affect training plot
            'mkpts0_c': mkpts0_c[mconf != 0],
            'mkpts1_c': mkpts1_c[mconf != 0],
            'mconf': mconf[mconf != 0],
            'mkpts0_c_pad': mkpts0_c,
            'mkpts1_c_pad': mkpts1_c,
            'mconf_pad': mconf,
            'i_ids_fine': i_ids_fine.long(),
            # 'i_ids_fine_offset': i_ids_fine_offset.long(),
            'i_ids_fine_x': data['pts_0'][b_ids, i_ids][:, 0].long(),
            'i_ids_fine_y': data['pts_0'][b_ids, i_ids][:, 1].long()
            # 'num_each': num_matches_pred+num_to_padding
        })
        if self.training_stage == 'only_coarse':
            coarse_matches.update({'b_ids': b_ids[mconf != 0]})

        return coarse_matches


class FinalTrans(nn.Module):
    def __init__(self, config):
        super(FinalTrans, self).__init__()
        self.sigma_spat = config['trans']['sigma_d']
        self.angel_k = config['trans']['angel_k']
        self.num_iterations = config['trans']['num_iterations']
        self.sigma_angle = config['trans']['sigma_a']
        self.inlier_threshold = config['trans']['inlier_threshold']
        self.gama = config['trans']['gama']

    @torch.no_grad()
    def forward(self, data):

        all_bs = data['pts_0'].shape[0]
        for b_id in range(all_bs):
            b_mask = data['m_bids'] == b_id
            src_keypts = data['mkpts0_f'][b_mask].unsqueeze(0).to(torch.float32)  # [bs, num_corr, 2]
            tgt_keypts = data['mkpts1_f'][b_mask].unsqueeze(0).to(torch.float32)  # [bs, num_corr, 2]
            final_trans, final_labels = self.cal_trans(src_keypts, tgt_keypts)
            final_trans = self.post_refinement(final_trans, src_keypts, tgt_keypts)
            # print('final transformation:\n', final_trans)
            # update
            warped_src_keypts = self.transform(src_keypts, final_trans)
            data['mkpts1_f'][b_mask] = warped_src_keypts.squeeze(0)

    def cal_trans(self, src_keypts, tgt_keypts):
        corr_pos = torch.cat((src_keypts, tgt_keypts), dim=-1)  # [bs, num_corr, 4]
        bs, num_corr = corr_pos.shape[0], corr_pos.shape[1]
        k = num_corr

        #################################
        # construct the spatial consistency matrix
        #################################
        src_dist = torch.norm((src_keypts[:, :, None, :] - src_keypts[:, None, :, :]), dim=-1)
        corr_compatibility = src_dist - torch.norm((tgt_keypts[:, :, None, :] - tgt_keypts[:, None, :, :]),
                                                   dim=-1)
        corr_compatibility = torch.clamp(1.0 - corr_compatibility ** 2 / self.sigma_spat ** 2, min=0)  # bs,L,l

        #################################
        # Power iteratation to get the inlier probability
        #################################
        corr_compatibility[:, torch.arange(corr_compatibility.shape[1]), torch.arange(corr_compatibility.shape[1])] = 0

        total_weight = self.cal_leading_eigenvector(corr_compatibility, method='power')

        total_weight = total_weight.view([bs, k])
        total_weight = total_weight / (torch.sum(total_weight, dim=-1, keepdim=True) + 1e-6)
        #################################
        # calculate the transformation by weighted least-squares for each subsets in parallel
        #################################
        total_weight = total_weight.view([-1, k])

        src_knn, tgt_knn = src_keypts.view([-1, k, 2]), tgt_keypts.view([-1, k, 2])
        seed_as_center = False

        if seed_as_center:
            assert ("Not codes!")
        else:
            # not use seeds as neighborhood centers.
            seedwise_trans = self.rigid_transform_2d(src_knn, tgt_knn, total_weight)
            seedwise_trans = seedwise_trans.view([bs, 3, 3])

        #################################
        # calculate the inlier number for each hypothesis, and find the best transformation for each point cloud pair
        #################################
        pred_position = torch.einsum('bnm,bmk->bnk', seedwise_trans[:, :2, :2],
                                     src_keypts.permute(0, 2, 1)) + seedwise_trans[:, :2, 2:3]  # [bs, num_corr, 3]
        pred_position = pred_position.permute(0, 2, 1)
        L2_dis = torch.norm(pred_position - tgt_keypts[:, :, :], dim=-1)  # [bs, num_corr]
        seedwise_fitness = torch.mean((L2_dis < self.inlier_threshold).float(), dim=-1)  # [bs, 1]
        # seedwise_inlier_rmse = torch.sum(L2_dis * (L2_dis < config.inlier_threshold).float(), dim=1)
        batch_best_guess = seedwise_fitness

        # refine the pose by using all the inlier correspondences (done in the post-refinement step)
        final_trans = seedwise_trans
        final_labels = (L2_dis < self.inlier_threshold).float()

        return final_trans, final_labels

    def cal_trans_weight(self, src_keypts, tgt_keypts, weight):
        # weight:[bs,L]
        corr_pos = torch.cat((src_keypts, tgt_keypts), dim=-1)  # [bs, num_corr, 4]
        bs, num_corr = corr_pos.shape[0], corr_pos.shape[1]
        k = num_corr

        #################################
        # construct the spatial consistency matrix
        #################################
        src_dist = torch.norm((src_keypts[:, :, None, :] - src_keypts[:, None, :, :]), dim=-1)
        corr_compatibility = src_dist - torch.norm((tgt_keypts[:, :, None, :] - tgt_keypts[:, None, :, :]),
                                                   dim=-1)
        corr_compatibility = torch.clamp(1.0 - corr_compatibility ** 2 / self.sigma_spat ** 2, min=0)  # bs,L,l

        #################################
        # Power iteratation to get the inlier probability
        #################################
        corr_compatibility[:, torch.arange(corr_compatibility.shape[1]), torch.arange(corr_compatibility.shape[1])] = 0

        total_weight = self.cal_leading_eigenvector(corr_compatibility, method='power')

        total_weight = total_weight.view([bs, k]) * weight
        total_weight = total_weight / (torch.sum(total_weight, dim=-1, keepdim=True) + 1e-6)
        #################################
        # calculate the transformation by weighted least-squares for each subsets in parallel
        #################################
        total_weight = total_weight.view([-1, k])

        src_knn, tgt_knn = src_keypts.view([-1, k, 2]), tgt_keypts.view([-1, k, 2])
        seed_as_center = False

        if seed_as_center:
            assert ("Not codes!")
        else:
            # not use seeds as neighborhood centers.
            seedwise_trans = self.rigid_transform_2d(src_knn, tgt_knn, total_weight)
            seedwise_trans = seedwise_trans.view([bs, 3, 3])

        #################################
        # calculate the inlier number for each hypothesis, and find the best transformation for each point cloud pair
        #################################
        pred_position = torch.einsum('bnm,bmk->bnk', seedwise_trans[:, :2, :2],
                                     src_keypts.permute(0, 2, 1)) + seedwise_trans[:, :2, 2:3]  # [bs, num_corr, 3]
        pred_position_z = torch.einsum('bnm,bmk->bnk', seedwise_trans[:, 2:3, :2],
                                       src_keypts.permute(0, 2, 1)) + seedwise_trans[:, 2:3, 2:3]  # [bs, num_corr, 3]
        pred_position = pred_position / pred_position_z

        pred_position = pred_position.permute(0, 2, 1)
        L2_dis = torch.norm(pred_position - tgt_keypts[:, :, :], dim=-1)  # [bs, num_corr]
        seedwise_fitness = torch.mean((L2_dis < self.inlier_threshold).float(), dim=-1)  # [bs, 1]
        # seedwise_inlier_rmse = torch.sum(L2_dis * (L2_dis < config.inlier_threshold).float(), dim=1)
        batch_best_guess = seedwise_fitness

        # refine the pose by using all the inlier correspondences (done in the post-refinement step)
        final_trans = seedwise_trans
        final_labels = (L2_dis < self.inlier_threshold).float()

        return final_trans, final_labels

    def cal_angel(self, points, angel_k=3):
        r"""

        :param points: (B,N,2)
        :param angel_k: number of nearest neighbors
        :return: angle: the angel of each correspondence (B,N,N)
        """
        batch_size = points.shape[0]
        num_point = points.shape[1]
        points_dist = torch.norm((points[:, :, None, :] - points[:, None, :, :]), dim=-1)
        knn_indices = points_dist.topk(k=angel_k + 1, dim=2, largest=False)[1][:, :,
                      1:]  # (B,N,k)  k+1 : get out the itself indice
        # print(knn_indices)
        knn_indices = knn_indices.unsqueeze(3).expand(batch_size, num_point, angel_k, 2)  # (B,N,K,2)
        expanded_points = points.unsqueeze(1).expand(batch_size, num_point, num_point, 2)  # (B,N,K,2)
        knn_points = torch.gather(expanded_points, dim=2, index=knn_indices)  # (B,N,k,2)

        ref_vectors = knn_points - points.unsqueeze(2)  # (B, N, k, 2)
        anc_vectors = points.unsqueeze(1) - points.unsqueeze(2)  # (B, N, N, 2)
        ref_vectors = ref_vectors.unsqueeze(2).expand(batch_size, num_point, num_point, angel_k, 2)  # (B, N, N, k, 2)
        anc_vectors = anc_vectors.unsqueeze(3).expand(batch_size, num_point, num_point, angel_k, 2)  # (B, N, N, k, 2)
        zeros = torch.zeros((batch_size, num_point, num_point, angel_k, 1), device=ref_vectors.device)
        ref_vectors = torch.cat((ref_vectors, zeros), dim=-1)
        anc_vectors = torch.cat((anc_vectors, zeros), dim=-1)
        sin_values = torch.linalg.norm(torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)  # (B, N, N, k)
        cos_values = torch.sum(ref_vectors * anc_vectors, dim=-1)  # (B, N, N, k)
        angles = torch.atan2(sin_values, cos_values)  # (B, N, N, k)
        angle, _ = torch.max(angles, dim=-1)
        return angle

    def cal_trans_homo(self, src_keypts, tgt_keypts, weight, is_training=False):
        eps = 1e-6
        # weight:[bs,L]
        corr_pos = torch.cat((src_keypts, tgt_keypts), dim=-1)  # [bs, num_corr, 4]
        bs, num_corr = corr_pos.shape[0], corr_pos.shape[1]
        k = num_corr

        #################################
        # construct the spatial consistency matrix
        #################################

        # normalized-distance consistency matrix
        src_dist = torch.norm((src_keypts[:, :, None, :] - src_keypts[:, None, :, :]), dim=-1)
        tgt_dist = torch.norm((tgt_keypts[:, :, None, :] - tgt_keypts[:, None, :, :]), dim=-1)
        src_dist = F.normalize(src_dist, dim=-1)  # deal scale-change case
        tgt_dist = F.normalize(tgt_dist, dim=-1)  # deal scale-change case
        dis_compatibility = (src_dist + eps) / (tgt_dist + eps)
        dis_compatibility = torch.clamp(1 - (dis_compatibility - 1) ** 2 / (self.sigma_spat ** 2), min=0)  # bs,L,l

        # normalized-distance consistency matrix
        src_angle = self.cal_angel(src_keypts, angel_k=self.angel_k)
        tgt_angle = self.cal_angel(tgt_keypts, angel_k=self.angel_k)
        angle_compatibility = torch.abs(src_angle - tgt_angle)
        angle_compatibility = torch.clamp(1 - (angle_compatibility) ** 2 / (self.sigma_angle ** 2), min=0)  # bs,L,l
        corr_compatibility = (1 - self.gama) * angle_compatibility + self.gama * dis_compatibility

        #################################
        # Power iteratation to get the inlier probability
        #################################
        corr_compatibility[:, torch.arange(corr_compatibility.shape[1]), torch.arange(corr_compatibility.shape[1])] = 0

        total_weight = self.cal_leading_eigenvector(corr_compatibility, method='power')

        total_weight = total_weight.view([bs, k]) * weight
        total_weight = total_weight / (torch.sum(total_weight, dim=-1, keepdim=True) + eps)
        #################################
        # calculate the transformation by weighted least-squares for each subsets in parallel
        #################################
        total_weight = total_weight.view([-1, k])

        # src_knn, tgt_knn = src_keypts.view([-1, k, 2]), tgt_keypts.view([-1, k, 2])
        seed_as_center = False

        if seed_as_center:
            assert ("Not codes!")
        else:
            # not use seeds as neighborhood centers.
            # src_knn = (src_keypts/ wh) * 2 - 1
            # tgt_knn = (tgt_keypts/ wh) * 2 - 1
            seedwise_trans = kornia.geometry.homography.find_homography_dlt(src_keypts, tgt_keypts, total_weight)
        return seedwise_trans, None
        # return self.homo_refinement(seedwise_trans,src_keypts,tgt_keypts,total_weight),None
        #################################

        # calculate the inlier number for each hypothesis, and find the best transformation for each point cloud pair
        #################################
        pred_position = torch.einsum('bnm,bmk->bnk', seedwise_trans[:, :2, :2],
                                     src_keypts.permute(0, 2, 1)) + seedwise_trans[:, :2, 2:3]  # [bs, num_corr, 3]

        pred_position_z = torch.einsum('bnm,bmk->bnk', seedwise_trans[:, 2:3, :2],
                                       src_keypts.permute(0, 2, 1)) + seedwise_trans[:, 2:3, 2:3]  # [bs, num_corr, 3]

        pred_position = pred_position / pred_position_z
        pred_position = pred_position.permute(0, 2, 1)

        L2_dis = torch.norm(pred_position - tgt_keypts[:, :, :], dim=-1)  # [bs, num_corr]
        seedwise_fitness = torch.mean((L2_dis < self.inlier_threshold).float(), dim=-1)  # [bs, 1]
        # seedwise_inlier_rmse = torch.sum(L2_dis * (L2_dis < config.inlier_threshold).float(), dim=1)
        batch_best_guess = seedwise_fitness

        # refine the pose by using all the inlier correspondences (done in the post-refinement step)
        final_trans = seedwise_trans
        final_labels = (L2_dis < self.inlier_threshold).float()

        return final_trans, final_labels

    def homo_refinement(self, initial_trans, src_keypts, tgt_keypts, weights=None):
        """
        Perform post refinement using the initial transformation matrix, only adopted during testing.
        Input
            - initial_trans: [bs, 3, 3]
            - src_keypts:    [bs, num_corr, 2]
            - tgt_keypts:    [bs, num_corr, 2]
            - weights:       [bs, num_corr]
        Output:
            - final_trans:   [bs, 3, 3]
        """
        assert initial_trans.shape[0] == 1
        if self.inlier_threshold == 2:  # for 3DMatch
            inlier_threshold_list = [4] * 5
        else:  # for KITTI
            inlier_threshold_list = [4] * 5

        previous_inlier_num = 0
        for inlier_threshold in inlier_threshold_list:

            pred_position = torch.einsum('bnm,bmk->bnk', initial_trans[:, :2, :2],
                                         src_keypts.permute(0, 2, 1)) + initial_trans[:, :2, 2:3]  # [bs, num_corr, 3]

            pred_position_z = torch.einsum('bnm,bmk->bnk', initial_trans[:, 2:3, :2],
                                           src_keypts.permute(0, 2, 1)) + initial_trans[:, 2:3,
                                                                          2:3]  # [bs, num_corr, 3]
            pred_position = pred_position / pred_position_z
            pred_position = pred_position.permute(0, 2, 1)
            L2_dis = torch.norm(pred_position - tgt_keypts[:, :, :], dim=-1)  # [bs, num_corr]

            pred_inlier = (L2_dis < inlier_threshold)[0]  # assume bs = 1
            inlier_num = torch.sum(pred_inlier)
            if abs(int(inlier_num - previous_inlier_num)) < 1:
                # update
                break
            else:
                previous_inlier_num = inlier_num
                initial_trans = kornia.geometry.homography.find_homography_dlt(src_keypts[:, pred_inlier, :],
                                                                               tgt_keypts[:, pred_inlier, :],
                                                                               weights=weights[:, pred_inlier])
        return initial_trans

    def cal_leading_eigenvector(self, M, method='power'):
        """
        Calculate the leading eigenvector using power iteration algorithm or torch.symeig
        Input:
            - M:      [bs, num_corr, num_corr] the compatibility matrix
            - method: select different method for calculating the learding eigenvector.
        Output:
            - solution: [bs, num_corr] leading eigenvector
        """
        if method == 'power':
            # power iteration algorithm
            leading_eig = torch.ones_like(M[:, :, 0:1])
            leading_eig_last = leading_eig
            for i in range(self.num_iterations):
                leading_eig = torch.bmm(M, leading_eig)
                leading_eig = leading_eig / (torch.norm(leading_eig, dim=1, keepdim=True) + 1e-6)
                if torch.allclose(leading_eig, leading_eig_last):
                    break
                leading_eig_last = leading_eig
            leading_eig = leading_eig.squeeze(-1)
            return leading_eig
        elif method == 'eig':  # cause NaN during back-prop
            e, v = torch.symeig(M, eigenvectors=True)
            leading_eig = v[:, :, -1]
            return leading_eig
        else:
            exit(-1)

    def rigid_transform_2d(self, A, B, weights=None, weight_threshold=0):
        """
        Input:
            - A:       [bs, num_corr, 2], source point cloud
            - B:       [bs, num_corr, 2], target point cloud
            - weights: [bs, num_corr]     weight for each correspondence
            - weight_threshold: float,    clips points with weight below threshold
        Output:
            - R, t
        """
        bs = A.shape[0]
        if weights is None:
            weights = torch.ones_like(A[:, :, 0])
        weights[weights < weight_threshold] = 0

        # find mean of point cloud
        centroid_A = torch.sum(A * weights[:, :, None], dim=1, keepdim=True) / (
                torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)
        centroid_B = torch.sum(B * weights[:, :, None], dim=1, keepdim=True) / (
                torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)
        # subtract mean
        Am = A - centroid_A
        Bm = B - centroid_B

        # construct weight covariance matrix
        Weight = torch.diag_embed(weights)
        H = Am.permute(0, 2, 1) @ Weight @ Bm
        U, S, Vt = torch.svd(H.cpu())
        U, S, Vt = U.to(weights.device), S.to(weights.device), Vt.to(weights.device)

        delta_UV = torch.det(Vt @ U.permute(0, 2, 1))
        eye = torch.eye(2)[None, :, :].repeat(bs, 1, 1).to(A.device)
        eye[:, -1, -1] = delta_UV
        R = Vt @ eye @ U.permute(0, 2, 1)
        t = centroid_B.permute(0, 2, 1) - R @ centroid_A.permute(0, 2, 1)
        # print('Estimated R:\n', R)
        # print('Estimated T:\n', t)
        return self.integrate_trans(R, t)

    def integrate_trans(self, R, t):
        """
        Integrate SE2 transformations from R and t, support torch.Tensor and np.ndarry.
        Input
            - R: [2, 2] or [bs, 2, 3], rotation matrix
            - t: [2, 1] or [bs, 2, 1], translation matrix
        Output
            - trans: [3, 3] or [bs, 3, 3], SE2 transformation matrix
        """
        if len(R.shape) == 3:  # batch
            if isinstance(R, torch.Tensor):
                trans = torch.eye(3)[None].repeat(R.shape[0], 1, 1).to(R.device)
            else:
                trans = np.eye(3)[None]
            trans[:, :2, :2] = R
            trans[:, :2, 2:3] = t.view([-1, 2, 1])
        else:
            if isinstance(R, torch.Tensor):
                trans = torch.eye(3).to(R.device)
            else:
                trans = np.eye(3)
            trans[:2, :2] = R
            trans[:2, 2:3] = t
        # print('transformation:\n', trans)
        return trans

    def transform(self, pts, trans):

        if len(pts.shape) == 3:
            trans_pts = trans[:, :2, :2] @ pts.permute(0, 2, 1) + trans[:, :2, 2:3]
            return trans_pts.permute(0, 2, 1)
        else:
            trans_pts = trans[:2, :2] @ pts.T + trans[:2, 2:3]
            return trans_pts.T

    def post_refinement(self, initial_trans, src_keypts, tgt_keypts, weights=None):
        """
        Perform post refinement using the initial transformation matrix, only adopted during testing.
        Input
            - initial_trans: [bs, 3, 3]
            - src_keypts:    [bs, num_corr, 2]
            - tgt_keypts:    [bs, num_corr, 2]
            - weights:       [bs, num_corr]
        Output:
            - final_trans:   [bs, 3, 3]
        """
        assert initial_trans.shape[0] == 1
        if self.inlier_threshold == 5:  # for 3DMatch
            inlier_threshold_list = [5] * 20
        else:  # for KITTI
            inlier_threshold_list = [8] * 20

        previous_inlier_num = 0
        for inlier_threshold in inlier_threshold_list:
            warped_src_keypts = self.transform(src_keypts, initial_trans)
            L2_dis = torch.norm(warped_src_keypts - tgt_keypts, dim=-1)
            pred_inlier = (L2_dis < inlier_threshold)[0]  # assume bs = 1
            inlier_num = torch.sum(pred_inlier)
            if abs(int(inlier_num - previous_inlier_num)) < 1:
                # update
                break
            else:
                previous_inlier_num = inlier_num
            initial_trans = self.rigid_transform_2d(
                A=src_keypts[:, pred_inlier, :],
                B=tgt_keypts[:, pred_inlier, :],
                ## https://link.springer.com/article/10.1007/s10589-014-9643-2
                # weights=None,
                weights=1 / (1 + (L2_dis / inlier_threshold) ** 2)[:, pred_inlier],
                # weights=((1-L2_dis/inlier_threshold)**2)[:, pred_inlier]
            )
        return initial_trans


class FineMatching(nn.Module):
    """FineMatching with s2d paradigm"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.temperature = config['fine']['dsmax_temperature']
        self.subpixel = False
        self.thr = config['fine']['thr']
        self.only_test = False
        self.photometric = self.config['fine']['photometric']

    def forward(self, feat_f0, feat_f1, data):
        """
        Args:
            feat0 (torch.Tensor): [M, WW, C]
            feat1 (torch.Tensor): [M, WW, C]
            mask_f0:[M,WW]
            data (dict)
        Update:
            data (dict):{
                'expec_f' (torch.Tensor): [M, 3],
                'mkpts0_f' (torch.Tensor): [M, 2],
                'mkpts1_f' (torch.Tensor): [M, 2]}
        """
        M, WW, C = feat_f0.shape
        W = int(math.sqrt(WW))
        scale = data['hw0'][0] / data['hw0_f'][0]
        self.M, self.W, self.WW, self.C, self.scale = M, W, WW, C, scale
        device = feat_f0.device
        if M == 0:
            assert self.training == False, "M is always >0, when training, see coarse_matching.py (padding)"
            logging.warning('No matches found in coarse-level.')
            data.update({
                'expec_f': torch.empty(0, 3, device=feat_f0.device),
                'mkpts0_f': data['mkpts0_c'],
                'mkpts1_f': data['mkpts1_c'],
            })
            return
        # 1. dual-softmax

        # normalize
        feat_f0_picked = feat_f0[:, WW // 2, :]
        sim_matrix = torch.einsum('mc,mrc->mr', feat_f0_picked, feat_f1)
        softmax_temp = 1. / C ** .5
        heatmap = torch.softmax(softmax_temp * sim_matrix, dim=1).view(-1, W, W)
        # compute coordinates from heatmap
        coords_normalized = dsnt.spatial_expectation2d(heatmap[None], True)[0]  # [M, 2]
        grid_normalized = create_meshgrid(W, W, True, heatmap.device).reshape(1, -1, 2)  # [1, WW, 2]

        # compute std over <x, y>
        var = torch.sum(grid_normalized ** 2 * heatmap.view(-1, WW, 1), dim=1) - coords_normalized ** 2  # [M, 2]
        std = torch.sum(torch.sqrt(torch.clamp(var, min=1e-10)), -1)  # [M]  clamp needed for numerical stability

        # for fine-level supervision
        data.update({'expec_f': torch.cat([coords_normalized, std.unsqueeze(1)], -1)})
        data.update({'std': std})

        if self.photometric:
            self.get_photometric_loss(coords_normalized, data)
            # self.smooth_loss(coords_normalized, data)

        # compute absolute kpt coords
        self.get_fine_match(coords_normalized, data)

    def get_photometric_loss(self, coords_normed, data):
        device = coords_normed.device
        image0_unfold = data['image0_unfold'].squeeze(-1)
        image1_unfold = data['image1_unfold'].squeeze(-1)
        theta = torch.tensor([
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=torch.float).to(device)
        theta_batch = theta.repeat(coords_normed.shape[0], 1, 1)
        theta_batch[:, 0, 2] = coords_normed[:, 0].clone()
        theta_batch[:, 1, 2] = coords_normed[:, 1].clone()

        image1_unfold = image1_unfold.reshape(-1, 1, self.W, self.W)
        grid = F.affine_grid(theta_batch, image1_unfold.size(), align_corners=True)
        IWarp = F.grid_sample(image1_unfold, grid, align_corners=True)
        IWarp = IWarp.reshape(-1, self.W ** 2)
        # mask_image0
        mask_image0_unfold = image0_unfold > 0  # [0/1] tensor
        loss_fine = ((((image0_unfold - IWarp) ** 2) * mask_image0_unfold).sum(-1) / (self.W ** 2)).mean()
        data.update({'loss_photometric': loss_fine})

    def get_fine_match(self, coords_normed, data):
        # with torch.no_grad():
        W, WW, C, scale = self.W, self.WW, self.C, self.scale
        device = data['mkpts0_c'].device
        mkpts0_f = data['p_src'].float()  # (L,2)
        mkpts1_f = mkpts0_f + ((coords_normed) * (W // 2) * scale)
        data.update({
            "mkpts0_f": mkpts0_f,
            "mkpts1_f": mkpts1_f
        })
        data.update({
            "b_ids": data['new_b_ids']
        })

    def smooth_loss(self, coords_normed, data):
        b, h, w = data['smooth_mask'].shape
        smooth_map = torch.zeros((b, h, w, 2), device=data['smooth_mask'].device)
        smooth_b_ids = data['smooth_b_ids']
        smooth_y_ids = data['smooth_y_ids']
        smooth_x_ids = data['smooth_x_ids']
        smooth_map[smooth_b_ids, smooth_y_ids, smooth_x_ids, :] = coords_normed
        data.update({
            "smooth_map": smooth_map  # [b,W,W,2]
        })


class FinePreprocess(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.scale_fine = config['fine']['resolution'][1]
        self.cat_c_feat = config['fine_concat_coarse_feat']
        self.W = self.config['fine']['fine_window_size']
        self.photometric = self.config['fine']['photometric']
        d_model_c = self.config['coarse']['d_model']
        d_model_f = self.config['fine']['d_model']
        self.d_model_f = d_model_f

        if self.cat_c_feat:
            self.down_proj = nn.Linear(d_model_c, d_model_f, bias=True)
            self.merge_feat = nn.Linear(2 * d_model_f, d_model_f, bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def forward(self, feat_f0, feat_f1, data, feat_c0=None, feat_c1=None):
        """

        :param feat_f0: [bs,d_model_f,h_f,w_f](bs,64,240,320)
        :param feat_f1: [bs,d_model_f,h_f,w_f]
        :param data:
        :param feat_c0: [bs,d_model_f,h,w]
        :param feat_c1: [bs,d_model_f,h,w]
        :return:
        """
        scale = data['resolution'][0]
        w0c = torch.div(data['hw0'][1], scale, rounding_mode="floor")
        W = self.W
        device = feat_f0.device
        sacle_f_div_c = stride = data['hw0_f'][0] // data['hw0_c'][0]
        h, w = data['hw0_f'][0], data['hw0_f'][1]
        data.update({'W': W})

        # crop the features in edge
        if self.cat_c_feat is False:
            f_points = data['f_points']
            mask = data['mask_fine_point']
            # out of bundary
            mask = mask * (f_points[:, :, 0] >= W // 2) * (f_points[:, :, 0] < w - W // 2) * \
                   (f_points[:, :, 1] >= W // 2) * (f_points[:, :, 1] < h - W // 2)
            b_ids, i_ids = torch.where(mask)  # (L)

            if data['b_ids'].shape[0] == 0:  # TODO;no need here
                feat0 = torch.empty(0, self.W ** 2, self.d_model_f, device=feat_f0.device)
                feat1 = torch.empty(0, self.W ** 2, self.d_model_f, device=feat_f0.device)
                data.update({
                    'b_ids_mask_fine': b_ids,
                    'i_ids_mask_fine': i_ids
                })
                return feat0, feat1

            gridX = torch.arange(-W // 2 + 1, W // 2 + 1).view(1, 1, -1).expand(1, W, W).contiguous().reshape(1, -1).to(
                device)  # (1,W*W)
            gridY = torch.arange(-W // 2 + 1, W // 2 + 1).view(1, -1, 1).expand(1, W, W).contiguous().reshape(1, -1).to(
                device)  # (1,WW)

            p_src = torch.stack([f_points[b_ids, i_ids, 0], f_points[b_ids, i_ids, 1]], dim=-1)

            data.update({
                'b_ids_mask_fine': b_ids,
                'i_ids_mask_fine': i_ids,
                'p_src': p_src * self.scale_fine,
                'new_b_ids': b_ids
            })
            y_ids = f_points[b_ids, i_ids, 1][:, None] + gridY  # (L,W*W)
            x_ids = f_points[b_ids, i_ids, 0][:, None] + gridX  # (L,W*W)
            b_ids = b_ids[:, None].repeat(1, W * W)
            feat_f0_unfold = feat_f0[b_ids, :, y_ids, x_ids]  # [L,ww, c_f]
            feat_f1_unfold = feat_f1[b_ids, :, y_ids, x_ids]  # [L,ww, c_f]
            return feat_f0_unfold, feat_f1_unfold

        else:
            template_pool_image0 = F.max_pool2d(data['image0'], kernel_size=2)  # bs,1,w,h
            template_pool = F.unfold(template_pool_image0, kernel_size=(W - 1, W - 1), stride=stride, padding=0)
            template_pool = rearrange(template_pool, 'n (c ww) l -> n l ww c', ww=(W - 1) ** 2)
            b_ids_point, i_ids_point = torch.where(data['mask0'])  # data['mask0'] is the mask of pts_0
            i_ids_point_full = data['pts_0'][b_ids_point, i_ids_point][:, 1] * w0c + data['pts_0'][
                                                                                         b_ids_point, i_ids_point][:, 0]
            template_pool = template_pool[b_ids_point, i_ids_point_full].squeeze(-1)  # [L, ww]
            b_ids_patch, i_ids_patch = torch.where(template_pool > 0)  # (L')
            gridX = torch.arange(0, W - 1).view(1, 1, -1).expand(1, W - 1, W - 1).contiguous().reshape(1, -1).to(
                device)  # (1,W*W)
            gridY = torch.arange(0, W - 1).view(1, -1, 1).expand(1, W - 1, W - 1).contiguous().reshape(1, -1).to(
                device)  # (1,WW)
            f_points_x = data['pts_0'][b_ids_point, i_ids_point][:, 0][:,
                         None] * sacle_f_div_c + gridX  # (L,(W-1)*(W-1))
            f_points_y = data['pts_0'][b_ids_point, i_ids_point][:, 1][:, None] * sacle_f_div_c + gridY

            feat_c0 = feat_c0[b_ids_point, i_ids_point]  # (L,f_c) # out of padding position
            feat_c1 = feat_c1[b_ids_point, i_ids_point]  # (L,f_c)
            feat_c0 = repeat(feat_c0, 'n c -> n ww c', ww=(W - 1) ** 2)
            feat_c1 = repeat(feat_c1, 'n c -> n ww c', ww=(W - 1) ** 2)
            f_points_x = f_points_x[b_ids_patch, i_ids_patch]  # (L') select point on the edge
            f_points_y = f_points_y[b_ids_patch, i_ids_patch]  # (L')

            feat_c0 = feat_c0[b_ids_patch, i_ids_patch]  # (L',f_c)
            feat_c1 = feat_c1[b_ids_patch, i_ids_patch]  # (L',f_c)

            gridX = torch.arange(-W // 2 + 1, W // 2 + 1).view(1, 1, -1).expand(1, W, W).contiguous().reshape(1, -1).to(
                device)  # (1,(W+1)*(W+1))
            gridY = torch.arange(-W // 2 + 1, W // 2 + 1).view(1, -1, 1).expand(1, W, W).contiguous().reshape(1, -1).to(
                device)  # (1,(W+1)*(W+1))

            f_points = torch.stack([f_points_x, f_points_y], dim=-1)

            # if self.photometric:
            #     smooth_mask = torch.zeros(template_pool_image0.shape[0], template_pool_image0.shape[2],
            #                               template_pool_image0.shape[3], dtype=bool, device=device)
            #     smooth_mask[b_ids_point[b_ids_patch], f_points_y, f_points_x] = True  # [bs,h,w]
            #     data.update({
            #         'smooth_mask': smooth_mask,
            #         'smooth_b_ids': b_ids_point[b_ids_patch],
            #         'smooth_y_ids': f_points_y,
            #         'smooth_x_ids': f_points_x
            #     })

            f_points_y = f_points_y[:, None] + gridY  # (L',W*W)
            f_points_x = f_points_x[:, None] + gridX  # (L',W*W)
            b_ids = b_ids_point[b_ids_patch][:, None].repeat(1, W ** 2)
            feat_f0_unfold = feat_f0[b_ids, :, f_points_y, f_points_x]  # [L',ww, c_f]
            feat_f1_unfold = feat_f1[b_ids, :, f_points_y, f_points_x]  # [L',ww, c_f]
            # option: use coarse-level loftr feature as context: concat and linear

            feat_c_win = self.down_proj(torch.cat([feat_c0, feat_c1], 0))  # [2n, c]
            feat_cf_win = self.merge_feat(torch.cat([torch.cat([feat_f0_unfold, feat_f1_unfold], 0),  # [2n, ww, cf]
                                                     repeat(feat_c_win, 'n c -> n ww c', ww=(W) ** 2),  # [2n, ww, cf]
                                                     ], -1))
            feat_f0_unfold, feat_f1_unfold = torch.chunk(feat_cf_win, 2, dim=0)

            if self.photometric:
                template_pool_image1 = F.max_pool2d(data['edge_warped'], kernel_size=2)  # [bs,c,h,w]
                image0_unfold = template_pool_image0[b_ids, :, f_points_y, f_points_x]  # [L',ww, c_f]
                image1_unfold = template_pool_image1[b_ids, :, f_points_y, f_points_x]  # [L',ww, c_f]
                data.update({
                    'image0_unfold': image0_unfold,
                    'image1_unfold': image1_unfold
                })

            data.update({
                'p_src': f_points * self.scale_fine,
                'b_ids_mask_fine': b_ids_patch,
                'i_ids_mask_fine': i_ids_patch,
                'new_b_ids': b_ids_point[b_ids_patch]
            })
            return feat_f0_unfold, feat_f1_unfold


class Tm(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config

        self.nms_dist = self.config['tm']['superpoint']['nms_dist'],
        self.conf_thresh = self.config['tm']['superpoint']['conf_thresh']
        self.stn = self.config['tm']['stn']

        self.W = self.config['tm']['fine']['fine_window_size']
        self.training_stage = self.config['tm']['match_coarse']['train_stage']
        # Modules
        self.coarse_2_stage = self.config['tm']['coarse']['two_stage']
        self.pos_encoding = PositionEncodingSine_line(d_model=config['tm']['superpoint']['block_dims'][-1],
                                                      temp_bug_fix=True)

        self.rotaty_encoding = RoFormerPositionEncoding(d_model=config['tm']['superpoint']['block_dims'][-1])
        self.rotaty_encoding_fine = RoFormerPositionEncoding(d_model=config['tm']['superpoint']['block_dims'][1])
        # self.geo_pos_encoding = GeometryPositionEncodingSine(d_model=config['tm']['superpoint']['block_dims'][-1],
        #                                                                     temp_bug_fix=True)
        self.position = config['tm']['coarse']['position']
        self.cat_c_feat = config['tm']['fine_concat_coarse_feat']
        self.LM_coarse = LocalFeatureTransformer(config['tm']['coarse'])
        self.coarse_matching = CoarseMatching(config['tm']['match_coarse'])
        if self.training_stage == 'whole':
            self.fine_preprocess = FinePreprocess(config['tm'])
            if self.cat_c_feat:
                self.LM_fine = LocalFeatureTransformer(config['tm']['fine_global'])

            self.loftr_fine = LocalFeatureTransformer(config['tm']["fine"])
            self.fine_matching = FineMatching(config['tm'])
        self.cal_trans = FinalTrans(config['tm'])

    def cal_homography(self, data, is_training):
        bs = data['bs']
        h = data['image0'].shape[2]
        w = data['image0'].shape[3]
        theta = []
        theta_inv = []
        data['mconf_pad'][data['mconf_pad'] == 0] = 1
        for b_id in range(bs):
            b_mask = data['b_ids'] == b_id
            # try:
            point_A = data['mkpts0_c_pad'][b_mask].unsqueeze(0).float()
            point_B = data['mkpts1_c_pad'][b_mask].unsqueeze(0).float()
            weights = data['mconf_pad'][b_mask].unsqueeze(0).float()
            try:
                print(len(point_A[0]), len(point_B[0]))
                theta_each, lable = self.cal_trans.cal_trans_homo(point_A, point_B, weights, is_training)
                if torch.linalg.matrix_rank(theta_each) == 3:
                    theta_inv.append(theta_each.inverse())
                else:
                    theta_each = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float,
                                              device=data['image0'].device).view(1, 3, 3)
                    theta_inv.append(theta_each.inverse())
            except:
                # The diagonal element 1 is zero, the inversion could not be completed because the input matrix is singular.
                logging.warning("whole training is impossible:there is an error when calculating the H_c matrix")
                theta_each = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float,
                                          device=data['image0'].device).view(1, 3, 3)
                theta_inv.append(theta_each.inverse())
            theta.append(theta_each)
        try:
            theta_bs = torch.cat(theta, dim=0)
            theta_inv_bs = torch.cat(theta_inv, dim=0)
            warped = kornia.geometry.transform.warp_perspective(data['edge'], theta_inv_bs, dsize=[h, w])
            warped_image0 = kornia.geometry.transform.warp_perspective(data['image0'], theta_bs, dsize=[h, w])
        except:
            logging.warning("seldom warning!")
            warped = data['edge']
            warped_image0 = data['image0']
            theta_bs = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float,
                                    device=data['image0'].device).view(1, 3, 3).repeat(bs, 1, 1)
            theta_inv_bs = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float,
                                        device=data['image0'].device).view(1, 3, 3).repeat(bs, 1, 1)
        return theta_bs, theta_inv_bs, warped, warped_image0

    def cal_newpos(self, position, trans):
        '''
        :param position: bs*N*2
        :param transformation: bs*3*3
        :return:
        '''
        pts_x = trans[:, 0, 0, None] * position[:, :, 0] + trans[:, 0, 1, None] * position[:, :, 1] + trans[:, 0, 2,
                                                                                                      None]
        pts_y = trans[:, 1, 0, None] * position[:, :, 0] + trans[:, 1, 1, None] * position[:, :, 1] + trans[:, 1, 2,
                                                                                                      None]
        pts_z = trans[:, 2, 0, None] * position[:, :, 0] + trans[:, 2, 1, None] * position[:, :, 1] + trans[:, 2, 2,
                                                                                                      None]
        new_position = torch.cat([(pts_x / pts_z)[:, :, None], (pts_y / pts_z)[:, :, None]], dim=-1)
        return new_position

    def forward(self, data, outs_post0, outs_post1, backbone):
        # 3. positoin encoding
        assert (self.position == 'rotary')
        pos_encoding_0 = self.rotaty_encoding(data['pts_0'])  # pos0:[bs,N,2] # pos_encoding_0 [bs,N,dim,2]
        pos_encoding_1 = self.rotaty_encoding(data['pts_1'])  # pos1:[bs,N,2]

        feat_c0 = outs_post0['desc_c']  # bs,l,c
        feat_c1 = outs_post1['desc_c']

        # 4. attention stage
        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0 = data['mask0']  # [bs,L]
        if 'mask1' in data:
            mask_c1 = data['mask1'].flatten(-2)  # [bs,S]

        feat_c0, feat_c1 = self.LM_coarse(feat_c0, feat_c1, pos_encoding_0, pos_encoding_1, mask_c0, mask_c1)

        # 3. match coarse-level, cal_transformation match coarse-level

        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)
        theta, theta_inv, warped, warped_image0 = self.cal_homography(data, self.training)

        data.update({'edge_warped': warped})
        data.update({'warped_template': warped_image0})
        data.update({'theta': theta})
        data.update({'theta_inv': theta_inv})

        if self.training_stage == 'only_coarse':
            data.update({
                "mkpts0_f": data['mkpts0_c'],
                "mkpts1_f": data['mkpts1_c']
            })
            data.update({'trans_predict': data['theta']})
        else:
            h = data['image0'].shape[2]
            w = data['image0'].shape[3]
            feat_f0 = outs_post0['desc_f_2']
            feat_f0 = feat_f0.reshape(feat_f0.shape[0], data['hw0_f'][0], data['hw0_f'][1],
                                      -1).permute(0, 3, 1, 2)
            if self.cat_c_feat:
                feat_c0_stage2 = outs_post0['desc_c']  # bs,l,c
                outs_post1_stage2 = backbone(warped, c_points=data['c_points'])
                feat_c1_stage2 = outs_post1_stage2['desc_c']
                feat_f1 = outs_post1_stage2['desc_f_2']
                mask_c0 = mask_c1 = None  # mask is useful
                if 'mask0' in data:
                    mask_c1 = mask_c0 = data['mask0']  # [bs,L]
                feat_c0, feat_c1 = self.LM_fine(feat_c0_stage2, feat_c1_stage2, pos_encoding_0, pos_encoding_0, mask_c0,
                                                mask_c1)
            else:
                feat_f1 = backbone(warped, choose=False, early_return=True)

            feat_f1 = feat_f1.reshape(feat_f1.shape[0], data['hw1_f'][0], data['hw1_f'][1], -1).permute(0, 3, 1, 2)
            feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, data, feat_c0, feat_c1)  # [b,w*w,c]
            # 4.2 self-cross attention in local windows
            if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
                grid = np.mgrid[:self.W, :self.W]
                grid = grid.reshape((2, -1))
                grid = grid.transpose(1, 0)
                grid = grid[:, [1, 0]]
                pos = torch.tensor(grid).to(data['image0'].device).repeat(feat_f0_unfold.size(0), 1, 1)
                pos_encoding = self.rotaty_encoding_fine(pos)  # pos0:[bs,N,2] # pos_encoding_0 [bs,N,dim,2]
                feat_f0_unfold, feat_f1_unfold = self.loftr_fine(feat_f0_unfold, feat_f1_unfold, pos_encoding,
                                                                 pos_encoding)

            # 5. match fine-level
            self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)
            # 6. plot
            warped_template = []
            trans_bs = []
            mkpts1_f_bs = torch.Tensor().to(feat_f0.device)
            for b_id in range(data['bs']):
                b_mask = data['b_ids'] == b_id
                kpts0 = data['mkpts0_f'][b_mask].unsqueeze(0)
                kpts1 = data['mkpts1_f'][b_mask].unsqueeze(0)

                # weights = data['std'][b_mask].unsqueeze(0) # if mconf_patch is empty , eyes matrix is correct
                # trans, lable = self.cal_trans.cal_trans_homo(kpts0, kpts1, weights)

                theta_inv = data['theta'][b_id]
                pts_x = (theta_inv[0, 0] * kpts1[0, :, 0] + theta_inv[0, 1] * kpts1[0, :, 1] + theta_inv[0, 2])
                pts_y = (theta_inv[1, 0] * kpts1[0, :, 0] + theta_inv[1, 1] * kpts1[0, :, 1] + theta_inv[1, 2])
                pts_z = (theta_inv[2, 0] * kpts1[0, :, 0] + theta_inv[2, 1] * kpts1[0, :, 1] + theta_inv[2, 2])
                pts_x /= pts_z
                pts_y /= pts_z
                kpts1 = torch.stack([pts_x, pts_y], dim=1)
                try:
                    trans = kornia.geometry.homography.find_homography_dlt(kpts0, kpts1.unsqueeze(0))
                    warped_template.append(
                        (kornia.geometry.transform.warp_perspective(data['image0'][b_id].unsqueeze(0),
                                                                    trans, dsize=[h, w])))
                except:
                    logging.warning('seldom: there is an error calculating the fine matches for training padding')
                    trans = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float,
                                         device=data['image0'].device).view(1, 3, 3)
                    warped_template.append(
                        (kornia.geometry.transform.warp_perspective(data['image0'][b_id].unsqueeze(0),
                                                                    trans, dsize=[h, w])))
                trans_bs.append(trans)
                mkpts1_f_bs = torch.cat([mkpts1_f_bs, kpts1], dim=0)

            warped_template_bs = torch.cat(warped_template, dim=0)
            trans_bs = torch.cat(trans_bs, dim=0)

            data.update({'warped_template': warped_template_bs})
            data.update({'trans_predict': trans_bs})  # sourse -> target
            data.update({
                "mkpts0_f": data['mkpts0_f'].detach().cpu(),
                "mkpts1_f": mkpts1_f_bs.detach().cpu()
            })

        plot = False
        if plot:
            from plotting import _make_evaluation_figure, plot_warped, distance_M, eval_predict_homography
            output_path = './result_plot'
            scale = data['scale'][0].detach().cpu()

            mkpts0_f_scaled = data['mkpts0_f'].detach().cpu() * scale
            mkpts1_f_scaled = data['mkpts1_f'].detach().cpu() * scale
            H_pred = \
            kornia.geometry.homography.find_homography_dlt(mkpts0_f_scaled[None, :, :], mkpts1_f_scaled[None, :, :])[
                0].detach().cpu().numpy()

            mkpts0_f = data['mkpts0_f'].numpy()
            mkpts1_f = data['mkpts1_f'].numpy()
            img_name = data['pair_names'][0][0].split('/')[-4] + '_' + os.path.basename(data['pair_names'][0][0])
            trans = data['trans'][0].detach().cpu().numpy()
            mean_dist, correctness = eval_predict_homography(points=data['points_template'][0].detach().cpu().numpy(),
                                                             h_gt=trans,
                                                             H_pred=H_pred)

            keypoints_error = distance_M(mkpts0_f_scaled, mkpts1_f_scaled, trans).numpy()
            _make_evaluation_figure(data['image0_mask'][0].detach().cpu().numpy(),
                                    data['image1'][0][0].detach().cpu().numpy(),
                                    mkpts0_f, mkpts1_f, keypoints_error, mean_dist, 'Ours',
                                    path=output_path + '/' + img_name + '.png')

            plot_warped(data['image0_raw'][0].detach().cpu().numpy(), data['image1_raw'][0].detach().cpu().numpy(),
                        trans, H_pred, path1=output_path + '/' + img_name + '_gt.png',
                        path2=output_path + '/' + img_name + '_es.png')


class SuperPoint_glue(nn.Module):
    """SuperPoint Convolutional Detector and Descriptor

    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629

    """
    default_config = {
        'descriptor_dim': 256,
        'nms_dist': 2,
        'conf_thresh': 0.005,
        'out_num_points': -1,
        'remove_borders': 2,
    }

    def __init__(self, config, early_return=False):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256
        if early_return:
            self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
            self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
            self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
            self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        else:
            self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
            self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
            self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
            self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
            self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
            self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
            self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
            self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
            self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
            self.convDb = nn.Conv2d(
                c5, self.config['block_dims'][-1],
                kernel_size=1, stride=1, padding=0)

        # path = './pretrained/superpoint_v1.pth'
        # self.load_state_dict(torch.load(str(path)))

        mk = self.config['out_num_points']
        if mk == 0 or mk < -1:
            raise ValueError('\"out_num_points\" must be positive or \"-1\"')

    def forward(self, data_image, c_points=None, choose=True, early_return=False):
        """ Compute keypoints, scores, descriptors for image """
        # Shared Encoder
        x = self.relu(self.conv1a(data_image))
        x = self.relu(self.conv1b(x))
        feature_fine_1 = x

        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))

        feature_fine = x
        if early_return:
            descriptors_fine = torch.nn.functional.normalize(feature_fine, p=2, dim=1)  # [bs,C,h,w]
            descriptors_fine = rearrange(descriptors_fine, 'n c h w -> n (h w) c')
            return descriptors_fine
        # visualize_feature_map(data_image,feature_fine,'./save_imgs_plot/')

        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        b, _, h, w = x.shape
        # keypoints = []
        descriptors_coarse = []
        batch_size = data_image.shape[0]
        # H, W = h * 4, w * 4  # 8->2
        # grid = np.mgrid[:H, 0:W]
        # grid = grid.reshape((2, -1))
        # grid = grid.transpose(1, 0)
        # grid = grid[:, [0, 1]]
        # pts_int_b = torch.tensor(grid).to(data_image.device).float()
        if choose:
            assert self.config['out_num_points'] == c_points.shape[1]  # c_points [bs,N,2]
            # for i in range(batch_size):
            #     # tensor [N, 2(x,y)]
            #     keypoints.append(pts_int_b[:, [1, 0]]) # TODO：speed-up
            # Compute the dense descriptors
            cDa = self.relu(self.convDa(x))
            descriptors = self.convDb(cDa)
            descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)  # [ba,C,h,w]【60，80】
            for i in range(batch_size):
                b_x = torch.flatten(c_points[i][:, 0]).long()
                b_y = torch.flatten(c_points[i][:, 1]).long()
                descriptor_coarse = descriptors[i][:, b_y, b_x]  # [C,l]
                descriptors_coarse.append(descriptor_coarse)

            descriptors_fine = torch.nn.functional.normalize(feature_fine, p=2, dim=1)  # [bs,C,h,w]
            descriptors_fine_1 = torch.nn.functional.normalize(feature_fine_1, p=2, dim=1)  # [bs,C,h,w]

            descriptors_fine = rearrange(descriptors_fine, 'n c h w -> n (h w) c')
            descriptors_fine_1 = rearrange(descriptors_fine_1, 'n c h w -> n (h w) c')

            pts_int_c = c_points
            # pts_int_f = torch.stack(keypoints, dim=0)
            descriptors_coarse = torch.stack(descriptors_coarse, dim=0).transpose(1, 2)  # [bs,l,C]

        else:
            keypoints_c = []
            grid_c = np.mgrid[:h, 0:w]
            grid_c = grid_c.reshape((2, -1))
            grid_c = grid_c.transpose(1, 0)
            grid_c = grid_c[:, [0, 1]]
            pts_int_b_c = torch.tensor(grid_c).to(data_image.device).float()

            for i in range(batch_size):
                # keypoints.append(pts_int_b[:, [1, 0]])
                keypoints_c.append(pts_int_b_c[:, [1, 0]])  # tensor [N, 2(x,y)]

            # Compute the dense descriptors
            cDa = self.relu(self.convDa(x))
            descriptors = self.convDb(cDa)
            descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)  # [bs,C,h,w]
            descriptors_coarse = rearrange(descriptors, 'n c h w -> n (h w) c')
            pts_int_c = torch.stack(keypoints_c, dim=0)
            # pts_int_f = torch.stack(keypoints, dim=0)
            descriptors_fine = torch.nn.functional.normalize(feature_fine, p=2, dim=1)  # [bs,C,h,w]
            descriptors_fine_1 = torch.nn.functional.normalize(feature_fine_1, p=2, dim=1)  # [bs,C,h,w]
            descriptors_fine = rearrange(descriptors_fine, 'n c h w -> n (h w) c')
            descriptors_fine_1 = rearrange(descriptors_fine_1, 'n c h w -> n (h w) c')
        return {
            'desc_f_2': descriptors_fine,  # [bs,L,C],
            'desc_f_1': descriptors_fine_1,  # [bs,L,C],
            'desc_c': descriptors_coarse,  # [bs,l,C]
            'pts_int_c': pts_int_c,  # [bs,l,2]
            # 'pts_int_f': pts_int_f # [bs,L,2]
        }


class Backnone(nn.Module):
    def __init__(self, config, edge_ckpt=None):
        super().__init__()
        # Misc
        self.config = config

        self.patch_size = self.config['tm']['superpoint']['patch_size']
        self.nms_dist = self.config['tm']['superpoint']['nms_dist'],
        self.conf_thresh = self.config['tm']['superpoint']['conf_thresh']
        self.backbone_name = self.config['tm']['superpoint']['name']

        self.resolution = self.config['tm']['resolution']

        # Modules
        self.edge_net = Edge_Net(self.config['tm']['edge'], edge_ckpt)

        if self.backbone_name == 'SuperPointNet_gauss2':
            self.backbone = SuperPointNet_gauss2(config['tm']['superpoint'])
        elif self.backbone_name == 'superpoint_glue':
            self.backbone = SuperPoint_glue(config['tm']['superpoint'])
        else:
            raise Exception('please choose the right backbone.')

    def forward(self, data):
        # 1. Local Feature CNN
        """
           Update:
               data (dict): {
                   'image0': (torch.Tensor): (N, 1, H, W)  template
                   'image1': (torch.Tensor): (N, 1, H, W)   image
                   'mask0'(optional) : (torch.Tensor): (N, L) '0' indicates a padded position
                   'mask1'(optional) : (torch.Tensor): (N, H, W)
               }
        """
        data.update({
            'bs': data['image0'].size(0),
            'hw0': data['image0'].shape[2:], 'hw1': data['image1'].shape[2:]
        })
        data.update({
            'hw0_c': (data['image0'].shape[2] // self.resolution[0], data['image0'].shape[3] // self.resolution[0]),
            'hw1_c': (data['image1'].shape[2] // self.resolution[0], data['image1'].shape[3] // self.resolution[0]),
            'hw0_f': (data['image0'].shape[2] // self.resolution[1], data['image0'].shape[3] // self.resolution[1]),
            'hw1_f': (data['image1'].shape[2] // self.resolution[1], data['image1'].shape[3] // self.resolution[1])
        })
        # 2. get image's deep edge map
        # deep_edge = self.edge_net(data['image1_rgb'])  # [N, 1, H, W]
        # data.update({'pidinet_out': deep_edge})
        # mask = data['image1_edge'] > 0.1 # canny mask
        # and_edge = deep_edge * mask
        and_edge = data['image1_rgb'][:, 0, :, :].unsqueeze(1)
        # 2. get image's deep edge map
        data.update({'edge': and_edge})  # deep_edge

        # # save for baseline
        # path_template = data['pair_names'][0][0]
        # path_img = data['pair_names'][1][0]
        # import cv2
        # cv2.imwrite(path_template.replace('template', 'template_edge'),
        #             (data['image0'][0][0] * 255).detach().cpu().numpy())
        # cv2.imwrite(path_img.replace('homo', 'homo_edge'),
        #             (data['edge'][0][0] * 255).detach().cpu().numpy())

        # 2. get superpoint's feature
        outs_post0, outs_post1 = self.backbone(data['image0'], c_points=data['c_points']), self.backbone(data['edge'],
                                                                                                         choose=False)
        pos0 = outs_post0['pts_int_c']
        pos1 = outs_post1['pts_int_c']
        data.update({
            'pts_0': pos0,
            'pts_1': pos1,
        })
        return outs_post0, outs_post1


class TmLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # config under the global namespace
        self.loss_config = config['tm']['loss']
        self.match_type = self.config['tm']['match_coarse']['match_type']
        self.sparse_spvs = self.config['tm']['match_coarse']['sparse_spvs']
        self.train_stage = self.config['tm']['match_coarse']['train_stage']
        # coarse-level
        self.c_pos_w = self.loss_config['pos_weight']
        self.c_neg_w = self.loss_config['neg_weight']
        self.fine_type = self.loss_config['fine_type']
        self.correct_thr = self.loss_config['fine_correct_thr']

    def compute_coarse_loss(self, conf, conf_gt, weight=None):
        """ Point-wise CE / Focal Loss with 0 / 1 confidence as gt.
        Args:
            conf (torch.Tensor): (N, HW0, HW1) / (N, HW0+1, HW1+1)
            conf_gt (torch.Tensor): (N, HW0, HW1)
            weight (torch.Tensor): (N, HW0, HW1)
        """
        pos_mask, neg_mask = conf_gt == 1, conf_gt == 0
        c_pos_w, c_neg_w = self.c_pos_w, self.c_neg_w
        # corner case: no gt coarse-level match at all
        if not pos_mask.any():  # assign a wrong gt
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_neg_w = 0.

        if self.loss_config['coarse_type'] == 'cross_entropy':
            assert not self.sparse_spvs, 'Sparse Supervision for cross-entropy not implemented!'
            conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
            loss_pos = - torch.log(conf[pos_mask])
            loss_neg = - torch.log(1 - conf[neg_mask])
            if weight is not None:
                loss_pos = loss_pos * weight[pos_mask]
                loss_neg = loss_neg * weight[neg_mask]
            return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
        elif self.loss_config['coarse_type'] == 'focal':
            conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
            alpha = self.loss_config['focal_alpha']
            gamma = self.loss_config['focal_gamma']

            if self.sparse_spvs:  # True
                pos_conf = conf[:, :-1, :-1][pos_mask] \
                    if self.match_type == 'sinkhorn' \
                    else conf[pos_mask]

                loss_pos = - alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()
                # calculate losses for negative samples
                if self.match_type == 'sinkhorn':
                    neg0, neg1 = conf_gt.sum(-1) == 0, conf_gt.sum(1) == 0
                    neg_conf = torch.cat([conf[:, :-1, -1][neg0], conf[:, -1, :-1][neg1]], 0)
                    loss_neg = - alpha * torch.pow(1 - neg_conf, gamma) * neg_conf.log()
                else:
                    # These is no dustbin for dual_softmax, so we left unmatchable patches without supervision.
                    # we could also add 'pseudo negtive-samples'
                    pass
                # handle loss weights
                if weight is not None:
                    # Different from dense-spvs, the loss w.r.t. padded regions aren't directly zeroed out,
                    # but only through manually setting corresponding regions in sim_matrix to '-inf'.
                    loss_pos = loss_pos * weight[pos_mask]
                    if self.match_type == 'sinkhorn':
                        neg_w0 = (weight.sum(-1) != 0)[neg0]
                        neg_w1 = (weight.sum(1) != 0)[neg1]
                        neg_mask = torch.cat([neg_w0, neg_w1], 0)
                        loss_neg = loss_neg[neg_mask]

                loss = c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean() \
                    if self.match_type == 'sinkhorn' \
                    else c_pos_w * loss_pos.mean()
                return loss
                # positive and negative elements occupy similar propotions. => more balanced loss weights needed
            else:  # dense supervision (in the case of match_type=='sinkhorn', the dustbin is not supervised.)
                loss_pos = - alpha * torch.pow(1 - conf[pos_mask], gamma) * (conf[pos_mask]).log()
                loss_neg = - alpha * torch.pow(conf[neg_mask], gamma) * (1 - conf[neg_mask]).log()
                if weight is not None:
                    loss_pos = loss_pos * weight[pos_mask]
                    loss_neg = loss_neg * weight[neg_mask]
                return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
                # each negative element occupy a smaller propotion than positive elements. => higher negative loss weight needed
        else:
            raise ValueError('Unknown coarse loss: {type}'.format(type=self.loss_config['coarse_type']))

    def compute_fine_match_loss(self, conf, conf_gt, weight=None):

        """ Point-wise CE / Focal Loss with 0 / 1 confidence as gt.
        Args:
            conf (torch.Tensor): (N, HW0, HW1) / (N, HW0+1, HW1+1)
            conf_gt (torch.Tensor): (N, HW0, HW1)
            weight (torch.Tensor): (N, HW0, HW1)
        """
        pos_mask, neg_mask = conf_gt == 1, conf_gt == 0
        c_pos_w, c_neg_w = self.c_pos_w, self.c_neg_w
        # corner case: no gt coarse-level match at all
        if not pos_mask.any():  # assign a wrong gt
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_neg_w = 0.

        if self.loss_config['coarse_type'] == 'focal':
            conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
            alpha = self.loss_config['focal_alpha']
            gamma = self.loss_config['focal_gamma']

            if self.sparse_spvs:  # True
                pos_conf = conf[pos_mask]
                loss_pos = - alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()
                # handle loss weights
                if weight is not None:
                    # Different from dense-spvs, the loss w.r.t. padded regions aren't directly zeroed out,
                    # but only through manually setting corresponding regions in sim_matrix to '-inf'.
                    loss_pos = loss_pos * weight[pos_mask]

                loss = c_pos_w * loss_pos.mean()
                return loss
                # positive and negative elements occupy similar propotions. => more balanced loss weights needed
            else:  # dense supervision (in the case of match_type=='sinkhorn', the dustbin is not supervised.)
                loss_pos = - alpha * torch.pow(1 - conf[pos_mask], gamma) * (conf[pos_mask]).log()
                loss_neg = - alpha * torch.pow(conf[neg_mask], gamma) * (1 - conf[neg_mask]).log()
                if weight is not None:
                    loss_pos = loss_pos * weight[pos_mask]
                    loss_neg = loss_neg * weight[neg_mask]
                return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
                # each negative element occupy a smaller propotion than positive elements. => higher negative loss weight needed
        else:
            raise ValueError('Unknown coarse loss: {type}'.format(type=self.loss_config['coarse_type']))

    def compute_fine_loss(self, expec_f, expec_f_gt):
        if self.fine_type == 'l2_with_std':
            return self._compute_fine_loss_l2_std(expec_f, expec_f_gt)
        elif self.fine_type == 'l2':
            return self._compute_fine_loss_l2(expec_f, expec_f_gt)
        else:
            raise NotImplementedError()

    def _compute_fine_loss_l2(self, expec_f, expec_f_gt):
        """
        Args:
            expec_f (torch.Tensor): [M, 2] <x, y>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
        """
        correct_mask = torch.linalg.norm(expec_f_gt, ord=float('inf'), dim=1) < self.correct_thr

        if correct_mask.sum() == 0:
            if self.training:  # this seldomly happen when training, since we pad prediction with gt
                logger.warning("assign a false supervision to avoid ddp deadlock")
                correct_mask[0] = True
            else:
                return None
        offset_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask]) ** 2).sum(-1)
        return offset_l2.mean()

    def _compute_fine_loss_l2_std(self, expec_f, expec_f_gt):
        """
        Args:
            expec_f (torch.Tensor): [M, 3] <x, y, std>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
        """
        # correct_mask tells you which pair to compute fine-loss
        correct_mask = torch.linalg.norm(expec_f_gt, ord=float('inf'),
                                         dim=1) < self.correct_thr  # norm for matrices: max(sum(abs(x), dim=1))

        # use std as weight that measures uncertainty
        std = expec_f[:, 2]
        inverse_std = 1. / torch.clamp(std, min=1e-10)
        weight = (inverse_std / torch.mean(inverse_std)).detach()  # avoid minizing loss through increase std

        # corner case: no correct coarse match found
        if (not correct_mask.any()) or (expec_f.shape[0] == 0):
            # if self.training:  # this seldomly happen during training, since we pad prediction with gt
            # sometimes there is not coarse-level gt at all.
            # training and validation
            logger.warning("seldomly: assign a false supervision to avoid ddp deadlock,only the beginning of training")
            expec_f_gt = torch.tensor([[1, 1]], device=expec_f_gt.device)
            expec_f = torch.tensor([[1, 1, 1]], device=expec_f.device)
            correct_mask = torch.tensor([True], device=correct_mask.device)
            weight = torch.tensor([0.], device=weight.device)  #

        # l2 loss with std
        offset_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask, :2]) ** 2).sum(-1)
        loss = (offset_l2 * weight[correct_mask]).mean()

        return loss

    @torch.no_grad()
    def compute_c_weight(self, data):
        """ compute element-wise weights for computing coarse-level loss. """
        if 'mask1' in data:
            # c_weight = data['mask1'].flatten(-2)[:, None].expand(data['mask1'].shape[0], data['conf_matrix'].shape[1], data['mask1'].flatten(-2).shape[1]).float()  # N,1,S -> N, L,S

            c_weight = (data['mask0'][..., None] * data['mask1'].flatten(-2)[:, None]).float()
        else:
            c_weight = None
        return c_weight

    def chamfer_loss_unidirectional(self, points_src, points_tgt, weight=None):
        ''' Computes minimal distances of each point in points_src to points_tgt.

        Args:
            weight:[bs,L]
            points_src (torch tensor): source points  [bs,L,2]
            normals_src (torch tensor): source normals
            points_tgt (torch tensor): target points [bs,L,2]
            normals_tgt (torch tensor): target normals
        '''
        if weight is not None:
            # TODO: no support parreller now ,may there is a bug when coordinate is same and value is large ,distance is still 0
            # TODO: find out why image is empty,this is strange !
            weight = (1 + (torch.sigmoid(1 / weight - 1) - 0.5) * 5)
            dist_matrix = ((points_src.unsqueeze(2) - points_tgt.unsqueeze(1)) ** 2).sum(-1)  # [bs, L,S]
            dist_matrix = dist_matrix * (weight.unsqueeze(-1))
            dist_complete = (dist_matrix.min(-1)[0]).mean(-1)
            dist_acc = (dist_matrix.min(-2)[0]).mean(-1)
            dist = ((dist_acc + dist_complete) / 2).mean()
        else:
            dist_matrix = ((points_src.unsqueeze(2) - points_tgt.unsqueeze(1)) ** 2).sum(-1)
            dist_complete = (dist_matrix.min(-1)[0]).mean(-1)
            # dist_acc = (dist_matrix.min(-2)[0]).mean(-1)
            # dist = ((dist_acc + dist_complete) / 2).mean()
        return dist_complete.mean()

    def cross_entropy_loss_RCF(self, prediction, labelf, beta):

        # from PIL import Image
        # import matplotlib.pyplot as plt
        # from src.utils.utils import toNumpy
        # result = torch.squeeze(prediction[0]*255).detach().cpu().numpy() # H,W
        # plt.imshow(result,cmap='gray')
        # plt.show()
        #
        # result = torch.squeeze(labelf[0] * 255).detach().cpu().numpy()  # H,W
        # plt.imshow(result, cmap='gray')
        # plt.show()
        # import matplotlib.pyplot as plt
        # result = torch.squeeze(prediction[0] * 255).detach().cpu().numpy()  # H,W
        # plt.imshow(result, cmap='gray')
        # plt.show()
        # labelf = torchvision.transforms.GaussianBlur(kernel_size=(3, 3))(labelf)
        # prediction = prediction.detach()
        thr_high = 0.3
        thr_low = 0.03
        mask_positive = labelf > thr_high  #
        mask_negative = labelf < thr_low

        mask_positive = (prediction > thr_high) * mask_positive  # thin
        mask_negative = (prediction < thr_low) * mask_negative  #

        labelf[:] = 2
        labelf[mask_negative] = 0

        labelf[mask_positive] = 1

        label = labelf.detach().long()
        mask = labelf.clone()
        num_positive = torch.sum(label == 1).float()
        num_negative = torch.sum(label == 0).float()

        mask[label == 1] = 1.0 * num_negative / (num_positive + num_negative + 1)
        mask[label == 0] = beta * num_positive / (num_positive + num_negative + 1)
        mask[label == 2] = 0

        cost = F.binary_cross_entropy(
            prediction, labelf, weight=mask.detach(), reduction='sum')  # weight=mask,
        return cost

    def cross_entropy_loss_RCF_supervise(self, prediction, labelf, beta):
        # from PIL import Image
        # import matplotlib.pyplot as plt
        # from src.utils.utils import toNumpy
        # result = torch.squeeze(prediction[0]*255).detach().cpu().numpy() # H,W
        # plt.imshow(result, cmap='gray')
        # plt.show()
        #
        # result = torch.squeeze(canny_edge[0] * 255).detach().cpu().numpy()  # H,W
        # plt.imshow(result, cmap='gray')
        # plt.show()

        #
        # result = torch.squeeze(labelf[0] * 255).detach().cpu().numpy()  # H,W
        # plt.imshow(result, cmap='gray')
        # plt.show()
        # prediction = prediction.detach()
        label = labelf.long()
        mask = labelf.clone()
        num_positive = torch.sum(label == 1).float()
        num_negative = torch.sum(label == 0).float()

        mask[label == 1] = 1.0 * (num_negative + 1) / (num_positive + num_negative + 1)
        mask[label == 0] = beta * (num_positive + 1) / (num_positive + num_negative + 1)
        mask[label == 2] = 0
        cost = F.binary_cross_entropy(
            prediction, labelf, weight=mask.detach(), reduction='sum')
        return cost  # / (num_positive +1)

    def charbonnier(self, x, alpha=0.25, epsilon=1.e-9):
        return torch.pow(torch.pow(x, 2) + epsilon ** 2, alpha)

    # def smoothness_loss(self, flow, flow_mask):
    #     #flow: b,WW,2
    #     #flow_mask: b,WW
    #     b, ww, c = flow.size()
    #     w = int(math.sqrt(ww))
    #     flow = flow.reshape(b,w,w,c).permute(0,3,1,2)
    #     flow_mask = flow_mask.reshape(b,w,w)
    #     b, c, h, w = flow.size()
    #     v_translated = torch.cat((flow[:, :, 1:, :], torch.zeros(b, c, 1, w, device=flow.device)), dim=-2)
    #     v_flow_mask = torch.cat((flow_mask[:, 1:, :], torch.zeros(b, 1, w, dtype=bool,device=flow.device)), dim=-2)
    #     v_mask = v_flow_mask * flow_mask
    #     h_translated = torch.cat((flow[:, :, :, 1:], torch.zeros(b, c, h, 1, device=flow.device)), dim=-1)
    #     h_flow_mask = torch.cat((flow_mask[:, :, 1:], torch.zeros(b, h, 1, dtype=bool,device=flow.device)), dim=-1)
    #     h_mask = h_flow_mask * flow_mask
    #
    #     s_loss = self.charbonnier(flow - v_translated)*v_mask[:,None,:,:] + self.charbonnier(flow - h_translated)*h_mask[:,None,:,:]
    #     s_loss = torch.sum(s_loss, dim=1) / 2
    #
    #     return (torch.sum(s_loss) / (torch.sum(h_mask)+torch.sum(v_mask)+1)) / 4 # 4:is the scale in the fine stage

    def smoothness_loss(self, flow, flow_mask):
        # flow: b,h,w,2
        # flow_mask: b,h,w
        flow = flow.permute(0, 3, 1, 2)
        b, c, h, w = flow.size()
        v_translated = torch.cat((flow[:, :, 1:, :], torch.zeros(b, c, 1, w, device=flow.device)), dim=-2)
        v_flow_mask = torch.cat((flow_mask[:, 1:, :], torch.zeros(b, 1, w, dtype=bool, device=flow.device)), dim=-2)
        v_mask = v_flow_mask * flow_mask
        h_translated = torch.cat((flow[:, :, :, 1:], torch.zeros(b, c, h, 1, device=flow.device)), dim=-1)
        h_flow_mask = torch.cat((flow_mask[:, :, 1:], torch.zeros(b, h, 1, dtype=bool, device=flow.device)), dim=-1)
        h_mask = h_flow_mask * flow_mask

        s_loss = self.charbonnier(flow - v_translated) * v_mask[:, None, :, :] + self.charbonnier(
            flow - h_translated) * h_mask[:, None, :, :]
        s_loss = torch.sum(s_loss, dim=1) / 2
        loss = (torch.sum(s_loss) / (torch.sum(h_mask) + torch.sum(v_mask) + 1)) / 4
        return loss  # 4:is the scale in the fine stage

    def transform_poi(self, theta, court_poi, normalize=True):
        ''' Transform PoI with the homography '''
        bs = theta.shape[0]
        theta_inv = torch.inverse(theta[:bs])
        poi = transform_points(theta_inv, court_poi[:bs])

        # Apply inverse normalization to the transformed PoI (from [-1,1] to [0,1]):
        if normalize:
            poi = poi / 2.0 + 0.5

        return poi

    def reprojection_loss(self, predict, gt, court_poi, reduction='mean'):
        poi_pre = transform_points(predict, court_poi)
        poi_gt = transform_points(gt.float(), court_poi)
        # normalize
        poi_pre[:, :, 0], poi_pre[:, :, 1] = poi_pre[:, :, 0] / (2 * 640), poi_pre[:, :, 1] / (480 * 2)
        poi_gt[:, :, 0], poi_gt[:, :, 1] = poi_gt[:, :, 0] / (2 * 640), poi_gt[:, :, 1] / (480 * 2)
        '''
        Calculate the distance between the input points and target ones
        '''
        dist = torch.sqrt(torch.sum(torch.pow(poi_pre - poi_gt, 2), dim=2))
        loss = torch.sum(dist, dim=1) / court_poi.shape[0]

        if reduction == 'mean':
            loss = torch.mean(loss)
        elif reduction == 'sum':
            loss = torch.sum(loss)

        return loss

    def forward(self, data):

        """
        Update:
            data (dict): update{
                'loss': [1] the reduced loss across a batch,
                'loss_scalars' (dict): loss scalars for tensorboard_record
            }
        """
        reprojection_loss = False  # True
        loss_scalars = {}
        # 0. compute element-wise loss weight
        c_weight = self.compute_c_weight(data)
        loss = 0
        # 1. coarse-level loss
        loss_c = self.compute_coarse_loss(
            data['conf_matrix_with_bin'] if self.sparse_spvs and self.match_type == 'sinkhorn' \
                else data['conf_matrix'],
            data['conf_matrix_gt'],
            weight=c_weight)
        # print(loss_c)
        loss = loss_c * self.loss_config['coarse_weight'] * 10

        loss_scalars.update({"loss_c": loss_c.clone().detach().cpu()})

        if reprojection_loss:
            bs = data['bs']
            court_poi = torch.tensor([[0.0, 0], [640, 0], [0, 480], [640, 480]],
                                     device=loss_c.device).unsqueeze(0).repeat(bs, 1, 1)
            loss_c_reprojection = self.reprojection_loss(data['trans_predict'], data['homo'], court_poi=court_poi) * 10
            loss += loss_c_reprojection
            loss_scalars.update({"loss_reprojection": loss_c_reprojection.clone().detach().cpu()})
            print(loss_c, loss_c_reprojection)
        # 2. fine-level loss
        # TODO: fine-level loss
        if self.train_stage == 'whole':

            # 2.1 fine-level loss
            loss_f_flow = self.compute_fine_loss(data['expec_f'], data['expec_f_gt'])

            # loss_f_smooth = self.smoothness_loss(data['smooth_map'],data['smooth_mask'])
            loss_f_photometric = data['loss_photometric']

            if loss_f_flow is not None:
                loss += loss_f_flow * self.loss_config['fine_weight']
                loss_scalars.update({"loss_f_flow": loss_f_flow.clone().detach().cpu()})

                loss += loss_f_photometric * self.loss_config['fine_weight']
                loss_scalars.update({"loss_f_photometric": loss_f_photometric.clone().detach().cpu()})
                #
                # loss += loss_f_smooth * self.loss_config['fine_weight']
                # loss_scalars.update({"loss_f_smooth": loss_f_smooth.clone().detach().cpu()})

            else:
                assert self.training is False
                loss_scalars.update({'loss_f': torch.tensor(1.)})  # 1 is the upper bound

        # edge_loss = False
        # if edge_loss:
        #     h, w = data['image0'].shape[2],data['image0'].shape[3]
        #     label_edge = kornia.geometry.transform.warp_perspective(data['image0'], data['trans'].float(), dsize=[h, w])
        #     canny_edge = data['image1_edge']
        #     thr_high = 0.3
        #     thr_low = 0.1
        #     mask_positive = (label_edge > thr_high) * (data['pidinet_out'] > thr_high)  #
        #     mask_negative = (data['pidinet_out'] < thr_low)  # (canny_edge < thr_low) *
        #     label_edge[:] = 2
        #     label_edge[mask_negative] = 0
        #     label_edge[mask_positive] = 1
        #     label_edge.requires_grad = True
        #     loss_edge = 0.001 * self.cross_entropy_loss_RCF_supervise(data['pidinet_out'], label_edge,beta=1.1)
        #     # print('edge_loss', loss_edge)
        #     loss_scalars.update({"loss_edge": loss_edge.clone().detach().cpu()})
        #     loss += loss_edge

        loss_scalars.update({'loss': loss.clone().detach().cpu()})
        data.update({"loss": loss, "loss_scalars": loss_scalars})


def spvs_coarse(data, config):
    """
        Update:
            data (dict): {
                "conf_matrix_gt": [N, hw0, hw1],
                'spv_b_ids': [M]
                'spv_i_ids': [M]
                'spv_j_ids': [M]
                'spv_w_pt0_i': [N, hw0, 2], in original image resolution
                'spv_pt1_i': [N, hw1, 2], in original image resolution
            }

        NOTE:
            - for scannet dataset, there're 3 kinds of resolution {i, c, f}
            - for megadepth dataset, there're 4 kinds of resolution {i, i_resize, c, f}
        """
    # 1. misc
    device = data['image0'].device
    N, _, H0, W0 = data['image0'].shape
    _, _, H1, W1 = data['image1'].shape
    scale = config['TM']['RESOLUTION'][0]
    h0, w0, h1, w1 = map(lambda x: torch.div(x, scale, rounding_mode='trunc'), [H0, W0, H1, W1])  #

    scale_x = scale * data['scale'][:, 0] if 'scale' in data else scale
    scale_y = scale * data['scale'][:, 1] if 'scale' in data else scale

    if data['dataset_name'][0] == 'linemod_2d':
        # bias_mode
        bias = data['bias']
        # N,n,2
        pts_x = (data['pts_0'][:, :, 0] + torch.as_tensor(bias[:, 0, None] / scale_x[:, None],
                                                          device=device)).round().long()  # [N,L]x
        pts_y = (data['pts_0'][:, :, 1] + torch.as_tensor(bias[:, 1, None] / scale_y[:, None],
                                                          device=device)).round().long()  # [N,L]y
    else:
        # trans_mode
        trans = data['homo']
        pts_x = (trans[:, 0, 0, None] * (data['pts_0'][:, :, 0] * scale_x[:, None]) + trans[:, 0, 1, None] * (
                    data['pts_0'][:, :, 1] * scale_y[:, None]) + trans[:, 0, 2, None])
        pts_y = (trans[:, 1, 0, None] * (data['pts_0'][:, :, 0] * scale_x[:, None]) + trans[:, 1, 1, None] * (
                    data['pts_0'][:, :, 1] * scale_y[:, None]) + trans[:, 1, 2, None])
        pts_z = (trans[:, 2, 0, None] * (data['pts_0'][:, :, 0] * scale_x[:, None]) + trans[:, 2, 1, None] * (
                    data['pts_0'][:, :, 1] * scale_y[:, None]) + trans[:, 2, 2, None])
        pts_x /= pts_z
        pts_y /= pts_z

    pts_x = (pts_x / scale_x[:, None]).round().long()
    pts_y = (pts_y / scale_y[:, None]).round().long()

    pts_image = torch.stack((pts_x, pts_y), dim=-1)

    # construct a gt conf_matrix
    L, S = data['pts_0'].shape[1], data['pts_1'].shape[1]
    conf_matrix_gt = torch.zeros(N, L, S, device=device)
    # i_ids is tempalte ids ,j_inds is image ids
    x_ids = torch.flatten(pts_image[:, :, 0])
    y_ids = torch.flatten(
        pts_image[:, :, 1])  # inds in image coordinate,but the j_inds is flatten,they are ready for j_ids
    b_ids, i_ids = torch.where(pts_image[:, :, 0] > -1e8)  # get all index in sampled position

    mask_x = (x_ids < w1) * (x_ids >= 0)
    mask_y = (y_ids < h1) * (y_ids >= 0)

    mask = (mask_x * mask_y)  # filter the out box points in image
    j_ids = w1 * y_ids + x_ids
    # b_ids, _ = torch.where(pts_image[:, :, 0] >= 0) #
    b_ids = b_ids[mask]
    i_ids = i_ids[mask]
    j_ids = j_ids[mask]
    try:
        conf_matrix_gt[b_ids, i_ids, j_ids] = 1
    except:
        raise ('mask is not ok!')
    data.update({'conf_matrix_gt': conf_matrix_gt})

    # 5. save coarse matches(gt) for training fine level
    if len(b_ids) == 0:
        logger.warning(f"No groundtruth coarse match found for: {data['pair_names']}")
        # this won't affect fine-level loss calculation
        # TODO: there is a bug when len(b_ids)>0 while there is no data in an image
        b_ids = torch.arange(0, N, device=device).long()
        i_ids = torch.zeros(N, device=device).long()
        j_ids = torch.zeros(N, device=device).long()

    data.update({
        'spv_b_ids': b_ids,
        'spv_i_ids': i_ids,
        'spv_j_ids': j_ids
    })

    # TODO:to check correspondence


def spvs_fine(data, config):
    """
        Update:
            data (dict): {
                "conf_matrix_gt": [N, hw0, hw1],
                'spv_b_ids': [M]
                'spv_i_ids': [M]
                'spv_j_ids': [M]
                'spv_w_pt0_i': [N, hw0, 2], in original image resolution
                'spv_pt1_i': [N, hw1, 2], in original image resolution
            }

        NOTE:
            - for scannet dataset, there're 3 kinds of resolution {i, c, f}
            - for megadepth dataset, there're 4 kinds of resolution {i, i_resize, c, f}
        """
    # 1. misc
    W = config['FINE_WINDOW_SIZE']
    is_cat_coarse = config['FINE_CONCAT_COARSE_FEAT']
    device = data['image0'].device
    N, _, H0, W0 = data['image0'].shape
    _, _, H1, W1 = data['image1'].shape
    scale = config['RESOLUTION'][0]
    scale_gap = config['RESOLUTION'][0] / config['RESOLUTION'][1]  # 4
    h0, w0, h1, w1 = map(lambda x: torch.div(x, scale, rounding_mode='trunc'), [H0, W0, H1, W1])  #

    scale_x = scale * data['scale'][:, 0] if 'scale' in data else scale
    scale_y = scale * data['scale'][:, 1] if 'scale' in data else scale
    if is_cat_coarse:
        f_points = data['p_src'] / scale  # [bs,L,2]
        pt1_i = data['p_src']  # resized resolution
        w_pt0_i_bs = torch.Tensor().to(pt1_i.device)
        for b_id in range(data['bs']):
            b_mask = data['b_ids'] == b_id
            points_x = f_points[b_mask][:, 0]  # (L)
            points_y = f_points[b_mask][:, 1]

            points_x = (points_x * scale_x[b_id])  # L
            points_y = (points_y * scale_y[b_id])  # L
            # # gt trans_mode
            trans = data['homo'][b_id]  # origin resolution
            pts_x = (trans[0, 0, None] * points_x + trans[0, 1, None] * points_y + trans[0, 2, None])
            pts_y = (trans[1, 0, None] * points_x + trans[1, 1, None] * points_y + trans[1, 2, None])
            pts_z = (trans[2, 0, None] * points_x + trans[2, 1, None] * points_y + trans[2, 2, None])
            pts_x /= pts_z
            pts_y /= pts_z
            points_x_i = (pts_x / scale_x[b_id]) * scale
            points_y_i = (pts_y / scale_y[b_id]) * scale  # resized resolution

            trans = data['theta_inv'][b_id]  # resized resolution
            pts_x = (trans[0, 0, None] * points_x_i + trans[0, 1, None] * points_y_i + trans[0, 2, None])
            pts_y = (trans[1, 0, None] * points_x_i + trans[1, 1, None] * points_y_i + trans[1, 2, None])
            pts_z = (trans[2, 0, None] * points_x_i + trans[2, 1, None] * points_y_i + trans[2, 2, None])
            pts_x /= pts_z
            pts_y /= pts_z
            # 3. compute gt
            w_pt0_i = torch.stack((pts_x, pts_y),
                                  dim=-1)  # the template points transformed to the warped edge image(resized resolution)
            w_pt0_i_bs = torch.cat([w_pt0_i_bs, w_pt0_i], dim=0)

        scale = config['RESOLUTION'][1]
        radius = W // 2

        r'''
         `expec_f_gt` might exceed the window, i.e. abs(*) > 1, which would be filtered later(never)
         expec_f_gt : to push the network learning the coordinates offset from the correspondence-level accuracy to subpixel accuracy
        '''
        expec_f_gt = (w_pt0_i_bs - pt1_i) / scale / radius  # [M, 2]
        data.update({"expec_f_gt": expec_f_gt})
    else:
        f_points = data['f_points'] / scale_gap  # [bs,L,2]
        points_x = f_points[:, :, 0]  # (B,L)
        points_y = f_points[:, :, 1]
        pt1_i = torch.stack((points_x * scale, points_y * scale), dim=-1)  # resized resolution
        points_x = (points_x * scale_x[:, None])  # bs,L
        points_y = (points_y * scale_y[:, None])  # bs,L
        # # gt trans_mode
        trans = data['homo']  # origin resolution
        pts_x = (trans[:, 0, 0, None] * points_x + trans[:, 0, 1, None] * points_y + trans[:, 0, 2, None])
        pts_y = (trans[:, 1, 0, None] * points_x + trans[:, 1, 1, None] * points_y + trans[:, 1, 2, None])
        pts_z = (trans[:, 2, 0, None] * points_x + trans[:, 2, 1, None] * points_y + trans[:, 2, 2, None])
        pts_x /= pts_z
        pts_y /= pts_z
        points_x_i = (pts_x / scale_x[:, None]) * scale
        points_y_i = (pts_y / scale_y[:, None]) * scale  # resized resolution

        trans = data['theta_inv']  # resized resolution
        pts_x = (trans[:, 0, 0, None] * points_x_i + trans[:, 0, 1, None] * points_y_i + trans[:, 0, 2, None])
        pts_y = (trans[:, 1, 0, None] * points_x_i + trans[:, 1, 1, None] * points_y_i + trans[:, 1, 2, None])
        pts_z = (trans[:, 2, 0, None] * points_x_i + trans[:, 2, 1, None] * points_y_i + trans[:, 2, 2, None])
        pts_x /= pts_z
        pts_y /= pts_z
        # 3. compute gt
        w_pt0_i = torch.stack((pts_x, pts_y),
                              dim=-1)  # the template points transformed to the warped edge image(resized resolution)
        scale = config['RESOLUTION'][1]
        radius = W // 2

        r'''
         `expec_f_gt` might exceed the window, i.e. abs(*) > 1, which would be filtered later(never)
         expec_f_gt : to push the network learning the coordinates offset from the correspondence-level accuracy to subpixel accuracy
        '''

        b_ids, i_ids = data['b_ids_mask_fine'], data['i_ids_mask_fine']
        expec_f_gt = (w_pt0_i[b_ids, i_ids] - pt1_i[b_ids, i_ids]) / scale / radius  # [M, 2]
        data.update({"expec_f_gt": expec_f_gt})


def compute_supervision_coarse(data, config):
    assert len(set(data['dataset_name'])) == 1, "Do not support mixed datasets training!"
    data_source = data['dataset_name'][0]
    if data_source.lower() in ['synthetic', 'linemod_2d']:
        spvs_coarse(data, config)
    else:
        raise ValueError(f'Unknown data source: {data_source}')


def compute_supervision_fine(data, config):
    data_source = data['dataset_name'][0]
    if data_source.lower() in ['synthetic', 'linemod_2d']:
        spvs_fine(data, config)
    else:
        raise NotImplementedError


class PL_Tm(pl.LightningModule):
    def __init__(self, config, pretrained_ckpt_backbone=None, pretrain_ckpt=None, edge_ckpt=None, profiler=None,
                 dump_dir=None, training=True):
        super().__init__()
        # Misc
        self.config = config  # full config
        _config = lower_config(self.config)
        self.profiler = profiler or PassThroughProfiler()
        self.n_vals_plot = max(config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1)
        self.load = _config['tm']['edge']['load']
        # TM Module
        self.edge_ckpt = edge_ckpt
        self.backbone = Backnone(config=_config, edge_ckpt=self.edge_ckpt)
        self.Tm = Tm(config=_config)
        self.loss = TmLoss(_config)
        self.fine_config = _config['tm']['fine']
        self.training_stage = _config['tm']['match_coarse']['train_stage']
        # load Pretrained weights
        if pretrained_ckpt_backbone:
            model_dict = self.backbone.state_dict()
            model_dict2 = self.Tm.state_dict()
            if _config['tm']['superpoint']['name'] == 'SuperPointNet_gauss2':
                pre_state_dict = torch.load(pretrained_ckpt_backbone, map_location='cpu')['model_state_dict']
            else:
                pre_state_dict = torch.load(pretrained_ckpt_backbone, map_location='cpu')

            for k, v in pre_state_dict.items():
                if 'backbone.' + k in model_dict.keys() and v.shape == model_dict['backbone.' + k].shape:
                    model_dict['backbone.' + k] = v
                if 'backbone.' + k in model_dict2.keys() and v.shape == model_dict2['backbone.' + k].shape:
                    model_dict2['backbone.' + k] = v
            self.Tm.load_state_dict(model_dict2, strict=True)
            self.backbone.load_state_dict(model_dict, strict=True)
            logger.info(f"Load \'{pretrained_ckpt_backbone}\' as pretrained checkpoint_backbone")
        print('pretrain_ckpt', pretrain_ckpt)

        # load my trained weights
        if pretrain_ckpt and len(pretrain_ckpt) > 4:
            model_dict = self.backbone.state_dict()
            model_dict2 = self.Tm.state_dict()
            pre_state_dict = torch.load(pretrain_ckpt, map_location='cpu')['state_dict']
            for k, v in pre_state_dict.items():
                if k[9:] in model_dict.keys() and v.shape == model_dict[k[9:]].shape:
                    model_dict[k[9:]] = v  # # get out 'backbone.'
                    print(k, 'has beed load')
                    if k[9:] in model_dict2:
                        # mask sure two superglue module share the same parameters
                        # when the  pre_state_dict does not contain the
                        # second superglue parameters
                        # (do not change the network order,otherwise need change here)
                        model_dict2[k[9:]] = v

                if k[3:] in model_dict2.keys() and v.shape == model_dict2[k[3:]].shape:
                    model_dict2[k[3:]] = v  # # get out 'TM.'
                    print(k, 'has beed load')

            self.Tm.load_state_dict(model_dict2, strict=True)
            self.backbone.load_state_dict(model_dict, strict=True)
            logger.info(f"Load \'{pretrain_ckpt}\' as pretrained checkpoint")
        if training:
            if self.config.TM.MATCH_COARSE.TRAIN_STAGE == "only_coarse":
                to_freeze_dict = ['edge_net']  # ,'LM_coarse','coarse_matching','backbone'
            elif self.config.TM.MATCH_COARSE.TRAIN_STAGE == "whole":
                to_freeze_dict = ['backbone', 'edge_net']
            else:
                assert "training stage name spell wrong!"
            for (name, param) in self.backbone.named_parameters():
                if name.split('.')[0] in to_freeze_dict:
                    print(name, ': freezeed')
                    param.requires_grad = False
            # loftr
            for (name, param) in self.Tm.named_parameters():
                if name.split('.')[0] in to_freeze_dict:
                    print(name, ': freezeed')
                    param.requires_grad = False
        # Testing
        self.dump_dir = dump_dir

    def configure_optimizers(self):
        # FIXME: The scheduler did not work properly when `--resume_from_checkpoint`
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]  # 优化器定义，返回一个优化器，或数个优化器，或两个List（优化器，Scheduler）

    def optimizer_step(
            self, epoch, batch_idx, optimizer, optimizer_idx,
            optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # learning rate warm up
        warmup_step = self.config.TRAINER.WARMUP_STEP
        if self.trainer.global_step < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == 'linear':
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + \
                     (self.trainer.global_step / self.config.TRAINER.WARMUP_STEP) * \
                     abs(self.config.TRAINER.TRUE_LR - base_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
            elif self.config.TRAINER.WARMUP_TYPE == 'constant':
                pass
            else:
                raise ValueError(f'Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}')

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def _compute_metrics(self, batch):
        with self.profiler.profile("Copmute metrics"):
            compute_distance_errors_old(batch)
            compute_distance_errors(batch)
            rel_pair_names = list(zip(*batch['pair_names']))
            bs = batch['image0'].size(0)
            num = batch['points_template'][0].shape[0]
            metrics = {
                # to filter duplicate pairs caused by DistributedSampler
                'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)],
                'inliers': batch['inliers'],
                # 'dis_errs_evaluate': [batch['dis_errs'][batch['m_bids'] == b].cpu().numpy() for b in range(bs)]
                'dis_errs_evaluate': [batch['dis_errs_evaluate'][b * num:(b + 1) * num].cpu().numpy() for b in
                                      range(bs)]  # [batch['m_bids'] == b]
            }
            ret_dict = {'metrics': metrics}
        return ret_dict, rel_pair_names

    def _compute_metrics_test(self, batch):
        with self.profiler.profile("Copmute metrics"):
            compute_distance_errors_test(batch)
            rel_pair_names = list(zip(*batch['pair_names']))
            bs = batch['image0'].size(0)
            metrics = {
                # to filter duplicate pairs caused by DistributedSampler
                'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)],
                'inliers': batch['inliers'],
                'dis_errs_evaluate': [batch['dis_errs'][:].cpu().numpy() for b in range(bs)]
                # [batch['m_bids'] == b]
            }
            ret_dict = {'metrics': metrics}
        return ret_dict, rel_pair_names

    def _trainval_inference(self, batch):
        if self.training == True:
            with self.profiler.profile("get keypoint and descriptor from backbone"):
                outs_post0, outs_post1 = self.backbone(batch)

            with self.profiler.profile("Compute coarse supervision"):
                compute_supervision_coarse(batch, self.config)
            with self.profiler.profile("transformer matching module"):
                self.Tm(batch, outs_post0, outs_post1, self.backbone.backbone)

            if self.training_stage == 'whole':
                with self.profiler.profile("Compute fine supervision"):
                    compute_supervision_fine(batch, self.config.TM.FINE)

            with self.profiler.profile("Compute losses"):
                self.loss(batch)
        else:
            with self.profiler.profile("get keypoint and descriptor from backbone"):
                outs_post0, outs_post1 = self.backbone(batch)
            with self.profiler.profile("transformer matching module"):
                self.Tm(batch, outs_post0, outs_post1, self.backbone.backbone)

    def forward(self, batch):
        self._trainval_inference(batch)

    def training_step(self, batch, batch_idx):
        self._trainval_inference(batch)

        # logging
        if self.trainer.global_rank == 0 and self.global_step % self.trainer.log_every_n_steps == 0:
            # scalars
            for k, v in batch['loss_scalars'].items():
                self.logger.experiment.add_scalar(f'train/{k}', v, self.global_step)

            # net-params
            if self.config.TM.MATCH_COARSE.MATCH_TYPE == 'sinkhorn':
                self.logger.experiment.add_scalar(
                    f'skh_bin_score', self.Tm.coarse_matching.bin_score.clone().detach().cpu().data, self.global_step)

            # figures  (not plot in training)
            # if self.config.TRAINER.ENABLE_PLOTTING:
            #     # compute the error of each eatimate correspondence
            #     compute_distance_errors_old(batch)
            #     figures = make_matching_figures(batch, self.config, self.config.TRAINER.PLOT_MODE)
            #     for k, v in figures.items():
            #         self.logger.experiment.add_figure(f'train_match/{k}', v, self.global_step)

        return {'loss': batch['loss']}  # dict Can include any keys,but must include the key 'loss'

    def training_epoch_end(self, outputs):
        # 在一个训练epoch结尾处被调用。
        # 输入参数：一个List，List的内容是前面training_step()所返回的每次的内容。
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        if self.trainer.global_rank == 0:
            self.logger.experiment.add_scalar(
                'train/avg_loss_on_epoch', avg_loss,
                global_step=self.current_epoch)
        # if self.trainer.current_epoch == 20:
        #     to_train_dict = ['edge_net']
        #     for (name, param) in self.backbone.named_parameters():
        #         if name.split('.')[0] in to_train_dict:
        #             print(name, ':begin  train')
        #             param.requires_grad = True

    def validation_step(self, batch, batch_idx):
        self._trainval_inference(batch)
        ret_dict, _ = self._compute_metrics(batch)

        val_plot_interval = max(self.trainer.num_val_batches[0] // self.n_vals_plot, 1)
        # figures = make_matching_figures(batch, self.config, mode=self.config.TRAINER.PLOT_MODE)
        # for k, v in figures.items():
        #     v[0].show()
        figures = {self.config.TRAINER.PLOT_MODE: []}
        if batch_idx % val_plot_interval == 0:
            figures = make_matching_figures(batch, self.config, mode=self.config.TRAINER.PLOT_MODE)

        return {
            **ret_dict,
            'loss_scalars': batch['loss_scalars'],
            'figures': figures,
        }

    def validation_epoch_end(self, outputs):
        # handle multiple validation sets
        multi_outputs = [outputs] if not isinstance(outputs[0], (list, tuple)) else outputs
        multi_val_metrics = defaultdict(list)
        # TODO:
        for valset_idx, outputs in enumerate(multi_outputs):
            # since pl performs sanity_check at the very begining of the training
            cur_epoch = self.trainer.current_epoch
            if not self.trainer.resume_from_checkpoint and self.trainer.running_sanity_check:
                cur_epoch = -1

            # 1. loss_scalars: dict of list, on cpu
            _loss_scalars = [o['loss_scalars'] for o in outputs]
            loss_scalars = {k: flattenList(all_gather([_ls[k] for _ls in _loss_scalars])) for k in _loss_scalars[0]}

            # 2. val metrics: dict of list, numpy
            _metrics = [o['metrics'] for o in outputs]
            metrics = {k: flattenList(all_gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}
            # NOTE: all ranks need to `aggregate_merics`, but only log at rank-0
            val_metrics_4tb = aggregate_metrics(metrics, self.config.TRAINER.DIS_ERR_THR)

            for thr in [1, 3, 5, 10, 20]:
                multi_val_metrics[f'auc@{thr}'] = val_metrics_4tb[f'auc@{thr}']

            # 3. figures
            _figures = [o['figures'] for o in outputs]
            figures = {k: flattenList(gather(flattenList([_me[k] for _me in _figures]))) for k in _figures[0]}

            # tensorboard records only on rank 0
            if self.trainer.global_rank == 0:
                for k, v in loss_scalars.items():
                    mean_v = torch.stack(v).mean()
                    self.logger.experiment.add_scalar(f'val_{valset_idx}/avg_{k}', mean_v, global_step=cur_epoch)

                for k, v in val_metrics_4tb.items():
                    self.logger.experiment.add_scalar(f"metrics_{valset_idx}/{k}", v, global_step=cur_epoch)

                for k, v in figures.items():
                    if self.trainer.global_rank == 0:
                        for plot_idx, fig in enumerate(v):
                            self.logger.experiment.add_figure(
                                f'val_match_{valset_idx}/{k}/pair-{plot_idx}', fig, cur_epoch, close=True)
            plt.close('all')

        for thr in [1, 3, 5, 10, 20]:
            # log on all ranks for ModelCheckpoint callback to work properly
            self.log(f'auc@{thr}', torch.tensor(np.mean(multi_val_metrics[f'auc@{thr}'])))  # ckpt monitors on this

    def test_step(self, batch, batch_idx):
        plot = False
        with self.profiler.profile("get keypoint and descriptor from backbone"):
            outs_post0, outs_post1 = self.backbone(batch)
        with self.profiler.profile("transfomer matching module"):
            self.Tm(batch, outs_post0, outs_post1, self.backbone.backbone)

        if plot:
            with self.profiler.profile("Compute coarse supervision"):
                compute_supervision_coarse(batch, self.config)
            compute_distance_errors_old(batch)
            figures = make_matching_figures(batch, self.config, self.config.TRAINER.PLOT_MODE)
            for k, v in figures.items():
                v[0].show()

        ret_dict, rel_pair_names = self._compute_metrics(batch)  # _test

        return ret_dict

    def test_epoch_end(self, outputs):
        # metrics: dict of list, numpy
        _metrics = [o['metrics'] for o in outputs]
        metrics = {k: flattenList(gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}
        if self.trainer.global_rank == 0:
            print(self.profiler.summary())
            val_metrics_4tb = aggregate_metrics(metrics, self.config.TRAINER.DIS_ERR_THR)
            logger.info('\n' + pprint.pformat(val_metrics_4tb))


def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}


def get_contours_points(image):
    """
    :param image: (H,W)
    :return: (N,2)
    """
    assert (len(image.shape) == 2)
    ret, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    xcnts = np.vstack([x.reshape(-1, 2) for x in contours])
    # get out of the contours on  edge
    mask_x_0 = xcnts[:, 0] > 8
    mask_y_0 = xcnts[:, 1] > 8
    mask_x_1 = xcnts[:, 0] < (image.shape[1] - 8)
    mask_y_1 = xcnts[:, 1] < (image.shape[0] - 8)

    mask = mask_x_0 * mask_y_0 * mask_x_1 * mask_y_1
    if mask.sum(0) == 0:
        pass
    else:
        xcnts = xcnts[mask]

    return xcnts


def make_matching_figure_4(
        ave_aligned, img0, img1, img1_edge, mkpts0, mkpts1, color,
        kpts0=None, kpts1=None, text=[], dpi=75, path=None, is_visual=False):
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    fig, axes = plt.subplots(1, 4, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0, cmap='gray')
    axes[1].imshow(img1, cmap='gray')
    axes[2].imshow(img1_edge, cmap='gray')
    axes[3].imshow(ave_aligned, cmap='gray')
    for i in range(4):  # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)

    # if kpts0 is not None:
    #     assert kpts1 is not None
    #     axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=2)
    #     axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=2)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                             (fkpts0[i, 1], fkpts1[i, 1]),
                                             transform=fig.transFigure, c=color[i], linewidth=1)
                     for i in range(len(mkpts0))]

        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=1, alpha=0.1)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=1, alpha=0.1)

    # put txts
    txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[1].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    # save or return figure
    if path and is_visual:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
        plt.close()
        # pass
    return fig


def filter_match_point(mkpts0, mkpts1, mconf, shape, threshold_conf, threshold_win):
    # 过滤低于阈值的匹配点对
    for th in mconf[::-1]:
        if th < threshold_conf:
            del_index_ = np.array(np.where(mconf == th))
            del_index = del_index_.squeeze() if (del_index_.shape[-1] == 1) else del_index_.squeeze()[-1]
            mconf = np.delete(mconf, del_index, axis=0)
            mkpts0 = np.delete(mkpts0, del_index, axis=0)
            mkpts1 = np.delete(mkpts1, del_index, axis=0)

    # 过滤超出局部窗口的匹配点对
    # del_indices = np.where(np.mean(abs(mkpts0 - mkpts1) < (threshold_win / 2), axis=1) != 1)
    # for del_index in del_indices[::-1]:
    #     mconf = np.delete(mconf, del_index, axis=0)
    #     mkpts0 = np.delete(mkpts0, del_index, axis=0)
    #     mkpts1 = np.delete(mkpts1, del_index, axis=0)

    # 过滤超出图像边界的匹配点对
    for i in range(len(mkpts0))[::-1]:
        if ((0 + 10) <= mkpts0[i][0] <= (shape[1] - 10)) and ((0 + 10) <= mkpts0[i][1] <= (shape[0] - 10)) and (
                (0 + 10) <= mkpts1[i][0] <= (shape[1] - 10)) and ((0 + 10) <= mkpts1[i][1] <= (shape[0] - 10)):
            continue
        mkpts0 = np.delete(mkpts0, i, axis=0)
        mkpts1 = np.delete(mkpts1, i, axis=0)
        mconf = np.delete(mconf, i, axis=0)
    # for pt0, pt1 in zip(mkpts0, mkpts1):
    #     print("pt0", pt0)
    #     print("pt1", pt1)

    return mkpts0, mkpts1, mconf


def get_matcher(image1, image0, matcher, resize, threshold_conf, threshold_win, visual_file, is_visual):
    patch_size = 8  # coarse stage patch size is 8x8
    num = 128  # num of query points
    shape0 = image0.shape[:-1]
    shape1 = image1.shape[:-1]
    image1_rgb = cv2.resize(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB), resize[::-1])
    image1 = cv2.resize(cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY), resize[::-1])
    image0 = cv2.resize(cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY), resize[::-1])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    w, h = image1.shape[1], image1.shape[0]
    image1 = cv2.resize(image1, resize[::-1])
    image1_rgb = transform(image1_rgb)[None]  # c,h,w
    image1_edge = cv2.Canny(image1, 5, 10)

    scale = torch.tensor(np.array(shape0[::-1]) / np.array(resize[::-1]), dtype=torch.float)
    contours_points = get_contours_points(image0)
    contours_points_fine = np.round(contours_points)
    contours_points = np.round(contours_points) // patch_size
    contours_points = np.array(list(set([tuple(t) for t in contours_points])))

    mask_0 = np.zeros(num, dtype=bool)
    if num <= contours_points.shape[0]:
        gap = contours_points.shape[0] // num
        contours_points = contours_points[:num * gap:gap, :]
        mask_0[:num] = True
    else:
        # mask
        num_pad = num - contours_points.shape[0]
        pad = np.random.choice(contours_points.shape[0], num_pad, replace=True)
        choice = np.concatenate([range(contours_points.shape[0]), pad])
        mask_0[:contours_points.shape[0]] = True
        contours_points = contours_points[choice, :]
    contours_points[:, 0] = np.clip(contours_points[:, 0], 0, (resize[1] // patch_size) - 1)
    contours_points[:, 1] = np.clip(contours_points[:, 1], 0, (resize[0] // patch_size) - 1)
    contours_points = torch.tensor(contours_points.astype(np.long))

    image0_raw = cv2.Canny(image0, 5, 10)

    image0 = torch.from_numpy(image0)[None][None].cuda() / 255.
    image1 = torch.from_numpy(image1)[None][None].cuda() / 255.
    image1_edge = torch.from_numpy(image1_edge)[None][None].cuda() / 255.

    device = image0.device
    homo = torch.ones([3, 3], device=device)
    batch = {
        'dataset_name': ['synthetic'],
        'image0': image0,
        'image1': image1,
        'image1_edge': image1_edge.cuda(),
        'scale': scale[None].cuda(),
        'c_points': contours_points[None].cuda(),
        'image1_rgb': image1_rgb.cuda(),
        'resolution': [patch_size],
        'homo': homo[None].cuda(),
        # 'c_points_fine': contours_points_fine[None].cuda()
    }

    mask0 = torch.from_numpy(np.ones((image0.shape[2], image0.shape[3]), dtype=bool))
    mask1 = torch.from_numpy(np.ones((image1.shape[2], image1.shape[3]), dtype=bool))

    if mask1 is not None:  # img_padding is True
        coarse_scale = 1 / patch_size
        if coarse_scale:
            [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
                                                   scale_factor=coarse_scale,
                                                   mode='nearest',
                                                   recompute_scale_factor=False)[0].bool()

        batch.update({'mask1': ts_mask_1[None].cuda()})
        batch.update({'mask0': torch.from_numpy(mask_0)[None].cuda()})  # coarse_scale mask  [L]

    with torch.no_grad():
        time_infer_stt = time.time()
        matcher(batch)
        time_infer_end = time.time() - time_infer_stt
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        # mconf = batch['mconf'].cpu().numpy()
        mconf = torch.ones(mkpts1.shape[0]).cpu().numpy()
        img0 = (batch['image0'][0][0].cpu().numpy() * 255).round().astype(np.int32)
        img1 = (batch['image1'][0][0].cpu().numpy() * 255).round().astype(np.int32)
        img1_edge = (batch['edge'][0][0].cpu().detach().numpy() * 255).round().astype(np.int32)

        ave_aligned = ((batch['warped_template'][0][0] * 0.5 + batch['image1'][0][
            0] * 0.5).cpu().detach().numpy() * 255).round().astype(np.int32)

    filter_mkpts0, filter_mkpts1, filter_mconf = filter_match_point(mkpts0, mkpts1, mconf, img0.shape,
                                                                    threshold_conf=threshold_conf,
                                                                    threshold_win=threshold_win)
    color = cm.jet(filter_mconf)
    text = ['TM', 'Matches: {}'.format(len(filter_mkpts0)), ]
    # fig = make_matching_figure(image0.squeeze().to('cpu'), image1.squeeze().to('cpu'), filter_mkpts0, filter_mkpts1,
    #                            color, text=text, path=visual_file, is_visual=is_visual)
    fig = make_matching_figure_4(
        ave_aligned, img0, img1, img1_edge, filter_mkpts0, filter_mkpts1, color,
        kpts0=None, kpts1=None, text=text, dpi=75, path=visual_file, is_visual=is_visual)
    return filter_mkpts0, filter_mkpts1, filter_mconf, time_infer_end


def get_center_and_next_split_window(mkpts0, gt_center, mkpts1, mconf, window_size, steps, centroid):
    win_lx, win_ly = centroid[0] - (window_size // 2), centroid[1] - (window_size // 2)
    win_rx, win_ry = centroid[0] + (window_size // 2), centroid[1] + (window_size // 2)
    bbox = [[win_lx, win_ly], [win_rx, win_ry]]
    step_x = np.linspace(bbox[0][0], bbox[1][0], steps)
    step_y = np.linspace(bbox[0][1], bbox[1][1], steps)
    loss_list, center_list = [], []
    for x in step_x:
        for y in step_y:
            center = np.array((x, y))
            gt_distence = mconf * np.sqrt(np.sum((mkpts0 - gt_center) ** 2, axis=1))
            distence = mconf * np.sqrt(np.sum((mkpts1 - center) ** 2, axis=1))
            loss = abs(distence.sum() - gt_distence.sum())
            loss_list.append(loss)
            center_list.append((x, y))
    ids = list(np.argsort(loss_list))
    loss_list = [loss_list[i] for i in ids][:(window_size ** 2 // 4)]
    center_list = [center_list[i] for i in ids][:(window_size ** 2 // 4)]
    centers = np.array(center_list)
    centroid = np.mean(centers, axis=0)
    return center_list[0], loss_list[0], centroid


def get_center(mkpts0, gt_center, mkpts1, mconf, gt_center_shape, resize, mode, is_recover):
    # ids = list(np.argsort(mconf))
    # mconf = np.array([mconf[i] for i in ids])
    # mkpts0 = np.array([mkpts0[i] for i in ids])
    # mkpts1 = np.array([mkpts1[i] for i in ids])
    if is_recover:
        mkpts0 = mkpts0 * (np.array(gt_center_shape[:2])[::-1] / resize[::-1])
        mkpts1 = mkpts1 * (np.array(gt_center_shape[:2])[::-1] / resize[::-1])
    else:
        gt_center = gt_center * (resize[::-1] / np.array(gt_center_shape[:2])[::-1])
    if mode == 'direct':
        gt_vec = gt_center - mkpts0
        weights = np.expand_dims(mconf, axis=1)
        center = (weights * (mkpts1 + gt_vec) / weights.sum()).sum(0)
    elif mode == 'search':
        centroid = copy.deepcopy(gt_center)
        window_size, steps, iters = 10, 20, 5
        start_time = time.time()
        best_loss = np.inf
        for i in range(iters):
            center, loss, centroid = get_center_and_next_split_window(mkpts0, gt_center, mkpts1, mconf,
                                                                      window_size - 2 * i, steps, centroid)
            print(f'gt_center: {gt_center}, center: {center}, loss: {loss}, centroid: {centroid}')
        print(f'processed time: {time.time() - start_time}s')
    if is_recover:
        center = center
    else:
        center = center * (np.array(gt_center_shape[:2])[::-1] / resize[::-1])
    return center


def get_center_from_loftr(image_path, homo_path, cache_path, model1_path, model2_path, resize=(224, 224),
                          threshold_conf=0.5, threshold_win=8, is_visual=False, mode='direct',
                          is_recover=False, th=1, img_ext_name='.bmp'):
    if is_visual:
        cache_path.create(is_remove=True)
        Path(os.path.join(cache_path.get_path(), 'matchs')).create(is_remove=True)
        Path(os.path.join(cache_path.get_path(), 'visuals')).create(is_remove=True)
        Path(os.path.join(cache_path.get_path(), 'bads')).create(is_remove=True)
    config = get_cfg_defaults()
    model = PL_Tm(config, pretrain_ckpt=model2_path.get_path(), edge_ckpt=model1_path.get_path())
    matcher = model.eval().cuda()
    n_parameters = sum(p.numel() for p in matcher.parameters() if p.requires_grad)
    print('Number of Model Params (M): %.2f' % (n_parameters / 1.e6))

    img_list = image_path.get_list(mode='img')
    img_list = [file for file in img_list if 'mask' not in file]
    if len(img_list):
        img_ext_name = os.path.splitext(img_list[0])[-1]

    losses, count, time_stt, time_infer = 0, 0, time.time(), 0.0
    for file in img_list:
        img_file = Path(os.path.join(image_path.get_path(), file))
        mask_file = Path(os.path.join(image_path.get_path(), file.split('.')[0] + '_mask.' + file.split('.')[-1]))
        homo_file = os.path.join(homo_path.get_path(), file.split('.')[0] + '_homo.npy')
        img = img_file.get_image()
        mask = mask_file.get_image()
        homo = np.load(homo_file)
        mkpts0, mkpts1, mconf, time_infer_end = get_matcher(img, mask, matcher, resize=resize,
                                                            threshold_conf=threshold_conf,
                                                            threshold_win=threshold_win,
                                                            visual_file=os.path.join(cache_path.get_path(), 'matchs',
                                                                                     file.split('.')[0]),
                                                            is_visual=is_visual)
        print("{} / {} : {}".format(count, len(img_list), time_infer_end))
        time_infer += time_infer_end

        ellipse = cv2.fitEllipse(get_contour_points(mask))
        gt_center = np.array((ellipse[0][0], ellipse[0][1]))

        if not (all(mkpts0.shape) and all(mkpts1.shape)):
            continue
        center = get_center(mkpts0, gt_center, mkpts1, mconf, mask.shape, resize, mode, is_recover)
        cv2.circle(img, (round(center[0]), round(center[1])), 0, (0, 0, 255), 1)
        if is_visual:
            cv2.imwrite(os.path.join(cache_path.get_path(), 'visuals', file.split('.')[0] + img_ext_name), img)

        gt_center = np.delete(np.matmul(homo, np.insert(gt_center, 2, 1, axis=0).T).T, 2, axis=0)

        loss = np.sqrt((center[0] - gt_center[0]) ** 2 + (center[1] - gt_center[1]) ** 2)

        if loss > th:
            print(file)
            save_file = Path(
                os.path.join(cache_path.get_path(), 'bads', file.split('.')[0] + '_' + str(loss) + img_ext_name))
            save_file.set_image(img)

        losses += loss
        count += 1

        print(
            'count: {} / {}, gt_center: {}, center: {}, loss: {}'.format(count, len(img_list), gt_center, center, loss))

    sum_time = time.time() - time_stt
    print('count: {}, num: {}'.format(count, len(img_list)))
    print('sum_loss: {}, avg_loss: {}'.format(losses, losses / count))
    print('sum_time: {}, avg_time: {}'.format(sum_time, sum_time / count))


if __name__ == '__main__':
    print('Start processing...')

    image_path = Path('/home/zwb/local/zwb/projects/Others/TMv5/datasets/hole_dataset.bak/hole/images/test')
    homo_path = Path('/home/zwb/local/zwb/projects/Others/TMv5/datasets/hole_dataset.bak/hole/homos/test')
    cache_path = Path('/home/zwb/local/zwb/projects/Others/TMv5/datasets/hole_dataset.bak/hole/caches/test')
    model1_path = Path(
        '/home/zwb/local/zwb/projects/Others/TMv5/codes/pidinet/trained_models/table5_pidinet-tiny-l.pth')
    model2_path = Path('/home/zwb/local/zwb/projects/Others/TMv5/models/version_3_whole/checkpoints/last.ckpt')

    resize = (352, 352)  # (height, width)
    # resize = (480, 640)  # (height, width)
    threshold_conf = 0.4  # threshold of confidence
    threshold_win = 50  # threshold of window size
    mode = 'direct'  # ['direct', 'search', 'optim']
    get_center_from_loftr(image_path, homo_path, cache_path, model1_path, model2_path, resize, threshold_conf,
                          threshold_win, is_visual=True, mode=mode, is_recover=False)

    print('End processing...')