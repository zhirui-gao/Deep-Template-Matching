import sys
sys.path.append(r"./pidinet")
# sys.path.append(r"paddleslim/demo/models")
import argparse
import os
import time
import cv2
import models
from models.convert_pidinet import convert_pidinet
from utils import *
from edge_dataloader import BSDS_VOCLoader, BSDS_Loader, Multicue_Loader, NYUD_Loader, Custom_Loader
from torch.utils.data import DataLoader

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

class Arg():
    def __init__(self):
        self.config = 'carv4'
        self.evaluate = './pidinet/trained_models/table5_pidinet-tiny-l.pth'
        self.evaluate_converted = True
        self.dil = False
        self.sa = False
        self.savedir = './pidinet/savedir' # no use

class Edge_Net(nn.Module):
    def __init__(self, config):
        super().__init__()
        # parser = argparse.ArgumentParser(description='PyTorch Pixel Difference Convolutional Networks')

        # parser.add_argument('--only-bsds', action='store_true',
        #                     help='only use bsds for training')
        # parser.add_argument('--ablation', action='store_true',
        #                     help='not use bsds val set for training')
        # parser.add_argument('--dataset', type=str, default='Custom',
        #                     help='data settings for BSDS, Multicue and NYUD datasets')
        #
        #
        # parser.add_argument('--sa', action='store_true',
        #                     help='use CSAM in pidinet')
        # parser.add_argument('--dil', action='store_true',
        #                     help='use CDCM in pidinet')
        # parser.add_argument('--config', type=str, default='carv4',
        #                     help='model configurations, please refer to models/config.py for possible configurations')
        # parser.add_argument('--seed', type=int, default=None,
        #                     help='random seed (default: None)')
        #
        # parser.add_argument('--checkinfo', action='store_true',
        #                     help='only check the informations about the model: model size, flops')
        #
        # parser.add_argument('--lr', type=float, default=0.005,
        #                     help='initial learning rate for all weights')
        # parser.add_argument('--lr-type', type=str, default='multistep',
        #                     help='learning rate strategy [cosine, multistep]')
        # parser.add_argument('--lr-steps', type=str, default=None,
        #                     help='steps for multistep learning rate')
        # parser.add_argument('--opt', type=str, default='adam',
        #                     help='optimizer')
        # parser.add_argument('--wd', type=float, default=1e-4,
        #                     help='weight decay for all weights')
        # parser.add_argument('-j', '--workers', type=int, default=4,
        #                     help='number of data loading workers')
        # parser.add_argument('--eta', type=float, default=0.3,
        #                     help='threshold to determine the ground truth (the eta parameter in the paper)')
        # parser.add_argument('--lmbda', type=float, default=1.1,
        #                     help='weight on negative pixels (the beta parameter in the paper)')
        #
        # parser.add_argument('--resume', action='store_true',
        #                     help='use latest checkpoint if have any')
        # parser.add_argument('--print-freq', type=int, default=10,
        #                     help='print frequency')
        # parser.add_argument('--save-freq', type=int, default=1,
        #                     help='save frequency')
        # parser.add_argument('--evaluate', type=str, default='./trained_models/table5_pidinet-tiny-l.pth',
        #                     help='full path to checkpoint to be evaluated')
        # parser.add_argument('--evaluate-converted',type=str,default='true',
        #                     help='convert the checkpoint to vanilla cnn, then evaluate')
        args = Arg()
        self.model = getattr(models, config['name'])(args)
        checkpoint = load_checkpoint(args)
        if args.evaluate_converted:
            state_dict = convert_pidinet(checkpoint['state_dict'], args.config)
            model_dict = self.model.state_dict()
            for k, v in state_dict.items():
                if k[7:] in model_dict.keys() and v.shape == model_dict[k[7:]].shape:
                    model_dict[k[7:]] = v  # 7 is to get out of module
            self.model.load_state_dict(model_dict, strict=True)

    def forward(self,image):
        _, _, H, W = image.shape  # [bs,3,h,w]
        results = self.model(image) # [bs,1,h,w] list len :5

        return results[-1] # [bs.1,h,w]








