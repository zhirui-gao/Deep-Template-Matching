import sys

sys.path.append(r"./pidinet")
import argparse
import os
from pidinet import models
from pidinet.models.convert_pidinet import convert_pidinet
from src.utils.utils import load_checkpoint
import torch


class Arg():
    def __init__(self):
        self.config = 'carv4'
        self.evaluate = './pidinet/trained_models/table5_pidinet-tiny-l.pth'
        self.evaluate_converted = True
        self.dil = False
        self.sa = False
        self.savedir = './pidinet/savedir'  # no use


class Edge_Net(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
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

    def forward(self, image):
        _, _, H, W = image.shape  # [bs,3,h,w]
        results = self.model(image)  # [bs,1,h,w] list len :5

        return results[-1]  # [bs.1,h,w]
