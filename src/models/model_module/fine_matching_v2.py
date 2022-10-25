import math
import random
import logging
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid
from einops.einops import rearrange, repeat
from src.utils.supervision  import compute_supervision_fine

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
        data.update({'std':std})

        if self.photometric:
            self.get_photometric_loss(coords_normalized, data)
            # self.smooth_loss(coords_normalized, data)

        # compute absolute kpt coords
        self.get_fine_match(coords_normalized, data)

    def get_photometric_loss(self,coords_normed, data):
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
        grid = F.affine_grid(theta_batch, image1_unfold.size(),align_corners=True)
        IWarp = F.grid_sample(image1_unfold, grid,align_corners=True)
        IWarp = IWarp.reshape(-1, self.W ** 2)
        # mask_image0
        mask_image0_unfold = image0_unfold > 0 # [0/1] tensor
        loss_fine = ((((image0_unfold - IWarp) ** 2)*mask_image0_unfold).sum(-1) / (self.W ** 2)).mean()
        data.update({'loss_photometric': loss_fine})

    # @torch.no_grad()
    def get_fine_match(self, coords_normed, data):
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
        b,h,w = data['smooth_mask'].shape
        smooth_map = torch.zeros((b,h,w,2),device=data['smooth_mask'].device)
        smooth_b_ids= data['smooth_b_ids']
        smooth_y_ids= data['smooth_y_ids']
        smooth_x_ids= data['smooth_x_ids']
        smooth_map[smooth_b_ids,smooth_y_ids,smooth_x_ids,:] = coords_normed
        data.update({
            "smooth_map": smooth_map  # [b,W,W,2]
        })


