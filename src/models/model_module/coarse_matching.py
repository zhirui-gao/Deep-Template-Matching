import os
import itertools as it
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
import numpy as np
from PIL import Image, ImageDraw
INF = 1e9

def mask_border(m, b: int, v):
    """ Mask borders with value
    Args:
        m (torch.Tensor): [N, H0, W0, H1, W1]
        b (int)
        v (m.dtype)
    """
    if b <= 0:
        return

    m[:, :b] = v
    m[:, :, :b] = v
    m[:, :, :, :b] = v
    m[:, :, :, :, :b] = v
    m[:, -b:] = v
    m[:, :, -b:] = v
    m[:, :, :, -b:] = v
    m[:, :, :, :, -b:] = v


def mask_border_with_padding(m, bd, v, p_m0, p_m1):
    if bd <= 0:
        return

    m[:, :bd] = v
    m[:, :, :bd] = v
    m[:, :, :, :bd] = v
    m[:, :, :, :, :bd] = v

    h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
    h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()
    for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
        m[b_idx, h0 - bd:] = v
        m[b_idx, :, w0 - bd:] = v
        m[b_idx, :, :, h1 - bd:] = v
        m[b_idx, :, :, :, w1 - bd:] = v


def compute_max_candidates(p_m0, p_m1):
    """Compute the max candidates of all pairs within a batch

    Args:
        p_m0, p_m1 (torch.Tensor): padded masks
    """
    h0s, w0s = p_m0.sum(1).max(-1)[0], p_m0.sum(-1).max(-1)[0]
    h1s, w1s = p_m1.sum(1).max(-1)[0], p_m1.sum(-1).max(-1)[0]
    max_cand = torch.sum(
        torch.min(torch.stack([h0s * w0s, h1s * w1s], -1), -1)[0])
    return max_cand


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
            try:
                from superglue.superglue import log_optimal_transport
            except ImportError:
                raise ImportError("download superglue.py first!")
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
        feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1]**.5, [feat_c0, feat_c1])


        if self.match_type == 'dual_softmax':
            sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0,
                                      feat_c1) / self.temperature
            if mask_c1 is not None:
                sim_matrix.masked_fill_(
                    ~(mask_c0[..., None] * mask_c1[:, None]).bool(),
                    -INF)

            conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)  # dim is same


        elif self.match_type == 'sinkhorn':
            sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1)
            if mask_c1 is not None:
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



    @torch.no_grad()
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
        mask_v, all_j_ids = mask.max(dim=2)   # mask_v: [N,L]
        num_matches_pred = torch.sum(mask_v, dim=1).int()
        # truncated_num = torch.min(num_matches_pred).long()
        # num_bs = num_matches_pred.tolist()
        b_ids, i_ids = torch.where(mask_v) # i_ids : [N*L]
        j_ids = all_j_ids[b_ids, i_ids] #[n]
        mconf = conf_matrix[b_ids, i_ids, j_ids]

        if self.training and self.training_stage == 'whole':
            # the more predict,the less padding
            # num_cal = torch.clip(self.out_num_points-num_matches_pred*0.2, min=4, max=self.out_num_points)
            # only pad 4 point (in the case no enough points to calculate)
            num_to_padding = (torch.zeros(num_matches_pred.shape[0],device=_device)+4).long()
            # num_to_padding = torch.clip(self.out_num_points*0.8 - num_matches_pred, min=0)
            low = 0
            high = len(torch.where(data['spv_b_ids'] ==0)[0])
            mconf_gt = torch.zeros(len(data['spv_b_ids']), device=_device)  # set conf of gt paddings to all zero
            for i in range(len(num_to_padding)):
                if high > low:
                    gt_pad_indices = torch.randint(
                        low,
                        high,
                        (int(num_to_padding[i]),), device=_device)
                    b_ids = torch.cat([b_ids, data['spv_b_ids'][gt_pad_indices]],dim=0)
                    i_ids = torch.cat([i_ids, data['spv_i_ids'][gt_pad_indices]],dim=0)
                    j_ids = torch.cat([j_ids, data['spv_j_ids'][gt_pad_indices]],dim=0)
                    mconf = torch.cat([mconf, mconf_gt[gt_pad_indices]],dim=0)
                low = high
                high += len(torch.where(data['spv_b_ids'] == i+1)[0])
            sorted, indices = torch.sort(b_ids)
            b_ids = b_ids[indices]
            i_ids = i_ids[indices]
            j_ids = j_ids[indices]
            mconf = mconf[indices]

        # These matches select patches that feed into fine-level network
        coarse_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids}

        # 4. Update with matches in to original coordinate

        mkpts0_c = data['pts_0'][b_ids, i_ids]*scale  # [n,2]
        mkpts1_c = torch.stack(
            [j_ids % w1c, torch.div(j_ids, w1c, rounding_mode="floor")],  # j_ids // data['hw1_c'][1]
            dim=1)*scale  # [n,2]

        # for template,get it's real i_ids
        i_ids_fine = data['pts_0'][b_ids, i_ids][:, 1]*w0c + data['pts_0'][b_ids, i_ids][:,0]

        # These matches is the current prediction (for visualization)
        coarse_matches.update({
            'predict_mask': mconf != 0, # mconf == 0 => gt matches
            #'b_ids': b_ids[mconf != 0],  # if only train the coarse-stage ,use this instead !!!! but only affect training plot
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