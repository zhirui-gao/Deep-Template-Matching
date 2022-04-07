from math import log
from loguru import logger

import torch
from einops import repeat
from kornia.utils import create_meshgrid




def compute_supervision_coarse(data, config):
    assert len(set(data['dataset_name'])) == 1, "Do not support mixed datasets training!"
    data_source = data['dataset_name'][0]
    if data_source.lower() in ['scannet', 'linemod_2d']:
        spvs_coarse(data, config)
    else:
        raise ValueError(f'Unknown data source: {data_source}')


@torch.no_grad()
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
    scale = 1
    scale_x = data['scale'][:, 0] if 'scale' in data else scale
    scale_y = data['scale'][:, 1] if 'scale' in data else scale
    bias = data['bias']
    # N,n,2
    pts_x = (data['pts_0'][:, :, 0] + torch.tensor(bias[:, 0, None] / scale_x[:, None], device=device)).round().long() # [N,L]x
    pts_y = (data['pts_0'][:, :, 1] + torch.tensor(bias[:, 1, None] / scale_y[:, None], device=device)).round().long() # [N,L]y
    pts_image = torch.stack((pts_x, pts_y), dim=-1)

    # construct a gt conf_matrix
    _, L, S =data['conf_matrix'].shape
    conf_matrix_gt = torch.zeros(N, L, S, device=device)
    # i_ids is tempalte ids ,j_inds is image ids
    x_ids = torch.flatten(pts_image[:, :, 0])
    y_ids = torch.flatten(pts_image[:, :, 1]) # inds in image coordinate,but the j_inds is flatten,they are ready for j_ids
    b_ids, i_ids = torch.where(pts_image[:, :, 0] >= 0)

    mask_x = x_ids < W1
    mask_y = y_ids < H1
    mask = (mask_x * mask_y)# filter the out box points in image
    j_ids = data['hw1'][1]*y_ids + x_ids
    b_ids, _ = torch.where(pts_image[:, :, 0] >= 0) #
    conf_matrix_gt[b_ids[mask], i_ids[mask], j_ids[mask]] = 1
    data.update({'conf_matrix_gt': conf_matrix_gt})
    # TODO:to check correspondence
    pass
