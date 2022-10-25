from math import log
from loguru import logger

import torch
from einops import repeat
from kornia.utils import create_meshgrid


def compute_supervision_fine(data, config):
    data_source = data['dataset_name'][0]
    if data_source.lower() in ['synthetic', 'linemod_2d']:
        spvs_fine(data, config)
    else:
        raise NotImplementedError


def compute_supervision_coarse(data, config):
    assert len(set(data['dataset_name'])) == 1, "Do not support mixed datasets training!"
    data_source = data['dataset_name'][0]
    if data_source.lower() in ['synthetic', 'linemod_2d']:
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
    scale = config['TM']['RESOLUTION'][0]
    h0, w0, h1, w1 = map(lambda x: torch.div(x, scale, rounding_mode='trunc'), [H0, W0, H1, W1])  #

    scale_x = scale * data['scale'][:, 0] if 'scale' in data else scale
    scale_y = scale * data['scale'][:, 1] if 'scale' in data else scale

    if data['dataset_name'][0] == 'linemod_2d':
        # bias_mode
        bias = data['bias']
        # N,n,2
        pts_x = (data['pts_0'][:, :, 0] + torch.as_tensor(bias[:, 0, None] / scale_x[:, None], device=device)).round().long() # [N,L]x
        pts_y = (data['pts_0'][:, :, 1] + torch.as_tensor(bias[:, 1, None] / scale_y[:, None], device=device)).round().long() # [N,L]y
    else:
        # trans_mode
        trans = data['trans']
        pts_x = (trans[:,0,0,None] * (data['pts_0'][:, :, 0]*scale_x[:,None]) + trans[:,0,1,None] * (data['pts_0'][:, :, 1]*scale_y[:,None]) + trans[:,0,2,None])
        pts_y = (trans[:,1,0,None] * (data['pts_0'][:, :, 0]*scale_x[:,None]) + trans[:,1,1,None] * (data['pts_0'][:, :, 1]*scale_y[:,None]) + trans[:,1,2,None])
        pts_z = (trans[:,2,0,None] * (data['pts_0'][:, :, 0]*scale_x[:,None]) + trans[:,2,1,None] * (data['pts_0'][:, :, 1]*scale_y[:,None]) + trans[:,2,2,None])
        pts_x /= pts_z
        pts_y /= pts_z

    pts_x = (pts_x/scale_x[:,None]).round().long()
    pts_y = (pts_y/scale_y[:,None]).round().long()

    pts_image = torch.stack((pts_x, pts_y), dim=-1)

    # construct a gt conf_matrix
    L, S = data['pts_0'].shape[1],data['pts_1'].shape[1]
    conf_matrix_gt = torch.zeros(N, L, S, device=device)
    # i_ids is tempalte ids ,j_inds is image ids
    x_ids = torch.flatten(pts_image[:, :, 0])
    y_ids = torch.flatten(pts_image[:, :, 1]) # inds in image coordinate,but the j_inds is flatten,they are ready for j_ids
    b_ids, i_ids = torch.where(pts_image[:, :, 0] > -1e8) # get all index in sampled position

    mask_x = (x_ids < w1) * (x_ids>=0)
    mask_y = (y_ids < h1) * (y_ids>=0)

    mask = (mask_x * mask_y)# filter the out box points in image
    j_ids = w1*y_ids + x_ids
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

@torch.no_grad()
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
    scale_gap = config['RESOLUTION'][0]/config['RESOLUTION'][1] # 4
    h0, w0, h1, w1 = map(lambda x: torch.div(x, scale, rounding_mode='trunc'), [H0, W0, H1, W1])  #

    scale_x = scale * data['scale'][:, 0] if 'scale' in data else scale
    scale_y = scale * data['scale'][:, 1] if 'scale' in data else scale
    if is_cat_coarse:
        f_points = data['p_src'] /scale   # [bs,L,2]
        pt1_i = data['p_src']  # resized resolution
        w_pt0_i_bs = torch.Tensor().to(pt1_i.device)
        for b_id in range(data['bs']):
            b_mask = data['b_ids'] == b_id
            points_x = f_points[b_mask][:, 0] #(L)
            points_y = f_points[b_mask][:, 1]

            points_x = (points_x * scale_x[b_id])  # L
            points_y = (points_y * scale_y[b_id])  # L
        # # gt trans_mode
            trans = data['trans'][b_id]  # origin resolution
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
        points_x = f_points[:,:,0]  # (B,L)
        points_y = f_points[:,:,1]
        pt1_i = torch.stack((points_x*scale, points_y*scale), dim=-1) # resized resolution
        points_x = (points_x * scale_x[:, None]) # bs,L
        points_y = (points_y * scale_y[:, None]) # bs,L
        # # gt trans_mode
        trans = data['trans']  # origin resolution
        pts_x = (trans[:,0,0,None] * points_x + trans[:,0,1,None] * points_y + trans[:,0,2,None])
        pts_y = (trans[:,1,0,None] * points_x + trans[:,1,1,None] * points_y + trans[:,1,2,None])
        pts_z = (trans[:,2,0,None] * points_x + trans[:,2,1,None] * points_y + trans[:,2,2,None])
        pts_x /= pts_z
        pts_y /= pts_z
        points_x_i = (pts_x / scale_x[:, None])*scale
        points_y_i = (pts_y / scale_y[:, None])*scale # resized resolution

        trans = data['theta_inv']  # resized resolution
        pts_x = (trans[:, 0, 0,None] * points_x_i + trans[:, 0, 1,None] * points_y_i + trans[:, 0, 2,None])
        pts_y = (trans[:, 1, 0,None] * points_x_i + trans[:, 1, 1,None] * points_y_i + trans[:, 1, 2,None])
        pts_z = (trans[:, 2, 0,None] * points_x_i + trans[:, 2, 1,None] * points_y_i + trans[:, 2, 2,None])
        pts_x /= pts_z
        pts_y /= pts_z
        # 3. compute gt
        w_pt0_i = torch.stack((pts_x, pts_y), dim=-1)  # the template points transformed to the warped edge image(resized resolution)
        scale = config['RESOLUTION'][1]
        radius = W // 2

        r'''
         `expec_f_gt` might exceed the window, i.e. abs(*) > 1, which would be filtered later(never)
         expec_f_gt : to push the network learning the coordinates offset from the correspondence-level accuracy to subpixel accuracy
        '''

        b_ids, i_ids = data['b_ids_mask_fine'], data['i_ids_mask_fine']
        expec_f_gt = (w_pt0_i[b_ids, i_ids] - pt1_i[b_ids, i_ids]) / scale / radius  # [M, 2]
        data.update({"expec_f_gt": expec_f_gt})


