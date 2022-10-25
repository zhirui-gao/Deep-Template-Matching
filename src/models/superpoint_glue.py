# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from pathlib import Path
import torch
from torch import nn
import numpy as np
from einops.einops import rearrange

def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert(nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int, height: int, width: int):
    if k >= len(keypoints): # (y,x)
        _device = keypoints.device
        pad_num = k- len(keypoints)
        # random padding to k
        pad_0 = torch.tensor(np.random.randint(low=0, high=height, size=(pad_num, 1)),device=_device)
        pad_1 = torch.tensor(np.random.randint(low=0, high=width, size=(pad_num, 1)),device=_device)
        add = torch.cat([pad_0, pad_1], dim=1)  # [y,x]
        keypoints = torch.cat([keypoints, add], dim=0)
        pad_score = torch.zeros(pad_num,device=_device)
        scores = torch.cat([scores, pad_score], dim=0)
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w*s - s/2 - 0.5), (h*s - s/2 - 0.5)],
                              ).to(keypoints)[None]
    keypoints = keypoints*2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if int(torch.__version__[2]) > 2 else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)

    return descriptors


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
            descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1) #[ba,C,h,w]【60，80】
            for i in range(batch_size):
                b_x = torch.flatten(c_points[i][:, 0]).long()
                b_y = torch.flatten(c_points[i][:, 1]).long()
                descriptor_coarse = descriptors[i][:,b_y,b_x]# [C,l]
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
                keypoints_c.append(pts_int_b_c[:, [1, 0]])# tensor [N, 2(x,y)]

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
            'desc_f_2': descriptors_fine, # [bs,L,C],
            'desc_f_1': descriptors_fine_1, # [bs,L,C],
            'desc_c': descriptors_coarse, #[bs,l,C]
            'pts_int_c': pts_int_c,#  [bs,l,2]
            # 'pts_int_f': pts_int_f # [bs,L,2]
        }