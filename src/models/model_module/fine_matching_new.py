# 2022/8/10
import math
import random

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid
from einops.einops import rearrange, repeat

class Translate_network(nn.Module):
    """Builds an network that predicts the 2
     parameters ued in a affine transformation.
     """

    def __init__(self, config):
        super(Translate_network, self).__init__()
        W = config['fine']['fine_window_size']
        self.h, self.w = W,W


    def forward(self, img_a, img_b):
        x = torch.cat([img_a, img_b], 1)
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        dtheta = self.local(x)
        return dtheta



class FineMatching(nn.Module):
    """FineMatching with s2d paradigm"""

    def __init__(self,config):
        super(FineMatching, self).__init__()
        self.W = config['fine']['fine_window_size']

        self.conv1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256, eps=1e-05)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128, eps=1e-05)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64, eps=1e-05)

        self.fc1 = nn.Linear((self.W//2) ** 2 * 64, 128)
        self.fc2 = nn.Linear(128, 2)
        self.tanh = nn.Tanh()

        self._reset_parameters()
        self.fc2.bias.data.zero_()  # F.arctanh(0) is 0
        self.fc2.weight.data.zero_()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.in_channels != m.out_channels or m.out_channels != m.groups or m.bias is not None:
                    # don't want to reinitialize downsample layers, code assuming normal conv layers will not have these characteristics
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    print('Not initializing')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif type(m) == nn.Linear:
                nn.init.normal_(m.weight, std=0.01)

    def forward(self, feat_f0, feat_f1, data):
        """
        Args:
            feat0 (torch.Tensor): [M, WW, C]
            feat1 (torch.Tensor): [M, WW, C]
            data (dict)
        Update:
            data (dict):{
                'expec_f' (torch.Tensor): [M, 3],
                'mkpts0_f' (torch.Tensor): [M, 2],
                'mkpts1_f' (torch.Tensor): [M, 2]}s
        """
        M, WW, C = feat_f0.shape
        W = int(math.sqrt(WW))
        scale = data['resolution'][0]
        self.M, self.W, self.WW, self.C, self.scale = M, W, WW, C, scale
        device = feat_f0.device
        # self.paddingSize = self.W//2
        # corner case: if no coarse matches found
        if M == 0:
            assert self.training == False, "M is always >0, when training, see coarse_matching.py"

            # logger.warning('No matches found in coarse-level.')
            data.update({
                'expec_f': torch.empty(0, 3, device=feat_f0.device),
                'mkpts0_f': data['mkpts0_c'],
                'mkpts1_f': data['mkpts1_c'],
            })
            data.update({'warped_source': data['edge_warped']})
            loss_fine = (((data['edge_warped'] - data['warped_template']) ** 2).sum(-1) / (self.W ** 2)).mean()
            loss_fine.requires_grad_(True)
            data.update({'loss_f': loss_fine})
            return

        feat_f0 = nn.functional.normalize(feat_f0, p=2, dim=2)  # l2_normalize  # [bs,WW,256]
        feat_f1 = nn.functional.normalize(feat_f1, p=2, dim=2)  # l2_normalize

        feat_f0 = feat_f0.view(M, W, W, -1).permute(0, 3, 1, 2)
        feat_f1 = feat_f1.view(M, W, W, -1).permute(0, 3, 1, 2)
        x = torch.cat((feat_f0, feat_f1), dim=1)  # [bs,512,H,W]

        n, c, w, h = x.size()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.avg_pool2d(x, 2)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)


        x = x.contiguous().view(M, -1)
        x = self.fc1(x)  # [M 2] # offset
        x = F.dropout(x, training=self.training)
        offset = self.fc2(x)
        offset = self.tanh(offset)  # the output is in range[-2,2]

        # flow = offset[:,:,None,None] # [M,2,1,1]

        theta = torch.tensor([
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=torch.float).to(device)
        theta_batch = theta.repeat(offset.shape[0],1,1)
        # theta_batch = einops.repeat(theta, 'h w ->c h w', c=offset.shape[0])

        theta_batch[:, 0, 2] = offset[:, 0].clone()
        theta_batch[:, 1, 2] = offset[:, 1].clone()

        W, WW, C, scale = self.W, self.WW, self.C, self.scale
        stride = data['hw0_f'][0] // data['hw0_c'][0]

        image0_unfold = F.unfold(data['image0'], kernel_size=(W, W), stride=stride,padding=0)  # input: tensor数据，四维， Batchsize, channel, height, width
        image0_unfold = rearrange(image0_unfold, 'n (c ww) l -> n l ww c', ww=W ** 2)
        image0_unfold = image0_unfold[data['b_ids'], data['i_ids_fine']].squeeze(-1) # L,ww

        image1_unfold = F.unfold(data['edge_warped'], kernel_size=(W, W), stride=stride,
                                 padding=0)  # input: tensor数据，四维， Batchsize, channel, height, width
        image1_unfold = rearrange(image1_unfold, 'n (c ww) l -> n l ww c', ww=W ** 2)
        image1_unfold = image1_unfold[data['b_ids'], data['i_ids_fine']].squeeze(-1)  # L,ww


        self.get_fine_match(offset, data,image0_unfold, image1_unfold)
        self.update_loss(data, theta_batch, image0_unfold,image1_unfold)

    # @torch.no_grad()
    def get_fine_match(self, offset, data, image0_unfold=None,image1_unfold=None):
        # mkpts0_f and mkpts1_f
        if self.training:
            mkpts0_f = (data['mkpts0_c_pad']+self.W/2) + 0.0
            mkpts1_f = (data['mkpts0_c_pad']+self.W/2) - offset*(self.W/2)
        else:
            mkpts0_f = (data['mkpts0_c'] + self.W / 2) + 0.0
            mkpts1_f = (data['mkpts0_c'] + self.W / 2) - offset * (self.W / 2)
        data.update({
            "mkpts0_f": mkpts0_f,
            "mkpts1_f": mkpts1_f
        })



    def update_loss(self,data,theta_batch, image0_unfold, image1_unfold):
        stride = data['hw0_f'][0] // data['hw0_c'][0]

        # loss
        image1_unfold = image1_unfold.reshape(-1, 1, self.W, self.W)

        # IWarp = F.grid_sample(image0_unfold, flow, mode='nearest')

        grid = F.affine_grid(theta_batch, image1_unfold.size())
        IWarp = F.grid_sample(image1_unfold, grid)

        IWarp = IWarp.reshape(-1, self.W **2)

        # import matplotlib.pyplot as plt
        # import numpy as np
        # for i in range(20):
        #     i = random.randint(0, 100)
        #     plt.matshow(image0_unfold[i].reshape(self.W,self.W).detach().cpu().numpy())
        #     plt.savefig('./image/'+ str(i)+'_rawimg', bbox_inches='tight', pad_inches=0)
        #     plt.close()
        #
        #     plt.matshow(image1_unfold[i].reshape(self.W, self.W).detach().cpu().numpy())
        #     plt.savefig('./image/' + str(i) + '_edge', bbox_inches='tight', pad_inches=0)
        #     plt.close()
        #
        #     plt.matshow(IWarp[i].reshape(self.W, self.W).detach().cpu().numpy())
        #     plt.savefig('./image/' + str(i) + '_edge_warped', bbox_inches='tight', pad_inches=0)
        #     plt.close()


        # # visualize the patch in image
        # for i in range(8):
        #     import cv2
        #     cv2.imwrite('id_image' +str(i)+'.jpg', (255* I_target_unfold[i]).reshape(W, W).detach().cpu().numpy())

        loss_fine = (((image0_unfold - IWarp) ** 2).sum(-1) / (self.W ** 2)).mean()
        data.update({'loss_f': loss_fine})


































# import math
# import random
#
# import einops
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from kornia.geometry.subpix import dsnt
# from kornia.utils.grid import create_meshgrid
# from einops.einops import rearrange, repeat
#
# class Translate_network(nn.Module):
#     """Builds an network that predicts the 2
#      parameters ued in a affine transformation.
#      """
#
#     def __init__(self, config):
#         super(Translate_network, self).__init__()
#         W = config['fine_window_size']
#         self.h, self.w = W,W
#
#
#     def forward(self, img_a, img_b):
#         x = torch.cat([img_a, img_b], 1)
#         x = self.convs(x)
#         x = x.view(x.size(0), -1)
#         dtheta = self.local(x)
#         return dtheta
#
#
#
# class FineMatching(nn.Module):
#     """FineMatching with s2d paradigm"""
#
#     def __init__(self,config):
#         super(FineMatching, self).__init__()
#         self.W = config['fine_window_size']
#
#
#
#         self.conv1 = nn.Conv2d(in_channels=self.W ** 2, out_channels=256, kernel_size=3, stride=1, padding=1,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(256, eps=1e-05)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(128, eps=1e-05)
#
#         self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(64, eps=1e-05)
#
#         self.fc1 = nn.Linear((self.W//2) ** 2 * 64, 128)
#         self.fc2 = nn.Linear(128, 2)
#         self.tanh = nn.Tanh()
#
#         self._reset_parameters()
#         self.fc2.bias.data.zero_()  # F.arctanh(0) is 0
#         self.fc2.weight.data.zero_()
#
#     def _reset_parameters(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 if m.in_channels != m.out_channels or m.out_channels != m.groups or m.bias is not None:
#                     # don't want to reinitialize downsample layers, code assuming normal conv layers will not have these characteristics
#                     nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 else:
#                     print('Not initializing')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif type(m) == nn.Linear:
#                 nn.init.normal_(m.weight, std=0.01)
#
#     def forward(self, feat_f0, feat_f1, data):
#         """
#         Args:
#             feat0 (torch.Tensor): [M, WW, C]
#             feat1 (torch.Tensor): [M, WW, C]
#             data (dict)
#         Update:
#             data (dict):{
#                 'expec_f' (torch.Tensor): [M, 3],
#                 'mkpts0_f' (torch.Tensor): [M, 2],
#                 'mkpts1_f' (torch.Tensor): [M, 2]}s
#         """
#         M, WW, C = feat_f0.shape
#         W = int(math.sqrt(WW))
#         scale = data['resolution'][0]
#         self.M, self.W, self.WW, self.C, self.scale = M, W, WW, C, scale
#         device = feat_f0.device
#         # self.paddingSize = self.W//2
#         # corner case: if no coarse matches found
#         if M == 0:
#             assert self.training == False, "M is always >0, when training, see coarse_matching.py"
#
#             # logger.warning('No matches found in coarse-level.')
#             data.update({
#                 'expec_f': torch.empty(0, 3, device=feat_f0.device),
#                 'mkpts0_f': data['mkpts0_c'],
#                 'mkpts1_f': data['mkpts1_c'],
#             })
#             data.update({'warped_source': data['edge_warped']})
#             loss_fine = (((data['edge_warped'] - data['warped_template']) ** 2).sum(-1) / (self.W ** 2)).mean()
#             loss_fine.requires_grad_(True)
#             data.update({'loss_f': loss_fine})
#             return
#
#         feat_f0 = nn.functional.normalize(feat_f0, p=2, dim=2)  # l2_normalize
#         feat_f1 = nn.functional.normalize(feat_f1, p=2, dim=2)  # l2_normalize
#
#         sim_matrix = torch.einsum('mlc,mrc->mlr', feat_f0, feat_f1)  # [m,ww,ww]
#         coef = sim_matrix.view(M, W, W, -1).permute(0, 3, 1, 2)  # sim_matrix size is [M,ww,w,w] [128,64,8,8]
#
#         n, c, w, h = coef.size()
#         x = self.conv1(coef)
#         x = self.bn1(x)
#         x = self.relu(x)
#
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#
#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = self.relu(x)
#
#         x = F.avg_pool2d(x, 2)
#         x = x.contiguous().view(M, -1)
#         x = self.fc1(x)  # [M 2] # offset
#         x = F.dropout(x, training=self.training)
#         offset = self.fc2(x)
#         offset = self.tanh(offset)  # the output is in range[-2,2]
#
#         # flow = offset[:,:,None,None] # [M,2,1,1]
#
#         theta = torch.tensor([
#             [1, 0, 0],
#             [0, 1, 0]
#         ], dtype=torch.float).to(device)
#         theta_batch = theta.repeat(offset.shape[0],1,1)
#         # theta_batch = einops.repeat(theta, 'h w ->c h w', c=offset.shape[0])
#
#         theta_batch[:, 0, 2] = offset[:, 0].clone()
#         theta_batch[:, 1, 2] = offset[:, 1].clone()
#
#         W, WW, C, scale = self.W, self.WW, self.C, self.scale
#         stride = data['hw0_f'][0] // data['hw0_c'][0]
#
#         image0_unfold = F.unfold(data['image0'], kernel_size=(W, W), stride=stride,padding=0)  # input: tensor数据，四维， Batchsize, channel, height, width
#         image0_unfold = rearrange(image0_unfold, 'n (c ww) l -> n l ww c', ww=W ** 2)
#         image0_unfold = image0_unfold[data['b_ids'], data['i_ids_fine']].squeeze(-1) # L,ww
#
#         image1_unfold = F.unfold(data['edge_warped'], kernel_size=(W, W), stride=stride,
#                                  padding=0)  # input: tensor数据，四维， Batchsize, channel, height, width
#         image1_unfold = rearrange(image1_unfold, 'n (c ww) l -> n l ww c', ww=W ** 2)
#         image1_unfold = image1_unfold[data['b_ids'], data['i_ids_fine']].squeeze(-1)  # L,ww
#
#
#         self.get_fine_match(offset, data,image0_unfold, image1_unfold)
#         self.update_loss(data, theta_batch, image0_unfold,image1_unfold)
#
#     # @torch.no_grad()
#     def get_fine_match(self, offset, data, image0_unfold=None,image1_unfold=None):
#         # mkpts0_f and mkpts1_f
#         if self.training:
#             mkpts0_f = (data['mkpts0_c_pad']+self.W/2) + 0.0
#             mkpts1_f = (data['mkpts0_c_pad']+self.W/2) - offset*(self.W/2)
#         else:
#             mkpts0_f = (data['mkpts0_c'] + self.W / 2) + 0.0
#             mkpts1_f = (data['mkpts0_c'] + self.W / 2) - offset * (self.W / 2)
#         data.update({
#             "mkpts0_f": mkpts0_f,
#             "mkpts1_f": mkpts1_f
#         })
#
#
#
#     def update_loss(self,data,theta_batch, image0_unfold, image1_unfold):
#         stride = data['hw0_f'][0] // data['hw0_c'][0]
#
#         # loss
#         image1_unfold = image1_unfold.reshape(-1, 1, self.W, self.W)
#
#         # IWarp = F.grid_sample(image0_unfold, flow, mode='nearest')
#
#         grid = F.affine_grid(theta_batch, image1_unfold.size())
#         IWarp = F.grid_sample(image1_unfold, grid)
#
#         IWarp = IWarp.reshape(-1, self.W **2)
#
#         # import matplotlib.pyplot as plt
#         # import numpy as np
#         # for i in range(20):
#         #     i = random.randint(0, 100)
#         #     plt.matshow(image0_unfold[i].reshape(self.W,self.W).detach().cpu().numpy())
#         #     plt.savefig('./image/'+ str(i)+'_rawimg', bbox_inches='tight', pad_inches=0)
#         #     plt.close()
#         #
#         #     plt.matshow(image1_unfold[i].reshape(self.W, self.W).detach().cpu().numpy())
#         #     plt.savefig('./image/' + str(i) + '_edge', bbox_inches='tight', pad_inches=0)
#         #     plt.close()
#         #
#         #     plt.matshow(IWarp[i].reshape(self.W, self.W).detach().cpu().numpy())
#         #     plt.savefig('./image/' + str(i) + '_edge_warped', bbox_inches='tight', pad_inches=0)
#         #     plt.close()
#
#
#         # # visualize the patch in image
#         # for i in range(8):
#         #     import cv2
#         #     cv2.imwrite('id_image' +str(i)+'.jpg', (255* I_target_unfold[i]).reshape(W, W).detach().cpu().numpy())
#
#         loss_fine = (((image0_unfold - IWarp) ** 2).sum(-1) / (self.W ** 2)).mean()
#         data.update({'loss_f': loss_fine})
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
