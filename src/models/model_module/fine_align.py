import math

import einops
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid
from einops.einops import rearrange, repeat
from src.models.model_module.stn import Homo_Net,STN2D,Affine_Net,Affine_Net_Whole

class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """
    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred,_device):
        I = y_true
        J = y_pred

        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [15] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(_device)

        pad_no = math.floor(win[0]/2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1,1)
            padding = (pad_no, pad_no)
        else:
            stride = (1,1,1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J
        I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)

    def loss_mask(self, y_true, y_pred,data,_device):
        I = y_true
        J = y_pred

        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(_device)

        pad_no = math.floor(win[0]/2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1,1)
            padding = (pad_no, pad_no)
        else:
            stride = (1,1,1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J
        I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        # mask form [bs,c,h,w] - > [bs,c,L]
        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        cc = cc.squeeze(1)

        scale = data['resolution'][0]
        cc = cc[data['b_ids'], data['i_ids_fine_y']*scale, data['i_ids_fine_x']*scale] # L,ww

        return -torch.mean(cc)

class FineAlign(nn.Module):
    """FineMatching with s2d paradigm"""

    def __init__(self, config):
        super(FineAlign, self).__init__()
        self.W = config['fine']['fine_window_size']
        # self.stn = Homo_Net(input_size=[self.W, self.W], input_channels=2, warp_size=(640,480) ,config=config['fine'])
        self.stn = Affine_Net_Whole(input_size=[self.W, self.W], input_channels=2, config=config['fine'])
        self.ncc = NCC(win=[self.W, self.W])
    # def forward_apart(self, feat_f0, feat_f1, data):
    #     """
    #     Args:
    #         feat0 (torch.Tensor): [M, WW, C]
    #         feat1 (torch.Tensor): [M, WW, C]
    #         data (dict)
    #     Update:
    #         data (dict):{
    #             'expec_f' (torch.Tensor): [M, 3],
    #             'mkpts0_f' (torch.Tensor): [M, 2],
    #             'mkpts1_f' (torch.Tensor): [M, 2]}
    #     """
    #     M, _, C = feat_f0.shape
    #     W = self.W
    #     scale = data['resolution'][0]
    #     self.M, self.W, self.WW, self.C, self.scale = M, W, W*W, C, scale
    #     self.device = feat_f0.device
    #     self.paddingSize = self.W//2
    #     # corner case: if no coarse matches found
    #     if data['b_ids'].shape[0] == 0:
    #         # TODO： imitate the loftr,to pad some true correspondences.
    #         assert self.training == False, "M is always >0, when training, see coarse_matching.py"
    #         # logger.warning('No matches found in coarse-level.')
    #         data.update({
    #             'expec_f': torch.empty(0, 3, device=feat_f0.device),
    #             'mkpts0_f': data['mkpts0_c'].detach().cpu(),
    #             'mkpts1_f': data['mkpts1_c'].detach().cpu(),
    #             'loss_f': torch.tensor(1.0),
    #         })
    #         # data['loss_f'].requires_grad_(True)
    #         return
    #
    #
    #     W, WW, C, scale = self.W, self.WW, self.C, self.scale
    #     stride = data['hw0_f'][0] // data['hw0_c'][0]
    #
    #     image0_unfold = F.unfold(data['image0'], kernel_size=(W, W), stride=stride, padding=W//2)  # input: tensor数据，四维， Batchsize, channel, height, width
    #     image0_unfold = rearrange(image0_unfold, 'n (c ww) l -> n l ww c', ww=W ** 2)
    #     image0_unfold = image0_unfold[data['b_ids'], data['i_ids_fine']].squeeze(-1) # L,ww
    #     image0_unfold = image0_unfold.reshape(-1, 1, self.W, self.W)
    #     # tempalte patch show
    #     # import matplotlib.pyplot as plt
    #     # import numpy as np
    #     # for i in range(128,138):
    #     #     plt.matshow(image0_unfold[i].reshape(W,W).detach().cpu().numpy())
    #     #     plt.savefig(('./image/id_img' + str(i)+"_warp"), bbox_inches='tight', pad_inches=0)
    #     #     plt.close()
    #
    #     I_target_unfold = F.unfold(data['edge_warped'], kernel_size=(self.W, self.W), stride=stride,  # edge
    #                                padding=self.W // 2)  # input: tensor数据，四维， Batchsize, channel, height, width
    #     I_target_unfold = rearrange(I_target_unfold, 'n (c ww) l -> n l ww c', ww=self.W ** 2)
    #     I_target_unfold = I_target_unfold[data['b_ids'], data['j_ids_warped']].squeeze(-1)  # L,ww
    #     I_target_unfold = I_target_unfold.reshape(-1, 1, self.W, self.W)
    #
    #     # for i in range(128,150):
    #     #     plt.matshow(I_target_unfold[i].reshape(W, W).detach().cpu().numpy())
    #     #     plt.savefig(('./image/id_img' + str(i)+"_target"), bbox_inches='tight', pad_inches=0)
    #     #     plt.close()
    #
    #     self.stn(torch.cat((image0_unfold, I_target_unfold), dim=1),data) # forward function is to get self.theta
    #
    #     # Homography Net
    #     # warped_source = self.stn.warp_image_homo(data['image0_warped'], device=self.device) # need change
    #
    #     # Affine transformation
    #     # warped_source = self.stn.warp_image(data['image0'], device=self.device) #TODO: which is better??
    #     #
    #     # dis_warped = (warped_source - data['edge_warped']) ** 2
    #     #
    #     # dis_warped = F.unfold(dis_warped, kernel_size=(self.W, self.W), stride=stride, padding=self.W // 2)
    #     # dis_warped = rearrange(dis_warped, 'n (c ww) l -> n l ww c', ww=self.W ** 2)
    #     # dis_warped = dis_warped[data['b_ids'], data['i_ids_fine']].squeeze(-1)  # L,ww
    #     # loss_fine = (dis_warped.sum(-1) / (self.W ** 2)).mean()
    #
    #
    #     # loss
    #     warped_source = self.stn.warp_image(data['edge_warped'], device=self.device)  # TODO: which is better??
    #
    #     # loss_fine = self.chamfer_loss_one_patch(warped_source, image0_unfold.squeeze(1), self.W, stride, data)
    #
    #     # patch loss
    #     # dis_warped = (warped_source - data['image0']) ** 2
    #     # dis_warped = F.unfold(dis_warped, kernel_size=(self.W, self.W), stride=stride, padding=self.W // 2)
    #     # dis_warped = rearrange(dis_warped, 'n (c ww) l -> n l ww c', ww=self.W ** 2)
    #     # dis_warped = dis_warped[data['b_ids'], data['i_ids_fine']].squeeze(-1)  # L,ww
    #     # loss_fine_2 = (dis_warped.sum(-1) / (self.W ** 2)).mean()
    #     # data.update({'loss_f': loss_fine_2})
    #
    #     # whole image  mes loss
    #     # dis_warped = (warped_source - data['image0']) ** 2
    #     # loss_fine_2 = (dis_warped.sum(-1)).mean()
    #     # data.update({'loss_f': loss_fine_2})
    #
    #     # ncc loss
    #     loss_ncc = self.ncc.loss_mask(warped_source, data['image0'],data, _device=self.device) + 1
    #     data.update({'loss_f': loss_ncc})
    #
    #     # data.update({'loss_f': 0.02*loss_fine + loss_fine_2*5})
    #
    #     self.get_fine_match_affine(self.stn.theta, data)

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
                'mkpts1_f' (torch.Tensor): [M, 2]}
        """
        M, _, C = feat_f0.shape
        W = self.W
        scale = data['resolution'][0]
        self.M, self.W, self.WW, self.C, self.scale = M, W, W*W, C, scale
        self.device = feat_f0.device
        self.paddingSize = self.W//2
        # corner case: if no coarse matches found
        if data['b_ids'].shape[0] == 0:
            assert self.training == False, "M is always >0, when training, see coarse_matching.py"
            # logger.warning('No matches found in coarse-level.')
            data.update({
                'expec_f': torch.empty(0, 3, device=feat_f0.device),
                'mkpts0_f': data['mkpts0_c'].detach().cpu(),
                'mkpts1_f': data['mkpts1_c'].detach().cpu(),
                'loss_f': torch.tensor(1.0),
            })
            warped_source = data['edge_warped']
            data.update({'warped_source': warped_source})

            data.update({'trans_predict': data['theta']})
            return



        # src: data['image0']   target: data['edge_warped']

        self.stn(data['image0'], data['edge_warped'], data)  # forward function is to get self.theta



        # loss
        warped_source = self.stn.warp_image(data['edge_warped'], device=self.device)  # TODO: which is better??
        warped_template = self.stn.warp_image_inverse(data['warped_template'], device=self.device)  # TODO: which is better??
        data.update({'warped_source': warped_source})
        data.update({'warped_template': warped_template})

        # patch loss
        # dis_warped = (warped_source - data['image0']) ** 2
        # dis_warped = F.unfold(dis_warped, kernel_size=(self.W, self.W), stride=stride, padding=self.W // 2)
        # dis_warped = rearrange(dis_warped, 'n (c ww) l -> n l ww c', ww=self.W ** 2)
        # dis_warped = dis_warped[data['b_ids'], data['i_ids_fine']].squeeze(-1)  # L,ww
        # loss_fine_2 = (dis_warped.sum(-1) / (self.W ** 2)).mean()
        # data.update({'loss_f': loss_fine_2})

        # whole image  mes loss
        self.get_fine_match_affine(self.stn.theta[:, 0:2, :], data)


    def chamfer_loss(self, warped_source, target, W, stride,data):
        # warped_source:[bs,1,H,W] 0~1
        # target:[bs,1,H,W] 0/1
        # TODO: maybe add weight in chamfer_loss
        warped_source = F.unfold(warped_source, kernel_size=(W, W), stride=stride, padding=self.W // 2) # [bs,ww*c,l]
        warped_source = rearrange(warped_source, 'n (c ww) l -> n l ww c', ww=self.W ** 2)
        warped_source = warped_source[data['b_ids'], data['i_ids_fine']].squeeze(-1) #[b,w*w]
        weight = warped_source.unsqueeze(-1)
        warped_source = warped_source.reshape(-1, W, W)  # L,w,w

        x, y = torch.meshgrid([torch.arange(W), torch.arange(W)])

        target = F.unfold(target, kernel_size=(W, W), stride=stride, padding=self.W // 2)
        pass

    def chamfer_loss_one_patch(self,warped_source_image, image0_unfold, W, stride,data):
        warped_source = F.unfold(warped_source_image, kernel_size=(W, W), stride=stride, padding=self.W // 2)  # [bs,ww*c,l]
        warped_source = rearrange(warped_source, 'n (c ww) l -> n l ww c', ww=self.W ** 2)
        warped_source = warped_source[data['b_ids'], data['i_ids_fine']].squeeze(-1)  # [b,w*w]
        warped_source = warped_source.reshape(-1, W, W)  # b,w,w
        loss = 0
        mun_patch = warped_source.shape[0]
        for i in range(mun_patch):
            points_src = torch.nonzero(torch.gt(warped_source[i], 1e-8))  #[n,2]
            weight = warped_source[i][points_src[:,0],points_src[:,1]]
            points_tgt = torch.nonzero(torch.gt(image0_unfold[i], 1e-8))  #[n,2]
            if points_src.shape[0]==0:
                points_src = torch.tensor([[0,0]],device=warped_source.device)
                weight = torch.tensor([1],device=warped_source.device)
            if points_tgt.shape[0]==0:
                points_tgt = torch.tensor([[W-1,W-1]],device=warped_source.device)
            if points_tgt.shape[0]==0 and points_src.shape[0]==0:
                points_src = torch.tensor([[0, 0]], device=warped_source.device)
                weight = torch.tensor([1], device=warped_source.device)
                points_tgt = torch.tensor([[W - 1, W - 1]], device=warped_source.device)
            loss += self.chamfer_loss(points_src.unsqueeze(0).float(), points_tgt.unsqueeze(0).float(),weight)
        return loss/mun_patch


    def chamfer_loss(self, points_src, points_tgt, weight=None):
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
            weight = (1 + (torch.sigmoid(1/weight-1)-0.5)*5)
            dist_matrix = ((points_src.unsqueeze(2) - points_tgt.unsqueeze(1)) ** 2).sum(-1) #[bs, L,S]
            dist_matrix = dist_matrix * (weight.unsqueeze(-1))
            dist_complete = (dist_matrix.min(-1)[0]).mean(-1)
            dist_acc = (dist_matrix.min(-2)[0]).mean(-1)
            dist = ((dist_acc + dist_complete) / 2).mean()
        else:
            dist_matrix = ((points_src.unsqueeze(2) - points_tgt.unsqueeze(1)) ** 2).sum(-1)
            dist_complete = (dist_matrix.min(-1)[0]).mean(-1)
            dist_acc = (dist_matrix.min(-2)[0]).mean(-1)
            dist = ((dist_acc + dist_complete) / 2).mean()
        return dist

    @torch.no_grad()
    def get_fine_match_homo(self, theta, data):
        stride = data['hw0_f'][0] // data['hw0_c'][0]
        image0_unfold = F.unfold(data['image0'], kernel_size=(self.W, self.W), stride=stride,
                                 padding=self.W // 2)  # input: tensor数据，四维， Batchsize, channel, height, width
        image0_unfold = rearrange(image0_unfold, 'n (c ww) l -> n l ww c', ww=self.W ** 2)
        image0_unfold = image0_unfold[data['b_ids'], data['i_ids_fine']].squeeze(-1)  # L,ww


        _device = theta.device
        bs = theta.shape[0]

        # tempalte patch show
        # import matplotlib.pyplot as plt
        # import numpy as np
        # for i in range(8):
        #     plt.matshow(image0_unfold[i].reshape(self.W,self.W).detach().cpu().numpy())
        #     plt.title('id_raw'+str(i))
        #     plt.show()


        index_nonzero =torch.argmax(image0_unfold, dim=1)
        coords_x = (index_nonzero % self.W).reshape(-1, 1)
        coords_y = torch.div(index_nonzero, self.W, rounding_mode='trunc').reshape(-1, 1)
        coords_points = torch.cat((coords_x, coords_y), dim=-1)  # [x,y]
        _, C, H, W = data['image0'].shape

        M = torch.einsum('bij,bjk->bik', theta.inverse(), data['theta'])

        mkpts0_f = (data['mkpts0_c']-self.W//2) + coords_points

        h = data['image0'].shape[2]
        w = data['image0'].shape[3]
        wh = torch.tensor([w, h], device=data['image0'].device)
        mkpts0_f_normalized = (mkpts0_f.reshape(bs, mkpts0_f.size(0) // bs, -1) / wh) * 2 - 1
        ones = torch.ones([mkpts0_f_normalized.shape[0], mkpts0_f_normalized.shape[1], 1], device=data['image0'].device)
        mkpts0_f_normalized = torch.cat([mkpts0_f_normalized, ones], dim=2).unsqueeze(3)

        mkpts1_f = torch.einsum('bij,bLjk->bLik', M, mkpts0_f_normalized.float()).squeeze(-1)
        mkpts1_f = (mkpts1_f / (mkpts1_f[:, :, 2].unsqueeze(2)))[:, :, 0:2]
        mkpts1_f = ((mkpts1_f + 1) / 2 * wh)

        mkpts0_f = mkpts0_f
        mkpts1_f = mkpts1_f.reshape(mkpts1_f.shape[1]*bs,-1)[:, [0, 1]]

        # mkpts0_f = data['mkpts0_c']
        # mkpts1_f = data['mkpts1_c']
        # print('only coarse!!!')

        data.update({
            "mkpts0_f": mkpts0_f.detach().cpu(),
            "mkpts1_f": mkpts1_f.detach().cpu()
        })

    # @torch.no_grad()
    def get_fine_match_affine(self, theta_affine, data):
        theta = theta_affine
        stride = data['hw0_f'][0] // data['hw0_c'][0]
        image0_unfold = F.unfold(data['image0'], kernel_size=(self.W, self.W), stride=stride,
                                 padding=self.W // 2)  # input: tensor数据，四维， Batchsize, channel, height, width
        image0_unfold = rearrange(image0_unfold, 'n (c ww) l -> n l ww c', ww=self.W ** 2)
        image0_unfold = image0_unfold[data['b_ids'], data['i_ids_fine']].squeeze(-1)  # L,ww

        _device = theta.device
        bs = theta.shape[0]
        # print('theta:',theta)
        # tempalte patch show
        # import matplotlib.pyplot as plt
        # import numpy as np
        # for i in range(8):
        #     plt.matshow(image0_unfold[i].reshape(self.W,self.W).detach().cpu().numpy())
        #     plt.title('id_raw'+str(i))
        #     plt.show()


        index_nonzero =torch.argmax(image0_unfold, dim=1)
        coords_x = (index_nonzero % self.W).reshape(-1, 1)
        coords_y = torch.div(index_nonzero, self.W, rounding_mode='trunc').reshape(-1, 1)
        coords_points = torch.cat((coords_x, coords_y), dim=-1)  # [x,y]

        ones = torch.tensor([0, 0, 1], device=_device).unsqueeze(1).reshape(1, -1)
        ones = ones.repeat(bs, 1, 1)
        theta = torch.cat((theta, ones), dim=1)

        _, C, H, W = data['image0'].shape
        T = torch.tensor([[[2 / W, 0, -1],
                           [0, 2 / H, -1],
                           [0, 0, 1]]], device=_device)
        T_inv = torch.linalg.inv(T)
        T = T.repeat(bs, 1, 1)
        T_inv = T_inv.repeat(bs, 1, 1)
        M = torch.einsum('bij,bjk-> bik', T_inv, torch.linalg.inv(theta))
        M = torch.einsum('bij,bjk->bik', M, T)

        M = torch.einsum('bij,bjk->bik', M.inverse(), data['theta'])

        mkpts0_f = (data['mkpts0_c']-self.W//2) + coords_points
        coords_ones = torch.ones_like(coords_y)
        mkpts0_f = torch.cat((mkpts0_f, coords_ones), dim=-1)  # [x,y]
        mkpts0_f = mkpts0_f.unsqueeze(-1).reshape(bs, mkpts0_f.shape[0]//bs, -1, 1)

        data.update({'trans_predict': M})  # sourse -> target

        mkpts1_f = torch.einsum('bij,bLjk->bLik', M, mkpts0_f.float()).squeeze(-1)

        mkpts1_f = (mkpts1_f / (mkpts1_f[:, :, 2].unsqueeze(2)))

        mkpts0_f = mkpts0_f.reshape(mkpts0_f.shape[1]*bs, -1)[:, [0, 1]]

        mkpts1_f = mkpts1_f.reshape(mkpts1_f.shape[1]*bs, -1)[:, [0, 1]]



        # mkpts0_f = data['mkpts0_c']
        # mkpts1_f = data['mkpts1_c']
        # print('only coarse!!!')

        data.update({
            "mkpts0_f": mkpts0_f.detach().cpu(),
            "mkpts1_f": mkpts1_f.detach().cpu()
        })

