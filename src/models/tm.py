import logging
import kornia
from src.models.SuperPointNet_gs import SuperPointNet_gauss2
import torch
from src.models.superpoint_glue import SuperPoint_glue
import torch.nn as nn
import os
import numpy as np
from src.models.PositionEncodingSine import PositionEncodingSine_line,RoFormerPositionEncoding
from src.models.model_module.transformer import LocalFeatureTransformer
from src.models.model_module.fine_preprocess import FinePreprocess
from src.models.model_module.coarse_matching import CoarseMatching
from src.models.model_module.fine_matching_v2 import FineMatching
from src.models.model_module.final_trans import FinalTrans
from pidinet.edge_net import Edge_Net

class Backnone(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config

        self.patch_size = self.config['tm']['superpoint']['patch_size']
        self.nms_dist = self.config['tm']['superpoint']['nms_dist'],
        self.conf_thresh = self.config['tm']['superpoint']['conf_thresh']
        self.backbone_name = self.config['tm']['superpoint']['name']

        self.resolution = self.config['tm']['resolution']
        self.use_edge = self.config['tm']['match_coarse']['use_edge']
        # Modules
        self.edge_net = Edge_Net(self.config['tm']['edge'])

        if self.backbone_name =='SuperPointNet_gauss2':
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
            'hw0_c': (data['image0'].shape[2]//self.resolution[0], data['image0'].shape[3]//self.resolution[0]),
            'hw1_c': (data['image1'].shape[2]//self.resolution[0],data['image1'].shape[3]//self.resolution[0]),
            'hw0_f': (data['image0'].shape[2]//self.resolution[1], data['image0'].shape[3]//self.resolution[1]),
            'hw1_f': (data['image1'].shape[2]//self.resolution[1],data['image1'].shape[3]//self.resolution[1])
        })
        # 2. get image's deep edge map
        if self.use_edge:
            deep_edge = self.edge_net(data['image1_rgb'])  # [N, 1, H, W]
            data.update({'pidinet_out': deep_edge})
            mask = data['image1_edge'] > 0.1 # canny mask
            and_edge = deep_edge * mask
        else:
            and_edge = data['image1_rgb'][:,0,:,:].unsqueeze(1)
        # 2. get image's deep edge map
        data.update({'edge': and_edge}) #deep_edge
        # 3. get superpoint's feature
        outs_post0, outs_post1 = self.backbone(data['image0'],c_points=data['c_points']), self.backbone(data['edge'], choose=False)
        pos0 = outs_post0['pts_int_c']
        pos1 = outs_post1['pts_int_c']
        data.update({
            'pts_0': pos0,
            'pts_1': pos1,
        })
        return outs_post0, outs_post1


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
                theta_each, lable = self.cal_trans.cal_trans_homo(point_A, point_B, weights, is_training)
                theta_inv.append(theta_each.inverse())
            except:
                # The diagonal element 1 is zero, the inversion could not be completed because the input matrix is singular.
                logging.warning("whole training is impossible:there is an error when calculating the H_c matrix")
                theta_each = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float,
                                     device=data['image0'].device).view(1,3,3)
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
        pts_x = trans[:, 0, 0, None] * position[:, :, 0] + trans[:, 0, 1, None] * position[:, :, 1] + trans[:, 0, 2, None]
        pts_y = trans[:, 1, 0, None] * position[:, :, 0] + trans[:, 1, 1, None] * position[:, :, 1] + trans[:, 1, 2, None]
        pts_z = trans[:, 2, 0, None] * position[:, :, 0] + trans[:, 2, 1, None] * position[:, :, 1] + trans[:, 2, 2, None]
        new_position = torch.cat([(pts_x / pts_z)[:, :, None], (pts_y / pts_z)[:, :, None]], dim=-1)
        return new_position

    def forward(self, data,outs_post0,outs_post1, backbone):
        # 3. positional encoding
        assert (self.position == 'rotary')
        pos_encoding_0 = self.rotaty_encoding(data['pts_0'])  # pos0:[bs,N,2] # pos_encoding_0 [bs,N,dim,2]
        pos_encoding_1 = self.rotaty_encoding(data['pts_1'])  # pos1:[bs,N,2]

        feat_c0 = outs_post0['desc_c']  # bs,l,c
        feat_c1 = outs_post1['desc_c']

        # 4. attention stage
        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0 = data['mask0'] #[bs,L]
        if 'mask1' in data:
            mask_c1 = data['mask1'].flatten(-2) # [bs,S]

        feat_c0, feat_c1 = self.LM_coarse(feat_c0, feat_c1, pos_encoding_0, pos_encoding_1, mask_c0, mask_c1)
        #  get transformed feature(visualization)
        # from src.utils.feature_visualize_RGB import visualize_attention_feature_batch
        # visualize_attention_feature_batch(feat_c0,feat_c1,data,data['pts_0'])

        # 5. match coarse-level, cal_transformation match coarse-level

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
                feat_c0, feat_c1 = self.LM_fine(feat_c0_stage2, feat_c1_stage2, pos_encoding_0, pos_encoding_0, mask_c0, mask_c1)
            else:
                feat_f1 = backbone(warped, choose=False, early_return=True)

            feat_f1 = feat_f1.reshape(feat_f1.shape[0], data['hw1_f'][0], data['hw1_f'][1], -1).permute(0, 3,  1, 2)
            feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, data, feat_c0,feat_c1) #[b,w*w,c]
            # 4.2 self-cross attention in local windows
            if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
                grid = np.mgrid[:self.W, :self.W]
                grid = grid.reshape((2, -1))
                grid = grid.transpose(1, 0)
                grid = grid[:, [1, 0]]
                pos = torch.tensor(grid).to(data['image0'].device).repeat(feat_f0_unfold.size(0),1,1)
                pos_encoding = self.rotaty_encoding_fine(pos)  # pos0:[bs,N,2] # pos_encoding_0 [bs,N,dim,2]
                feat_f0_unfold, feat_f1_unfold = self.loftr_fine(feat_f0_unfold, feat_f1_unfold, pos_encoding, pos_encoding)


            # 5. match fine-level
            self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)
            # 6. plot
            warped_template =[]
            trans_bs = []
            mkpts1_f_bs = torch.Tensor().to(feat_f0.device)
            for b_id in range(data['bs']):
                b_mask = data['b_ids'] == b_id
                kpts0 = data['mkpts0_f'][b_mask].unsqueeze(0)
                kpts1 = data['mkpts1_f'][b_mask].unsqueeze(0)

                theta_inv = data['theta'][b_id]
                pts_x = (theta_inv[0, 0] * kpts1[0, :, 0] + theta_inv[0, 1] * kpts1[0, :, 1] + theta_inv[0, 2])
                pts_y = (theta_inv[1, 0] * kpts1[0, :, 0] + theta_inv[1, 1] * kpts1[0, :, 1] + theta_inv[1, 2])
                pts_z = (theta_inv[2, 0] * kpts1[0, :, 0] + theta_inv[2, 1] * kpts1[0, :, 1] + theta_inv[2, 2])
                pts_x /= pts_z
                pts_y /= pts_z
                kpts1 = torch.stack([pts_x, pts_y],dim=1)
                try:
                    # trans = kornia.geometry.homography.find_homography_dlt(kpts0, kpts1.unsqueeze(0))
                    weights = data['std'][b_mask].unsqueeze(0)  # if mconf_patch is empty , eyes matrix is correct
                    trans, lable = self.cal_trans.cal_trans_homo(kpts0, kpts1.unsqueeze(0), weights)
                    warped_template.append(
                        (kornia.geometry.transform.warp_perspective(data['image0'][b_id].unsqueeze(0),
                                                                    trans, dsize=[h, w])))
                except:
                    logging.warning('seldom: there is an error calculating the fine matches for training padding')
                    trans = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float,
                                         device=data['image0'].device).view(1, 3, 3)
                    warped_template.append((kornia.geometry.transform.warp_perspective(data['image0'][b_id].unsqueeze(0),
                                                                                   trans, dsize=[h, w])))
                trans_bs.append(trans)
                mkpts1_f_bs = torch.cat([mkpts1_f_bs,kpts1],dim=0)

            warped_template_bs = torch.cat(warped_template, dim=0)
            trans_bs = torch.cat(trans_bs, dim=0)

            data.update({'warped_template': warped_template_bs})
            data.update({'trans_predict': trans_bs})  # sourse -> target
            data.update({
                "mkpts0_f": data['mkpts0_f'].detach().cpu(),
                "mkpts1_f": mkpts1_f_bs.detach().cpu()
            })
        # save result image
        plot = False
        if plot:
            from plotting import _make_evaluation_figure, plot_warped, distance_M, eval_predict_homography
            output_path = './result_plot'
            scale = data['scale'][0].detach().cpu()

            mkpts0_f_scaled = data['mkpts0_f'].detach().cpu()* scale
            mkpts1_f_scaled = data['mkpts1_f'].detach().cpu()* scale
            H_pred = kornia.geometry.homography.find_homography_dlt(mkpts0_f_scaled[None,:,:], mkpts1_f_scaled[None,:,:])[0].detach().cpu().numpy()

            mkpts0_f = data['mkpts0_f'].cpu().numpy()
            mkpts1_f = data['mkpts1_f'].cpu().numpy()
            img_name =data['pair_names'][0][0].split('/')[-4]+'_'+os.path.basename(data['pair_names'][0][0])
            trans = data['trans'][0].detach().cpu().numpy()
            mean_dist, correctness = eval_predict_homography(points=data['points_template'][0].detach().cpu().numpy(),
                                                             h_gt=trans,
                                                             H_pred=H_pred)

            keypoints_error = distance_M(mkpts0_f_scaled, mkpts1_f_scaled, trans).numpy()
            _make_evaluation_figure(data['image0_mask'][0].detach().cpu().numpy(), data['image1'][0][0].detach().cpu().numpy(),
                                    mkpts0_f, mkpts1_f, keypoints_error, mean_dist, 'Ours',
                                    path=output_path + '/' + img_name + '.png')

            # plot_warped(data['image0_raw'][0].detach().cpu().numpy(), data['image1_raw'][0].detach().cpu().numpy(), trans, H_pred, path1=output_path + '/' + img_name + '_gt.png',
                        # path2=output_path + '/' + img_name + '_es.png')




