from src.models.SuperPointNet_gs import SuperPointNet_gauss2
import torch
import torch.nn as nn
from einops.einops import rearrange
import os
import subprocess
import matplotlib.pyplot as plt
from src.models.model_utils import SuperPointNet_process
from src.models.PositionEncodingSine import PositionEncodingSine_line
from src.models.model_module.transformer import LocalFeatureTransformer
from src.models.model_module.coarse_matching import CoarseMatching
class Tm(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config
        self.out_num_points = self.config['tm']['superpoint']['out_num_points']
        self.patch_size = self.config['tm']['superpoint']['patch_size']
        self.nms_dist = self.config['tm']['superpoint']['nms_dist'],
        self.conf_thresh = self.config['tm']['superpoint']['conf_thresh']
        # Modules
        self.backbone = SuperPointNet_gauss2(config['tm']['superpoint'])
        self.pos_encoding = PositionEncodingSine_line(d_model=config['tm']['superpoint']['block_dims'][-1],
                                                                            temp_bug_fix=True)

        self.LM_coarse = LocalFeatureTransformer(config['tm']['coarse'])
        self.coarse_matching = CoarseMatching(config['tm']['match_coarse'])


    def forward(self, data):
        # 1. Local Feature CNN
        """
           Update:
               data (dict): {
                   'image0': (torch.Tensor): (N, 1, H, W)  template
                   'image1': (torch.Tensor): (N, 1, H, W)   image
                   'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                   'mask1'(optional) : (torch.Tensor): (N, H, W)
               }
        """
        data.update({
            'bs': data['image0'].size(0),
            'hw0': data['image0'].shape[2:], 'hw1': data['image1'].shape[2:]
        })
        """ 
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # TODO:support
        # if data['hw0'] == data['hw1']:  # faster & better BN convergence
        #     output = self.backbone(torch.cat([data['image0'], data['image1']], dim=0))
        #     semi, desc = output['semi'], output['desc']
        #     (semi0, semi1), (desc0, desc1) = semi.split(data['bs']), desc.split(data['bs'])
        # else:  # handle different input shapes
        output0,  output1 = self.backbone(data['image0']), self.backbone(data['image1'])
        semi0, desc0 = output0['semi'], output0['desc']
        semi1, desc1 = output1['semi'], output1['desc']
        # 2. process the semi and desc from the super point net

        params = {
            'out_num_points': self.config['tm']['superpoint']['out_num_points'],
            'patch_size': self.config['tm']['superpoint']['patch_size'],
            'device': data['image0'].device,
            'nms_dist': self.config['tm']['superpoint']['nms_dist'],
            'conf_thresh': self.config['tm']['superpoint']['conf_thresh']
        }
        sp_processer = SuperPointNet_process(**params)
        outs_post0 = self.backbone.process_output(sp_processer, output0)
        outs_post1 = self.backbone.process_output(sp_processer, output1, choice=False)

        self.draw_keypoint_two(outs_post0, outs_post1,data)  # self.draw_keypoint(outs_post,data)

        # 3. positoin encoding
        pos0 = outs_post0['pts_int'] + outs_post0['pts_offset']
        pos1 = outs_post1['pts_int'] + outs_post1['pts_offset']
        feat_c0 = rearrange(outs_post0['pts_desc'], 'n l c -> n c l')
        feat_c1 = rearrange(outs_post1['pts_desc'], 'n l c -> n c l')
        feat_c0 = rearrange(self.pos_encoding(feat_c0, pos0), 'n c l -> n l c')  # template
        feat_c1 = rearrange(self.pos_encoding(feat_c1, pos1), 'n c l -> n l c') # image
        data.update({
            'pts_0': pos0,
            'pts_1': pos1,
        })


        # 4. attention stage
        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0 = data['mask0'].flatten(-2)
        if 'mask1' in data:
            mask_c1 = data['mask1'].flatten(-2)

        feat_c0, feat_c1 = self.LM_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        # matching
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)






    def draw_keypoint(self, outs_post, data):
        from src.utils.print_tool import print_dict_attr
        print_dict_attr(outs_post, 'shape')
        from src.utils.draw import draw_keypoints
        from src.utils.utils import toNumpy
        pts_int = outs_post['pts_int']
        pts_offset = outs_post['pts_offset']
        pts_desc = outs_post['pts_desc']
        for i in range(2):
            img = draw_keypoints(toNumpy(data["image" + f"{i}"].squeeze()),
                                 toNumpy((pts_int[i] + pts_offset[i]).squeeze()).transpose())
            # print("img: ", img_0)
            plt.imshow(img)
            plt.show()
        from src.models.model_wrap import PointTracker

        tracker = PointTracker(max_length=2, nn_thresh=0.7)

        for i in range(2):
            f = lambda x: toNumpy(x.squeeze())
            tracker.update(f(pts_int[i]).transpose(), f(pts_desc[i]).transpose())

        matches = tracker.get_matches().T
        print("matches: ", matches.transpose().shape)

        from src.utils.draw import draw_matches
        # filename = path_match + '/' + f_num + 'm.png'
        draw_matches(f(data["image0"]), f(data["image1"]), matches, filename='', show=True)

    def draw_keypoint_two(self, outs_post0,outs_post1, data):
        from src.utils.print_tool import print_dict_attr
        print_dict_attr(outs_post1, 'shape')
        print_dict_attr(outs_post0, 'shape')
        from src.utils.draw import draw_keypoints
        from src.utils.utils import toNumpy
        pts_int, pts_offset, pts_desc = [], [], []
        pts_int.append(outs_post0['pts_int'])
        pts_offset.append(outs_post0['pts_offset'])
        pts_desc.append(outs_post0['pts_desc'])

        pts_int.append(outs_post1['pts_int'])
        pts_offset.append(outs_post1['pts_offset'])
        pts_desc.append(outs_post1['pts_desc'])

        print(pts_int[1])
        print(pts_int[0])
        for i in range(2):
            img = draw_keypoints(toNumpy(data["image" + f"{i}"].squeeze()),
                                 toNumpy((pts_int[i] + pts_offset[i]).squeeze()).transpose())
            # print("img: ", img_0)
            plt.imshow(img)
            plt.show()
        from src.models.model_wrap import PointTracker

        tracker = PointTracker(max_length=2, nn_thresh=0.7)

        for i in range(2):
            f = lambda x: toNumpy(x.squeeze())
            tracker.update(f(pts_int[i]).transpose(), f(pts_desc[i]).transpose())

        matches = tracker.get_matches().T
        print("matches: ", matches.transpose().shape)

        from src.utils.draw import draw_matches
        # filename = path_match + '/' + f_num + 'm.png'
        draw_matches(f(data["image0"]), f(data["image1"]), matches, filename='', show=True)

