import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange, repeat

class FinePreprocess(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.scale_fine = config['fine']['resolution'][1]
        self.cat_c_feat = config['fine_concat_coarse_feat']
        self.W = self.config['fine']['fine_window_size']
        self.photometric = self.config['fine']['photometric']
        d_model_c = self.config['coarse']['d_model']
        d_model_f = self.config['fine']['d_model']
        self.d_model_f = d_model_f

        if self.cat_c_feat:
            self.down_proj = nn.Linear(d_model_c, d_model_f, bias=True)
            self.merge_feat = nn.Linear(2 * d_model_f, d_model_f, bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def forward(self, feat_f0, feat_f1, data,feat_c0=None, feat_c1=None):
        """

        :param feat_f0: [bs,d_model_f,h_f,w_f](bs,64,240,320)
        :param feat_f1: [bs,d_model_f,h_f,w_f]
        :param data:
        :param feat_c0: [bs,d_model_f,h,w]
        :param feat_c1: [bs,d_model_f,h,w]
        :return:
        """
        scale = data['resolution'][0]
        w0c = torch.div(data['hw0'][1], scale, rounding_mode="floor")
        W = self.W
        device = feat_f0.device
        sacle_f_div_c= stride = data['hw0_f'][0] // data['hw0_c'][0]
        h, w = data['hw0_f'][0],data['hw0_f'][1]
        data.update({'W': W})

        # crop the features in edge
        if self.cat_c_feat is False:
            f_points = data['f_points']
            mask = data['mask_fine_point']
            # out of bundary
            mask = mask * (f_points[:, :, 0] >= W // 2) * (f_points[:, :, 0] < w - W // 2) * \
                   (f_points[:, :, 1] >= W // 2) * (f_points[:, :, 1] < h - W // 2)
            b_ids, i_ids = torch.where(mask)  # (L)

            if data['b_ids'].shape[0] == 0: #TODO;no need here
                feat0 = torch.empty(0, self.W**2, self.d_model_f, device=feat_f0.device)
                feat1 = torch.empty(0, self.W**2, self.d_model_f, device=feat_f0.device)
                data.update({
                    'b_ids_mask_fine': b_ids,
                    'i_ids_mask_fine': i_ids
                })
                return feat0, feat1

            gridX = torch.arange(-W // 2 + 1, W // 2 + 1).view(1, 1, -1).expand(1, W, W).contiguous().reshape(1, -1).to(
                device)  # (1,W*W)
            gridY = torch.arange(-W // 2 + 1, W // 2 + 1).view(1, -1, 1).expand(1, W, W).contiguous().reshape(1, -1).to(
                device)  # (1,WW)

            p_src = torch.stack([f_points[b_ids, i_ids, 0],f_points[b_ids, i_ids, 1]],dim=-1)

            data.update({
                'b_ids_mask_fine':b_ids,
                'i_ids_mask_fine':i_ids,
                'p_src': p_src * self.scale_fine,
                'new_b_ids': b_ids
            })
            y_ids = f_points[b_ids, i_ids, 1][:, None] + gridY  # (L,W*W)
            x_ids = f_points[b_ids, i_ids, 0][:, None] + gridX  # (L,W*W)
            b_ids = b_ids[:, None].repeat(1, W * W)
            feat_f0_unfold = feat_f0[b_ids, :, y_ids, x_ids]  # [L,ww, c_f]
            feat_f1_unfold = feat_f1[b_ids, :, y_ids, x_ids]  # [L,ww, c_f]
            return feat_f0_unfold, feat_f1_unfold

        else:
            template_pool_image0 = F.max_pool2d(data['image0'], kernel_size=2)  # bs,1,w,h
            template_pool = F.unfold(template_pool_image0, kernel_size=(W-1, W-1), stride=stride, padding=0)
            template_pool = rearrange(template_pool, 'n (c ww) l -> n l ww c', ww=(W-1) ** 2)
            b_ids_point, i_ids_point = torch.where(data['mask0']) # data['mask0'] is the mask of pts_0
            i_ids_point_full = data['pts_0'][b_ids_point, i_ids_point][:, 1] * w0c + data['pts_0'][
                                                                                         b_ids_point, i_ids_point][:, 0]
            template_pool = template_pool[b_ids_point, i_ids_point_full].squeeze(-1)  # [L, ww]
            b_ids_patch, i_ids_patch = torch.where(template_pool > 0) #(L')
            gridX = torch.arange(0, W-1).view(1, 1, -1).expand(1, W-1, W-1).contiguous().reshape(1, -1).to(
                device)  # (1,W*W)
            gridY = torch.arange(0,W-1).view(1, -1, 1).expand(1, W-1, W-1).contiguous().reshape(1, -1).to(
                device)  # (1,WW)
            f_points_x = data['pts_0'][b_ids_point, i_ids_point][:,0][:,None]*sacle_f_div_c + gridX  #(L,(W-1)*(W-1))
            f_points_y = data['pts_0'][b_ids_point, i_ids_point][:,1][:,None]*sacle_f_div_c + gridY

            feat_c0 = feat_c0[b_ids_point, i_ids_point]  #(L,f_c) # out of padding position
            feat_c1 = feat_c1[b_ids_point, i_ids_point] #(L,f_c)
            feat_c0 = repeat(feat_c0, 'n c -> n ww c', ww=(W-1) ** 2)
            feat_c1 = repeat(feat_c1, 'n c -> n ww c', ww=(W-1) ** 2)
            f_points_x = f_points_x[b_ids_patch,i_ids_patch] # (L') select point on the edge
            f_points_y = f_points_y[b_ids_patch,i_ids_patch] # (L')

            feat_c0 = feat_c0[b_ids_patch,i_ids_patch] # (L',f_c)
            feat_c1 = feat_c1[b_ids_patch,i_ids_patch]  #(L',f_c)

            gridX = torch.arange(-W // 2 + 1, W // 2 + 1).view(1, 1, -1).expand(1, W, W).contiguous().reshape(1, -1).to(
                device)  # (1,(W+1)*(W+1))
            gridY = torch.arange(-W // 2 + 1, W // 2 + 1).view(1, -1, 1).expand(1, W, W).contiguous().reshape(1, -1).to(
                device)  # (1,(W+1)*(W+1))

            f_points = torch.stack([f_points_x, f_points_y], dim=-1)

            # if self.photometric:
            #     smooth_mask = torch.zeros(template_pool_image0.shape[0], template_pool_image0.shape[2],
            #                               template_pool_image0.shape[3], dtype=bool, device=device)
            #     smooth_mask[b_ids_point[b_ids_patch], f_points_y, f_points_x] = True  # [bs,h,w]
            #     data.update({
            #         'smooth_mask': smooth_mask,
            #         'smooth_b_ids': b_ids_point[b_ids_patch],
            #         'smooth_y_ids': f_points_y,
            #         'smooth_x_ids': f_points_x
            #     })

            f_points_y = f_points_y[:, None] + gridY  # (L',W*W)
            f_points_x = f_points_x[:, None] + gridX  # (L',W*W)
            b_ids = b_ids_point[b_ids_patch][:,None].repeat(1, W**2)
            feat_f0_unfold = feat_f0[b_ids, :, f_points_y, f_points_x]  # [L',ww, c_f]
            feat_f1_unfold = feat_f1[b_ids, :, f_points_y, f_points_x]  # [L',ww, c_f]
            # option: use coarse-level loftr feature as context: concat and linear

            feat_c_win = self.down_proj(torch.cat([feat_c0, feat_c1], 0)) # [2n, c]
            feat_cf_win = self.merge_feat(torch.cat([torch.cat([feat_f0_unfold, feat_f1_unfold], 0),  # [2n, ww, cf]
                repeat(feat_c_win, 'n c -> n ww c', ww=(W)**2),  # [2n, ww, cf]
            ], -1))
            feat_f0_unfold, feat_f1_unfold = torch.chunk(feat_cf_win, 2, dim=0)

            if self.photometric:
                template_pool_image1 = F.max_pool2d(data['edge_warped'], kernel_size=2) # [bs,c,h,w]
                image0_unfold = template_pool_image0[b_ids, :, f_points_y, f_points_x]  # [L',ww, c_f]
                image1_unfold = template_pool_image1[b_ids, :, f_points_y, f_points_x]  # [L',ww, c_f]
                data.update({
                    'image0_unfold': image0_unfold,
                    'image1_unfold': image1_unfold
                })

            data.update({
                'p_src': f_points* self.scale_fine,
                'b_ids_mask_fine': b_ids_patch,
                'i_ids_mask_fine': i_ids_patch,
                'new_b_ids': b_ids_point[b_ids_patch]
            })
            return feat_f0_unfold, feat_f1_unfold


