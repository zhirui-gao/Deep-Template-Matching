import kornia
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
class FinalTrans(nn.Module):
    def __init__(self, config):
        super(FinalTrans, self).__init__()
        self.sigma_spat = config['trans']['sigma_d']
        self.angel_k = config['trans']['angel_k']
        self.num_iterations = config['trans']['num_iterations']
        self.sigma_angle = config['trans']['sigma_a']
        self.inlier_threshold = config['trans']['inlier_threshold']
        self.gama = config['trans']['gama']
    @torch.no_grad()
    def forward(self,data):

        all_bs = data['pts_0'].shape[0]
        for b_id in range(all_bs):
            b_mask = data['m_bids'] == b_id
            src_keypts = data['mkpts0_f'][b_mask].unsqueeze(0).to(torch.float32) # [bs, num_corr, 2]
            tgt_keypts = data['mkpts1_f'][b_mask].unsqueeze(0).to(torch.float32)  # [bs, num_corr, 2]
            final_trans, final_labels = self.cal_trans(src_keypts, tgt_keypts)
            final_trans = self.post_refinement(final_trans, src_keypts, tgt_keypts)
            # print('final transformation:\n', final_trans)
            # update
            warped_src_keypts = self.transform(src_keypts, final_trans)
            data['mkpts1_f'][b_mask] = warped_src_keypts.squeeze(0)



    def cal_trans(self,src_keypts,tgt_keypts):
        corr_pos = torch.cat((src_keypts, tgt_keypts), dim=-1)  # [bs, num_corr, 4]
        bs, num_corr = corr_pos.shape[0], corr_pos.shape[1]
        k = num_corr

        #################################
        # construct the spatial consistency matrix
        #################################
        src_dist = torch.norm((src_keypts[:, :, None, :] - src_keypts[:, None, :, :]), dim=-1)
        corr_compatibility = src_dist - torch.norm((tgt_keypts[:, :, None, :] - tgt_keypts[:, None, :, :]),
                                                   dim=-1)
        corr_compatibility = torch.clamp(1.0 - corr_compatibility ** 2 / self.sigma_spat ** 2, min=0)  # bs,L,l

        #################################
        # Power iteratation to get the inlier probability
        #################################
        corr_compatibility[:, torch.arange(corr_compatibility.shape[1]), torch.arange(corr_compatibility.shape[1])] = 0

        total_weight = self.cal_leading_eigenvector(corr_compatibility, method='power')

        total_weight = total_weight.view([bs, k])
        total_weight = total_weight / (torch.sum(total_weight, dim=-1, keepdim=True) + 1e-6)
        #################################
        # calculate the transformation by weighted least-squares for each subsets in parallel
        #################################
        total_weight = total_weight.view([-1, k])

        src_knn, tgt_knn = src_keypts.view([-1, k, 2]), tgt_keypts.view([-1, k, 2])
        seed_as_center = False

        if seed_as_center:
            assert ("Not codes!")
        else:
            # not use seeds as neighborhood centers.
            seedwise_trans = self.rigid_transform_2d(src_knn, tgt_knn, total_weight)
            seedwise_trans = seedwise_trans.view([bs, 3, 3])

        #################################
        # calculate the inlier number for each hypothesis, and find the best transformation for each point cloud pair
        #################################
        pred_position = torch.einsum('bnm,bmk->bnk', seedwise_trans[:,:2, :2], src_keypts.permute(0,2,1)) + seedwise_trans[:,:2, 2:3] # [bs, num_corr, 3]
        pred_position = pred_position.permute(0,2,1)
        L2_dis = torch.norm(pred_position - tgt_keypts[:, :, :], dim=-1)  # [bs, num_corr]
        seedwise_fitness = torch.mean((L2_dis < self.inlier_threshold).float(), dim=-1)  # [bs, 1]
        # seedwise_inlier_rmse = torch.sum(L2_dis * (L2_dis < config.inlier_threshold).float(), dim=1)
        batch_best_guess = seedwise_fitness

        # refine the pose by using all the inlier correspondences (done in the post-refinement step)
        final_trans = seedwise_trans
        final_labels = (L2_dis < self.inlier_threshold).float()

        return final_trans, final_labels


    def cal_trans_weight(self,src_keypts,tgt_keypts,weight):
        #weight:[bs,L]
        corr_pos = torch.cat((src_keypts, tgt_keypts), dim=-1)  # [bs, num_corr, 4]
        bs, num_corr = corr_pos.shape[0], corr_pos.shape[1]
        k = num_corr

        #################################
        # construct the spatial consistency matrix
        #################################
        src_dist = torch.norm((src_keypts[:, :, None, :] - src_keypts[:, None, :, :]), dim=-1)
        corr_compatibility = src_dist - torch.norm((tgt_keypts[:, :, None, :] - tgt_keypts[:, None, :, :]),
                                                   dim=-1)
        corr_compatibility = torch.clamp(1.0 - corr_compatibility ** 2 / self.sigma_spat ** 2, min=0)  # bs,L,l

        #################################
        # Power iteratation to get the inlier probability
        #################################
        corr_compatibility[:, torch.arange(corr_compatibility.shape[1]), torch.arange(corr_compatibility.shape[1])] = 0

        total_weight = self.cal_leading_eigenvector(corr_compatibility, method='power')

        total_weight = total_weight.view([bs, k]) * weight
        total_weight = total_weight / (torch.sum(total_weight, dim=-1, keepdim=True) + 1e-6)
        #################################
        # calculate the transformation by weighted least-squares for each subsets in parallel
        #################################
        total_weight = total_weight.view([-1, k])

        src_knn, tgt_knn = src_keypts.view([-1, k, 2]), tgt_keypts.view([-1, k, 2])
        seed_as_center = False

        if seed_as_center:
            assert ("Not codes!")
        else:
            # not use seeds as neighborhood centers.
            seedwise_trans = self.rigid_transform_2d(src_knn, tgt_knn, total_weight)
            seedwise_trans = seedwise_trans.view([bs, 3, 3])

        #################################
        # calculate the inlier number for each hypothesis, and find the best transformation for each point cloud pair
        #################################
        pred_position = torch.einsum('bnm,bmk->bnk', seedwise_trans[:,:2, :2], src_keypts.permute(0,2,1)) + seedwise_trans[:,:2, 2:3] # [bs, num_corr, 3]
        pred_position_z = torch.einsum('bnm,bmk->bnk', seedwise_trans[:,2:3, :2], src_keypts.permute(0,2,1)) + seedwise_trans[:,2:3, 2:3] # [bs, num_corr, 3]
        pred_position = pred_position/pred_position_z

        pred_position = pred_position.permute(0,2,1)
        L2_dis = torch.norm(pred_position - tgt_keypts[:, :, :], dim=-1)  # [bs, num_corr]
        seedwise_fitness = torch.mean((L2_dis < self.inlier_threshold).float(), dim=-1)  # [bs, 1]
        # seedwise_inlier_rmse = torch.sum(L2_dis * (L2_dis < config.inlier_threshold).float(), dim=1)
        batch_best_guess = seedwise_fitness

        # refine the pose by using all the inlier correspondences (done in the post-refinement step)
        final_trans = seedwise_trans
        final_labels = (L2_dis < self.inlier_threshold).float()

        return final_trans, final_labels


    def cal_angel(self, points,angel_k=3):
        r"""

        :param points: (B,N,2)
        :param angel_k: number of nearest neighbors
        :return: angle: the angel of each correspondence (B,N,N)
        """
        batch_size = points.shape[0]
        num_point = points.shape[1]
        points_dist = torch.norm((points[:, :, None, :] - points[:, None, :, :]), dim=-1)
        knn_indices = points_dist.topk(k=angel_k + 1, dim=2, largest=False)[1][:, :,
                      1:]  # (B,N,k)  k+1 : get out the itself indice
        # print(knn_indices)
        knn_indices = knn_indices.unsqueeze(3).expand(batch_size, num_point, angel_k, 2)  # (B,N,K,2)
        expanded_points = points.unsqueeze(1).expand(batch_size, num_point, num_point, 2)  # (B,N,K,2)
        knn_points = torch.gather(expanded_points, dim=2, index=knn_indices)  # (B,N,k,2)

        ref_vectors = knn_points - points.unsqueeze(2)  # (B, N, k, 2)
        anc_vectors = points.unsqueeze(1) - points.unsqueeze(2)  # (B, N, N, 2)
        ref_vectors = ref_vectors.unsqueeze(2).expand(batch_size, num_point, num_point, angel_k, 2)  # (B, N, N, k, 2)
        anc_vectors = anc_vectors.unsqueeze(3).expand(batch_size, num_point, num_point, angel_k, 2)  # (B, N, N, k, 2)
        zeros = torch.zeros((batch_size, num_point, num_point, angel_k, 1), device=ref_vectors.device)
        ref_vectors = torch.cat((ref_vectors, zeros), dim=-1)
        anc_vectors = torch.cat((anc_vectors, zeros), dim=-1)
        sin_values = torch.linalg.norm(torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)  # (B, N, N, k)
        cos_values = torch.sum(ref_vectors * anc_vectors, dim=-1)  # (B, N, N, k)
        angles = torch.atan2(sin_values, cos_values)  # (B, N, N, k)
        angle, _ = torch.max(angles, dim=-1)
        return angle
    def cal_trans_homo(self, src_keypts, tgt_keypts, weight,is_training=False):
        eps = 1e-6
        # weight:[bs,L]
        corr_pos = torch.cat((src_keypts, tgt_keypts), dim=-1)  # [bs, num_corr, 4]
        bs, num_corr = corr_pos.shape[0], corr_pos.shape[1]
        k = num_corr

        #################################
        # construct the spatial consistency matrix
        #################################

        # normalized-distance consistency matrix
        src_dist = torch.norm((src_keypts[:, :, None, :] - src_keypts[:, None, :, :]), dim=-1)
        tgt_dist = torch.norm((tgt_keypts[:, :, None, :] - tgt_keypts[:, None, :, :]), dim=-1)
        src_dist = F.normalize(src_dist, dim=-1)  # deal scale-change case
        tgt_dist = F.normalize(tgt_dist, dim=-1)  # deal scale-change case
        dis_compatibility = (src_dist + eps) / (tgt_dist + eps)
        dis_compatibility = torch.clamp(1 - (dis_compatibility - 1) ** 2 / (self.sigma_spat ** 2), min=0)  # bs,L,l

        # normalized-distance consistency matrix
        src_angle = self.cal_angel(src_keypts, angel_k=self.angel_k)
        tgt_angle = self.cal_angel(tgt_keypts, angel_k=self.angel_k)
        angle_compatibility = torch.abs(src_angle - tgt_angle)
        angle_compatibility = torch.clamp(1 - (angle_compatibility) ** 2 / (self.sigma_angle ** 2), min=0)  # bs,L,l
        corr_compatibility = (1 - self.gama) * angle_compatibility + self.gama * dis_compatibility

        #################################
        # Power iteratation to get the inlier probability
        #################################
        corr_compatibility[:, torch.arange(corr_compatibility.shape[1]), torch.arange(corr_compatibility.shape[1])] = 0

        total_weight = self.cal_leading_eigenvector(corr_compatibility, method='power')

        total_weight = total_weight.view([bs, k]) * weight
        total_weight = total_weight / (torch.sum(total_weight, dim=-1, keepdim=True) + eps)
        #################################
        # calculate the transformation by weighted least-squares for each subsets in parallel
        #################################
        total_weight = total_weight.view([-1, k])

        # src_knn, tgt_knn = src_keypts.view([-1, k, 2]), tgt_keypts.view([-1, k, 2])
        seed_as_center = False

        if seed_as_center:
            assert ("Not codes!")
        else:
            # not use seeds as neighborhood centers.
            # src_knn = (src_keypts/ wh) * 2 - 1
            # tgt_knn = (tgt_keypts/ wh) * 2 - 1
            seedwise_trans = kornia.geometry.homography.find_homography_dlt(src_keypts, tgt_keypts,total_weight)
        return seedwise_trans, None
        # return self.homo_refinement(seedwise_trans,src_keypts,tgt_keypts,total_weight),None
        #################################

        # calculate the inlier number for each hypothesis, and find the best transformation for each point cloud pair
        #################################
        pred_position = torch.einsum('bnm,bmk->bnk', seedwise_trans[:, :2, :2],
                                     src_keypts.permute(0, 2, 1)) + seedwise_trans[:, :2, 2:3]  # [bs, num_corr, 3]

        pred_position_z = torch.einsum('bnm,bmk->bnk', seedwise_trans[:, 2:3, :2],
                                       src_keypts.permute(0, 2, 1)) + seedwise_trans[:, 2:3, 2:3]  # [bs, num_corr, 3]

        pred_position = pred_position / pred_position_z
        pred_position = pred_position.permute(0, 2, 1)

        L2_dis = torch.norm(pred_position - tgt_keypts[:, :, :], dim=-1)  # [bs, num_corr]
        seedwise_fitness = torch.mean((L2_dis < self.inlier_threshold).float(), dim=-1)  # [bs, 1]
        # seedwise_inlier_rmse = torch.sum(L2_dis * (L2_dis < config.inlier_threshold).float(), dim=1)
        batch_best_guess = seedwise_fitness

        # refine the pose by using all the inlier correspondences (done in the post-refinement step)
        final_trans = seedwise_trans
        final_labels = (L2_dis < self.inlier_threshold).float()

        return final_trans, final_labels

    def homo_refinement(self, initial_trans, src_keypts, tgt_keypts, weights=None):
        """
        Perform post refinement using the initial transformation matrix, only adopted during testing.
        Input
            - initial_trans: [bs, 3, 3]
            - src_keypts:    [bs, num_corr, 2]
            - tgt_keypts:    [bs, num_corr, 2]
            - weights:       [bs, num_corr]
        Output:
            - final_trans:   [bs, 3, 3]
        """
        assert initial_trans.shape[0] == 1
        if self.inlier_threshold == 2: # for 3DMatch
            inlier_threshold_list = [4] * 5
        else: # for KITTI
            inlier_threshold_list = [4] * 5

        previous_inlier_num = 0
        for inlier_threshold in inlier_threshold_list:

            pred_position = torch.einsum('bnm,bmk->bnk', initial_trans[:,:2, :2], src_keypts.permute(0,2,1)) + initial_trans[:,:2, 2:3] # [bs, num_corr, 3]

            pred_position_z = torch.einsum('bnm,bmk->bnk', initial_trans[:, 2:3, :2],
                                           src_keypts.permute(0, 2, 1)) + initial_trans[:, 2:3, 2:3]  # [bs, num_corr, 3]
            pred_position = pred_position / pred_position_z
            pred_position = pred_position.permute(0, 2, 1)
            L2_dis = torch.norm(pred_position - tgt_keypts[:, :, :], dim=-1)  # [bs, num_corr]

            pred_inlier = (L2_dis < inlier_threshold)[0]  # assume bs = 1
            inlier_num = torch.sum(pred_inlier)
            if abs(int(inlier_num - previous_inlier_num)) < 1:
                # update
                break
            else:
                previous_inlier_num = inlier_num
                initial_trans = kornia.geometry.homography.find_homography_dlt(src_keypts[:, pred_inlier, :], tgt_keypts[:, pred_inlier, :],
                                                                               weights=weights[:, pred_inlier])
        return initial_trans


    def cal_leading_eigenvector(self, M, method='power'):
        """
        Calculate the leading eigenvector using power iteration algorithm or torch.symeig
        Input:
            - M:      [bs, num_corr, num_corr] the compatibility matrix
            - method: select different method for calculating the learding eigenvector.
        Output:
            - solution: [bs, num_corr] leading eigenvector
        """
        if method == 'power':
            # power iteration algorithm
            leading_eig = torch.ones_like(M[:, :, 0:1])
            leading_eig_last = leading_eig
            for i in range(self.num_iterations):
                leading_eig = torch.bmm(M, leading_eig)
                leading_eig = leading_eig / (torch.norm(leading_eig, dim=1, keepdim=True) + 1e-6)
                if torch.allclose(leading_eig, leading_eig_last):
                    break
                leading_eig_last = leading_eig
            leading_eig = leading_eig.squeeze(-1)
            return leading_eig
        elif method == 'eig':  # cause NaN during back-prop
            e, v = torch.symeig(M, eigenvectors=True)
            leading_eig = v[:, :, -1]
            return leading_eig
        else:
            exit(-1)

    def rigid_transform_2d(self,A, B, weights=None, weight_threshold=0):
        """
        Input:
            - A:       [bs, num_corr, 2], source point cloud
            - B:       [bs, num_corr, 2], target point cloud
            - weights: [bs, num_corr]     weight for each correspondence
            - weight_threshold: float,    clips points with weight below threshold
        Output:
            - R, t
        """
        bs = A.shape[0]
        if weights is None:
            weights = torch.ones_like(A[:, :, 0])
        weights[weights < weight_threshold] = 0

        # find mean of point cloud
        centroid_A = torch.sum(A * weights[:, :, None], dim=1, keepdim=True) / (
                    torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)
        centroid_B = torch.sum(B * weights[:, :, None], dim=1, keepdim=True) / (
                    torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)
        # subtract mean
        Am = A - centroid_A
        Bm = B - centroid_B

        # construct weight covariance matrix
        Weight = torch.diag_embed(weights)
        H = Am.permute(0, 2, 1) @ Weight @ Bm
        U, S, Vt = torch.svd(H.cpu())
        U, S, Vt = U.to(weights.device), S.to(weights.device), Vt.to(weights.device)

        delta_UV = torch.det(Vt @ U.permute(0, 2, 1))
        eye = torch.eye(2)[None, :, :].repeat(bs, 1, 1).to(A.device)
        eye[:, -1, -1] = delta_UV
        R = Vt @ eye @ U.permute(0, 2, 1)
        t = centroid_B.permute(0, 2, 1) - R @ centroid_A.permute(0, 2, 1)
        # print('Estimated R:\n', R)
        # print('Estimated T:\n', t)
        return self.integrate_trans(R,t)

    def integrate_trans(self,R, t):
        """
        Integrate SE2 transformations from R and t, support torch.Tensor and np.ndarry.
        Input
            - R: [2, 2] or [bs, 2, 3], rotation matrix
            - t: [2, 1] or [bs, 2, 1], translation matrix
        Output
            - trans: [3, 3] or [bs, 3, 3], SE2 transformation matrix
        """
        if len(R.shape) == 3: # batch
            if isinstance(R, torch.Tensor):
                trans = torch.eye(3)[None].repeat(R.shape[0], 1, 1).to(R.device)
            else:
                trans = np.eye(3)[None]
            trans[:, :2, :2] = R
            trans[:, :2, 2:3] = t.view([-1, 2, 1])
        else:
            if isinstance(R, torch.Tensor):
                trans = torch.eye(3).to(R.device)
            else:
                trans = np.eye(3)
            trans[:2, :2] = R
            trans[:2, 2:3] = t
        # print('transformation:\n', trans)
        return trans

    def transform(self, pts, trans):

        if len(pts.shape) == 3:
            trans_pts = trans[:, :2, :2] @ pts.permute(0, 2, 1) + trans[:, :2, 2:3]
            return trans_pts.permute(0, 2, 1)
        else:
            trans_pts = trans[:2, :2] @ pts.T + trans[:2, 2:3]
            return trans_pts.T

    def post_refinement(self, initial_trans, src_keypts, tgt_keypts, weights=None):
        """
        Perform post refinement using the initial transformation matrix, only adopted during testing.
        Input
            - initial_trans: [bs, 3, 3]
            - src_keypts:    [bs, num_corr, 2]
            - tgt_keypts:    [bs, num_corr, 2]
            - weights:       [bs, num_corr]
        Output:
            - final_trans:   [bs, 3, 3]
        """
        assert initial_trans.shape[0] == 1
        if self.inlier_threshold == 5: # for 3DMatch
            inlier_threshold_list = [5] * 20
        else: # for KITTI
            inlier_threshold_list = [8] * 20

        previous_inlier_num = 0
        for inlier_threshold in inlier_threshold_list:
            warped_src_keypts = self.transform(src_keypts, initial_trans)
            L2_dis = torch.norm(warped_src_keypts - tgt_keypts, dim=-1)
            pred_inlier = (L2_dis < inlier_threshold)[0]  # assume bs = 1
            inlier_num = torch.sum(pred_inlier)
            if abs(int(inlier_num - previous_inlier_num)) < 1:
                # update
                break
            else:
                previous_inlier_num = inlier_num
            initial_trans = self.rigid_transform_2d(
                A=src_keypts[:, pred_inlier, :],
                B=tgt_keypts[:, pred_inlier, :],
                ## https://link.springer.com/article/10.1007/s10589-014-9643-2
                # weights=None,
                weights=1/(1 + (L2_dis/inlier_threshold)**2)[:, pred_inlier],
                # weights=((1-L2_dis/inlier_threshold)**2)[:, pred_inlier]
            )
        return initial_trans