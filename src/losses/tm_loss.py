import math

from loguru import logger
import kornia
import torch
import cv2
import torch.nn as nn
from kornia.geometry.linalg import transform_points
import torch.nn.functional as F
import numpy as np
import torchvision.transforms
class TmLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # config under the global namespace
        self.loss_config = config['tm']['loss']
        self.match_type = self.config['tm']['match_coarse']['match_type']
        self.sparse_spvs = self.config['tm']['match_coarse']['sparse_spvs']
        self.train_stage = self.config['tm']['match_coarse']['train_stage']
        # coarse-level
        self.c_pos_w = self.loss_config['pos_weight']
        self.c_neg_w = self.loss_config['neg_weight']
        self.fine_type = self.loss_config['fine_type']
        self.correct_thr = self.loss_config['fine_correct_thr']

    def compute_coarse_loss(self, conf, conf_gt, weight=None):
        """ Point-wise CE / Focal Loss with 0 / 1 confidence as gt.
        Args:
            conf (torch.Tensor): (N, HW0, HW1) / (N, HW0+1, HW1+1)
            conf_gt (torch.Tensor): (N, HW0, HW1)
            weight (torch.Tensor): (N, HW0, HW1)
        """
        pos_mask, neg_mask = conf_gt == 1, conf_gt == 0
        c_pos_w, c_neg_w = self.c_pos_w, self.c_neg_w
        # corner case: no gt coarse-level match at all
        if not pos_mask.any():  # assign a wrong gt
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_neg_w = 0.

        if self.loss_config['coarse_type'] == 'cross_entropy':
            assert not self.sparse_spvs, 'Sparse Supervision for cross-entropy not implemented!'
            conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
            loss_pos = - torch.log(conf[pos_mask])
            loss_neg = - torch.log(1 - conf[neg_mask])
            if weight is not None:
                loss_pos = loss_pos * weight[pos_mask]
                loss_neg = loss_neg * weight[neg_mask]
            return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
        elif self.loss_config['coarse_type'] == 'focal':
            conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
            alpha = self.loss_config['focal_alpha']
            gamma = self.loss_config['focal_gamma']

            if self.sparse_spvs:  # True
                pos_conf = conf[:, :-1, :-1][pos_mask] \
                    if self.match_type == 'sinkhorn' \
                    else conf[pos_mask]

                loss_pos = - alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()
                # calculate losses for negative samples
                if self.match_type == 'sinkhorn':
                    neg0, neg1 = conf_gt.sum(-1) == 0, conf_gt.sum(1) == 0
                    neg_conf = torch.cat([conf[:, :-1, -1][neg0], conf[:, -1, :-1][neg1]], 0)
                    loss_neg = - alpha * torch.pow(1 - neg_conf, gamma) * neg_conf.log()
                else:
                    # These is no dustbin for dual_softmax, so we left unmatchable patches without supervision.
                    # we could also add 'pseudo negtive-samples'
                    pass
                # handle loss weights
                if weight is not None:
                    # Different from dense-spvs, the loss w.r.t. padded regions aren't directly zeroed out,
                    # but only through manually setting corresponding regions in sim_matrix to '-inf'.
                    loss_pos = loss_pos * weight[pos_mask]
                    if self.match_type == 'sinkhorn':
                        neg_w0 = (weight.sum(-1) != 0)[neg0]
                        neg_w1 = (weight.sum(1) != 0)[neg1]
                        neg_mask = torch.cat([neg_w0, neg_w1], 0)
                        loss_neg = loss_neg[neg_mask]

                loss = c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean() \
                    if self.match_type == 'sinkhorn' \
                    else c_pos_w * loss_pos.mean()
                return loss
                # positive and negative elements occupy similar propotions. => more balanced loss weights needed
            else:  # dense supervision (in the case of match_type=='sinkhorn', the dustbin is not supervised.)
                loss_pos = - alpha * torch.pow(1 - conf[pos_mask], gamma) * (conf[pos_mask]).log()
                loss_neg = - alpha * torch.pow(conf[neg_mask], gamma) * (1 - conf[neg_mask]).log()
                if weight is not None:
                    loss_pos = loss_pos * weight[pos_mask]
                    loss_neg = loss_neg * weight[neg_mask]
                return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
                # each negative element occupy a smaller propotion than positive elements. => higher negative loss weight needed
        else:
            raise ValueError('Unknown coarse loss: {type}'.format(type=self.loss_config['coarse_type']))


    def compute_fine_match_loss(self, conf, conf_gt, weight=None):

        """ Point-wise CE / Focal Loss with 0 / 1 confidence as gt.
        Args:
            conf (torch.Tensor): (N, HW0, HW1) / (N, HW0+1, HW1+1)
            conf_gt (torch.Tensor): (N, HW0, HW1)
            weight (torch.Tensor): (N, HW0, HW1)
        """
        pos_mask, neg_mask = conf_gt == 1, conf_gt == 0
        c_pos_w, c_neg_w = self.c_pos_w, self.c_neg_w
        # corner case: no gt coarse-level match at all
        if not pos_mask.any():  # assign a wrong gt
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_neg_w = 0.

        if self.loss_config['coarse_type'] == 'focal':
            conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
            alpha = self.loss_config['focal_alpha']
            gamma = self.loss_config['focal_gamma']

            if self.sparse_spvs:  # True
                pos_conf = conf[pos_mask]
                loss_pos = - alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()
                # handle loss weights
                if weight is not None:
                    # Different from dense-spvs, the loss w.r.t. padded regions aren't directly zeroed out,
                    # but only through manually setting corresponding regions in sim_matrix to '-inf'.
                    loss_pos = loss_pos * weight[pos_mask]

                loss = c_pos_w * loss_pos.mean()
                return loss
                # positive and negative elements occupy similar propotions. => more balanced loss weights needed
            else:  # dense supervision (in the case of match_type=='sinkhorn', the dustbin is not supervised.)
                loss_pos = - alpha * torch.pow(1 - conf[pos_mask], gamma) * (conf[pos_mask]).log()
                loss_neg = - alpha * torch.pow(conf[neg_mask], gamma) * (1 - conf[neg_mask]).log()
                if weight is not None:
                    loss_pos = loss_pos * weight[pos_mask]
                    loss_neg = loss_neg * weight[neg_mask]
                return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
                # each negative element occupy a smaller propotion than positive elements. => higher negative loss weight needed
        else:
            raise ValueError('Unknown coarse loss: {type}'.format(type=self.loss_config['coarse_type']))


    def compute_fine_loss(self, expec_f, expec_f_gt):
        if self.fine_type == 'l2_with_std':
            return self._compute_fine_loss_l2_std(expec_f, expec_f_gt)
        elif self.fine_type == 'l2':
            return self._compute_fine_loss_l2(expec_f, expec_f_gt)
        else:
            raise NotImplementedError()

    def _compute_fine_loss_l2(self, expec_f, expec_f_gt):
        """
        Args:
            expec_f (torch.Tensor): [M, 2] <x, y>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
        """
        correct_mask = torch.linalg.norm(expec_f_gt, ord=float('inf'), dim=1) < self.correct_thr

        if correct_mask.sum() == 0:
            if self.training:  # this seldomly happen when training, since we pad prediction with gt
                logger.warning("assign a false supervision to avoid ddp deadlock")
                correct_mask[0] = True
            else:
                return None
        offset_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask]) ** 2).sum(-1)
        return offset_l2.mean()

    def _compute_fine_loss_l2_std(self, expec_f, expec_f_gt):
        """
        Args:
            expec_f (torch.Tensor): [M, 3] <x, y, std>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
        """
        # correct_mask tells you which pair to compute fine-loss
        correct_mask = torch.linalg.norm(expec_f_gt, ord=float('inf'), dim=1) < self.correct_thr  # norm for matrices: max(sum(abs(x), dim=1))

        # use std as weight that measures uncertainty
        std = expec_f[:, 2]
        inverse_std = 1. / torch.clamp(std, min=1e-10)
        weight = (inverse_std / torch.mean(inverse_std)).detach()  # avoid minizing loss through increase std

        # corner case: no correct coarse match found
        if (not correct_mask.any()) or (expec_f.shape[0]==0):
            # if self.training:  # this seldomly happen during training, since we pad prediction with gt
                # sometimes there is not coarse-level gt at all.
            # training and validation
            logger.warning("seldomly: assign a false supervision to avoid ddp deadlock,only the beginning of training")
            expec_f_gt = torch.tensor([[1, 1]], device=expec_f_gt.device)
            expec_f = torch.tensor([[1, 1, 1]], device=expec_f.device)
            correct_mask = torch.tensor([True], device=correct_mask.device)
            weight = torch.tensor([0.], device=weight.device)  #


        # l2 loss with std
        offset_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask, :2]) ** 2).sum(-1)
        loss = (offset_l2 * weight[correct_mask]).mean()


        return loss


    @torch.no_grad()
    def compute_c_weight(self, data):
        """ compute element-wise weights for computing coarse-level loss. """
        if 'mask1' in data:
            # c_weight = data['mask1'].flatten(-2)[:, None].expand(data['mask1'].shape[0], data['conf_matrix'].shape[1], data['mask1'].flatten(-2).shape[1]).float()  # N,1,S -> N, L,S

            c_weight = (data['mask0'][..., None] * data['mask1'].flatten(-2)[:, None]).float()
        else:
            c_weight = None
        return c_weight


    def chamfer_loss_unidirectional(self, points_src, points_tgt, weight=None):
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
            # dist_acc = (dist_matrix.min(-2)[0]).mean(-1)
            # dist = ((dist_acc + dist_complete) / 2).mean()
        return dist_complete.mean()

    def cross_entropy_loss_RCF(self, prediction, labelf, beta):

        # from PIL import Image
        # import matplotlib.pyplot as plt
        # from src.utils.utils import toNumpy
        # result = torch.squeeze(prediction[0]*255).detach().cpu().numpy() # H,W
        # plt.imshow(result,cmap='gray')
        # plt.show()
        #
        # result = torch.squeeze(labelf[0] * 255).detach().cpu().numpy()  # H,W
        # plt.imshow(result, cmap='gray')
        # plt.show()
        # import matplotlib.pyplot as plt
        # result = torch.squeeze(prediction[0] * 255).detach().cpu().numpy()  # H,W
        # plt.imshow(result, cmap='gray')
        # plt.show()
        # labelf = torchvision.transforms.GaussianBlur(kernel_size=(3, 3))(labelf)
        # prediction = prediction.detach()
        thr_high = 0.3
        thr_low = 0.03
        mask_positive = labelf > thr_high  #
        mask_negative = labelf < thr_low

        mask_positive = (prediction > thr_high) * mask_positive  # thin
        mask_negative = (prediction < thr_low) * mask_negative  #

        labelf[:] = 2
        labelf[mask_negative] = 0


        labelf[mask_positive] = 1

        label = labelf.detach().long()
        mask = labelf.clone()
        num_positive = torch.sum(label==1).float()
        num_negative = torch.sum(label == 0).float()

        mask[label == 1] = 1.0 * num_negative / (num_positive + num_negative+1)
        mask[label == 0] = beta * num_positive / (num_positive + num_negative+1)
        mask[label == 2] = 0

        cost = F.binary_cross_entropy(
            prediction, labelf, weight=mask.detach(), reduction='sum') # weight=mask,
        return cost


    def cross_entropy_loss_RCF_supervise(self, prediction, labelf, beta):
        # from PIL import Image
        # import matplotlib.pyplot as plt
        # from src.utils.utils import toNumpy
        # result = torch.squeeze(prediction[0]*255).detach().cpu().numpy() # H,W
        # plt.imshow(result, cmap='gray')
        # plt.show()
        #
        # result = torch.squeeze(canny_edge[0] * 255).detach().cpu().numpy()  # H,W
        # plt.imshow(result, cmap='gray')
        # plt.show()

        #
        # result = torch.squeeze(labelf[0] * 255).detach().cpu().numpy()  # H,W
        # plt.imshow(result, cmap='gray')
        # plt.show()
        # prediction = prediction.detach()
        label = labelf.long()
        mask = labelf.clone()
        num_positive = torch.sum(label == 1).float()
        num_negative = torch.sum(label == 0).float()

        mask[label == 1] = 1.0 * (num_negative+1) / (num_positive + num_negative+1)
        mask[label == 0] = beta * (num_positive+1) / (num_positive + num_negative+1)
        mask[label == 2] = 0
        cost = F.binary_cross_entropy(
            prediction, labelf, weight=mask.detach(), reduction='sum')
        return cost #/ (num_positive +1)

    def charbonnier(self,x, alpha=0.25, epsilon=1.e-9):
        return torch.pow(torch.pow(x, 2) + epsilon ** 2, alpha)

    # def smoothness_loss(self, flow, flow_mask):
    #     #flow: b,WW,2
    #     #flow_mask: b,WW
    #     b, ww, c = flow.size()
    #     w = int(math.sqrt(ww))
    #     flow = flow.reshape(b,w,w,c).permute(0,3,1,2)
    #     flow_mask = flow_mask.reshape(b,w,w)
    #     b, c, h, w = flow.size()
    #     v_translated = torch.cat((flow[:, :, 1:, :], torch.zeros(b, c, 1, w, device=flow.device)), dim=-2)
    #     v_flow_mask = torch.cat((flow_mask[:, 1:, :], torch.zeros(b, 1, w, dtype=bool,device=flow.device)), dim=-2)
    #     v_mask = v_flow_mask * flow_mask
    #     h_translated = torch.cat((flow[:, :, :, 1:], torch.zeros(b, c, h, 1, device=flow.device)), dim=-1)
    #     h_flow_mask = torch.cat((flow_mask[:, :, 1:], torch.zeros(b, h, 1, dtype=bool,device=flow.device)), dim=-1)
    #     h_mask = h_flow_mask * flow_mask
    #
    #     s_loss = self.charbonnier(flow - v_translated)*v_mask[:,None,:,:] + self.charbonnier(flow - h_translated)*h_mask[:,None,:,:]
    #     s_loss = torch.sum(s_loss, dim=1) / 2
    #
    #     return (torch.sum(s_loss) / (torch.sum(h_mask)+torch.sum(v_mask)+1)) / 4 # 4:is the scale in the fine stage

    def smoothness_loss(self, flow, flow_mask):
        # flow: b,h,w,2
        # flow_mask: b,h,w
        flow = flow.permute(0,3,1,2)
        b, c, h, w = flow.size()
        v_translated = torch.cat((flow[:, :, 1:, :], torch.zeros(b, c, 1, w, device=flow.device)), dim=-2)
        v_flow_mask = torch.cat((flow_mask[:, 1:, :], torch.zeros(b, 1, w, dtype=bool, device=flow.device)), dim=-2)
        v_mask = v_flow_mask * flow_mask
        h_translated = torch.cat((flow[:, :, :, 1:], torch.zeros(b, c, h, 1, device=flow.device)), dim=-1)
        h_flow_mask = torch.cat((flow_mask[:, :, 1:], torch.zeros(b, h, 1, dtype=bool, device=flow.device)), dim=-1)
        h_mask = h_flow_mask * flow_mask

        s_loss = self.charbonnier(flow - v_translated) * v_mask[:, None, :, :] + self.charbonnier(
            flow - h_translated) * h_mask[:, None, :, :]
        s_loss = torch.sum(s_loss, dim=1) / 2
        loss = (torch.sum(s_loss) / (torch.sum(h_mask) + torch.sum(v_mask) + 1)) /4
        return loss # 4:is the scale in the fine stage

    def transform_poi(self, theta, court_poi, normalize=True):
        ''' Transform PoI with the homography '''
        bs = theta.shape[0]
        theta_inv = torch.inverse(theta[:bs])
        poi = transform_points(theta_inv, court_poi[:bs])

        # Apply inverse normalization to the transformed PoI (from [-1,1] to [0,1]):
        if normalize:
            poi = poi / 2.0 + 0.5

        return poi

    def reprojection_loss(self, predict, gt, court_poi, reduction='mean'):
        r6 = torch.nn.ReLU6(inplace=True)
        poi_pre = transform_points(predict, court_poi)
        poi_gt = transform_points(gt.float(), court_poi)
        # normalize
        poi_pre[:,:,0],poi_pre[:,:,1] = poi_pre[:,:,0]/(640), poi_pre[:,:,1]/(480)
        poi_gt[:,:,0],poi_gt[:,:,1] = poi_gt[:,:,0]/(640), poi_gt[:,:,1]/(480)
        '''
        Calculate the distance between the input points and target ones
        '''
        dist = torch.sum(torch.pow(poi_pre - poi_gt, 2), dim=2)
        dist = r6(dist)
        loss = torch.sum(dist, dim=1) / court_poi.shape[0]

        if reduction == 'mean':
            loss = torch.mean(loss)
        elif reduction == 'sum':
            loss = torch.sum(loss)

        return loss

    def forward(self, data):

        """
        Update:
            data (dict): update{
                'loss': [1] the reduced loss across a batch,
                'loss_scalars' (dict): loss scalars for tensorboard_record
            }
        """
        reprojection_loss = False
        loss_scalars = {}
        # 0. compute element-wise loss weight
        c_weight = self.compute_c_weight(data)
        loss = 0
        # 1. coarse-level loss
        loss_c = self.compute_coarse_loss(
            data['conf_matrix_with_bin'] if self.sparse_spvs and self.match_type == 'sinkhorn' \
                else data['conf_matrix'],
            data['conf_matrix_gt'],
            weight=c_weight)
        # print(loss_c)
        loss = loss_c * self.loss_config['coarse_weight']*10

        loss_scalars.update({"loss_c": loss_c.clone().detach().cpu()})


        if reprojection_loss:
            bs = data['bs']
            court_poi = torch.tensor([[0.0, 0], [640, 0], [0, 480], [640, 480]],
                                     device=loss_c.device).unsqueeze(0).repeat(bs, 1, 1)
            loss_c_reprojection = self.reprojection_loss(data['trans_predict'],data['trans'],court_poi=court_poi)
            loss +=loss_c_reprojection
            loss_scalars.update({"loss_reprojection": loss_c_reprojection.clone().detach().cpu()})
            print(loss_c_reprojection)

        # 2. fine-level loss
        # TODO: fine-level loss
        if self.train_stage=='whole':

            # 2.1 fine-level loss
            loss_f_flow = self.compute_fine_loss(data['expec_f'], data['expec_f_gt'])

            # loss_f_smooth = self.smoothness_loss(data['smooth_map'],data['smooth_mask'])
            loss_f_photometric = data['loss_photometric']

            if loss_f_flow is not None:
                loss += loss_f_flow * self.loss_config['fine_weight']
                loss_scalars.update({"loss_f_flow": loss_f_flow.clone().detach().cpu()})

                loss += loss_f_photometric * self.loss_config['fine_weight']
                loss_scalars.update({"loss_f_photometric": loss_f_photometric.clone().detach().cpu()})
                #
                # loss += loss_f_smooth * self.loss_config['fine_weight']
                # loss_scalars.update({"loss_f_smooth": loss_f_smooth.clone().detach().cpu()})
            else:
                assert self.training is False
                loss_scalars.update({'loss_f': torch.tensor(1.)})  # 1 is the upper bound


        # edge_loss = False
        # if edge_loss:
        #     h, w = data['image0'].shape[2],data['image0'].shape[3]
        #     label_edge = kornia.geometry.transform.warp_perspective(data['image0'], data['trans'].float(), dsize=[h, w])
        #     canny_edge = data['image1_edge']
        #     thr_high = 0.3
        #     thr_low = 0.1
        #     mask_positive = (label_edge > thr_high) * (data['pidinet_out'] > thr_high)  #
        #     mask_negative = (data['pidinet_out'] < thr_low)  # (canny_edge < thr_low) *
        #     label_edge[:] = 2
        #     label_edge[mask_negative] = 0
        #     label_edge[mask_positive] = 1
        #     label_edge.requires_grad = True
        #     loss_edge = 0.001 * self.cross_entropy_loss_RCF_supervise(data['pidinet_out'], label_edge,beta=1.1)
        #     # print('edge_loss', loss_edge)
        #     loss_scalars.update({"loss_edge": loss_edge.clone().detach().cpu()})
        #     loss += loss_edge

        loss_scalars.update({'loss': loss.clone().detach().cpu()})
        data.update({"loss": loss, "loss_scalars": loss_scalars})
