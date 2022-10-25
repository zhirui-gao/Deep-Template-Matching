import numpy as np
import math
import kornia
import torch
from einops.einops import rearrange
import torch.nn as nn
import torch.nn.functional as F
from src.models.PositionEncodingSine import PositionEncodingSine_line,GeometryPositionEncodingSine,RoFormerPositionEncoding
from src.models.model_module.transformer import RefineTransformer
class STN2D(nn.Module):

    def __init__(self, input_size, input_channels):
        super(STN2D, self).__init__()
        # cnn-->cnn->cnn
        num_features = torch.prod((((torch.tensor(input_size) - 2) - 2) ) - 2)
        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-05)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(16, eps=1e-05)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(32, eps=1e-05)
        self.fc = nn.Linear(32 * num_features, 32)

        self.theta = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).view(2, 3)



        # Regressor for the 2 * 3 affine matrix
        # self.affine_regressor = nn.Linear(32, 2 * 3)

        # initialize the weights/bias with identity transformation
        # self.affine_regressor.weight.data.zero_()
        # self.affine_regressor.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        # Regressor for individual parameters
        self.translation = nn.Linear(32, 2)
        self.rotation = nn.Linear(32, 1)
        # self.scaling = nn.Linear(32, 2)
        # self.shearing = nn.Linear(32, 1)

        # initialize the weights/bias with identity transformation
        self._reset_parameters()
        self.translation.weight.data.zero_()
        self.translation.bias.data.copy_(torch.tensor([0, 0], dtype=torch.float))
        self.rotation.weight.data.zero_() # new add!
        self.rotation.bias.data.copy_(torch.tensor([0], dtype=torch.float))
        # self.scaling.weight.data.zero_()
        # self.scaling.bias.data.copy_(torch.tensor([0, 0], dtype=torch.float))
        # self.shearing.weight.data.zero_()
        # self.shearing.bias.data.copy_(torch.tensor([0], dtype=torch.float))

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

    def forward(self, x):
        xs = F.relu(self.bn1(self.conv1(x)))
        xs = F.relu(self.bn2(self.conv2(xs)))
        xs = F.relu(self.bn3(self.conv3(xs)))
        xs = xs.view(xs.size(0), -1)
        xs = F.relu(self.fc(xs))
        # theta = self.affine_regressor(xs).view(-1, 2, 3)
        self.theta = self.affine_matrix(xs)

        # extract first channel for warping
        # img = x.narrow(dim=1, start=0, length=1)
        #
        # # warp image
        # return self.warp_image(img,device)

    def warp_image(self, img, device):
        grid = F.affine_grid(self.theta, img.size(),align_corners=True).to(device)
        wrp = F.grid_sample(img, grid,align_corners=True)

        return wrp

    def affine_matrix(self, x):
        b = x.size(0)

        # trans = self.translation(x)
        trans = torch.tanh(self.translation(x))*0.5  # 0.1
        translation_matrix = torch.zeros([b, 3, 3], dtype=torch.float)
        translation_matrix[:, 0, 0] = 1.0
        translation_matrix[:, 1, 1] = 1.0
        translation_matrix[:, 0, 2] = trans[:, 0].view(-1)
        translation_matrix[:, 1, 2] = trans[:, 1].view(-1)
        translation_matrix[:, 2, 2] = 1.0

        # # rot = self.rotation(x)
        rot = torch.tanh(self.rotation(x)) * (math.pi / 4.0)  #[-45,45]
        rotation_matrix = torch.zeros([b, 3, 3], dtype=torch.float)
        rotation_matrix[:, 0, 0] = torch.cos(rot.view(-1))
        rotation_matrix[:, 0, 1] = -torch.sin(rot.view(-1))
        rotation_matrix[:, 1, 0] = torch.sin(rot.view(-1))
        rotation_matrix[:, 1, 1] = torch.cos(rot.view(-1))
        rotation_matrix[:, 2, 2] = 1.0

        matrix = torch.bmm(rotation_matrix, translation_matrix)
        return matrix[:, 0:2, :]
        #
        # # scale = F.softplus(self.scaling(x), beta=np.log(2.0))
        # # scale = self.scaling(x)
        # scale = torch.tanh(self.scaling(x)) * 0.2
        # scaling_matrix = torch.zeros([b, 3, 3], dtype=torch.float)
        # # scaling_matrix[:, 0, 0] = scale[:, 0].view(-1)
        # # scaling_matrix[:, 1, 1] = scale[:, 1].view(-1)
        # scaling_matrix[:, 0, 0] = torch.exp(scale[:, 0].view(-1))
        # scaling_matrix[:, 1, 1] = torch.exp(scale[:, 1].view(-1))
        # scaling_matrix[:, 2, 2] = 1.0
        #
        # # shear = self.shearing(x)
        # shear = torch.tanh(self.shearing(x)) * (math.pi / 4.0)
        # shearing_matrix = torch.zeros([b, 3, 3], dtype=torch.float)
        # shearing_matrix[:, 0, 0] = torch.cos(shear.view(-1))
        # shearing_matrix[:, 0, 1] = -torch.sin(shear.view(-1))
        # shearing_matrix[:, 1, 0] = torch.sin(shear.view(-1))
        # shearing_matrix[:, 1, 1] = torch.cos(shear.view(-1))
        # shearing_matrix[:, 2, 2] = 1.0
        #
        # # Affine transform
        # matrix = torch.bmm(shearing_matrix, scaling_matrix)
        # matrix = torch.bmm(matrix, torch.transpose(shearing_matrix, 1, 2))
        # matrix = torch.bmm(matrix, rotation_matrix)
        # matrix = torch.bmm(matrix, translation_matrix)
        #
        # # matrix = torch.bmm(translation_matrix, rotation_matrix)
        # # matrix = torch.bmm(matrix, torch.transpose(shearing_matrix, 1, 2))
        # # matrix = torch.bmm(matrix, scaling_matrix)
        # # matrix = torch.bmm(matrix, shearing_matrix)
        #
        # # No-shear transform
        # # matrix = torch.bmm(scaling_matrix, rotation_matrix)
        # # matrix = torch.bmm(matrix, translation_matrix)
        #
        # # Rigid-body transform
        # # matrix = torch.bmm(rotation_matrix, translation_matrix)
        #
        # return matrix[:, 0:2, :]

class Encoder(nn.Module):
    def __init__(self, input_size, input_channels):
        super(Encoder, self).__init__()
        # cnn-->cnn->cnn
        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-05)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(16, eps=1e-05)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.bn3 = nn.BatchNorm2d(32, eps=1e-05)
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                # Initialize the weights/bias of the reg layer with identity transformation:

    def forward(self, x,bs):
        self.bs = bs
        xs = F.relu(self.bn1(self.conv1(x)))
        xs = F.relu(self.bn2(self.conv2(xs)))
        xs = F.avg_pool2d(F.relu(self.bn3(self.conv3(xs))), 2) # content is loss
        xs = xs.view(self.bs, xs.size(0)//self.bs, -1)  # [bs,num_features*64]
        return xs


class Homo_Net(nn.Module):

    def __init__(self, input_size, input_channels, warp_size,config):
        super(Homo_Net, self).__init__()
        self.encoder = Encoder(input_size, input_channels)

        self.num_features = 576
        self.rotaty_encoding = RoFormerPositionEncoding(d_model=self.num_features)
        self.attention_fine = RefineTransformer(config)
        self.reg_token = nn.Parameter(torch.randn(1, 1, self.num_features))
        self.fc = nn.Linear(self.num_features, 128)
        # Regression layer outputs transformation matrix 3*3:
        self.reg = nn.Linear(128, 3 * 3)
        self._reset_parameters()

        # Homography warper:
        h, w = warp_size[1], warp_size[0]
        self.warper = kornia.geometry.transform.HomographyWarper(h, w, normalized_coordinates=True)


    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                # Initialize the weights/bias of the reg layer with identity transformation:
            self.reg.weight.data.zero_()
            self.reg.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float))


    def forward(self, x, data):
        bs = data['bs']
        xs = self.encoder(x,bs) # xs -> [bs,L,C] C->1600

        pos0 = data['mkpts1_c'].reshape(bs, data['mkpts1_c'].size(0)//bs, -1)  # [bs,N+1,2]
        pos0 = torch.cat((torch.zeros(bs, 1, 2,device=pos0.device), pos0), dim=1)
        reg_token = self.reg_token.repeat(bs, 1, 1)
        xs = torch.cat((reg_token, xs), dim=1)

        pos_encoding_0 = self.rotaty_encoding(pos0)  # pos0:[bs,N,2] # pos_encoding_0 [bs,N,dim,2]
        if pos_encoding_0 is not None:
            q_cos, q_sin = pos_encoding_0[..., 0], pos_encoding_0[..., 1]
            xs = RoFormerPositionEncoding.embed_rotary(xs, q_cos, q_sin)



        xs = self.attention_fine(xs, pos_encoding_0)[:,0,:]
        xs = F.relu(self.fc(xs))
        # self.theta = self.affine_matrix(xs)
        self.theta = self.homo_matrix(xs)



    def warp_image_homo(self, img, device):
        '''
        Warp teamplate image by predicted homographies
        '''
        warped = self.warper(img, self.theta)

        return warped


    def warp_image(self, img, device):


        grid = F.affine_grid(self.theta, img.size(),align_corners=True).to(device)
        wrp = F.grid_sample(img, grid,align_corners=True)

        return wrp

    def homo_matrix(self,x):
        # Regression layer outputs transformation matrix 3*3:
        # Regress the 3x3 transformation matrix:
        x = self.reg(x)
        theta = x.view(-1, 3, 3)
        return theta

    def affine_matrix(self, x):
        b = x.size(0)
        trans = torch.tanh(self.translation(x)) * 0.5
        _device = trans.device

        translation_matrix = torch.zeros([b, 3, 3], dtype=torch.float, device=_device)
        translation_matrix[:, 0, 0] = 1.0
        translation_matrix[:, 1, 1] = 1.0
        translation_matrix[:, 0, 2] = trans[:, 0].view(-1)
        translation_matrix[:, 1, 2] = trans[:, 1].view(-1)
        translation_matrix[:, 2, 2] = 1.0

        # rot = self.rotation(x)
        rot = torch.tanh(self.rotation(x)) * (math.pi / 2.0)
        rotation_matrix = torch.zeros([b, 3, 3], dtype=torch.float, device=_device)
        rotation_matrix[:, 0, 0] = torch.cos(rot.view(-1))
        rotation_matrix[:, 0, 1] = -torch.sin(rot.view(-1))
        rotation_matrix[:, 1, 0] = torch.sin(rot.view(-1))
        rotation_matrix[:, 1, 1] = torch.cos(rot.view(-1))
        rotation_matrix[:, 2, 2] = 1.0

        # scale = F.softplus(self.scaling(x), beta=np.log(2.0))
        # scale = self.scaling(x)
        scale = torch.tanh(self.scaling(x)) * 0.1
        scaling_matrix = torch.zeros([b, 3, 3], dtype=torch.float, device=_device)
        # scaling_matrix[:, 0, 0] = scale[:, 0].view(-1)
        # scaling_matrix[:, 1, 1] = scale[:, 1].view(-1)
        scaling_matrix[:, 0, 0] = torch.exp(scale[:, 0].view(-1))
        scaling_matrix[:, 1, 1] = torch.exp(scale[:, 1].view(-1))
        scaling_matrix[:, 2, 2] = 1.0

        # shear = self.shearing(x)
        shear = torch.tanh(self.shearing(x)) * (math.pi / 4.0)
        shearing_matrix = torch.zeros([b, 3, 3], dtype=torch.float, device=_device)
        shearing_matrix[:, 0, 0] = torch.cos(shear.view(-1))
        shearing_matrix[:, 0, 1] = -torch.sin(shear.view(-1))
        shearing_matrix[:, 1, 0] = torch.sin(shear.view(-1))
        shearing_matrix[:, 1, 1] = torch.cos(shear.view(-1))
        shearing_matrix[:, 2, 2] = 1.0

        # Affine transform
        matrix = torch.bmm(shearing_matrix, scaling_matrix)
        matrix = torch.bmm(matrix, torch.transpose(shearing_matrix, 1, 2))
        matrix = torch.bmm(matrix, rotation_matrix)
        matrix = torch.bmm(matrix, translation_matrix)

        return matrix[:, 0:2, :]


class Affine_Net(nn.Module):

    def __init__(self, input_size, input_channels, config):
        super(Affine_Net, self).__init__()
        self.encoder = Encoder(input_size, input_channels)

        self.num_features = 512
        self.rotaty_encoding = RoFormerPositionEncoding(d_model=self.num_features)
        self.attention_fine = RefineTransformer(config)

        self.fc = nn.Linear(self.num_features, self.num_features//2)

        self.translation = nn.Linear(self.num_features//2, 2)
        self.rotation = nn.Linear(self.num_features//2, 1)
        self.scaling = nn.Linear(self.num_features//2, 2)
        self.shearing = nn.Linear(self.num_features//2, 1) # 2?
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize the weights/bias with identity transformation
        self.translation.weight.data.zero_()
        self.translation.bias.data.copy_(torch.tensor([0, 0], dtype=torch.float))
        self.rotation.weight.data.zero_()
        self.rotation.bias.data.copy_(torch.tensor([0], dtype=torch.float))
        self.scaling.weight.data.zero_()
        self.scaling.bias.data.copy_(torch.tensor([0, 0], dtype=torch.float))
        self.shearing.weight.data.zero_()
        self.shearing.bias.data.copy_(torch.tensor([0], dtype=torch.float))

    def forward(self, x, data):
        bs = data['bs']
        xs = self.encoder(x, bs)  # xs -> [bs,L,C] C->1600

        pos0 = data['mkpts0_c'].reshape(bs, data['mkpts0_c'].size(0) // bs, -1)  # [bs,N+1,2]

        pos_encoding_0 = self.rotaty_encoding(pos0)  # pos0:[bs,N,2] # pos_encoding_0 [bs,N,dim,2]
        if pos_encoding_0 is not None:
            q_cos, q_sin = pos_encoding_0[..., 0], pos_encoding_0[..., 1]
            xs = RoFormerPositionEncoding.embed_rotary(xs, q_cos, q_sin)

        xs = self.attention_fine(xs, pos_encoding_0)# [bs,N,L]


        xs = F.relu(self.fc(xs.mean(dim=1)))
        self.theta = self.affine_matrix(xs)

    def warp_image(self, img, device):
        grid = F.affine_grid(self.theta, img.size(),align_corners=True).to(device)
        wrp = F.grid_sample(img, grid,align_corners=True)
        return wrp

    def affine_matrix(self, x):
        b = x.size(0)

        trans = torch.tanh(self.translation(x)) * 0.1
        _device = trans.device
        # trans[0][0] = torch.tensor(0.05)
        # trans[0][1] = torch.tensor(0.05)

        translation_matrix = torch.zeros([b, 3, 3], dtype=torch.float, device=_device)
        translation_matrix[:, 0, 0] = 1.0
        translation_matrix[:, 1, 1] = 1.0
        translation_matrix[:, 0, 2] = trans[:, 0].view(-1)
        translation_matrix[:, 1, 2] = trans[:, 1].view(-1)
        translation_matrix[:, 2, 2] = 1.0

        # rot = self.rotation(x)
        rot = torch.tanh(self.rotation(x)) * (math.pi / 10.0)
        rotation_matrix = torch.zeros([b, 3, 3], dtype=torch.float, device=_device)
        rotation_matrix[:, 0, 0] = torch.cos(rot.view(-1))
        rotation_matrix[:, 0, 1] = -torch.sin(rot.view(-1))
        rotation_matrix[:, 1, 0] = torch.sin(rot.view(-1))
        rotation_matrix[:, 1, 1] = torch.cos(rot.view(-1))
        rotation_matrix[:, 2, 2] = 1.0

        # scale = F.softplus(self.scaling(x), beta=np.log(2.0))
        # scale = self.scaling(x)
        scale = torch.tanh(self.scaling(x)) * 0.1
        scaling_matrix = torch.zeros([b, 3, 3], dtype=torch.float, device=_device)
        # scaling_matrix[:, 0, 0] = scale[:, 0].view(-1)
        # scaling_matrix[:, 1, 1] = scale[:, 1].view(-1)
        scaling_matrix[:, 0, 0] = torch.exp(scale[:, 0].view(-1))
        scaling_matrix[:, 1, 1] = torch.exp(scale[:, 1].view(-1))
        scaling_matrix[:, 2, 2] = 1.0

        # shear = self.shearing(x)
        shear = torch.tanh(self.shearing(x)) * (math.pi / 10.0)
        shearing_matrix = torch.zeros([b, 3, 3], dtype=torch.float, device=_device)
        shearing_matrix[:, 0, 0] = torch.cos(shear.view(-1))
        shearing_matrix[:, 0, 1] = -torch.sin(shear.view(-1))
        shearing_matrix[:, 1, 0] = torch.sin(shear.view(-1))
        shearing_matrix[:, 1, 1] = torch.cos(shear.view(-1))
        shearing_matrix[:, 2, 2] = 1.0

        # Affine transform
        matrix = torch.bmm(shearing_matrix, scaling_matrix)
        matrix = torch.bmm(matrix, torch.transpose(shearing_matrix, 1, 2))
        matrix = torch.bmm(matrix, rotation_matrix)
        matrix = torch.bmm(matrix, translation_matrix)

        return matrix[:, 0:2, :]

# class Affine_Net_old(nn.Module):
#   # cls_token to regress
#     def __init__(self, input_size, input_channels,config):
#         super(Affine_Net_old, self).__init__()
#         self.encoder = Encoder(input_size, input_channels)
#
#         self.num_features = 512
#         self.rotaty_encoding = RoFormerPositionEncoding(d_model=self.num_features)
#         self.attention_fine = RefineTransformer(config)
#         self.reg_token = nn.Parameter(torch.randn(1, 1, self.num_features))
#         self.fc = nn.Linear(self.num_features, 128)
#
#         self.translation = nn.Linear(128, 2)
#         self.rotation = nn.Linear(128, 1)
#         self.scaling = nn.Linear(128, 2)
#         self.shearing = nn.Linear(128, 1)
#         self._reset_parameters()
#
#
#
#
#
#     def _reset_parameters(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#         # initialize the weights/bias with identity transformation
#         self.translation.weight.data.zero_()
#         self.translation.bias.data.copy_(torch.tensor([0, 0], dtype=torch.float))
#         self.rotation.weight.data.zero_()
#         self.rotation.bias.data.copy_(torch.tensor([0], dtype=torch.float))
#         self.scaling.weight.data.zero_()
#         self.scaling.bias.data.copy_(torch.tensor([0, 0], dtype=torch.float))
#         self.shearing.weight.data.zero_()
#         self.shearing.bias.data.copy_(torch.tensor([0], dtype=torch.float))
#
#     def forward(self, x, data):
#         bs = data['bs']
#         xs = self.encoder(x,bs) # xs -> [bs,L,C] C->1600
#
#         pos0 = data['mkpts0_c'].reshape(bs, data['mkpts0_c'].size(0)//bs, -1)  # [bs,N+1,2]
#         pos0 = torch.cat((torch.zeros(bs, 1, 2,device=pos0.device), pos0), dim=1)
#         reg_token = self.reg_token.repeat(bs, 1, 1)
#         xs = torch.cat((reg_token, xs), dim=1)
#
#         pos_encoding_0 = self.rotaty_encoding(pos0)  # pos0:[bs,N,2] # pos_encoding_0 [bs,N,dim,2]
#         if pos_encoding_0 is not None:
#             q_cos, q_sin = pos_encoding_0[..., 0], pos_encoding_0[..., 1]
#             xs = RoFormerPositionEncoding.embed_rotary(xs, q_cos, q_sin)
#
#
#
#         xs = self.attention_fine(xs, pos_encoding_0)[:,0,:]
#         xs = F.relu(self.fc(xs))
#         self.theta = self.affine_matrix(xs)
#
#
#     def warp_image(self, img, device):
#         grid = F.affine_grid(self.theta, img.size()).to(device)
#         wrp = F.grid_sample(img, grid)
#         return wrp
#
#
#     def affine_matrix(self, x):
#         b = x.size(0)
#
#         trans = torch.tanh(self.translation(x)) * 0.1
#         _device = trans.device
#         # trans[0][0] = torch.tensor(0.05)
#         # trans[0][1] = torch.tensor(0.05)
#
#
#         translation_matrix = torch.zeros([b, 3, 3], dtype=torch.float, device=_device)
#         translation_matrix[:, 0, 0] = 1.0
#         translation_matrix[:, 1, 1] = 1.0
#         translation_matrix[:, 0, 2] = trans[:, 0].view(-1)
#         translation_matrix[:, 1, 2] = trans[:, 1].view(-1)
#         translation_matrix[:, 2, 2] = 1.0
#
#         # rot = self.rotation(x)
#         rot = torch.tanh(self.rotation(x)) * (math.pi / 10.0)
#         rotation_matrix = torch.zeros([b, 3, 3], dtype=torch.float, device=_device)
#         rotation_matrix[:, 0, 0] = torch.cos(rot.view(-1))
#         rotation_matrix[:, 0, 1] = -torch.sin(rot.view(-1))
#         rotation_matrix[:, 1, 0] = torch.sin(rot.view(-1))
#         rotation_matrix[:, 1, 1] = torch.cos(rot.view(-1))
#         rotation_matrix[:, 2, 2] = 1.0
#
#         # scale = F.softplus(self.scaling(x), beta=np.log(2.0))
#         # scale = self.scaling(x)
#         scale = torch.tanh(self.scaling(x)) * 0.1
#         scaling_matrix = torch.zeros([b, 3, 3], dtype=torch.float, device=_device)
#         # scaling_matrix[:, 0, 0] = scale[:, 0].view(-1)
#         # scaling_matrix[:, 1, 1] = scale[:, 1].view(-1)
#         scaling_matrix[:, 0, 0] = torch.exp(scale[:, 0].view(-1))
#         scaling_matrix[:, 1, 1] = torch.exp(scale[:, 1].view(-1))
#         scaling_matrix[:, 2, 2] = 1.0
#
#         # shear = self.shearing(x)
#         shear = torch.tanh(self.shearing(x)) * (math.pi / 10.0)
#         shearing_matrix = torch.zeros([b, 3, 3], dtype=torch.float, device=_device)
#         shearing_matrix[:, 0, 0] = torch.cos(shear.view(-1))
#         shearing_matrix[:, 0, 1] = -torch.sin(shear.view(-1))
#         shearing_matrix[:, 1, 0] = torch.sin(shear.view(-1))
#         shearing_matrix[:, 1, 1] = torch.cos(shear.view(-1))
#         shearing_matrix[:, 2, 2] = 1.0
#
#         # Affine transform
#         matrix = torch.bmm(shearing_matrix, scaling_matrix)
#         matrix = torch.bmm(matrix, torch.transpose(shearing_matrix, 1, 2))
#         matrix = torch.bmm(matrix, rotation_matrix)
#         matrix = torch.bmm(matrix, translation_matrix)
#
#         return matrix[:, 0:2, :]

class Encoder_whole(nn.Module):
    def __init__(self, input_size, input_channels):
        super(Encoder_whole, self).__init__()
        # cnn-->cnn->cnn
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, padding=2)


        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        xs = F.avg_pool2d(F.relu(self.conv1(x)), 2)
        xs = F.avg_pool2d(F.relu(self.conv2(xs)), 2)
        xs = F.avg_pool2d(F.relu(self.conv3(xs)), 2)
        xs = rearrange(xs, 'n c h w -> n (h w) c ')
        return xs

class Affine_Net_Whole(nn.Module):

    def __init__(self, input_size, input_channels, config):
        super(Affine_Net_Whole, self).__init__()
        self.encoder = Encoder_whole(input_size, input_channels)

        self.num_features = 256
        self.rotaty_encoding = RoFormerPositionEncoding(d_model=self.num_features)

        self.pos_encoding = PositionEncodingSine_line(d_model=self.num_features,
                                                                            temp_bug_fix=True)
        self.attention_fine = RefineTransformer(config)

        self.fc = nn.Linear(self.num_features, self.num_features//2)

        self.translation = nn.Linear(self.num_features//2, 2)
        self.rotation = nn.Linear(self.num_features//2, 1)
        self.scaling = nn.Linear(self.num_features//2, 2)
        self.shearing = nn.Linear(self.num_features//2, 1) # 2?
        self._reset_parameters()


    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize the weights/bias with identity transformation
        self.translation.weight.data.zero_()
        self.translation.bias.data.copy_(torch.tensor([0, 0], dtype=torch.float))
        self.rotation.weight.data.zero_()
        self.rotation.bias.data.copy_(torch.tensor([0], dtype=torch.float))
        self.scaling.weight.data.zero_()
        self.scaling.bias.data.copy_(torch.tensor([0, 0], dtype=torch.float))
        self.shearing.weight.data.zero_()
        self.shearing.bias.data.copy_(torch.tensor([0], dtype=torch.float))

    def forward(self, x0, x1, data):

        win_size = 20
        sum_filt = torch.ones([1, 1, win_size, win_size]).to(x0.device)

        x0_patch_sum = F.conv2d(x0, sum_filt, stride=8,padding=8).flatten(-2)
        b_ids, _, ids = torch.where(x0_patch_sum > 0)
        mask = torch.zeros([x0_patch_sum.shape[0], x0_patch_sum.shape[2]], dtype=bool,device=x0.device)
        mask[b_ids,ids] = True


        x = torch.cat((x0 , x1), dim=1) #[bs,2,H,W]

        bs = data['bs']
        xs = self.encoder(x)  # xs

        # pos0:[bs,N,2] # pos_encoding_0 [bs,N,dim,2]

        # rotaty encoding
        # pos_encoding = self.rotaty_encoding(data['pts_1']) #TODO: put in init to speed-up
        # if pos_encoding is not None:
        #     q_cos, q_sin = pos_encoding[..., 0], pos_encoding[..., 1]
        #     xs = RoFormerPositionEncoding.embed_rotary(xs, q_cos, q_sin)

        pos = data['pts_1']
        xs = rearrange(xs, 'n l c -> n c l')
        xs = rearrange(self.pos_encoding(xs, pos), 'n c l -> n l c')  # template
        xs = self.attention_fine(xs)# [bs,N,L]


        xs = F.relu(self.fc(xs.mean(dim=1)))
        self.theta = self.affine_matrix(xs)

    def warp_image(self, img, device):
        grid = F.affine_grid(self.theta[:, 0:2, :], img.size(), align_corners=True).to(device)
        wrp = F.grid_sample(img, grid, align_corners=True)
        return wrp

    def warp_image_inverse(self, img, device):
        grid = F.affine_grid(self.theta.inverse()[:, 0:2, :], img.size(), align_corners=True).to(device)
        wrp = F.grid_sample(img, grid, align_corners=True)
        return wrp

    def affine_matrix(self, x):
        b = x.size(0)

        trans = torch.tanh(self.translation(x)) * 0.1
        _device = trans.device
        # trans[0][0] = torch.tensor(0.05)
        # trans[0][1] = torch.tensor(0.05)

        translation_matrix = torch.zeros([b, 3, 3], dtype=torch.float, device=_device)
        translation_matrix[:, 0, 0] = 1.0
        translation_matrix[:, 1, 1] = 1.0
        translation_matrix[:, 0, 2] = trans[:, 0].view(-1)
        translation_matrix[:, 1, 2] = trans[:, 1].view(-1)
        translation_matrix[:, 2, 2] = 1.0

        # rot = self.rotation(x)
        rot = torch.tanh(self.rotation(x)) * (math.pi / 8.0)
        rotation_matrix = torch.zeros([b, 3, 3], dtype=torch.float, device=_device)
        rotation_matrix[:, 0, 0] = torch.cos(rot.view(-1))
        rotation_matrix[:, 0, 1] = -torch.sin(rot.view(-1))
        rotation_matrix[:, 1, 0] = torch.sin(rot.view(-1))
        rotation_matrix[:, 1, 1] = torch.cos(rot.view(-1))
        rotation_matrix[:, 2, 2] = 1.0

        # scale = F.softplus(self.scaling(x), beta=np.log(2.0))
        # scale = self.scaling(x)
        scale = torch.tanh(self.scaling(x)) * 0.1
        scaling_matrix = torch.zeros([b, 3, 3], dtype=torch.float, device=_device)
        # scaling_matrix[:, 0, 0] = scale[:, 0].view(-1)
        # scaling_matrix[:, 1, 1] = scale[:, 1].view(-1)
        scaling_matrix[:, 0, 0] = torch.exp(scale[:, 0].view(-1))
        scaling_matrix[:, 1, 1] = torch.exp(scale[:, 1].view(-1))
        scaling_matrix[:, 2, 2] = 1.0

        # shear = self.shearing(x)
        shear = torch.tanh(self.shearing(x)) * (math.pi / 8.0)
        shearing_matrix = torch.zeros([b, 3, 3], dtype=torch.float, device=_device)
        shearing_matrix[:, 0, 0] = torch.cos(shear.view(-1))
        shearing_matrix[:, 0, 1] = -torch.sin(shear.view(-1))
        shearing_matrix[:, 1, 0] = torch.sin(shear.view(-1))
        shearing_matrix[:, 1, 1] = torch.cos(shear.view(-1))
        shearing_matrix[:, 2, 2] = 1.0

        # Affine transform
        matrix = torch.bmm(shearing_matrix, scaling_matrix)
        matrix = torch.bmm(matrix, torch.transpose(shearing_matrix, 1, 2))
        matrix = torch.bmm(matrix, rotation_matrix)
        matrix = torch.bmm(matrix, translation_matrix)
        return matrix