import math
import torch
from torch import nn


class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(512, 512), temp_bug_fix=True):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        if temp_bug_fix:
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / (d_model//2)))
        else:  # a buggy implementation (for backward compatability only)
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / d_model//2))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, C, H, W]

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)]

class PositionEncodingSine_line(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 1-dimensional sequences
    """

    def __init__(self, d_model, temp_bug_fix=True):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()
        if temp_bug_fix:
            div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() * (-math.log(10000.0) / (d_model // 2)))
        else:  # a buggy implementation (for backward compatability only)
            div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() * (-math.log(10000.0) / d_model // 2))

        self.div_term = div_term[:, None, None]  # [C//4, 1]

    def forward(self, x, pts_int):
        """
        Args:
            x: [bs, C, L]
            pts_int:[bs,L,2]
        """
        device = x.device
        d_model = x.shape[1]
        x_position = pts_int[:, :, 0].unsqueeze(0)
        y_position = pts_int[:, :, 1].unsqueeze(0)
        self.div_term = self.div_term.to(device)
        pe = torch.zeros((x.shape[0], d_model, x.shape[2]),device=device)
        pe[:, 0::4, :] = torch.sin(x_position * self.div_term).permute((1, 0, 2))
        pe[:, 1::4, :] = torch.cos(x_position * self.div_term).permute((1, 0, 2))
        pe[:, 2::4, :] = torch.sin(y_position * self.div_term).permute((1, 0, 2))
        pe[:, 3::4, :] = torch.cos(y_position * self.div_term).permute((1, 0, 2))
        return x + pe




class GeometryPositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, temp_bug_fix=True):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / (d_model)))
        self.div_term = div_term[None,:, None, None]  # [1,C//2, 1, 1]
        self.d_model = d_model
        print('dim_div shape',self.div_term.size())

    def forward(self, x):
        """
        Args:
            x: [N, L, S]
            return [N,C,L,S]
        """
        dis = x.unsqueeze(1)
        device = x.device
        self.div_term = self.div_term.to(device)
        pe = torch.zeros((x.shape[0],self.d_model, x.shape[1],x.shape[2]), device=device)
        pe[:,0::2, :, :] = torch.sin(dis * self.div_term)
        pe[:,1::2, :, :] = torch.cos(dis * self.div_term)
        return pe


class RoFormerPositionEncoding(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        self.d_model = d_model

    @staticmethod
    def embed_rotary(x, cos,sin):
        '''

        :param x: [bs, N, d]
        :param cos: [bs, N, d]
        :param sin: [bs, N, d]
        :return:
        '''
        x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x).contiguous()
        x = x * cos + x2 * sin

        # x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x).contiguous()
        # x = x * cos + x2 * sin

        return x

    @staticmethod
    def embed_pos(x, pe):
        """
        conbline feature and position code
        :param x: [bs, N, d]
        :param pe: [bs, N, d,2]
        :return:
        """
        # ... 表示省略前面所有的维度
        return RoFormerPositionEncoding.embed_rotaty(x, pe[..., 0], pe[..., 1])

    def forward(self, pts_int):
        '''
        @param XYZ: [B,N,2]
        @return:[B,N,dim,2]
        '''
        bsize, npoint, _ = pts_int.shape

        x_position, y_position = pts_int[..., 0:1], pts_int[..., 1:2]


        div_term = torch.exp(torch.arange(0, self.d_model // 2, 2, dtype=torch.float, device=pts_int.device) * (
                    -math.log(10000.0) / (self.d_model // 2)))
        div_term = div_term.view(1, 1, -1)  # [1, 1, d//4]

        sinx = torch.sin(x_position * div_term)  # [B, N, d//4]
        cosx = torch.cos(x_position * div_term)
        siny = torch.sin(y_position * div_term)
        cosy = torch.cos(y_position * div_term)


        # sin/cos [θ0,θ1,θ2......θd/4-1] -> sin/cos [θ0,θ0,θ1,θ1,θ2,θ2......θd/4-1,θd/4-1]
        sinx, cosx, siny, cosy= map(
            lambda feat: torch.stack([feat, feat], dim=-1).view(bsize, npoint, -1),
            [sinx, cosx, siny, cosy])
        sin_pos = torch.cat([sinx, siny], dim=-1)
        cos_pos = torch.cat([cosx, cosy], dim=-1)
        position_code = torch.stack([cos_pos, sin_pos], dim=-1)


        if position_code.requires_grad:
            position_code = position_code.detach()

        return position_code