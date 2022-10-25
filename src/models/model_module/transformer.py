import copy
import torch
import torch.nn as nn
from .linear_attention import LinearAttention, FullAttention,GeoAttention

from src.models.PositionEncodingSine import RoFormerPositionEncoding as RoFPE
from einops import rearrange
def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1
class LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_pe, source_pe,  x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source
        qp, kvp = x_pe, source_pe
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)
        if qp is not None:
            q_cos, q_sin = qp[..., 0], qp[..., 1]
            k_cos, k_sin = kvp[..., 0], kvp[..., 1]
            Q_pos = RoFPE.embed_rotary(query, q_cos, q_sin)
            K_pos = RoFPE.embed_rotary(key, k_cos, k_sin)

        # multi-head attention
        Q_pos = Q_pos.view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        K_pos = K_pos.view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]

        value = value.view(bs, -1, self.nhead, self.dim)
        # set padded position to zero
        q_mask,kv_mask = x_mask,source_mask
        if q_mask is not None:
            Q_pos = Q_pos * q_mask[:, :, None, None]
        if kv_mask is not None:
            K_pos = K_pos * kv_mask[:, :, None, None]
            value = value * kv_mask[:, :, None, None]

        message = self.attention(Q_pos, K_pos, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)
        return x + message


    # def forward(self, x, source, x_mask=None, source_mask=None):
    #     """
    #     Args:
    #         x (torch.Tensor): [N, L, C]
    #         source (torch.Tensor): [N, S, C]
    #         x_mask (torch.Tensor): [N, L] (optional)
    #         source_mask (torch.Tensor): [N, S] (optional)
    #     """
    #     bs = x.size(0)
    #     query, key, value = x, source, source
    #
    #     # multi-head attention
    #     query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
    #     key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
    #     value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
    #
    #     message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
    #     message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
    #     message = self.norm1(message)
    #
    #     # feed-forward network
    #     message = self.mlp(torch.cat([x, message], dim=2))
    #     message = self.norm2(message)
    #
    #     return x + message

class Rotary_LoFTREncoderLayer_vis(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(Rotary_LoFTREncoderLayer_vis, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead
        self.eps = 1e-6
        self.feature_map = elu_feature_map

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_pe, source_pe,  x_mask=None, source_mask=None, name='s_0_0',data=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source
        qp, kvp = x_pe, source_pe
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        Q = self.feature_map(query)
        K = self.feature_map(key)

        if qp is not None:
            q_cos, q_sin = qp[..., 0], qp[..., 1]
            k_cos, k_sin = kvp[..., 0], kvp[..., 1]
            Q_pos = RoFPE.embed_rotary(query, q_cos, q_sin)
            K_pos = RoFPE.embed_rotary(key, k_cos, k_sin)
            value = RoFPE.embed_rotary(value, k_cos, k_sin)
        # multi-head attention
        Q = Q.view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        K = K.view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]

        Q_pos = Q_pos.view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        K_pos = K_pos.view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]

        value = value.view(bs, -1, self.nhead, self.dim)
        # set padded position to zero
        q_mask, kv_mask = x_mask, source_mask
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
            Q_pos = Q_pos * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            K_pos = K_pos * kv_mask[:, :, None, None]
            value = value * kv_mask[:, :, None, None]

        # attention map for visualization
        # Compute the attention and the weighted average
        if name.split('_')[0]=='s':
            QK = torch.einsum("nlhd,nshd->nlsh", Q_pos, K_pos)
            softmax_temp = 1. / Q_pos.size(3) ** .5  # sqrt(D)
            A = torch.softmax(softmax_temp * QK, dim=2)
            if q_mask is not None:
                QK.masked_fill_(~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]), float('-inf'))
            from src.utils.feature_visualize_RGB import visualize_self_attention_map_batch

            if name.split('_')[1]=='0':
                pos_list = data['pts_0']
                visualize_self_attention_map_batch(A,data['image0_raw'],pos_list,name)
            else:
                pos_list = data['pts_1']
                visualize_self_attention_map_batch(A, data['image1_raw'],pos_list, name)
        elif name.split('_')[0]=='c':
            QK = torch.einsum("nlhd,nshd->nlsh", Q_pos, K_pos)
            softmax_temp = 1. / Q_pos.size(3) ** .5  # sqrt(D)
            A = torch.softmax(softmax_temp * QK, dim=2)
            if q_mask is not None:
                QK.masked_fill_(~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]), float('-inf'))
            from src.utils.feature_visualize_RGB import visualize_cross_attention_map_batch

            if name.split('_')[1] == '01':
                pos_list0 = data['pts_0']
                pos_list1 = data['pts_1']
                visualize_cross_attention_map_batch(A, data['image0_raw'],data['image1_raw'], pos_list0,pos_list1, name)
            # else:
            #     pos_list = data['pts_1']
            #     visualize_self_attention_map_batch(A, data['image1_raw'], pos_list, name)

        v_length = value.size(1)
        values = value / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K_pos, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = (torch.einsum("nlhd,nhdv,nlh->nlhv", Q_pos, KV, Z) * v_length)

        message = queried_values.contiguous()
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message


class Geo_LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='geo'):
        super(Geo_LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.p_proj = nn.Linear(d_model, d_model, bias=False)
        self.p_proj_2 = nn.Linear(d_model, nhead, bias=False)
        self.relu = nn.ReLU(inplace=True) # inplaec=True will change input
        self.attention = GeoAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, embed_qk, x, x_mask=None, source_mask=None):
        """
        Args:
            embed_qk: torch.Tensor (B, l, s, C), relative positional embedding
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        # only for self attention
        query, key, value = x,x,x
        bs = embed_qk.size(0)
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)

        embed_qk = self.p_proj(embed_qk)  # b l s C *  C C -> b l s C
        embed_qk = self.p_proj_2(embed_qk) # b l s C * C h -> b l s h
        # embed_qk = self.relu(embed_qk) # b l s h

        message = self.attention(embed_qk, query, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message


class Rotary_LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(Rotary_LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead
        self.eps = 1e-6
        self.feature_map = elu_feature_map

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_pe, source_pe,  x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source
        qp, kvp = x_pe, source_pe
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        Q = self.feature_map(query)
        K = self.feature_map(key)

        if qp is not None:
            q_cos, q_sin = qp[..., 0], qp[..., 1]
            k_cos, k_sin = kvp[..., 0], kvp[..., 1]
            Q_pos = RoFPE.embed_rotary(query, q_cos, q_sin)
            K_pos = RoFPE.embed_rotary(key, k_cos, k_sin)
            value = RoFPE.embed_rotary(value, k_cos, k_sin)
        # multi-head attention
        Q = Q.view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        K = K.view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]

        Q_pos = Q_pos.view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        K_pos = K_pos.view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]

        value = value.view(bs, -1, self.nhead, self.dim)
        # set padded position to zero
        q_mask, kv_mask = x_mask, source_mask
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
            Q_pos = Q_pos * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            K_pos = K_pos * kv_mask[:, :, None, None]
            value = value * kv_mask[:, :, None, None]

        v_length = value.size(1)
        values = value / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K_pos, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = (torch.einsum("nlhd,nhdv,nlh->nlhv", Q_pos, KV, Z) * v_length)

        message = queried_values.contiguous()
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message

class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(LocalFeatureTransformer, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        self.attention = config['attention']
        if self.attention =='full':
            encoder_layer = LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'])
        else:
            encoder_layer = Rotary_LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'])
            # encoder_layer = Rotary_LoFTREncoderLayer_vis(config['d_model'], config['nhead'], config['attention'])
        # geo_encoder_layer = Geo_LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'])
        attention_module_list = []
        for i in range(len(self.layer_names)):
            attention_module_list.append(copy.deepcopy(encoder_layer))
        self.layers = nn.ModuleList(attention_module_list)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, pos_encoding_0, pos_encoding_1, mask0=None, mask1=None, only_self=False,data=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"
        cnt = 0
        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                cnt = cnt + 1
                if only_self:
                    feat0 = layer(feat0, feat0, pos_encoding_0, pos_encoding_0, mask0, mask0)
                    pass
                feat0 = layer(feat0, feat0, pos_encoding_0, pos_encoding_0, mask0, mask0) # ,'s_0_'+str(cnt),data
                feat1 = layer(feat1, feat1, pos_encoding_1, pos_encoding_1, mask1, mask1)
                pass
            elif name == 'cross':
                feat0 = layer(feat0, feat1, pos_encoding_0, pos_encoding_1, mask0, mask1)
                feat1 = layer(feat1, feat0, pos_encoding_1, pos_encoding_0, mask1, mask0)
            else:
                raise KeyError

        if only_self:
            return feat0
        return feat0, feat1


class RefineTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(RefineTransformer, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        encoder_layer = LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'])

        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0,  pos_encoding_0=None, mask0=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0)
            else:
                raise KeyError
        return feat0