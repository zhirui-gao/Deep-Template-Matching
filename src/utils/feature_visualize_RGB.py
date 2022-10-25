import random
from src.utils.visualize_feature import BilinearInterpolation
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from matplotlib import gridspec
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from einops.einops import rearrange
import torch
import bisect
import matplotlib.pyplot as plt
import matplotlib

torch.random.seed()
random.seed(66)
np.random.seed(66)
#PCA
from sklearn.decomposition import PCA
#TSNE
from sklearn.manifold import TSNE

def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w*s - s/2 - 0.5), (h*s - s/2 - 0.5)],
                              ).to(keypoints)[None]
    keypoints = keypoints*2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if int(torch.__version__[2]) > 2 else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors


def visualize_attention_feature_batch(feat_0,feat_1, data, pos_0,pos_1=None):
    assert pos_1 == None

    for i in range(feat_0.shape[0]):
        visualize_attention_feature(i, feat_0[i], feat_1[i], data, pos_0[i])


# def visualize_attention_feature(rk, feat_0, feat_1, data, pos_0):
#
#     r"""
#     :param feat_0: L,C
#     :param feat_1: S,C
#     :param data:
#     :param pos_0: L,2
#     :return:
#     """
#     L, S = feat_0.shape[0], feat_1.shape[0]
#     data_feat = torch.concat([feat_0, feat_1],dim=0)
#
#     pca = PCA(n_components=3)
#     # tsne = TSNE(random_state = 42, n_components=3,verbose=0, perplexity=30, n_iter=400)
#     principalComponents = pca.fit_transform(data_feat.detach().cpu().numpy())
#
#     mm = MinMaxScaler() # [0-1] normalize
#     principalComponents = mm.fit_transform(principalComponents)
#
#     feat_0 = principalComponents[0:L, :]
#     feat_1 = principalComponents[L:, :]
#
#     feat_1 = rearrange(feat_1, '(h w) c -> h w c', h=data['hw1_c'][0])
#
#
#     feat_temp = np.zeros_like(feat_1)
#
#     pos_0_x = pos_0[:, 0].detach().cpu().numpy()
#     pos_0_y = pos_0[:, 1].detach().cpu().numpy()
#
#     feat_temp[pos_0_y,pos_0_x,:] = feat_0
#     feat_0 = feat_temp
#
#
#     # BilinearInterpolation
#     BI = BilinearInterpolation(8,8)
#     feat_0 = BI.transform(feat_0)
#     feat_1 = BI.transform(feat_1)
#
#
#
#     ax = plt.subplot(221)
#     plt.imshow(data['image0'][rk][0].detach().cpu().numpy(), cmap='gray')
#     plt.axis('off')
#
#     ax = plt.subplot(222)
#     plt.imshow(data['image1'][rk][0].detach().cpu().numpy(), cmap='gray')
#     plt.axis('off')
#
#     ax = plt.subplot(223)
#     plt.imshow(feat_0)
#     plt.axis('off')
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#
#     ax = plt.subplot(224)
#     plt.imshow(feat_1)
#     plt.axis('off')
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#
#     rk = random.randint(0, 1000)
#     plt.savefig('./save_imgs_plot/' + str(rk) + '.png', bbox_inches='tight', dpi=200, pad_inches=0.0)
#     pass
def visualize_attention_feature(rk, feat_0, feat_1, data, pos_0):
    r"""
    :param feat_0: L,C
    :param feat_1: S,C
    :param data:
    :param pos_0: L,2
    :return:
    """


    feat_1 = rearrange(feat_1, '(h w) c -> h w c', h=data['hw1_c'][0]).detach().cpu().numpy()
    feat_0 = feat_0.detach().cpu().numpy()


    # # BilinearInterpolation
    # BI = BilinearInterpolation(8, 8)
    # feat_0 = BI.transform(feat_0)
    # feat_1 = BI.transform(feat_1)

    L = feat_0.shape[0]#*feat_0.shape[1]
    S = feat_1.shape[0]*feat_1.shape[1]

    # feat_0 = feat_0.reshape(L, feat_0.shape[2])
    feat_1 = feat_1.reshape(S, feat_1.shape[2])

    data_feat = np.concatenate((feat_0, feat_1), axis=0)

    # pca = PCA(n_components=3)
    # principalComponents = pca.fit_transform(data_feat)

    tsne = TSNE(random_state = 42, n_components=3,verbose=0, perplexity=30, n_iter=400)
    principalComponents = tsne.fit_transform(data_feat)

    mm = MinMaxScaler()  # [0-1] normalize
    principalComponents = mm.fit_transform(principalComponents)

    feat_0 = principalComponents[0:L, :]
    feat_1 = principalComponents[L:, :]

    feat_1 = rearrange(feat_1, '(h w) c -> h w c', h=data['hw1_c'][0])

    pos_0_x = pos_0[:, 0].detach().cpu().numpy()
    pos_0_y = pos_0[:, 1].detach().cpu().numpy()
    feat_temp = np.zeros_like(feat_1)
    feat_temp[pos_0_y, pos_0_x, :] = feat_0
    feat_0 = feat_temp

    # m = torch.nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
    # feat_0 = torch.tensor(feat_0)[None].permute(0,3,1,2)
    # feat_1 = torch.tensor(feat_1)[None].permute(0,3,1,2)
    # feat_0 = m(feat_0).permute(0,2,3,1).numpy()[0]
    # feat_1 = m(feat_1).permute(0,2,3,1).numpy()[0]


    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=100)
    axes[0].imshow(feat_0, cmap='gray')
    axes[1].imshow(feat_1, cmap='gray')
    for i in range(2):  # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    fig.canvas.draw()

    rk = data['pair_names'][0][0].split('/')[-1].split('_')[0]
    fig.tight_layout()  # 调整整体空白
    plt.subplots_adjust(wspace=0.005, hspace=0)  # 调整子图间距
    plt.savefig('./' + rk + '.png', bbox_inches='tight', dpi=100, pad_inches=0.0)
    return

def visualize_self_attention_map_batch(attention_map_batch,img_batch,pos_list,ID):
    r"""
    :param attention_map: (B,L,S,h)
    :param img0:
    :return:
    """
    pos_0 = 35
    if ID.split('_')[1]=='1':
        pos_0 = 830+80+80-2+80*15+8

    for i in range(attention_map_batch.shape[0]):
        visualize_self_attention_map(attention_map_batch[i], img_batch[i], i, pos_0, pos_list[i],ID)

def visualize_self_attention_map(att_mat_0, img0, rk ,pos_0, pos_list, ID):
    r"""
    :param att_mat: (L,S,h)
    :param img:
    :param rk:
    :return:
    """

    img0 = img0.detach().cpu().numpy()
    pos_list = pos_list.detach().cpu().numpy()

    h, w = 60, 80
    scale = 8
    top_n = 50
    # 0. Average the attention weights across all heads.
    att_mat_0 = torch.mean(att_mat_0, dim=-1)  # att_mat_0[:,:,0]#

    # 1. normalize the weights
    att_mat_0 = att_mat_0/att_mat_0.sum(dim=-1) # (L,S)


    # center position
    att_val_0 = att_mat_0[pos_0]

    sorted, indices = torch.sort(att_val_0,descending=True)
    sorted0, indices0 = sorted[0:top_n], indices[0:top_n]
    print('value:', sorted0)
    indices0 = indices0.detach().cpu().numpy()
    sorted0 = sorted0.detach().cpu().numpy()

    mkpts0 = pos_list[indices0]*scale + scale//2
    pos_0 = pos_list[pos_0] * scale + scale // 2
    sorted0 = np.array(sorted0).reshape(-1,1)
    mm = MinMaxScaler([0.1,0.999])  # [0-1] normalize
    sorted0 = mm.fit_transform(sorted0)


    fig, axes = plt.subplots(1, 1, figsize=(10, 6), dpi=100)
    axes.imshow(img0, cmap='gray')
    axes.get_yaxis().set_ticks([])
    axes.get_xaxis().set_ticks([])
    for spine in axes.spines.values():
        spine.set_visible(False)
    plt.tight_layout(pad=1)
    fig.canvas.draw()
    transFigure = fig.transFigure.inverted()


    fkpts0 = transFigure.transform(axes.transData.transform(mkpts0))

    fpos_0 = transFigure.transform(axes.transData.transform(pos_0))


    fig.lines = [matplotlib.lines.Line2D((fpos_0[0], fkpts0[i, 0]),
                                         (fpos_0[1], fkpts0[i, 1]),zorder=1,
                                         transform=fig.transFigure,color=[1,0,0,sorted0[i][0]],  linewidth=1) #
                 for i in range(len(mkpts0))
                 ]


    axes.scatter(fkpts0[:, 0], fkpts0[:, 1], transform=fig.transFigure,c='green', s=10,alpha=1)
    #
    axes.scatter(fpos_0[0], fpos_0[1],transform=fig.transFigure, c='blue', s=40,alpha=1)


    plt.savefig('./'+ID+'.png', bbox_inches='tight', pad_inches=0)
    # plt.show()
    plt.close()

def visualize_cross_attention_map_batch(attention_map_batch,img0_batch,img1_batch,pos_list0,pos_list1,ID):
    r"""

    :param attention_map: (B,L,S,h)
    :param img0:
    :return:
    """
    pos_0 = 35
    # if ID.split('_')[1]=='1':
    #     pos_0 = 830+160
    for i in range(attention_map_batch.shape[0]):
        visualize_cross_attention_map(attention_map_batch[i], img0_batch[i],img1_batch[i], i, pos_0, pos_list0[i],pos_list1[i],ID)

def visualize_cross_attention_map(att_mat_0, img0, img1,rk ,pos_0, pos_list0,pos_list1, ID):
    r"""

    :param att_mat: (L,S,h)
    :param img:
    :param rk:
    :return:
    """
    img0 = img0.detach().cpu().numpy()
    img1 = img1.detach().cpu().numpy()
    pos_list0 = pos_list0.detach().cpu().numpy()
    pos_list1 = pos_list1.detach().cpu().numpy()

    h, w = 60, 80
    scale = 8
    top_n = 40
    # 0. Average the attention weights across all heads.
    att_mat_0 = torch.mean(att_mat_0, dim=-1)  # att_mat_0[:,:,0]#

    # 1. normalize the weights
    # cc = att_mat_0.sum(dim=-1)
    # att_mat_0 = att_mat_0/att_mat_0.sum(dim=-1) # (L,S)


    # center position
    att_val_0 = att_mat_0[pos_0]

    sorted, indices = torch.sort(att_val_0,descending=True)
    sorted0, indices0 = sorted[0:top_n], indices[0:top_n]
    print('value:', sorted0)
    indices0 = indices0.detach().cpu().numpy()
    sorted0 = sorted0.detach().cpu().numpy()


    pos_0 = pos_list0[pos_0] * scale + scale // 2
    mkpts1 = pos_list1[indices0] * scale + scale // 2

    sorted0 = np.array(sorted0).reshape(-1,1)
    mm = MinMaxScaler([0.1,0.999])  # [0-1] normalize
    sorted0 = mm.fit_transform(sorted0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=100)
    axes[0].imshow(img0, cmap='gray')
    axes[1].imshow(img1, cmap='gray')
    for i in range(2):  # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    fig.canvas.draw()
    transFigure = fig.transFigure.inverted()

    fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
    fpos_0 = transFigure.transform(axes[0].transData.transform(pos_0))


    fig.lines = [matplotlib.lines.Line2D((fpos_0[0], fkpts1[i, 0]),
                                         (fpos_0[1], fkpts1[i, 1]),zorder=1,
                                         transform=fig.transFigure,color=[1,0,0,sorted0[i][0]],  linewidth=1) #
                 for i in range(len(mkpts1))
                 ]
    axes[1].scatter(fkpts1[:, 0], fkpts1[:, 1], transform=fig.transFigure,c='green', s=10,alpha=1)
    #
    axes[0].scatter(fpos_0[0], fpos_0[1], transform=fig.transFigure, c='blue', s=40,alpha=1)
    fig.tight_layout()  # 调整整体空白
    plt.subplots_adjust(wspace=0.005, hspace=0)  # 调整子图间距
    plt.savefig('./'+ID+'.png', bbox_inches='tight', dpi=100, pad_inches=0)
    # plt.show()
    plt.close()