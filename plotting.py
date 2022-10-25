from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import os
import argparse
import numpy as np
import cv2
import torch
import time
import bisect


def dynamic_alpha(n_matches,
                  milestones=[0, 300, 1000, 2000],
                  alphas=[1.0, 0.8, 0.4, 0.2]):
    if n_matches == 0:
        return 1.0
    ranges = list(zip(alphas, alphas[1:] + [None]))
    loc = bisect.bisect_right(milestones, n_matches) - 1
    _range = ranges[loc]
    if _range[1] is None:
        return _range[0]
    return _range[1] + (milestones[loc + 1] - n_matches) / (
        milestones[loc + 1] - milestones[loc]) * (_range[0] - _range[1])


def error_colormap(err, thr, alpha=1.0):
    assert alpha <= 1.0 and alpha > 0, f"Invaid alpha value: {alpha}"
    x = 1 - np.clip(err / (thr * 2), 0, 1)
    return np.clip(
        np.stack([2-x*2, x*2, np.zeros_like(x), np.ones_like(x)*alpha], -1), 0, 1)

def _make_evaluation_figure(img0,img1, kpts0,kpts1, epi_errs,homo_error, name_algorithm,path,alpha='dynamic'):
    # ?
    conf_thr = 4
    correct_mask = epi_errs < conf_thr
    n_correct = np.sum(correct_mask)

    # matching info
    if alpha == 'dynamic':
        alpha = dynamic_alpha(len(correct_mask))
    color = error_colormap(epi_errs, conf_thr, alpha=alpha)

    text = [
        name_algorithm,
        f'error:{homo_error:.2f}',
        f'inliers:{n_correct}/{len(kpts0)}'
    ]

    # make the figure
    figure = make_matching_figure(img0, img1, kpts0, kpts1,
                                  color, text=text,path=path)
    return figure


def make_matching_figure(
        img0, img1, mkpts0, mkpts1, color,
        kpts0=None, kpts1=None, text=[], dpi=75, path=None):
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0, cmap='gray')
    axes[1].imshow(img1, cmap='gray')
    for i in range(2):  # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)

    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=2)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                             (fkpts0[i, 1], fkpts1[i, 1]),
                                             transform=fig.transFigure, c=color[i], linewidth=1)
                     for i in range(len(mkpts0))]

        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=4)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=4)

    # put txts
    txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    # save or return figure
    plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
    plt.close()


def add_alpha_channel(img):
    """ 为jpg图像添加alpha通道 """

    b_channel, g_channel, r_channel = cv2.split(img)  # 剥离jpg图像通道
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255  # 创建Alpha通道

    img_new = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))  # 融合通道
    return img_new

def plot_warped(img0, img1, gt_h,estimate_h,path1,path2):
    if img0.ndim==2:
        img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2RGB)
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    w_new, h_new = 640, 480
    img0_gt_warped = cv2.warpPerspective(img0,gt_h,(img0.shape[1], img0.shape[0]))
    mask = img0_gt_warped[:,:,1] > 125
    # img0_gt_warped[:,:,1][mask]= 255
    img0_gt_warped[:,:,0][mask]= 255 - img0_gt_warped[:,:,1][mask]
    img0_gt_warped[:,:,2][mask]= 255 - img0_gt_warped[:,:,1][mask]

    # ave_aligned_gt = ((img0_gt_warped * 0.5 + img1 * 0.5)).round().astype(np.int32)

    alpha = 0.2
    cv2.addWeighted(img0_gt_warped, alpha, img1, 1 - alpha, 0, img0_gt_warped)
    img0_gt_warped = cv2.resize(img0_gt_warped, (w_new, h_new))
    cv2.imwrite(path1, img0_gt_warped)
    if estimate_h is None:

        img1 = cv2.resize(img1, (w_new, h_new))
        cv2.imwrite(path2, img1)
    else:
        img0_es_warped = cv2.warpPerspective(img0,estimate_h,(img0.shape[1], img0.shape[0]))
        mask = img0_es_warped[:, :, 1] > 125
        img0_es_warped[:, :, 0][mask] = 255 - img0_es_warped[:, :, 1][mask]
        img0_es_warped[:, :, 2][mask] = 255 - img0_es_warped[:, :, 1][mask]
        cv2.addWeighted(img0_es_warped, alpha, img1, 1 - alpha, 0, img0_es_warped)
        img0_es_warped = cv2.resize(img0_es_warped, (w_new, h_new))
        cv2.imwrite(path2, img0_es_warped)

def distance_M(pts0, pts1, trans):
    pts_x = (trans[0, 0] * (pts0[:, 0] ) + trans[0, 1] * (pts0[:, 1]) + trans[0, 2])
    pts_y = (trans[1, 0] * (pts0[:, 0] ) + trans[1, 1] * (pts0[:, 1] ) + trans[1, 2])
    pts_z = (trans[2, 0] * (pts0[:, 0] ) + trans[2, 1] * (pts0[:, 1] ) + trans[2, 2])
    pts_x /= pts_z
    pts_y /= pts_z
    dis = np.sqrt((pts1[:, 0] - pts_x[:]) ** 2 + (pts1[:, 1] - pts_y[:]) ** 2)  # n
    return dis

def eval_predict_homography(points, h_gt, H_pred):
    # Estimate the homography between the matches using RANSAC
    max_error = 1e6
    ones = np.ones((points.shape[0],1))
    points = np.concatenate((points, ones),axis=1)
    if H_pred is None:
        correctness = np.zeros(5)
        mean_dist = max_error
    else:
        real_warped_points = np.dot(points, np.transpose(h_gt))
        real_warped_points = real_warped_points[:, :2] / real_warped_points[:, 2:]
        warped_points = np.dot(points, np.transpose(H_pred))
        warped_points = warped_points[:, :2] / warped_points[:, 2:]
        dist = np.linalg.norm(real_warped_points - warped_points, axis=1)
        mean_dist = np.mean(dist)
        if mean_dist>max_error:
            mean_dist = max_error
        correctness = np.array([float(np.mean(dist<i)) for i in [1, 2, 3,5, 8]])
    return mean_dist, correctness