import bisect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2

def _compute_conf_thresh(data):
    dataset_name = data['dataset_name'][0].lower()
    if dataset_name == 'scannet':
        thr = 5e-4
    elif dataset_name == 'linemod_2d':
        thr = 5
    elif dataset_name == 'synthetic':
        thr = 5
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    return thr


# --- VISUALIZATION --- #

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
        0.01, 0.99, '\n'.join(text), transform=fig.axes[1].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return fig

def make_matching_figure_4(
        ave_aligned,img0, img1,img1_edge ,mkpts0, mkpts1, color,
        kpts0=None, kpts1=None, text=[], dpi=75, path=None):
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    fig, axes = plt.subplots(1, 4, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0, cmap='gray')
    axes[1].imshow(img1, cmap='gray')
    axes[2].imshow(img1_edge, cmap='gray')
    axes[3].imshow(ave_aligned, cmap='gray')
    for i in range(4):  # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    # draw matches
    # a mask for out of border of image
    mask0 = (mkpts0[:,0] > 0) * (mkpts0[:,0] < 512)
    mask1 = (mkpts0[:,1] > 0) * (mkpts0[:,1] < 512)
    mask2 = (mkpts1[:,0] > 0) * (mkpts1[:,0] < 512)
    mask3 = (mkpts1[:,1] > 0) * (mkpts1[:,1] < 512)
    mask = mask3*mask2*mask1*mask0
    mkpts1 = mkpts1[mask]
    mkpts0 = mkpts0[mask]
    color = color[mask]
    # step
    step = 8
    mkpts0 = mkpts0[0:-1:step]
    mkpts1 = mkpts1[0:-1:step]
    color = color[0:-1:step]
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                             (fkpts0[i, 1], fkpts1[i, 1]),
                                             transform=fig.transFigure, c=color[i], linewidth=1)
                     for i in range(len(mkpts0))]

        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=1,alpha=0.1)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=1,alpha=0.1)

    # put txts
    txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=10, va='top', ha='left', color=txt_color)

    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
        plt.close()
    plt.close("all")
    return fig

def _make_evaluation_figure(data, b_id, alpha='dynamic'):
    # 'mkpts0_c': mkpts0_c[mconf != 0],
    # 'mkpts1_c': mkpts1_c[mconf != 0],
    # 'mconf': mconf[mconf != 0]

    b_mask = data['b_ids'] == b_id
    conf_thr = _compute_conf_thresh(data)
    img0 = (data['image0'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    img1 = (data['image1'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    img1_edge = (data['edge'][b_id][0].cpu().detach().numpy() * 255).round().astype(np.int32)
    kpts0 = data['mkpts0_f'][b_mask].cpu().numpy()
    kpts1 = data['mkpts1_f'][b_mask].cpu().numpy()
    ave_aligned = ((data['warped_template'][b_id][0]*0.5 + data['image1'][b_id][0]*0.5).cpu().detach().numpy() * 255).round().astype(np.int32)
    dis_errs = data['dis_errs'][b_mask].cpu().numpy()
    correct_mask = dis_errs < conf_thr
    precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0
    n_correct = np.sum(correct_mask)
    n_gt_matches = int(data['conf_matrix_gt'][b_id].sum().cpu())
    recall = 0 if n_gt_matches == 0 else n_correct / (n_gt_matches)
    # recall might be larger than 1, since the calculation of conf_matrix_gt
    # uses groundtruth depths and camera poses, but epipolar distance is used here.

    # matching info
    if alpha == 'dynamic':
        alpha = dynamic_alpha(len(correct_mask))
    color = error_colormap(dis_errs, conf_thr, alpha=alpha)

    text = [
        f'#Matches {len(kpts0)}',
        f'Precision({conf_thr:.2e}) ({100 * precision:.1f}%): {n_correct}/{len(kpts0)}',
        f'Recall({conf_thr:.2e}) ({100 * recall:.1f}%): {n_correct}/{n_gt_matches}'
    ]

    # make the figure
    # figure = make_matching_figure(img0, img1, kpts0, kpts1,
    #                               color, text=text)
    figure = make_matching_figure_4(ave_aligned, img0, img1,img1_edge, kpts0, kpts1,
                                  color, text=text)

    return figure


def _make_confidence_figure(data, b_id):
    # TODO: Implement confidence figure
    raise NotImplementedError()


def make_matching_figures(data, config, mode='evaluation'):
    """ Make matching figures for a batch.

    Args:
        data (Dict): a batch updated by PL_LoFTR.
        config (Dict): matcher config
    Returns:
        figures (Dict[str, List[plt.figure]]
    """
    assert mode in ['evaluation', 'confidence']  # 'confidence'
    figures = {mode: []}
    for b_id in range(data['image0'].size(0)):
        if mode == 'evaluation':
            fig = _make_evaluation_figure(
                data, b_id,
                alpha=config.TRAINER.PLOT_MATCHES_ALPHA)
        elif mode == 'confidence':
            fig = _make_confidence_figure(data, b_id)
        else:
            raise ValueError(f'Unknown plot mode: {mode}')
    figures[mode].append(fig)
    return figures


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
        np.stack([2 - x * 2, x * 2, np.zeros_like(x), np.ones_like(x) * alpha], -1), 0, 1)



def _make_matching_plot_fast(data,b_id, alpha='dynamic'):
    b_mask = data['b_ids'] == b_id
    conf_thr = _compute_conf_thresh(data)
    img0 = data['image0_raw'][b_id].detach().cpu().numpy()
    img1 = data['image1_raw'][b_id].detach().cpu().numpy()

    kpts0 = data['mkpts0_f'][b_mask].cpu().numpy()
    kpts1 = data['mkpts1_f'][b_mask].cpu().numpy()
    estimate_h = data['trans_predict'][b_id].detach().cpu().numpy()

    if img0.ndim==2:
        img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2RGB)
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    img0_es_warped = cv2.warpPerspective(img0, estimate_h, (img0.shape[1], img0.shape[0]))
    mask = img0_es_warped[:, :, 1] > 125
    img0_es_warped[:, :, 0][mask] = 255 - img0_es_warped[:, :, 1][mask]
    img0_es_warped[:, :, 2][mask] = 255 - img0_es_warped[:, :, 1][mask]
    cv2.addWeighted(img0_es_warped, 0.2, img1, 0.8, 0, img0_es_warped)

    dis_errs = data['dis_errs'][b_mask].cpu().numpy()
    correct_mask = dis_errs < conf_thr
    precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0
    n_correct = np.sum(correct_mask)
    n_gt_matches = int(data['conf_matrix_gt'][b_id].sum().cpu())
    recall = 0 if n_gt_matches == 0 else n_correct / (n_gt_matches)
    # recall might be larger than 1, since the calculation of conf_matrix_gt
    # uses groundtruth depths and camera poses, but epipolar distance is used here.

    # matching info
    if alpha == 'dynamic':
        alpha = dynamic_alpha(len(correct_mask))
    color = error_colormap(dis_errs, conf_thr, alpha=alpha)

    text = [
        f'#Matches {len(kpts0)}',
        f'Precision({conf_thr:.2e}) ({100 * precision:.1f}%): {n_correct}/{len(kpts0)}'
    ]
    return make_matching_plot_fast(img0,img1,img0_es_warped,kpts0,kpts1,color,text,)





def make_matching_plot_fast(image0, image1, img_ave, mkpts0,
                            mkpts1, color, text, path=None,
                            margin=10,
                            opencv_display=True, opencv_title='',
                            small_text=[]):
    H0, W0,_ = image0.shape
    H1, W1,_ = image1.shape
    H, W = max(H0, H1), W0 + W1 + W1 + 2* margin

    out = 255*np.ones((H, W, 3), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0+margin:W0+margin+W1] = image1
    out[:H1, W0+W1+2*margin:] = img_ave
    # out = np.stack([out]*3, -1)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                   lineType=cv2.LINE_AA)

    # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)

    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_fg, 1, cv2.LINE_AA)

    cv2.putText(out, 'Template registration',(1500,Ht), cv2.FONT_HERSHEY_DUPLEX,
                    1.2*sc, txt_color_fg, 2, cv2.LINE_AA)

    # Small text.
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_fg, 1, cv2.LINE_AA)



    if path is not None:
        cv2.imwrite(str(path), out)

    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)

    return out


def make_matching_plot_fast_2(image0, image1,ave_img, mkpts0,
                            mkpts1, error, conf_thr=3,path=None,
                            margin=10,
                            opencv_display=True, opencv_title='',
                            small_text=[]):
    # matching info
    mkpts0 = mkpts0[0].cpu().numpy()
    mkpts1 = mkpts1[0].cpu().numpy()
    correct_mask = error < conf_thr
    precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0
    n_correct = np.sum(correct_mask)
    text = [
        f'#Matches {len(mkpts0)}'#,
        # f'Precision ({100 * precision:.1f}%): {n_correct}/{len(mkpts0)}'
        # f'Precision({conf_thr:.2e}) ({100 * precision:.1f}%): {n_correct}/{len(mkpts0)}'
    ]

    alpha = dynamic_alpha(len(correct_mask))
    color = error_colormap(error, conf_thr, alpha=alpha)




    H0, W0, _ = image0.shape
    H1, W1, _ = image1.shape
    H, W = max(H0, H1), W0 + W1 + W1 + 2 * margin

    out = 255 * np.ones((H, W, 3), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0 + margin:W0 + margin + W1] = image1
    out[:H1, W0 + W1 + 2 * margin:] = ave_img



    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                   lineType=cv2.LINE_AA)

    # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)

    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_fg, 1, cv2.LINE_AA)

    cv2.putText(out, 'Template registration',(1500,Ht), cv2.FONT_HERSHEY_DUPLEX,
                    1.2*sc, txt_color_fg, 2, cv2.LINE_AA)

    # Small text.
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_fg, 1, cv2.LINE_AA)



    if path is not None:
        cv2.imwrite(str(path), out)

    # if opencv_display:
    #     cv2.imshow(opencv_title, out)
    #     cv2.waitKey(100)

    return precision,out