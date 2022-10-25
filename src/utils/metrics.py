import torch
import cv2
import numpy as np
from collections import OrderedDict
from loguru import logger
# from kornia.geometry.epipolar import numeric
from kornia.geometry.conversions import convert_points_to_homogeneous

def distance_M_metric(pts0, pts1, trans):
    trans = trans.to(device=pts0.device)
    pts_x = (trans[0, 0] * (pts0[:, 0] ) + trans[0, 1] * (pts0[:, 1] ) + trans[0, 2])
    pts_y = (trans[1, 0] * (pts0[:, 0] ) + trans[1, 1] * (pts0[:, 1] ) + trans[1, 2])
    pts_z = (trans[2, 0] * (pts0[:, 0] ) + trans[2, 1] * (pts0[:, 1] ) + trans[2, 2])
    pts_x /= pts_z
    pts_y /= pts_z
    dis = torch.sqrt((pts1[:, 0] - pts_x[:]) ** 2 + (pts1[:, 1] - pts_y[:]) ** 2)  # n
    # dis = torch.mean(dis,dim=0,keepdim=True)
    return dis


def distance(pts0, pts1, bias, scale):
    """this is a easy version to compute the distance,no consider scale and rotation

    :param pts0: [N, 2]
    :param pts1: [N, 2]
    :param bias: [2, 1]
    :return: dis
    """
    # get gt x,y
    pts_image = pts0 + torch.tensor([bias[0] / scale[0], bias[1] /scale[1]], device=pts0.device)
    # compute distance between gt and prediction
    dis = torch.sqrt((pts1[:,0]-pts_image[:,0])**2 +(pts1[:,1]-pts_image[:,1])**2) # n
    return dis

def distance_M(pts0, pts1, trans, scale):
    trans = trans.to(device=pts0.device)
    scale = scale.to(device=pts0.device)
    pts_x = (trans[0, 0] * (pts0[:, 0] * scale[0]) + trans[0, 1] * (pts0[:, 1] * scale[1]) + trans[0, 2])
    pts_y = (trans[1, 0] * (pts0[:, 0] * scale[0]) + trans[1, 1] * (pts0[:, 1] * scale[1]) + trans[1, 2])
    pts_z = (trans[2, 0] * (pts0[:, 0] * scale[0]) + trans[2, 1] * (pts0[:, 1] * scale[1]) + trans[2, 2])
    pts_x /= pts_z
    pts_y /= pts_z

    pts_x = pts_x / scale[0]
    pts_y = pts_y / scale[1]


    dis = torch.sqrt((pts1[:, 0] - pts_x[:]) ** 2 + (pts1[:, 1] - pts_y[:]) ** 2)  # n
    return dis

def distance_M_test(pts0, pts1, trans, scale):
    trans = trans.to(device=pts0.device)
    scale = scale.to(device=pts0.device)
    pts_x = (trans[0, 0] * (pts0[:, 0] /scale[0]) + trans[0, 1] * (pts0[:, 1] / scale[1]) + trans[0, 2])
    pts_y = (trans[1, 0] * (pts0[:, 0] / scale[0]) + trans[1, 1] * (pts0[:, 1] / scale[1]) + trans[1, 2])
    pts_z = (trans[2, 0] * (pts0[:, 0] / scale[0]) + trans[2, 1] * (pts0[:, 1] / scale[1]) + trans[2, 2])
    pts_x /= pts_z
    pts_y /= pts_z

    pts_x = pts_x * scale[0]
    pts_y = pts_y * scale[1]


    dis = torch.sqrt((pts1[:, 0] - pts_x[:]) ** 2 + (pts1[:, 1] - pts_y[:]) ** 2)  # n
    return dis

def compute_distance_errors_old(data):
    """
    Update:
        data (dict):{"dict_errs": [M]}
    """
    m_bids = data['b_ids']
    pts0 = data['mkpts0_f']
    pts1 = data['mkpts1_f']
    dis_errs = []
    num_inliers = []
    data.update({'inliers': []})
    for bs in range(data['image0'].shape[0]):
        mask = m_bids == bs
        if data['dataset_name'][0] == 'linemod_2d':
            # bias mode
            dis_errs.append(
                distance(pts0[mask], pts1[mask],data['bias'][bs],data['scale'][bs])
            )
        else:
            # trans mode
            dis_errs.append(
                distance_M(pts0[mask], pts1[mask], data['trans'][bs], data['scale'][bs])
            )
        dis_errs_np = dis_errs[bs].cpu().numpy()
        correct_mask = dis_errs_np < 4
        num_inliers.append(np.sum(correct_mask))
        data['inliers'].append(num_inliers)
    dis_errs = torch.cat(dis_errs, dim=0)

    data.update({'dis_errs': dis_errs})


def compute_distance_errors_test(data):
    """
    Update:
        data (dict):{"dict_errs": [M]}
    """
    m_bids = data['b_ids']
    dis_errs = []
    num_inliers = []
    data.update({'inliers': []})
    for bs in range(data['image0'].shape[0]):
        mask = m_bids == bs
        pts0 = data['points_template'][bs]
        pts1 = data['points_homo'][bs]

        dis_errs.append(
            distance_M_test(pts0, pts1, data['trans_predict'][bs], data['scale'][bs])
        )
        dis_errs_np = dis_errs[bs].cpu().numpy()
        correct_mask = dis_errs_np < 4
        num_inliers.append(np.sum(correct_mask))
        data['inliers'].append(num_inliers)
    dis_errs = torch.cat(dis_errs, dim=0)

    data.update({'dis_errs': dis_errs})


# 得到轮廓点--亚像素
def get_contour_points(frame, scale_percent=10, is_reverse=False, is_open=False, is_close=False):
    if is_reverse: # 是否翻转二值图
        where_0 = np.where(frame == 0)
        where_255 = np.where(frame == 255)
        frame[where_0] = 255
        frame[where_255] = 0
    ret, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    if is_open: # 是否开操作
        frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel, iterations=1)
    if is_close: # 是否闭操作
        frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel, iterations=1)
    # 改变尺寸
    width = frame.shape[1] * scale_percent
    height = frame.shape[0] * scale_percent
    dim = (width, height)
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)  # 改变像素尺寸
    frame = cv2.GaussianBlur(frame, (17, 17), 0)  # 图像去噪
    edge_output = cv2.Canny(frame, 50, 150)  # 获取轮廓
    contours, heriachy = cv2.findContours(edge_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # contours将所有轮廓值放进一个列表内
    # get max contour points
    max_len, max_len_index = 0, 0
    for index in range(len(contours)):
        if len(contours[index]) >= max_len:
            max_len = len(contours[index])
            max_len_index = index
    points = contours[max_len_index].squeeze() / scale_percent
    return np.array(points, dtype=np.float32)

def compute_distance_errors(data):
    """
    Update:
        data (dict):{"dict_errs": [M]}
    """
    m_bids = data['b_ids']
    points_template = data['points_template']

    dis_errs = []
    dis_center_errs = []
    num_inliers = []

    for bs in range(data['image0'].shape[0]):
        mask = m_bids == bs
        M = data['trans_predict'][bs]
        scale = data['scale'][bs]
        pts0 = points_template[bs]/scale
        pts_x = (M[0, 0] * pts0[:, 0] + M[0, 1] * pts0[:, 1] + M[0, 2])
        pts_y = (M[1, 0] * pts0[:, 0]+ M[1, 1] * pts0[:, 1] + M[1, 2])
        pts_z = (M[2, 0] * pts0[:, 0]+ M[2, 1] * pts0[:, 1] + M[2, 2])
        pts_x = (pts_x/pts_z)
        pts_y = (pts_y/pts_z)
        pts1 = torch.stack([pts_x, pts_y], dim=1)
        pts1 = pts1*scale
        # trans mode
        dis_errs.append(
            distance_M_metric(points_template[bs], pts1, data['trans'][bs])
        )
        # dis_errs.append(
        #     distance_M(pts0, pts1, data['trans'][bs],scale)
        # )

        # calculate center loss
        mask = (data['image0'][bs][0].cpu().numpy() * 255).round().astype(np.uint8)
        # ellipse = cv2.fitEllipse(get_contour_points(mask))
        gt_center =torch.tensor([320, 240]).to(device=M.device)
        pts_x = (M[0, 0] * gt_center[0] + M[0, 1] * gt_center[1] + M[0, 2])
        pts_y = (M[1, 0] * gt_center[0] + M[1, 1] * gt_center[1] + M[1, 2])
        pts_z = (M[2, 0] * gt_center[0] + M[2, 1] * gt_center[1] + M[2, 2])
        pts_x = (pts_x / pts_z)
        pts_y = (pts_y / pts_z)
        pts1 = torch.stack([pts_x, pts_y], dim=0)
        pts1 = pts1 * scale
        trans = data['trans'][bs]
        gt_center = gt_center * scale
        pts_x = (trans[0, 0] * gt_center[0]+ trans[0, 1] * gt_center[1]  + trans[0, 2])
        pts_y = (trans[1, 0] * gt_center[0] + trans[1, 1] * gt_center[1] + trans[1, 2])
        pts_z = (trans[2, 0] * gt_center[0] + trans[2, 1] * gt_center[1] + trans[2, 2])
        pts_x /= pts_z
        pts_y /= pts_z
        dis_center = torch.sqrt((pts1[0] - pts_x) ** 2 + (pts1[1] - pts_y) ** 2)  # n
        dis_center_errs.append(dis_center)
        if dis_center>2:
            print(data['pair_names'][0])
            print('gt_center: {}, center: {}, loss: {}'.format(gt_center, [pts_x,pts_y], dis_center))
        if dis_center>10:
            dis_center_errs[0]=torch.tensor(10)
            print('fail to matching')
    dis_errs = torch.cat(dis_errs, dim=0)
    data.update({'dis_errs_evaluate': dis_errs})
    data.update({'dis_errs_evaluate_center': dis_center_errs})




def epidist_prec(errors, thresholds, ret_dict=False):
    precs = []
    for thr in thresholds:
        if thr==0:
            errors_ave = np.clip(errors,a_min=0, a_max=10)
            precs.append(np.mean(errors_ave))
            continue

        prec_ = []
        for errs in errors:
            correct_mask = errs < thr
            prec_.append(np.mean(correct_mask) if len(correct_mask) > 0 else 0)
        precs.append(np.mean(prec_) if len(prec_) > 0 else 0)
    if ret_dict:
        return {f'prec_dis_errs@{t}': prec for t, prec in zip(thresholds, precs)}
    else:
        return precs

def error_auc(errors):
    """
    Args:
        errors (list): [N,]
        thresholds (list)
    """
    errors = errors.flatten()
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    thresholds = [1, 3, 5, 10, 20]
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)
    auc = {f'auc@{t}': auc for t, auc in zip(thresholds, aucs)}
    print(auc)
    return auc

def aggregate_metrics(metrics, dis_err_thr=3):
    """ Aggregate metrics for the whole dataset:
    (This method should be called once per dataset)
    1. AUC of the pose error (angular) at the threshold [5, 10, 20]
    2. Mean matching precision at the threshold 5e-4(ScanNet), 1e-4(MegaDepth)
    """
    # # filter duplicates
    # unq_ids = OrderedDict((iden, id) for id, iden in enumerate(metrics['identifiers']))
    # unq_ids = list(unq_ids.values())
    # logger.info(f'Aggregating metrics over {len(unq_ids)} unique items...')
    #
    # # inlines
    # # TODO: inlines
    # # matching precision
    # dist_thresholds = [0, 1, 2, dis_err_thr, 5,8,15]
    # # dist_thresholds = [1, 2, dis_err_thr, 5,8,15]
    # precs = epidist_prec(np.array(metrics['dis_errs_evaluate'], dtype=object)[unq_ids], dist_thresholds, True)  # (prec@err_thr)
    #
    # return {**precs}

    # filter duplicates
    unq_ids = OrderedDict((iden, id) for id, iden in enumerate(metrics['identifiers']))
    unq_ids = list(unq_ids.values())
    logger.info(f'Aggregating metrics over {len(unq_ids)} unique items...')

    precs = error_auc(np.array(metrics['dis_errs_evaluate'], dtype=object)[unq_ids])


    return {**precs}