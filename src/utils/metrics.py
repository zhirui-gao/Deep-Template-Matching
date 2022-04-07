import torch
import cv2
import numpy as np
from collections import OrderedDict
from loguru import logger
from kornia.geometry.epipolar import numeric
from kornia.geometry.conversions import convert_points_to_homogeneous

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

def compute_distance_errors(data):
    """
    Update:
        data (dict):{"dict_errs": [M]}
    """
    m_bids = data['m_bids']
    pts0 = data['mkpts0_c']
    pts1 = data['mkpts1_c']
    dis_errs = []
    num_inliers = []
    data.update({'inliers': []})
    for bs in range(data['image0'].shape[0]):
        mask = m_bids == bs
        dis_errs.append(
            distance(pts0[mask], pts1[mask],data['bias'][bs],data['scale'][bs])
        )
        dis_errs_np = dis_errs[bs].cpu().numpy()
        correct_mask = dis_errs_np < 2
        num_inliers.append(np.sum(correct_mask))
        data['inliers'].append(num_inliers)
    dis_errs = torch.cat(dis_errs, dim=0)

    data.update({'dis_errs': dis_errs})


def epidist_prec(errors, thresholds, ret_dict=False):
    precs = []
    for thr in thresholds:
        prec_ = []
        for errs in errors:
            correct_mask = errs < thr
            prec_.append(np.mean(correct_mask) if len(correct_mask) > 0 else 0)
        precs.append(np.mean(prec_) if len(prec_) > 0 else 0)
    if ret_dict:
        return {f'prec_dis_errs@{t}': prec for t, prec in zip(thresholds, precs)}
    else:
        return precs

def aggregate_metrics(metrics, dis_err_thr=2):
    """ Aggregate metrics for the whole dataset:
    (This method should be called once per dataset)
    1. AUC of the pose error (angular) at the threshold [5, 10, 20]
    2. Mean matching precision at the threshold 5e-4(ScanNet), 1e-4(MegaDepth)
    """
    # filter duplicates
    unq_ids = OrderedDict((iden, id) for id, iden in enumerate(metrics['identifiers']))
    unq_ids = list(unq_ids.values())
    logger.info(f'Aggregating metrics over {len(unq_ids)} unique items...')

    # inlines
    # TODO: inlines
    # matching precision
    dist_thresholds = [1, dis_err_thr, 3]
    precs = epidist_prec(np.array(metrics['dis_errs'], dtype=object)[unq_ids], dist_thresholds, True)  # (prec@err_thr)

    return {**precs}