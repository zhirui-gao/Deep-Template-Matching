
import sys
import os

import kornia.geometry

from src.utils.plotting import make_matching_plot_fast_2
os.chdir("/home/gzr/workspace/Template_Matching_v5_loftr")
from src.lightning.lightning_tm import PL_Tm
from src.config.default import get_cfg_defaults
import torch
import cv2
import torch.nn.functional as F
from  src.lightning.data import get_contours_points,pad_bottom_right
import torchvision.transforms as transforms
import numpy as np
config = get_cfg_defaults()
ckpt_path = '/home/gzr/Data/ckpt/steel_data/last.ckpt'
model = PL_Tm(config, pretrain_ckpt = ckpt_path)
matcher = model.eval().cuda()
#1.pre config
Resize = [480,640] # h,w
h,w =Resize[0],Resize[1]
patch_size = 8  # coarse stage patch size is 8x8
num =128 # num of query points

img0_pth = "/home/gzr/Data/generative_steel/steel_dataset_now/0/images/test/20_template.png"
img1_pth = "/home/gzr/Data/generative_steel/steel_dataset_now/0/images/test/20_homo.png"


def test_pair(img0_pth, img1_pth, matcher):
    conf_thr = 5
    image0 = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE) #tamplate
    image0 = cv2.resize(image0, (Resize[1], Resize[0]))

    image1 = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
    image1 = cv2.resize(image1, (Resize[1], Resize[0]))
    image0_raw, image1_raw = image0, image1

    scale = torch.tensor([image0.shape[1]/w,image0.shape[0]/h],dtype=torch.float)


    image1_rgb = cv2.imread(img1_pth)
    image1_rgb = cv2.cvtColor(image1_rgb, cv2.COLOR_BGR2RGB)
    image1_rgb = cv2.resize(image1_rgb, (Resize[1], Resize[0]))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
                transforms.ToTensor(),
                normalize])
    image1_rgb = transform(image1_rgb)[None] # c,h,w
    image1_edge = cv2.Canny(image1, 5, 10)

    #5. template resize

    #6. get contours
    image0 = cv2.Canny(image0, 5, 10)
    contours_points = get_contours_points(image0)
    contours_points = np.round(contours_points)//patch_size
    contours_points = np.array(list(set([tuple(t) for t in contours_points])))

    mask_0 = np.zeros(num, dtype=bool)
    if num <= contours_points.shape[0]:
        gap = contours_points.shape[0] // num
        contours_points = contours_points[:num * gap:gap, :]
        mask_0[:num] = True
    else:
        # mask
        num_pad = num - contours_points.shape[0]
        pad = np.random.choice(contours_points.shape[0], num_pad, replace=True)
        choice = np.concatenate([range(contours_points.shape[0]), pad])
        mask_0[:contours_points.shape[0]] = True
        contours_points = contours_points[choice, :]

    contours_points[:,0] = np.clip(contours_points[:,0], 0, (w//patch_size)-1)
    contours_points[:,1] = np.clip(contours_points[:,1], 0, (h//patch_size)-1)
    contours_points = torch.tensor(contours_points.astype(np.long))


    image0 = torch.from_numpy(image0)[None][None].cuda() / 255.
    image1 = torch.from_numpy(image1)[None][None].cuda() / 255.
    image1_edge = torch.from_numpy(image1_edge)[None][None].cuda() / 255.

    device = image0.device
    trans = torch.ones([3,3],device=device)
    batch = {'dataset_name': ['synthetic'],'image0': image0, 'image1': image1,'image1_edge':image1_edge.cuda(),
             'image0_raw':torch.from_numpy(image0_raw).cuda(),'image1_raw':torch.from_numpy(image1_raw).cuda(),
             'scale':scale[None].cuda(),'c_points':contours_points[None].cuda(),
             'image1_rgb':image1_rgb.cuda(),'resolution':[patch_size],'trans':trans[None].cuda()}

    mask0 = torch.from_numpy(np.ones((image0.shape[2], image0.shape[3]), dtype=bool))
    mask1 = torch.from_numpy(np.ones((image1.shape[2], image1.shape[3]), dtype=bool))

    if mask1 is not None:  # img_padding is True
        coarse_scale = 1/patch_size
        if coarse_scale:
            [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
                                                   scale_factor=coarse_scale,
                                                   mode='nearest',
                                                   recompute_scale_factor=False)[0].bool()

        batch.update({'mask1': ts_mask_1[None].cuda()})
        batch.update({'mask0': torch.from_numpy(mask_0)[None].cuda()}) # coarse_scale mask  [L]

    # coarse stage matching
    with torch.no_grad():
        matcher(batch)
        mkpts0 = batch['mkpts0_f'].unsqueeze(0)+0.0
        mkpts1 = batch['mkpts1_f'].unsqueeze(0)+0.0
        # calculate the error under the estimate matrix
        trans_predict = batch['trans_predict']
        mkpts0_warped = kornia.geometry.transform_points(trans_predict,mkpts0)
        error = torch.norm(mkpts1-mkpts0_warped,p=2,dim=2).squeeze(0).detach().cpu().numpy()
        # mconf = batch['std'].cpu().numpy()
        img0 = batch['image0_raw'].detach().cpu().numpy()
        img1 = batch['image1_raw'].detach().cpu().numpy()
        precision,out = make_matching_plot_fast_2(img0,img1,mkpts0,mkpts1,error,conf_thr=conf_thr)
        return precision,out

if __name__ == '__main__':
    path_base = '/home/gzr/Data/generative_steel/steel_dataset_now/application_test'
    template_candidate = ['0_template.png','1_template.png','2_template.png','3_template.png','4_template.png','5_template.png','6_template.png','7_template.png','8_template.png']
    image_candidate =  ['0_homo.png','1_homo.png','2_homo.png','3_homo.png','4_homo.png','5_homo.png','6_homo.png','7_homo.png','8_homo.png']
    for image_path in image_candidate:
        precision_max = 0.2
        best_candidate = ''
        out_img = None
        for template_path in template_candidate:
            precision,out = test_pair(os.path.join(path_base,template_path), os.path.join(path_base,image_path), matcher)
            if precision > precision_max:
                precision_max = precision
                best_candidate = template_path
                out_img = out

        if best_candidate=='':
            print('no proper template in candidate set')
        else:
            print('best candidate path is:', best_candidate)
            cv2.imwrite(os.path.join(path_base, image_path).replace('.png','_result.png'), out_img)
