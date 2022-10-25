import random
import sys
import os

import kornia.geometry

from src.utils.plotting import make_matching_plot_fast_2
os.chdir("/home/gzr/workspace/Template_Matching_v5_loftr")
from src.lightning.lightning_tm import PL_Tm
from src.config.default import get_cfg_defaults
from src.config.default_whole import get_cfg_defaults_application
import torch
import cv2
import torch.nn.functional as F
from  src.lightning.data import get_contours_points,pad_bottom_right
import torchvision.transforms as transforms
import numpy as np
random.seed(67)
np.random.seed(67)
config = get_cfg_defaults()
ckpt_path = '/home/gzr/Data/ckpt/steel_data/last.ckpt'
model = PL_Tm(config, pretrain_ckpt = ckpt_path)
matcher = model.eval().cuda()

config_application = get_cfg_defaults_application()
model_appication = PL_Tm(config_application, pretrain_ckpt = ckpt_path)
matcher_application = model_appication.eval().cuda()



#1.pre config
Resize = [480,640] #
H,W =Resize[0],Resize[1]
patch_size = 8  # coarse stage patch size is 8x8
num =128 # num of query points

img0_pth = "/home/gzr/Data/generative_steel/steel_dataset_now/0/images/test/20_template.png"
img1_pth = "/home/gzr/Data/generative_steel/steel_dataset_now/0/images/test/20_homo.png"

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 2]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)     # 采样点矩阵（B, npoint）
    distance = torch.ones(B, N).to(device) * 1e10                       # 采样点到所有点距离（B, N）
    batch_indices = torch.arange(B, dtype=torch.long).to(device)        # batch_size 数组
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)  # 初始时随机选择一点

    for i in range(npoint):
        centroids[:, i] = farthest                                      # 更新第i个最远点
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 2)        # 取出这个最远点的xyz坐标
        dist = torch.sum((xyz - centroid) ** 2, -1).float()                   # 计算点集中的所有点到这个最远点的欧式距离
        mask = dist < distance
        distance[mask] = dist[mask]                                     # 更新distance，记录样本中每个点距离所有已出现的采样点的最小距离
        farthest = torch.max(distance, -1)[1]                           # 返回最远点索引

    return centroids

def test_pair(img0_pth, img1_pth, matcher,whole=False):
    conf_thr = 5

    image0 = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE) #tamplate
    image0 = cv2.resize(image0, (Resize[1], Resize[0]))

    image1 = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
    image1 = cv2.resize(image1, (Resize[1], Resize[0]))
    image0_raw, image1_raw = image0, image1

    scale = torch.tensor([image0.shape[1]/W,image0.shape[0]/H],dtype=torch.float)


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
        indices = farthest_point_sample(torch.tensor(contours_points)[None, :], num)[0]
        contours_points = contours_points[indices]
        mask_0[:num] = True
    else:
        num_pad = num - contours_points.shape[0]
        pad = np.random.choice(contours_points.shape[0], num_pad, replace=True)
        choice = np.concatenate([range(contours_points.shape[0]), pad])
        mask_0[:contours_points.shape[0]] = True
        contours_points = contours_points[choice, :]

    contours_points[:,0] = np.clip(contours_points[:,0], 0, (W//patch_size)-1)
    contours_points[:,1] = np.clip(contours_points[:,1], 0, (H//patch_size)-1)
    contours_points = torch.tensor(contours_points.astype(np.long))


    image0 = torch.from_numpy(image0)[None][None].cuda() / 255.
    image1 = torch.from_numpy(image1)[None][None].cuda() / 255.
    image1_edge = torch.from_numpy(image1_edge)[None][None].cuda() / 255.

    device = image0.device
    trans = torch.ones([3,3],device=device)
    batch = {'dataset_name': ['synthetic'],'image0': image0, 'image1': image1,'image1_edge':image1_edge.cuda(),
             'image0_raw':torch.from_numpy(image0_raw).cuda(),'image1_raw':torch.from_numpy(image1_raw).cuda(),
             'scale':scale[None].cuda(),'c_points':contours_points[None].cuda(),
             'image1_rgb':image1_rgb.cuda(),'resolution':[patch_size],'trans':trans[None].cuda(),
             'pair_names': (img0_pth,
                            img1_pth)
             }

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
        if whole is True:
            trans_predict = batch['trans_predict'].detach().cpu()
        else:
            trans_predict = batch['trans_predict']
        mkpts0_warped = kornia.geometry.transform_points(trans_predict,mkpts0)
        error = torch.norm(mkpts1-mkpts0_warped,p=2,dim=2).squeeze(0).detach().cpu().numpy()
        # mconf = batch['std'].cpu().numpy()
        img0 = batch['image0_raw'].detach().cpu().numpy()
        img1 = batch['image1_raw'].detach().cpu().numpy()

        if img0.ndim == 2:
            img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2RGB)
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        img0_es_warped = cv2.warpPerspective(img0, trans_predict[0].detach().cpu().numpy(), (img0.shape[1], img0.shape[0]))
        mask = img0_es_warped[:, :, 1] > 125
        img0_es_warped[:, :, 0][mask] = 255 - img0_es_warped[:, :, 1][mask]
        img0_es_warped[:, :, 2][mask] = 255 - img0_es_warped[:, :, 1][mask]
        cv2.addWeighted(img0_es_warped, 0.2, img1, 0.8, 0, img0_es_warped)

        precision, out = make_matching_plot_fast_2(img0,img1,img0_es_warped,mkpts0,mkpts1,error,conf_thr=conf_thr)


        return precision,out,trans_predict


def random_choice(num,len):
    assert num<=len
    arr = np.array(range(0, len, 1))
    np.random.shuffle(arr)
    return arr[0:num]

def load_template_set(origin_path,candinate_path='/home/gzr/Data/generative_steel/real_dataset/0307-0313/process/template_candidate',num_fix=10):
    list_ans = []
    list_tem = os.listdir(origin_path)
    for tem in list_tem:
        if 'template.png' in tem:
            list_ans.append(os.path.join(origin_path,tem))

    list_tem_candinate = os.listdir(candinate_path)
    choose_num = random_choice(num_fix - len(list_ans), len(list_tem_candinate))
    choose = np.array(list_tem_candinate)[choose_num]
    for tem in choose:
        if 'template' in tem:
            list_ans.append(os.path.join(candinate_path,tem))
    np.random.shuffle(list_ans)
    img1 = cv2.hconcat([cv2.imread(list_ans[0]),cv2.imread(list_ans[1]),cv2.imread(list_ans[2]),cv2.imread(list_ans[3]),cv2.imread(list_ans[4])])
    img2 = cv2.hconcat([cv2.imread(list_ans[5]),cv2.imread(list_ans[6]),cv2.imread(list_ans[7]),cv2.imread(list_ans[8]),cv2.imread(list_ans[9])])
    tem_imgs = cv2.vconcat([img1,img2])

    return list_ans,tem_imgs

def load_template_set_9(origin_test,candinate_path='/home/gzr/Data/generative_steel/real_dataset/0307-0313/process/template_candidate',num_fix=10):
    list_ans = []

    list_ans.append(origin_test)

    list_tem_candinate = os.listdir(candinate_path)
    choose_num = random_choice(num_fix - len(list_ans), len(list_tem_candinate))
    choose = np.array(list_tem_candinate)[choose_num]
    for tem in choose:
        if 'template' in tem:
            list_ans.append(os.path.join(candinate_path,tem))
    np.random.shuffle(list_ans)
    img1 = cv2.hconcat([cv2.imread(list_ans[0]),cv2.imread(list_ans[1]),cv2.imread(list_ans[2]),cv2.imread(list_ans[3]),cv2.imread(list_ans[4])])
    img2 = cv2.hconcat([cv2.imread(list_ans[5]),cv2.imread(list_ans[6]),cv2.imread(list_ans[7]),cv2.imread(list_ans[8]),cv2.imread(list_ans[9])])
    tem_imgs = cv2.vconcat([img1,img2])

    return list_ans,tem_imgs



def test_all_image():
    for test_id in range(0,285):
        path_base = '/home/gzr/Data/generative_steel/real_dataset/0307-0313/process/all_data/'+str(test_id)+'/images/test'
        crop_base = '/home/gzr/Data/generative_steel/real_dataset/0307-0313/process/all_data/'+str(test_id)+'/crop/test'
        if os.path.isdir(path_base) is False:
            print('wrong path!')
            continue
        str_id = '%05d' % test_id
        fold_num = len(os.listdir(crop_base))

        raw_img = cv2.imread(os.path.join('/home/gzr/Data/generative_steel/real_dataset/0307-0313/imgs', str_id+'.jpg'))
        raw_img = cv2.resize(raw_img, dsize=(2400, 640))
        raw_img_bbox = raw_img
        # cv2.namedWindow('overall display', 0)
        # cv2.resizeWindow('overall display', 1200, 320)
        # cv2.imshow('overall display', raw_img_bbox)
        for rk_img in range(fold_num):
            precision_max = 0.6
            best_candidate = None
            out_img = None
            trans_predict = None
            template_best_path = None
            image_best_path = None

            from_id = max(0, rk_img-5) # max is 10 template
            to_id = min(fold_num, rk_img+5)

            template_paths,_ = load_template_set_9(os.path.join(path_base,str(rk_img)+'_template.png'))


            for rk_tem, template_path in enumerate(template_paths):
                # print(template_path)
                precision,out,trans = test_pair(template_path, os.path.join(path_base,str(rk_img)+'_homo.png'), matcher)
                if precision > precision_max:
                    precision_max = precision
                    out_img = out
                    best_candidate = rk_tem
                    trans_predict = trans[0].detach().cpu().numpy()
                    template_best_path = template_path
                    image_best_path = os.path.join(path_base,str(rk_img)+'_homo.png')
            if best_candidate is None:
                print('No suitable template in the candidate set')
            else:

                precision, out, trans = test_pair(template_best_path,
                                                  image_best_path, matcher_application,whole=True)
                print('The optimal template id is:', best_candidate, '\t the coarse matching precision is : ',round(precision_max,2))
                # cv2.imwrite(os.path.join(path_base,str(rk_img)+'_homo.png').replace('.png','_result.png'), out_img)

                template = cv2.imread(template_best_path)  # tamplate
                image = cv2.imread(image_best_path)
                img0_es_warped = cv2.warpPerspective(template, trans_predict, (template.shape[1], template.shape[0]))
                mask = img0_es_warped[:, :, 1] > 125
                img0_es_warped[:, :, 0][mask] = 255 - img0_es_warped[:, :, 1][mask]
                img0_es_warped[:, :, 2][mask] = 255 - img0_es_warped[:, :, 1][mask]
                cv2.addWeighted(img0_es_warped, 0.2, image, 1.0, 0, img0_es_warped)

                origin_coordinate = np.load(crop_base + '/' + str(rk_img) + '.npy')
                min_x_new, min_y_new, max_x_new, max_y_new = origin_coordinate[0], origin_coordinate[1], origin_coordinate[
                    2], origin_coordinate[3]

                # cv2.rectangle(raw_img_bbox,(min_x_new,min_y_new),(max_x_new,max_y_new),(0,0,255),1)
                # text = 'Confidence: ' + str(round(precision_max,2))
                # cv2.putText(raw_img_bbox,text,(min_x_new,min_y_new),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                # cv2.imshow('overall display', raw_img_bbox)
                # cv2.waitKey(100)
                w, h = max_x_new-min_x_new,max_y_new-min_y_new


                # cv2.namedWindow('single dosplay', 0)
                # cv2.resizeWindow('single dosplay',1300,400)
                # cv2.imshow('single dosplay', out)
                # cv2.waitKey(100)
                result = cv2.resize(img0_es_warped,[w,h])

                # raw_img[min_y_new:min_y_new+h, min_x_new:min_x_new+w,:] = result
                # avoid overlap
                overlap_area = raw_img[min_y_new:min_y_new+h, min_x_new:min_x_new+w, :]
                mask = (result[:, :, 1] != result[:, :, 0])
                overlap_area[:, :, 0][mask] = result[:, :, 0][mask]
                overlap_area[:, :, 1][mask] = result[:, :, 1][mask]
                overlap_area[:, :, 2][mask] = result[:, :, 2][mask]
                raw_img[min_y_new:min_y_new + h, min_x_new:min_x_new + w, :] = overlap_area

        # cv2.imwrite('/home/gzr/Data/generative_steel/real_dataset/0307-0313/process/'+str(test_id)+'/images/test/result.png',raw_img)
        # cv2.imshow('overall display', raw_img)
        # cv2.waitKey(100)

def vidio():
    # 6,14,18,19,21,28,34,42,43,48,49,51,60,61,62,65,66,71,72,75,78,79,80,81,84,85,87,90,92,118,200
    for test_id in [65]:
        path_base = '/home/gzr/Data/generative_steel/real_dataset/0307-0313/process/' + str(test_id) + '/images/test'
        crop_base = '/home/gzr/Data/generative_steel/real_dataset/0307-0313/process/' + str(test_id) + '/crop/test'
        str_id = '%05d' % test_id
        fold_num = len(os.listdir(crop_base))

        raw_img = cv2.imread(
            os.path.join('/home/gzr/Data/generative_steel/real_dataset/0307-0313/imgs', str_id + '.jpg'))
        raw_img = cv2.resize(raw_img, dsize=(2400, 640))
        raw_img_bbox = raw_img

        cv2.imshow('overall display', raw_img_bbox)

        template_paths,tem_imgs = load_template_set(path_base)
        tem_imgs = cv2.resize(tem_imgs, None, fx=0.5, fy=0.5)
        sc = min(480 / 640., 2.0)
        cv2.putText(tem_imgs, 'Template Candidates', (700, 20), cv2.FONT_HERSHEY_DUPLEX,
                    1.2 * sc, (255, 255, 255), 2, cv2.LINE_AA)

        for rk_img in reversed(range(fold_num)):
            precision_max = 0.
            best_candidate = None
            out_img = None
            trans_predict = None
            template_best_path = None
            image_best_path = None

            from_id = max(0, rk_img - 3)  # max is 10 template
            to_id = min(fold_num, rk_img + 3)
            # for rk_tem in template_paths:
            for index, template_path in enumerate(template_paths):
                if index==4 and rk_img==25-6 and test_id==65:
                    continue
                if test_id==65 and index!=4 and rk_img==25-19:
                    continue
                print('index:',index)
                precision, out, trans = test_pair(template_path,
                                                  os.path.join(path_base, str(rk_img) + '_homo.png'), matcher)
                if precision > precision_max:
                    precision_max = precision
                    out_img = out
                    best_candidate = index
                    trans_predict = trans[0].detach().cpu().numpy()
                    template_best_path = template_path
                    image_best_path = os.path.join(path_base, str(rk_img) + '_homo.png')
            if best_candidate is None:
                print('No suitable template in the candidate set')
            else:

                precision, out, trans = test_pair(template_best_path,
                                                  image_best_path, matcher_application, whole=True)
                print('The optimal template id is:', best_candidate, '\t the coarse matching precision is : ',
                      round(precision_max, 2))
                # cv2.imwrite(os.path.join(path_base,str(rk_img)+'_homo.png').replace('.png','_result.png'), out_img)

                template = cv2.imread(template_best_path)  # tamplate
                image = cv2.imread(image_best_path)
                img0_es_warped = cv2.warpPerspective(template, trans_predict, (template.shape[1], template.shape[0]))
                mask = img0_es_warped[:, :, 1] > 125
                img0_es_warped[:, :, 0][mask] = 255 - img0_es_warped[:, :, 1][mask]
                img0_es_warped[:, :, 2][mask] = 255 - img0_es_warped[:, :, 1][mask]
                cv2.addWeighted(img0_es_warped, 0.2, image, 1.0, 0, img0_es_warped)

                origin_coordinate = np.load(crop_base + '/' + str(rk_img) + '.npy')
                min_x_new, min_y_new, max_x_new, max_y_new = origin_coordinate[0], origin_coordinate[1], \
                                                             origin_coordinate[
                                                                 2], origin_coordinate[3]
                # fx = fy = 2400/out.shape[1]
                # out = cv2.resize(out,None, fx=fx, fy=fy*0.5)
                # out = cv2.rotate(out,cv2.ROTATE_90_COUNTERCLOCKWISE)


                w, h = max_x_new - min_x_new, max_y_new - min_y_new

                # cv2.namedWindow('single dosplay', 0)
                # cv2.resizeWindow('single dosplay', 1300, 400)
                # cv2.imshow('single dosplay', out)
                # cv2.waitKey(100)
                result = cv2.resize(img0_es_warped, [w, h])

                # raw_img[min_y_new:min_y_new+h, min_x_new:min_x_new+w,:] = result
                # avoid overlap
                overlap_area = raw_img[min_y_new:min_y_new + h, min_x_new:min_x_new + w, :]
                mask = (result[:, :, 1] != result[:, :, 0])
                overlap_area[:, :, 0][mask] = result[:, :, 0][mask]
                overlap_area[:, :, 1][mask] = result[:, :, 1][mask]
                overlap_area[:, :, 2][mask] = result[:, :, 2][mask]
                raw_img[min_y_new:min_y_new + h, min_x_new:min_x_new + w, :] = overlap_area

                cv2.rectangle(raw_img_bbox, (min_x_new, min_y_new), (max_x_new, max_y_new), (0, 0, 255), 1)
                text = 'Confidence: ' + str(round(precision_max, 2))
                cv2.putText(raw_img_bbox, text, (min_x_new, min_y_new), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                text = 'Template id: ' + str(best_candidate)
                cv2.putText(raw_img_bbox, text, (min_x_new, min_y_new+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                out = cv2.hconcat([out, tem_imgs])

                fx = fy = 2400/out.shape[1]
                out = cv2.resize(out,None, fx=fx, fy=fy)

                dis_play_matrix = cv2.vconcat([out, raw_img_bbox])

                cv2.imshow('overall display', dis_play_matrix)
                cv2.waitKey(100)

        cv2.imwrite('/home/gzr/Data/generative_steel/real_dataset/0307-0313/process/'+str(test_id)+'/images/test/result.png',dis_play_matrix)
        pass
        # cv2.imshow('overall display', raw_img)
        # cv2.waitKey(100)

if __name__ == '__main__':
    # best_candidate
    vidio()



