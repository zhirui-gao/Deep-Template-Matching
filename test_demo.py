import cv2
import torch
import torchvision.transforms as transforms
from src.lightning.lightning_tm import PL_Tm
from src.config.default import get_cfg_defaults
import numpy as np
from src.lightning.data import get_contours_points
import torch.nn.functional as F
import matplotlib.cm as cm
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
from src.utils.utils import farthest_point_sample
from src.utils.plotting import make_matching_figure_4
import random

CFG = {
    'IMG_PATHS': {
        'template': './data/test_case/case0_template.png',
        'image': './data/test_case/case0.png',
    },
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'RESIZE': (480, 640),  # H,W
    'PATCH_SIZE': 8,
    'NUM_POINTS': 128,
    'NORMALIZE': transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
}


def read_image(path, grayscale=True):
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
        img = cv2.resize(img, (CFG['RESIZE'][1], CFG['RESIZE'][0]))
        return img
    except Exception as e:
        print(f"Error reading image: {path}", e)
        return None


def preprocess_image(img, grayscale=True):
    if grayscale:
        img = torch.from_numpy(img)[None][None].cuda() / 255.
    else:
        img = transforms.Compose([
            transforms.ToTensor(),
            CFG['NORMALIZE']
        ])(img)
        img = img[None]
    return img


def generate_contours_points(image, num):
    contours_points = get_contours_points(image)
    contours_points = np.round(contours_points) // CFG['PATCH_SIZE']
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

    contours_points[:, 0] = np.clip(contours_points[:, 0], 0, (CFG['RESIZE'][1] // CFG['PATCH_SIZE']) - 1)
    contours_points[:, 1] = np.clip(contours_points[:, 1], 0, (CFG['RESIZE'][0] // CFG['PATCH_SIZE']) - 1)
    contours_points = torch.tensor(contours_points, dtype=torch.long)
    return contours_points, mask_0


if __name__ == "__main__":
    config = get_cfg_defaults()
    device = torch.device(CFG['DEVICE'])
    ckpt_path = './weights/epoch=17-auc@1=0.175-auc@3=0.591-auc@5=0.748-auc@1=auc@10=0.874.ckpt'
    model = PL_Tm(config, pretrain_ckpt=ckpt_path)
    matcher = model.eval().to(device)

    img0 = read_image(CFG['IMG_PATHS']['template'], grayscale=True)
    img1 = read_image(CFG['IMG_PATHS']['image'], grayscale=True)
    img1_RGB = read_image(CFG['IMG_PATHS']['image'], grayscale=False)
    img1_RGB = cv2.cvtColor(img1_RGB, cv2.COLOR_BGR2RGB)
    if img0 is None or img1 is None:
        print("One or both images could not be read.")
        exit(1)

    image0_raw, image1_raw = img0, img1
    scale = torch.tensor([img0.shape[1] / CFG['RESIZE'][1], img0.shape[0] / CFG['RESIZE'][0]], dtype=torch.float)
    image1_edge = cv2.Canny(img1, 5, 10)
    img0 = cv2.Canny(img0, 5, 10)
    contours_points, mask_0 = generate_contours_points(img0, CFG['NUM_POINTS'])
    image0, image1 = preprocess_image(img0), preprocess_image(img1)
    image1_edge = preprocess_image(image1_edge)
    image1_rgb = preprocess_image(img1_RGB, grayscale=False)
    batch = {
        'dataset_name': ['synthetic'],
        'image0': image0,
        'image1': image1,
        'image1_edge': image1_edge.to(device),
        'image0_raw': torch.from_numpy(image0_raw)[None].to(device),
        'image1_raw': torch.from_numpy(image1_raw)[None].to(device),
        'scale': scale[None].to(device),
        'c_points': contours_points[None].to(device),
        'image1_rgb': image1_rgb.to(device),
        'resolution': [CFG['PATCH_SIZE']],
        'trans': torch.ones([3, 3], device=device)[None],
        'pair_names': [(CFG['IMG_PATHS']['template'], CFG['IMG_PATHS']['image'])],
    }

    mask0 = torch.from_numpy(np.ones((image0.shape[2], image0.shape[3]), dtype=bool))
    mask1 = torch.from_numpy(np.ones((image1.shape[2], image1.shape[3]), dtype=bool))

    if mask1 is not None:
        coarse_scale = 1 / CFG['PATCH_SIZE']
        if coarse_scale:
            [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
                                                   scale_factor=coarse_scale,
                                                   mode='nearest',
                                                   recompute_scale_factor=False)[0].bool()
            batch.update({'mask1': ts_mask_1[None].to(device)})
            batch.update({'mask0': torch.from_numpy(mask_0)[None].cuda()})  # coarse_scale mask [L]

    with torch.no_grad():
        matcher(batch)
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        mconf = torch.ones(mkpts1.shape[0]).cpu().numpy() * 0.5
        img0 = (batch['image0'][0][0].cpu().numpy() * 255).round().astype(np.int32)
        img1 = (batch['image1'][0][0].cpu().numpy() * 255).round().astype(np.int32)
        img1_edge = (batch['edge'][0][0].cpu().detach().numpy() * 255).round().astype(np.int32)

        ave_aligned = ((batch['warped_template'][0][0] * 0.5 + batch['image1'][0][
            0] * 0.5).cpu().detach().numpy() * 255).round().astype(np.int32)

        color = cm.jet(mconf)
        print(color.shape)
        text = [
            'Tm',
            f'Matches: {len(mkpts0)}',
        ]
        fig = make_matching_figure_4(ave_aligned, img0, img1, img1_edge,
                                     mkpts0, mkpts1, color, text=text)
        plt.show()
