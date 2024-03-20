import os
import math
from collections import abc
from loguru import logger
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from os import path as osp
from pathlib import Path
from joblib import Parallel, delayed
import cv2
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import distributed as dist
from torch.utils.data import (
    Dataset,
    DataLoader,
    ConcatDataset,
    DistributedSampler,
    RandomSampler,
    dataloader
)
import torch
import re
import collections
from torch._six import string_classes

np_str_obj_array_pattern = re.compile(r'[SaUO]')
from src.utils.augment import build_augmentor
from src.utils.dataloader import get_local_split
from src.utils.misc import tqdm_joblib
from src.utils import comm
from src.dataset.linemod_2d import SyntheticDataset
from src.dataset.sampler import RandomConcatSampler


class MultiSceneDataModule(pl.LightningDataModule):
    """
    For distributed training, each training process is assgined
    only a part of the training scenes to reduce memory overhead.
    """

    def __init__(self, args, config):
        super().__init__()

        # 1. data config
        # Train and Val should from the same data source
        self.trainval_data_source = config.DATASET.TRAINVAL_DATA_SOURCE
        self.test_data_source = config.DATASET.TEST_DATA_SOURCE
        # training and validating
        self.train_data_root = config.DATASET.TRAIN_DATA_ROOT
        self.train_npz_root = config.DATASET.TRAIN_NPZ_ROOT
        self.train_list_path = config.DATASET.TRAIN_LIST_PATH
        self.val_data_root = config.DATASET.VAL_DATA_ROOT
        self.val_npz_root = config.DATASET.VAL_NPZ_ROOT
        self.val_list_path = config.DATASET.VAL_LIST_PATH
        self.img_resize = config.DATASET.IMG_RESIZE
        # testing
        self.test_data_root = config.DATASET.TEST_DATA_ROOT
        self.test_npz_root = config.DATASET.TEST_NPZ_ROOT
        self.test_list_path = config.DATASET.TEST_LIST_PATH

        # 2. dataset config
        # general options
        self.augment_fn = build_augmentor(config.DATASET.AUGMENTATION_TYPE)  # None, options: [None, 'dark', 'mobile']
        # 3.loader parameters
        self.train_loader_params = {
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'pin_memory': getattr(args, 'pin_memory', True),
            'collate_fn': my_collator,  # use custum collate
        }
        self.val_loader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': args.num_workers,
            'pin_memory': getattr(args, 'pin_memory', True),
            'collate_fn': my_collator,  # use custum collate
        }
        self.test_loader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': args.num_workers,
            'pin_memory': True,
            'collate_fn': my_collator,  # use custum collate
        }

        # 4. sampler
        self.data_sampler = config.TRAINER.DATA_SAMPLER
        self.n_samples_per_subset = config.TRAINER.N_SAMPLES_PER_SUBSET
        self.subset_replacement = config.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT
        self.shuffle = config.TRAINER.SB_SUBSET_SHUFFLE
        self.repeat = config.TRAINER.SB_REPEAT
        # (optional) RandomSampler for debugging

        # misc configurations
        self.parallel_load_data = getattr(args, 'parallel_load_data', False)
        self.seed = config.TRAINER.SEED  # 66

    def setup(self, stage=None):
        """
        Setup train / val / test dataset. This method will be called by PL automatically.
        Args:
            stage (str): 'fit' in training phase, and 'test' in testing phase.
        """
        assert stage in ['fit', 'test'], "stage must be either fit or test"

        try:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            logger.info(f"[rank:{self.rank}] world_size: {self.world_size}")
        except AssertionError as ae:
            self.world_size = 1
            self.rank = 0
            logger.warning(str(ae) + " (set wolrd_size=1 and rank=0)")

        if stage == 'fit':
            self.train_dataset = self._setup_dataset(
                self.train_data_root,
                self.train_npz_root,
                self.train_list_path,
                mode='train')
            # setup multiple (optional) validation subsets
            if isinstance(self.val_list_path, (list, tuple)):
                self.val_dataset = []
                if not isinstance(self.val_npz_root, (list, tuple)):
                    self.val_npz_root = [self.val_npz_root for _ in range(len(self.val_list_path))]
                for npz_list, npz_root in zip(self.val_list_path, self.val_npz_root):
                    self.val_dataset.append(self._setup_dataset(
                        self.val_data_root,
                        npz_root,
                        npz_list,
                        mode='val'))
            else:
                self.val_dataset = self._setup_dataset(
                    self.val_data_root,
                    self.val_npz_root,
                    self.val_list_path,
                    mode='val')
            logger.info(f'[rank:{self.rank}] Train & Val Dataset loaded!')
        else:  # stage == 'test
            self.test_dataset = self._setup_dataset(
                self.test_data_root,
                self.test_npz_root,
                self.test_list_path,
                mode='test')
            logger.info(f'[rank:{self.rank}]: Test Dataset loaded!')

    def _setup_dataset(self,
                       data_root,
                       split_npz_root,
                       scene_list_path,
                       mode='train',
                       ):
        """ Setup train / val / test set"""
        with open(scene_list_path, 'r') as f:
            npz_names = [name.split()[0] for name in f.readlines()]

        if mode == 'train':
            local_npz_names = get_local_split(npz_names, self.world_size, self.rank, self.seed)
        else:
            local_npz_names = npz_names
        logger.info(f'[rank {self.rank}]: {len(local_npz_names)} scene(s) assigned.')

        dataset_builder = self._build_concat_dataset_parallel \
            if self.parallel_load_data \
            else self._build_concat_dataset
        return dataset_builder(data_root, local_npz_names, split_npz_root,
                               mode=mode)

    def _build_concat_dataset(
            self,
            data_root,
            npz_names,
            npz_dir,
            mode,
    ):
        datasets = []
        augment_fn = self.augment_fn if mode == 'train' else None
        data_source = self.trainval_data_source if mode in ['train', 'val'] else self.test_data_source

        for npz_name in tqdm(npz_names,
                             desc=f'[rank:{self.rank}] loading {mode} datasets',
                             disable=int(self.rank) != 0):
            npz_path = osp.join(npz_dir, npz_name)
            data_root_name = osp.join(data_root, npz_name)

            datasets.append(
                SyntheticDataset(data_root_name,
                                 npz_path,
                                 mode=mode,
                                 img_resize=self.img_resize,
                                 augment_fn=augment_fn))

        return ConcatDataset(datasets)

    def _build_concat_dataset_parallel(
            self,
            data_root,
            npz_names,
            npz_dir,
            mode
    ):

        augment_fn = self.augment_fn if mode == 'train' else None
        data_source = self.trainval_data_source if mode in ['train', 'val'] else self.test_data_source

        with tqdm_joblib(tqdm(desc=f'[rank:{self.rank}] loading {mode} datasets',
                              total=len(npz_names), disable=int(self.rank) != 0)):
            if data_source == 'linemod_2d':
                datasets = Parallel(math.floor(len(os.sched_getaffinity(0)) * 0.9 / comm.get_local_size()))(
                    delayed(lambda x: _build_dataset(
                        Linemod2dDataset,
                        osp.join(data_root, x),
                        osp.join(npz_dir, x),
                        mode=mode,
                        img_resize=self.img_resize,
                        augment_fn=augment_fn))(name)
                    for name in npz_names)
            else:
                raise ValueError(f'Unknown dataset: {data_source}')
        return ConcatDataset(datasets)

    def train_dataloader(self):
        """ Build training dataloader for ScanNet / MegaDepth. """
        # assert self.data_sampler in ['scene_balance']
        logger.info(
            f'[rank:{self.rank}/{self.world_size}]: Train Sampler and DataLoader re-init (should not re-init between epochs!).')
        if self.data_sampler == 'scene_balance':
            sampler = RandomConcatSampler(self.train_dataset,
                                          self.n_samples_per_subset,
                                          self.subset_replacement,
                                          self.shuffle, self.repeat, self.seed)
        else:
            sampler = None
        # TODO: for muti-scene ,we should take sampler
        sampler = None
        dataloader = DataLoader(self.train_dataset, sampler=sampler, shuffle=True, **self.train_loader_params)
        return dataloader

    def val_dataloader(self):
        """ Build validation dataloader for ScanNet / MegaDepth. """
        logger.info(f'[rank:{self.rank}/{self.world_size}]: Val Sampler and DataLoader re-init.')
        if not isinstance(self.val_dataset, abc.Sequence):
            sampler = DistributedSampler(self.val_dataset, shuffle=False)
            return DataLoader(self.val_dataset, sampler=sampler, **self.val_loader_params)
        else:
            dataloaders = []
            for dataset in self.val_dataset:
                sampler = DistributedSampler(dataset, shuffle=False)
                dataloaders.append(DataLoader(dataset, sampler=sampler, **self.val_loader_params))
            return dataloaders

    def test_dataloader(self, *args, **kwargs):
        logger.info(f'[rank:{self.rank}/{self.world_size}]: Test Sampler and DataLoader re-init.')
        sampler = DistributedSampler(self.test_dataset, shuffle=False)
        return DataLoader(self.test_dataset, sampler=sampler, **self.test_loader_params)


def _build_dataset(dataset: Dataset, *args, **kwargs):
    return dataset(*args, **kwargs)


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
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)  # 采样点矩阵（B, npoint）
    distance = torch.ones(B, N).to(device) * 1e10  # 采样点到所有点距离（B, N）
    batch_indices = torch.arange(B, dtype=torch.long).to(device)  # batch_size 数组
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)  # 初始时随机选择一点

    for i in range(npoint):
        centroids[:, i] = farthest  # 更新第i个最远点
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 2)  # 取出这个最远点的xyz坐标
        dist = torch.sum((xyz - centroid) ** 2, -1).float()  # 计算点集中的所有点到这个最远点的欧式距离
        mask = dist < distance
        distance[mask] = dist[mask]  # 更新distance，记录样本中每个点距离所有已出现的采样点的最小距离
        farthest = torch.max(distance, -1)[1]  # 返回最远点索引

    return centroids


def my_collator(batch):
    # TODO: sampler by image size
    r"""Puts each data field into a tensor with outer dimension batch size"""
    # resize image:
    is_whole = True
    if isinstance(batch[0], dict) and 'image1' in batch[0] and batch[0]['image1'].ndim == 2:

        df, df_fine = 8, 2  # 8
        num = 128  # 128 # num of query point
        num_fine = 1024
        Resize = [480, 640]  # h,w

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        for i in range(len(batch)):
            batch[i]['resolution'] = df
            w, h = batch[i]['image1'].shape[1], batch[i]['image1'].shape[0]
            w_new = Resize[1]
            h_new = Resize[0]
            pad_to = Resize  # [h_max, w_max]
            # rgb image resize
            image1_rgb = cv2.resize(batch[i]['image1_rgb'], (w_new, h_new))
            image1_rgb = transform(image1_rgb)  # c,h,w
            image1 = cv2.resize(batch[i]['image1'], (w_new, h_new))
            image1_edge = cv2.Canny(image1, 5, 10)  # 20,50 for coco
            image1 = torch.from_numpy(image1).float()[None] / 255  # (h, w) -> (1, h, w) and normalized
            image1_edge = torch.from_numpy(image1_edge).float()[None] / 255  # (h, w) -> (1, h, w) and normalized

            mask1 = torch.from_numpy(np.ones((image1.shape[1], image1.shape[2]), dtype=bool))

            scale = torch.tensor([w / w_new, h / h_new], dtype=torch.float)
            image0 = cv2.resize(batch[i]['image0'], dsize=(0, 0), fx=1 / float(scale[0]), fy=1 / float(scale[1]))
            batch[i]['image0_mask'] = image0  # (1, h, w)
            image0 = cv2.Canny(image0, 5, 10)
            contours_points = get_contours_points(image0)

            if is_whole:
                contours_points_fine = np.round(contours_points) // df_fine
                contours_points_fine = np.array(list(set([tuple(t) for t in contours_points_fine])))

            contours_points = np.round(contours_points) // df
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
            contours_points[:, 0] = np.clip(contours_points[:, 0], 0, (w_new // df) - 1)
            contours_points[:, 1] = np.clip(contours_points[:, 1], 0, (h_new // df) - 1)
            contours_points = contours_points.astype(np.int_)
            if is_whole:
                mask_fine = np.zeros(num_fine, dtype=bool)
                if num_fine <= contours_points_fine.shape[0]:
                    indices = farthest_point_sample(torch.tensor(contours_points_fine)[None, :], num_fine)[0]
                    contours_points_fine = contours_points_fine[indices]
                    mask_fine[:] = True
                else:
                    mask_fine[:contours_points_fine.shape[0]] = True
                    num_pad = num_fine - contours_points_fine.shape[0]
                    pad = np.random.choice(contours_points_fine.shape[0], num_pad, replace=True)
                    choice = np.concatenate([range(contours_points_fine.shape[0]), pad])
                    contours_points_fine = contours_points_fine[choice, :]

                contours_points_fine[:, 0] = np.clip(contours_points_fine[:, 0], 0, (w_new // df_fine) - 1)
                contours_points_fine[:, 1] = np.clip(contours_points_fine[:, 1], 0, (h_new // df_fine) - 1)
                contours_points_fine = contours_points_fine.astype(np.int_)
                batch[i]['f_points'] = contours_points_fine
                batch[i]['mask_fine_point'] = mask_fine

            image0 = torch.from_numpy(image0).float()[None] / 255  # (h, w) -> (1, h, w) and normalized
            batch[i]['image0'] = image0  # (1, h, w)
            batch[i]['image1'] = image1  # (1, h, w)
            batch[i]['image1_edge'] = image1_edge  # (1, h, w)
            batch[i]['scale'] = scale
            batch[i]['c_points'] = contours_points
            batch[i]['image1_rgb'] = image1_rgb
            mask0 = torch.from_numpy(np.ones((image0.shape[1], image0.shape[2]), dtype=bool))
            if mask1 is not None:  # img_padding is True
                coarse_scale = 1 / df
                if coarse_scale:
                    [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
                                                           scale_factor=coarse_scale,
                                                           mode='nearest',
                                                           recompute_scale_factor=False)[0].bool()
                batch[i].update({'mask1': ts_mask_1})
                batch[i].update({'mask0': mask_0})  # coarse_scale mask  [L]

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
            out = torch.stack(batch, 0)
        return out
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return my_collator([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: my_collator([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(my_collator(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [my_collator(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def pad_bottom_right(inp, pad_size, ret_mask=False):
    mask = None
    if inp.ndim == 2:
        assert isinstance(pad_size[0], int) and pad_size[0] >= inp.shape[0] and pad_size[1] >= inp.shape[1]
        padded = np.zeros((pad_size[0], pad_size[1]), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1]] = inp
        if ret_mask:
            mask = np.zeros((pad_size[0], pad_size[1]), dtype=bool)
            mask[:inp.shape[0], :inp.shape[1]] = True
    if inp.ndim == 3:  # for h,w,3,
        assert isinstance(pad_size[0], int) and pad_size[0] >= inp.shape[0] and pad_size[1] >= inp.shape[1]
        padded = np.zeros((pad_size[0], pad_size[1], inp.shape[2]), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1], :] = inp
        if ret_mask:
            mask = np.zeros((pad_size[0], pad_size[1], inp.shape[2]), dtype=bool)
            mask[:inp.shape[0], :inp.shape[1], :] = True
    return padded, mask


def get_contours_points(image):
    """
    :param image: (H,W)
    :return: (N,2)
    """
    assert (len(image.shape) == 2)
    ret, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    xcnts = np.vstack([x.reshape(-1, 2) for x in contours])
    # get out of the contours on  edge
    mask_x_0 = xcnts[:, 0] > 8
    mask_y_0 = xcnts[:, 1] > 8
    mask_x_1 = xcnts[:, 0] < (image.shape[1] - 8)
    mask_y_1 = xcnts[:, 1] < (image.shape[0] - 8)

    mask = mask_x_0 * mask_y_0 * mask_x_1 * mask_y_1
    if mask.sum(0) == 0:
        pass
    else:
        xcnts = xcnts[mask]

    return xcnts
