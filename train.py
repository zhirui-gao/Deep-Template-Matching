import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import math
import argparse
import pprint
from distutils.util import strtobool
from pathlib import Path
from loguru import logger as loguru_logger
from src.config.default import get_cfg_defaults
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin
from src.lightning.lightning_tm import PL_Tm
from src.utils.misc import setup_gpus,get_rank_zero_only_logger
from src.utils.profiler import build_profiler
import torch
from src.lightning.data import MultiSceneDataModule
def parse_args():
    # init a costum parser which will be added into pl.Trainer parser
    # check documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--data_cfg_path', type=str, default='./config/Synthetic_train.py', help='data config path') # linemod2d_train
    parser.add_argument(
        '--main_cfg_path', type=str, default='./config/model_tm.py', help='main config path')
    parser.add_argument(
        '--batch_size', type=int, default=1, help='batch_size per gpu')
    parser.add_argument(
        '--num_workers', type=int, default=1)
    parser.add_argument(
        '--pin_memory', type=lambda x: bool(strtobool(x)),
        nargs='?', default=True, help='whether loading data to pinned memory or not')
    parser.add_argument(
        '--ckpt_path_backbone', type=str, default='./pretrained/superpoint_v1.pth', #superpoint_v1.pth  superPointNet_170000_checkpoint.pth.tar
        help='pretrained checkpoint path')
    parser.add_argument(
        '--ckpt_path', type=str, default='',
        help='pretrained checkpoint path')
    parser.add_argument(
        '--disable_ckpt', action='store_true',
        help='disable checkpoint saving (useful for debugging).')
    parser.add_argument(
        '--profiler_name', type=str, default=None,
        help='options: [inference, pytorch], or leave it unset')
    parser.add_argument(
        '--parallel_load_data', action='store_true',
        help='load datasets in with multiple processes.')

    parser.add_argument(
        '--exp_name', type=str, default='default_exp_name')

    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()

def main():
    args = parse_args()
    rank_zero_only(pprint.pprint)(vars(args))
    # init default-cfg and merge it with the main- and data-cfg
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    pl.seed_everything(config.TRAINER.SEED)  # reproducibility
    # scale lr and warmup-step automatically
    args.gpus = _n_gpus = setup_gpus(args.gpus)
    config.TRAINER.WORLD_SIZE = _n_gpus * args.num_nodes
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
    _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    config.TRAINER.SCALING = _scaling
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * _scaling
    config.TRAINER.WARMUP_STEP = math.floor(config.TRAINER.WARMUP_STEP / _scaling)

    # lightning module
    profiler = build_profiler(args.profiler_name)
    model = PL_Tm(config, pretrained_ckpt_backbone=args.ckpt_path_backbone, pretrain_ckpt=args.ckpt_path, profiler=profiler)

    loguru_logger.info(f"TM LightningModule initialized!")

    # lightning data
    data_module = MultiSceneDataModule(args, config)
    loguru_logger.info(f"TM DataModule initialized!")


    # TensorBoard Logger
    logger = TensorBoardLogger(save_dir='logs/tb_logs', name=args.exp_name, default_hp_metric=False)
    ckpt_dir = Path(logger.log_dir) / 'checkpoints'
    # save network graph
    # batch_test = {}
    # img0 = torch.rand((1, 1, 256, 256))
    # img1 = torch.rand((1, 1, 256, 256))
    # data = {
    #     'image0': img0,  # (1, h, w)
    #     'image1': img1,
    #     'dataset_name': 'linemod_2d'
    #
    # }
    # logger.experiment.add_graph(model, data)

    # Callbacks
    # TODO: update ModelCheckpoint to monitor multiple metrics
    ckpt_callback = ModelCheckpoint(monitor='auc@3', verbose=True, save_top_k=3, mode='max',
                                    save_last=True,
                                    dirpath=str(ckpt_dir),
                                    filename='{epoch}-{auc@1:.3f}-{auc@3:.3f}-{auc@5:.3f}-{auc@10:.3f}')

    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [lr_monitor]
    if not args.disable_ckpt:
        callbacks.append(ckpt_callback)

    # Lightning Trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        plugins=DDPPlugin(find_unused_parameters=False,
                          num_nodes=args.num_nodes,
                          sync_batchnorm=config.TRAINER.WORLD_SIZE > 0),
        gradient_clip_val=config.TRAINER.GRADIENT_CLIPPING,
        callbacks=callbacks,
        logger=logger,
        #resume_from_checkpoint=args.ckpt_path,
        sync_batchnorm=config.TRAINER.WORLD_SIZE > 0,
        replace_sampler_ddp=False,  # use custom sampler
        reload_dataloaders_every_epoch=False,  # avoid repeated samples!
        weights_summary='full',
        profiler=profiler)
    loguru_logger.info(f"Trainer initialized!")
    loguru_logger.info(f"Start training!")
    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    main()