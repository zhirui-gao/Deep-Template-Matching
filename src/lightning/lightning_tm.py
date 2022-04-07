import torch
import numpy as np
import pytorch_lightning as pl
from src.models.tm import Tm
import pprint
from loguru import logger
import matplotlib.cm as cm
from collections import defaultdict
from src.utils.misc import lower_config, flattenList
from src.utils.profiler import PassThroughProfiler
from src.utils.supervision  import compute_supervision_coarse
from src.losses.tm_loss import TmLoss
from src.utils.comm import gather, all_gather
from src.utils.plotting import make_matching_figures
from src.optimizers import build_optimizer,build_scheduler
from src.utils.plotting import make_matching_figure
from src.utils.metrics import compute_distance_errors,aggregate_metrics
from matplotlib import pyplot as plt
class PL_Tm(pl.LightningModule):
    def __init__(self, config, pretrained_ckpt_backbone=None, pretrain_ckpt=None, profiler=None, dump_dir=None):
        super().__init__()
        # Misc
        self.config = config  # full config
        _config = lower_config(self.config)
        self.profiler = profiler or PassThroughProfiler()
        self.n_vals_plot = max(config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1)

        # TM Module
        self.Tm = Tm(config=_config)
        self.loss = TmLoss(_config)
        # Pretrained weights
        if pretrained_ckpt_backbone:
            pre_state_dict = torch.load(pretrained_ckpt_backbone, map_location='cpu')['model_state_dict']
            model_dict = self.Tm.state_dict()
            for k, v in pre_state_dict.items():
                if 'backbone.'+k in model_dict.keys() and v.shape == model_dict['backbone.'+k].shape:
                    model_dict['backbone.'+k] = v
            self.Tm.load_state_dict(model_dict, strict=True)
            logger.info(f"Load \'{pretrained_ckpt_backbone}\' as pretrained checkpoint_backbone")

        if pretrain_ckpt:
            pre_state_dict = torch.load(pretrain_ckpt, map_location='cpu')['state_dict']
            model_dict = self.Tm.state_dict()
            for k, v in pre_state_dict.items():
                if k[3:] in model_dict.keys() and v.shape == model_dict[k[3:]].shape:
                    model_dict[k[3:]] = v  # # get out '.TM'
            self.Tm.load_state_dict(model_dict, strict=True)
            logger.info(f"Load \'{pretrain_ckpt}\' as pretrained checkpoint")


        #
        # to_freeze_dict = ['backbone']
        # # loftr
        # for (name, param) in self.Tm.named_parameters():
        #     if name.split('.')[0] in to_freeze_dict:
        #         print(name, ': freezeed')
        #         param.requires_grad = False
        #     else:
        #         pass
        # Testing
        self.dump_dir = dump_dir

    def configure_optimizers(self):
        # FIXME: The scheduler did not work properly when `--resume_from_checkpoint`
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]  # 优化器定义，返回一个优化器，或数个优化器，或两个List（优化器，Scheduler）

    def optimizer_step(
            self, epoch, batch_idx, optimizer, optimizer_idx,
            optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # learning rate warm up
        warmup_step = self.config.TRAINER.WARMUP_STEP
        if self.trainer.global_step < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == 'linear':
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + \
                     (self.trainer.global_step / self.config.TRAINER.WARMUP_STEP) * \
                     abs(self.config.TRAINER.TRUE_LR - base_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
            elif self.config.TRAINER.WARMUP_TYPE == 'constant':
                pass
            else:
                raise ValueError(f'Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}')

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def _compute_metrics(self, batch):
        with self.profiler.profile("Copmute metrics"):
            compute_distance_errors(batch)
            rel_pair_names = list(zip(*batch['pair_names']))
            bs = batch['image0'].size(0)
            metrics = {
                # to filter duplicate pairs caused by DistributedSampler
                'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)],
                'inliers':  batch['inliers'],
                'dis_errs': [batch['dis_errs'][batch['m_bids'] == b].cpu().numpy() for b in range(bs)]
            }
            ret_dict = {'metrics': metrics}
        return ret_dict, rel_pair_names

    def _trainval_inference(self, batch):
        with self.profiler.profile("get keypoint and descriptor from backbone"):
            outs_post = self.Tm(batch)

        with self.profiler.profile("Compute coarse supervision"):
            compute_supervision_coarse(batch, self.config)

        with self.profiler.profile("Compute losses"):
            self.loss(batch)

    def forward(self, batch):
        self._trainval_inference(batch)


    def training_step(self, batch, batch_idx):
        self._trainval_inference(batch)

        # logging
        if self.trainer.global_rank == 0 and self.global_step % self.trainer.log_every_n_steps == 0:
            # scalars
            for k, v in batch['loss_scalars'].items():
                self.logger.experiment.add_scalar(f'train/{k}', v, self.global_step)

            # net-params
            if self.config.TM.MATCH_COARSE.MATCH_TYPE == 'sinkhorn':
                self.logger.experiment.add_scalar(
                    f'skh_bin_score', self.Tm.coarse_matching.bin_score.clone().detach().cpu().data, self.global_step)

            # figures
            if self.config.TRAINER.ENABLE_PLOTTING:
                # compute the error of each eatimate correspondence
                compute_distance_errors(batch)
                figures = make_matching_figures(batch, self.config, self.config.TRAINER.PLOT_MODE)
                for k, v in figures.items():
                    self.logger.experiment.add_figure(f'train_match/{k}', v, self.global_step)

        return {'loss': batch['loss']} # dict Can include any keys,but must include the key 'loss'

    def training_epoch_end(self, outputs):
        # 在一个训练epoch结尾处被调用。
        # 输入参数：一个List，List的内容是前面training_step()所返回的每次的内容。
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        if self.trainer.global_rank == 0:
            self.logger.experiment.add_scalar(
                'train/avg_loss_on_epoch', avg_loss,
                global_step=self.current_epoch)

    def validation_step(self, batch, batch_idx):
        self._trainval_inference(batch)
        ret_dict, _ = self._compute_metrics(batch)

        val_plot_interval = max(self.trainer.num_val_batches[0] // self.n_vals_plot, 1)
        figures = {self.config.TRAINER.PLOT_MODE: []}
        if batch_idx % val_plot_interval == 0:
            figures = make_matching_figures(batch, self.config, mode=self.config.TRAINER.PLOT_MODE)

        return {
            **ret_dict,
            'loss_scalars': batch['loss_scalars'],
            'figures': figures,
        }

    def validation_epoch_end(self, outputs):
        # handle multiple validation sets
        multi_outputs = [outputs] if not isinstance(outputs[0], (list, tuple)) else outputs
        multi_val_metrics = defaultdict(list)
        # TODO:
        for valset_idx, outputs in enumerate(multi_outputs):
            # since pl performs sanity_check at the very begining of the training
            cur_epoch = self.trainer.current_epoch
            if not self.trainer.resume_from_checkpoint and self.trainer.running_sanity_check:
                cur_epoch = -1

            # 1. loss_scalars: dict of list, on cpu
            _loss_scalars = [o['loss_scalars'] for o in outputs]
            loss_scalars = {k: flattenList(all_gather([_ls[k] for _ls in _loss_scalars])) for k in _loss_scalars[0]}

            # 2. val metrics: dict of list, numpy
            _metrics = [o['metrics'] for o in outputs]
            metrics = {k: flattenList(all_gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}
            # NOTE: all ranks need to `aggregate_merics`, but only log at rank-0
            val_metrics_4tb = aggregate_metrics(metrics, self.config.TRAINER.DIS_ERR_THR)

            for thr in [1, 2, 3]:
                multi_val_metrics[f'prec_dis_errs@{thr}'] = val_metrics_4tb[f'prec_dis_errs@{thr}']

            # 3. figures
            _figures = [o['figures'] for o in outputs]
            figures = {k: flattenList(gather(flattenList([_me[k] for _me in _figures]))) for k in _figures[0]}

            # tensorboard records only on rank 0
            if self.trainer.global_rank == 0:
                for k, v in loss_scalars.items():
                    mean_v = torch.stack(v).mean()
                    self.logger.experiment.add_scalar(f'val_{valset_idx}/avg_{k}', mean_v, global_step=cur_epoch)

                for k, v in val_metrics_4tb.items():
                    self.logger.experiment.add_scalar(f"metrics_{valset_idx}/{k}", v, global_step=cur_epoch)

                for k, v in figures.items():
                    if self.trainer.global_rank == 0:
                        for plot_idx, fig in enumerate(v):
                            self.logger.experiment.add_figure(
                                f'val_match_{valset_idx}/{k}/pair-{plot_idx}', fig, cur_epoch, close=True)
            plt.close('all')
        for thr in [1, 2, 3]:
            # log on all ranks for ModelCheckpoint callback to work properly
            self.log(f'prec_dis_errs@{thr}', torch.tensor(np.mean(multi_val_metrics[f'prec_dis_errs@{thr}'])))  # ckpt monitors on this

    def test_step(self, batch, batch_idx):
        with self.profiler.profile("get keypoint and descriptor from backbone"):
            self.Tm(batch)
        with self.profiler.profile("Compute coarse supervision"):
            compute_supervision_coarse(batch, self.config)
        compute_distance_errors(batch)
        figures = make_matching_figures(batch, self.config, self.config.TRAINER.PLOT_MODE)
        for k, v in figures.items():
            v[0].show()

        ret_dict, rel_pair_names = self._compute_metrics(batch)
        # save_visualize = True
        # if save_visualize:
        #     mkpts0 = batch['mkpts0_c'].cpu().numpy()
        #     mkpts1 = batch['mkpts1_c'].cpu().numpy()
        #     mconf = batch['mconf'].cpu().numpy()
        #     img0_raw = batch['image0'].cpu().numpy() * 255
        #
        #     img1_raw = batch['image1'].cpu().numpy() * 255
        #     img0_raw = np.squeeze(img0_raw)
        #     img1_raw = np.squeeze(img1_raw)
        #
        #     # Draw
        #     color = cm.jet(mconf)
        #     text = [
        #         'TM',
        #         'Matches: {}'.format(len(mkpts0)),
        #     ]
        #     fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, text=text)
        #     fig.show()

        return ret_dict

    def test_epoch_end(self, outputs):

        # metrics: dict of list, numpy
        _metrics = [o['metrics'] for o in outputs]
        metrics = {k: flattenList(gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}
        if self.trainer.global_rank == 0:
            print(self.profiler.summary())
            val_metrics_4tb = aggregate_metrics(metrics, self.config.TRAINER.DIS_ERR_THR)
            logger.info('\n' + pprint.pformat(val_metrics_4tb))





