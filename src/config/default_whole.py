from yacs.config import CfgNode as CN
import torch
_CN = CN()

##############  ↓  SuperPonit  ↓  ##############
_CN.TM = CN()
_CN.TM.SUPERPOINT = CN()
_CN.TM.SUPERPOINT.NAME =  'superpoint_glue' # 'SuperPointNet_gauss2'
_CN.TM.SUPERPOINT.BLOCK_DIMS = [64, 64, 128, 128, 256, 256]  # c1, c2, c3, c4, c5, d1
_CN.TM.SUPERPOINT.DET_DIM = 65
_CN.TM.SUPERPOINT.OUT_NUM_POINTS = 128  #  change num=128 in data.py
_CN.TM.SUPERPOINT.PATCH_SIZE = 5
_CN.TM.SUPERPOINT.NMS_DIST = 2
_CN.TM.SUPERPOINT.CONF_THRESH = 0.004
_CN.TM.SUPERPOINT.subpixel_channel = 1


_CN.TM.FINE_CONCAT_COARSE_FEAT = True


# 2. LoFTR-coarse module config


_CN.TM.COARSE = CN()
_CN.TM.COARSE.TWO_STAGE = False
_CN.TM.COARSE.D_MODEL = 256
_CN.TM.COARSE.D_FFN = 256
_CN.TM.COARSE.NHEAD = 8
_CN.TM.RESOLUTION = (8, 2)
if _CN.TM.COARSE.TWO_STAGE:
    _CN.TM.COARSE.LAYER_NAMES = ['self', 'cross'] * 2
else:
    _CN.TM.COARSE.LAYER_NAMES = ['self', 'cross'] * 4
_CN.TM.COARSE.ATTENTION = 'linear'  # options: ['linear', 'full']
_CN.TM.COARSE.POSITION = 'rotary'  # options: ['rotary', 'sin']
_CN.TM.COARSE.TEMP_BUG_FIX = True

# fine_global
_CN.TM.FINE_GLOBAL = CN()
_CN.TM.FINE_GLOBAL.D_MODEL = 256
_CN.TM.FINE_GLOBAL.D_FFN = 256
_CN.TM.FINE_GLOBAL.NHEAD = 8
_CN.TM.FINE_GLOBAL.LAYER_NAMES = ['self', 'cross'] * 2
_CN.TM.FINE_GLOBAL.ATTENTION = 'linear'
_CN.TM.FINE_GLOBAL.POSITION = 'rotary'



# 3. Coarse-Matching config
_CN.TM.MATCH_COARSE = CN()
_CN.TM.MATCH_COARSE.THR = 0.2
_CN.TM.MATCH_COARSE.TRAIN_STAGE = 'whole'  # 'whole' or 'only_coarse'

_CN.TM.MATCH_COARSE.OUT_NUM_POINTS = _CN.TM.SUPERPOINT.OUT_NUM_POINTS
_CN.TM.MATCH_COARSE.BORDER_RM = 2
_CN.TM.MATCH_COARSE.MATCH_TYPE = 'sinkhorn'  # options: ['dual_softmax, 'sinkhorn']
_CN.TM.MATCH_COARSE.DSMAX_TEMPERATURE = 0.1
_CN.TM.MATCH_COARSE.SKH_ITERS = 3
_CN.TM.MATCH_COARSE.SKH_INIT_BIN_SCORE = 1.0
_CN.TM.MATCH_COARSE.SKH_PREFILTER = False
_CN.TM.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.2  # training tricks: save GPU memory
_CN.TM.MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN = 400  # training tricks: avoid DDP deadlock
_CN.TM.MATCH_COARSE.SPARSE_SPVS = True

# 3.1 Fine module config
_CN.TM.FINE = CN()
_CN.TM.STN = False
_CN.TM.FINE.PHOTOMETRIC = True
_CN.TM.FINE.THR = 0.0
_CN.TM.FINE.D_MODEL = 64
_CN.TM.FINE.D_FFN = 64
_CN.TM.FINE.NHEAD = 4
_CN.TM.FINE.LAYER_NAMES = ['self', 'cross'] * 2
_CN.TM.FINE.ATTENTION = 'linear'  # 'full' #TODO:change it to full
_CN.TM.FINE.POSITION = 'rotary'  # options: ['rotary', 'sin']
_CN.TM.FINE.RESOLUTION = _CN.TM.RESOLUTION
_CN.TM.FINE.FINE_CONCAT_COARSE_FEAT = _CN.TM.FINE_CONCAT_COARSE_FEAT
_CN.TM.FINE.FINE_WINDOW_SIZE = 5 # window_size in fine_level, must be odd
_CN.TM.FINE.DSMAX_TEMPERATURE = 0.025
# 3.2 transformation

_CN.TM.TRANS = CN()
_CN.TM.TRANS.SIGMA_D = 0.4 # 0.5
_CN.TM.TRANS.SIGMA_A = 1.0 # 1.5
_CN.TM.TRANS.GAMA = 0.5
_CN.TM.TRANS.ANGEL_K = 3
_CN.TM.TRANS.INLIER_THRESHOLD = 4 # 5

_CN.TM.TRANS.NUM_ITERATIONS = 10



# 4. Losses
# -- # coarse-level
_CN.TM.LOSS = CN()
_CN.TM.LOSS.FINE_TYPE = 'l2_with_std'
_CN.TM.LOSS.COARSE_TYPE = 'focal'  # ['focal', 'cross_entropy']
_CN.TM.LOSS.COARSE_WEIGHT = 1.0
_CN.TM.LOSS.COARSE_WEIGHT = 1.0

# _CN.TM.LOSS.SPARSE_SPVS = False

# -- # fine-level
_CN.TM.LOSS.FINE_TYPE = 'l2_with_std'  # ['l2_with_std', 'l2']
_CN.TM.LOSS.FINE_WEIGHT = 1.0
_CN.TM.LOSS.FINE_CORRECT_THR = 1.0  # for filtering valid fine-leve



# -- - -- # focal loss (coarse)
_CN.TM.LOSS.FOCAL_ALPHA = 0.25
_CN.TM.LOSS.FOCAL_GAMMA = 2.0
_CN.TM.LOSS.POS_WEIGHT = 1.0
_CN.TM.LOSS.NEG_WEIGHT = 1.0

# 5.edge config
_CN.TM.EDGE = CN()
_CN.TM.EDGE.NAME = 'pidinet_tiny_converted'
_CN.TM.EDGE.LOAD = True  # load

##############  Dataset  ##############
_CN.DATASET = CN()
# 1. data config
# training and validating
_CN.DATASET.TRAINVAL_DATA_SOURCE = None  # options: ['ScanNet', 'MegaDepth']
_CN.DATASET.TRAIN_DATA_ROOT = None
_CN.DATASET.TRAIN_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.TRAIN_NPZ_ROOT = None
_CN.DATASET.TRAIN_LIST_PATH = None
_CN.DATASET.TRAIN_INTRINSIC_PATH = None
_CN.DATASET.VAL_DATA_ROOT = None
_CN.DATASET.VAL_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.VAL_NPZ_ROOT = None
_CN.DATASET.VAL_LIST_PATH = None    # None if val data from all scenes are bundled into a single npz file
_CN.DATASET.VAL_INTRINSIC_PATH = None
_CN.DATASET.IMG_RESIZE = 256

# testing
_CN.DATASET.TEST_DATA_SOURCE = None
_CN.DATASET.TEST_DATA_ROOT = None
_CN.DATASET.TEST_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.TEST_NPZ_ROOT = None
_CN.DATASET.TEST_LIST_PATH = None   # None if test data from all scenes are bundled into a single npz file
_CN.DATASET.TEST_INTRINSIC_PATH = None

# 2. dataset config
_CN.DATASET.AUGMENTATION_TYPE = None  # options: [None, 'dark', 'mobile']


##############  Trainer  ##############
_CN.TRAINER = CN()
if _CN.TM.MATCH_COARSE.TRAIN_STAGE =='only_coarse':
    _CN.TRAINER.WORLD_SIZE = 1
    _CN.TRAINER.CANONICAL_BS = 64 #
    _CN.TRAINER.CANONICAL_LR = 6e-3
    _CN.TRAINER.SCALING = None  # this will be calculated automatically
    _CN.TRAINER.FIND_LR = False  # use learning rate finder from pytorch-lightning

    # optimizer
    _CN.TRAINER.OPTIMIZER = "adamw"  # [adam, adamw]
    _CN.TRAINER.TRUE_LR = None  # this will be calculated automatically at runtime
    _CN.TRAINER.ADAM_DECAY = 0.  # ADAM: for adam
    _CN.TRAINER.ADAMW_DECAY = 0.1

    # step-based warm-up
    _CN.TRAINER.WARMUP_TYPE = 'linear'  # [linear, constant]
    _CN.TRAINER.WARMUP_RATIO = 0.
    _CN.TRAINER.WARMUP_STEP = 4800
else:
    _CN.TRAINER.WORLD_SIZE = 1
    _CN.TRAINER.CANONICAL_BS = 64  #
    _CN.TRAINER.CANONICAL_LR = 6e-4
    _CN.TRAINER.SCALING = None  # this will be calculated automatically
    _CN.TRAINER.FIND_LR = False  # use learning rate finder from pytorch-lightning
    # optimizer
    _CN.TRAINER.OPTIMIZER = "adamw"  # [adam, adamw]
    _CN.TRAINER.TRUE_LR = None  # this will be calculated automatically at runtime
    _CN.TRAINER.ADAM_DECAY = 0.  # ADAM: for adam
    _CN.TRAINER.ADAMW_DECAY = 0.1
    # step-based warm-up
    _CN.TRAINER.WARMUP_TYPE = 'linear'  # [linear, constant]
    _CN.TRAINER.WARMUP_RATIO = 0.
    _CN.TRAINER.WARMUP_STEP = 4800

# learning rate scheduler
_CN.TRAINER.SCHEDULER = 'MultiStepLR'  # [MultiStepLR, CosineAnnealing, ExponentialLR]
_CN.TRAINER.SCHEDULER_INTERVAL = 'epoch'    # [epoch, step]
_CN.TRAINER.MSLR_MILESTONES = [3, 6, 9, 12]  # MSLR: MultiStepLR
_CN.TRAINER.MSLR_GAMMA = 0.5
_CN.TRAINER.COSA_TMAX = 30  # COSA: CosineAnnealing
_CN.TRAINER.ELR_GAMMA = 0.999992  # ELR: ExponentialLR, this value for 'step' interval

# plotting related
_CN.TRAINER.ENABLE_PLOTTING = True
_CN.TRAINER.N_VAL_PAIRS_TO_PLOT = 12     # number of val/test paris for plotting
_CN.TRAINER.PLOT_MODE = 'evaluation'  # ['evaluation', 'confidence']
_CN.TRAINER.PLOT_MATCHES_ALPHA = 'dynamic'

# geometric metrics and pose solver
_CN.TRAINER.DIS_ERR_THR = 3
_CN.TRAINER.EPI_ERR_THR = 5e-4  # recommendation: 5e-4 for ScanNet, 1e-4 for MegaDepth (from SuperGlue)
_CN.TRAINER.POSE_GEO_MODEL = 'E'  # ['E', 'F', 'H']
_CN.TRAINER.POSE_ESTIMATION_METHOD = 'RANSAC'  # [RANSAC, DEGENSAC, MAGSAC]
_CN.TRAINER.RANSAC_PIXEL_THR = 0.5
_CN.TRAINER.RANSAC_CONF = 0.99999
_CN.TRAINER.RANSAC_MAX_ITERS = 10000
_CN.TRAINER.USE_MAGSACPP = False

# data sampler for train_dataloader
_CN.TRAINER.DATA_SAMPLER = 'scene_balance'  # options: ['scene_balance', 'random', 'normal']
# 'scene_balance' config
_CN.TRAINER.N_SAMPLES_PER_SUBSET = 5000# for hole dataset
_CN.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT = True  # whether sample each scene with replacement or not
_CN.TRAINER.SB_SUBSET_SHUFFLE = True  # after sampling from scenes, whether shuffle within the epoch or not
_CN.TRAINER.SB_REPEAT = 1  # repeat N times for training the sampled data
# 'random' config
_CN.TRAINER.RDM_REPLACEMENT = True
_CN.TRAINER.RDM_NUM_SAMPLES = None

# gradient clipping
_CN.TRAINER.GRADIENT_CLIPPING = 0.5

# reproducibility
# This seed affects the data sampling. With the same seed, the data sampling is promised
# to be the same. When resume training from a checkpoint, it's better to use a different
# seed, otherwise the sampled data will be exactly the same as before resuming, which will
# cause less unique data items sampled during the entire training.
# Use of different seed values might affect the final training result, since not all data items
# are used during training on ScanNet. (60M pairs of images sampled during traing from 230M pairs in total.)
_CN.TRAINER.SEED = 68


def get_cfg_defaults_application():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _CN.clone()
