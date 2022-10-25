from src.config.default import _CN as cfg
# TRAIN_BASE_PATH = "/home/gzr/Data/synthetic_dataset_transfer" #/home/gzr/Data/generative_steel/process/data/final_train"
TRAIN_BASE_PATH = '/home/gzr/Data/zwb_data/hole_wb'#"/home/gzr/Data/generative_steel/process/data/final_train"
# TRAIN_BASE_PATH = '/home/gzr/下载/chorme_download/lm_train_pbr/train_pbr_select'#"/home/gzr/Data/generative_steel/process/data/final_train"

cfg.DATASET.TRAINVAL_DATA_SOURCE = "synthetic"
cfg.DATASET.TRAIN_DATA_ROOT = f"{TRAIN_BASE_PATH}"
cfg.DATASET.TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}"
cfg.DATASET.TRAIN_LIST_PATH = f"{TRAIN_BASE_PATH}/train_list/data_list.txt"



cfg.DATASET.TEST_DATA_SOURCE = "synthetic"
cfg.DATASET.VAL_DATA_ROOT = f"{TRAIN_BASE_PATH}"
cfg.DATASET.VAL_NPZ_ROOT = f"{TRAIN_BASE_PATH}"
cfg.DATASET.VAL_LIST_PATH = f"{TRAIN_BASE_PATH}/val_list/data_list.txt"