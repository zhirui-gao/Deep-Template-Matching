from src.config.default import _CN as cfg
TRAIN_BASE_PATH = "/home/gzr/Data/linemod_2d"

cfg.DATASET.TRAINVAL_DATA_SOURCE = "linemod_2d"
cfg.DATASET.TRAIN_DATA_ROOT = f"{TRAIN_BASE_PATH}/DATA"
cfg.DATASET.TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/DATA"
cfg.DATASET.TRAIN_LIST_PATH = f"{TRAIN_BASE_PATH}/train_list/data_list.txt"



cfg.DATASET.TEST_DATA_SOURCE = "linemod_2d"
cfg.DATASET.VAL_DATA_ROOT = f"{TRAIN_BASE_PATH}/DATA"
cfg.DATASET.VAL_NPZ_ROOT = f"{TRAIN_BASE_PATH}/DATA"
cfg.DATASET.VAL_LIST_PATH = f"{TRAIN_BASE_PATH}/val_list/data_list.txt"