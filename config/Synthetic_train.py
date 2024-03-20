from src.config.default import _CN as cfg

# dataset path
TRAIN_BASE_PATH = './data/train_data'
cfg.DATASET.TRAINVAL_DATA_SOURCE = "synthetic"
cfg.DATASET.TRAIN_DATA_ROOT = f"{TRAIN_BASE_PATH}"
cfg.DATASET.TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}"
cfg.DATASET.TRAIN_LIST_PATH = f"{TRAIN_BASE_PATH}/train_list/data_list.txt"
cfg.DATASET.TEST_DATA_SOURCE = "synthetic"
cfg.DATASET.VAL_DATA_ROOT = f"{TRAIN_BASE_PATH}"
cfg.DATASET.VAL_NPZ_ROOT = f"{TRAIN_BASE_PATH}"
cfg.DATASET.VAL_LIST_PATH = f"{TRAIN_BASE_PATH}/val_list/data_list.txt"
