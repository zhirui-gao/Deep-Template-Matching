from src.config.default import _CN as cfg
TEST_BASE_PATH = "/home/gzr/Data/linemod_2d"

cfg.DATASET.TEST_DATA_SOURCE = "linemod_2d"
cfg.DATASET.TEST_DATA_ROOT = f"{TEST_BASE_PATH}/DATA"
cfg.DATASET.TEST_NPZ_ROOT = f"{TEST_BASE_PATH}/DATA"
cfg.DATASET.TEST_LIST_PATH = f"{TEST_BASE_PATH}/test_list/data_list.txt"
