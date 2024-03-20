from src.config.default import _CN as cfg

TEST_BASE_PATH = './data/train_data'
cfg.DATASET.TEST_DATA_SOURCE = "synthetic"
cfg.DATASET.TEST_DATA_ROOT = f"{TEST_BASE_PATH}"
cfg.DATASET.TEST_NPZ_ROOT = f"{TEST_BASE_PATH}"
cfg.DATASET.TEST_LIST_PATH = f"{TEST_BASE_PATH}/test_list/data_list.txt"
