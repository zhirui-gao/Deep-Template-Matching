from src.config.default import _CN as cfg
# TEST_BASE_PATH = "/home/gzr/Data/hole_data/hole_dataset_new"#  "/home/gzr/Data/generative_steel/process/data/final_train"
# TEST_BASE_PATH = '/home/gzr/Data/generative_steel/steel_dataset_now' # #
TEST_BASE_PATH = '/home/gzr/Data/zwb_data/hole_wb'
# TEST_BASE_PATH = '/home/gzr/Data/generative_steel/real_dataset/0307-0313/process'

cfg.DATASET.TEST_DATA_SOURCE = "synthetic"
cfg.DATASET.TEST_DATA_ROOT = f"{TEST_BASE_PATH}"
cfg.DATASET.TEST_NPZ_ROOT = f"{TEST_BASE_PATH}"
cfg.DATASET.TEST_LIST_PATH = f"{TEST_BASE_PATH}/test_list/data_list.txt"


