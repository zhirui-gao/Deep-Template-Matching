#!/bin/bash -l
SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../"
# conda activate loftr
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

n_nodes=1
n_gpus_per_node=1
torch_num_workers=4
batch_size=4
pin_memory=true


exp_name="linemod2d-${TRAIN_IMG_SIZE}-bs=$(($n_gpus_per_node * $n_nodes * $batch_size))"
python -u ./train.py \
    --exp_name=${exp_name} \
    --gpus=${n_gpus_per_node}\
    --accelerator="ddp"\
    --num_nodes=${n_nodes}\
    --batch_size=${batch_size} --num_workers=${torch_num_workers} --pin_memory=${pin_memory} \
    --check_val_every_n_epoch=1 \
    --log_every_n_steps=200 \
    --flush_logs_every_n_steps=200 \
    --limit_val_batches=1. \
    --num_sanity_val_steps=10 \
    --benchmark=True \
    --max_epochs=30 \
#    --parallel_load_data
