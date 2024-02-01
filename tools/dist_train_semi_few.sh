#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
MODEL=$3
PERCENT=$4
SEED=$5
SHOT=$6
FOLD=$7
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch \
    --cfg-options model_type=${MODEL} percent=${PERCENT} seed=${SEED} shot=${SHOT} fold=${FOLD} ${@:8}
