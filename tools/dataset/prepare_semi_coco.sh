#!/usr/bin/env bash
# Example Usage:
# bash tools/dataset/prepare_semi_coco.sh data/coco/
set -x
OFFSET=0
for percent in 1 5 10; do
    for fold in 1 2 3 4 5; do
        python $(dirname "$0")/semi_coco.py \
        --percent ${percent} --seed ${fold} \
        --seed-offset ${OFFSET} --data-dir $1
    done
done
