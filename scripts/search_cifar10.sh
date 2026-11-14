#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
GPU="${GPU:-0}"
DATA_PATH="${DATA_PATH:-./data}"
# Set the visible physical GPU before torch is imported; the entry uses logical 0.
CUDA_VISIBLE_DEVICES="$GPU" python -u search_gen_arch.py --gpu_ids 0 --dataset cifar10 --data_path "$DATA_PATH" --bottom_width 4 --img_size 32 --my_max_epoch_G 200 --warmup 40 --num_individual 32 --num_selected 12 --Total_evolutionary_algebra 30 --num_eval_imgs 5000 --eval_batch_size 100 --max_model_size 15 --random_seed 12345 --final_evo_search
