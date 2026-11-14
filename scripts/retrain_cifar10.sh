#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${GENOTYPE:?Set GENOTYPE to the architecture path relative to exps/}"
GPU="${GPU:-0}"
DATA_PATH="${DATA_PATH:-./data}"
# This preserved entry runs max_epoch_G*n_critic outer data epochs: 120*5=600.
CUDA_VISIBLE_DEVICES="$GPU" python -u fully_train_arch.py --gpu_ids 0 --dataset cifar10 --data_path "$DATA_PATH" --genotype_of_G "$GENOTYPE" --genotypes_exp cifar10_D.npy --bottom_width 4 --img_size 32 --max_epoch_G 120 --n_critic 5 --gen_bs 256 --dis_bs 128 --latent_dim 120 --gf_dim 256 --df_dim 128 --g_lr 0.0002 --d_lr 0.0002 --beta1 0.0 --beta2 0.9 --num_eval_imgs 50000 --eval_batch_size 100 --random_seed 12345 --cr 1 --exp_name treeevo_cifar10_retrain
