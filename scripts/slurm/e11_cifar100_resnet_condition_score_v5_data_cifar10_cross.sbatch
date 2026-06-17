#!/usr/bin/env bash
#SBATCH --job-name=e11-condscore-v5-c10x
#SBATCH --partition=gpu
#SBATCH --gres=gpushare:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

cd /home/tyliu/ghworkspace/muonexperiment
mkdir -p outputs/slurm /tmp/matplotlib-e11-condscore-v5-c10x

export MPLCONFIGDIR=/tmp/matplotlib-e11-condscore-v5-c10x
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

/data/conda_envs/SpatialQuantization/bin/python scripts/e11_run_cifar100_resnet_layer_jvp_checkpoint_prediction.py \
  --device cuda \
  --dataset-name CIFAR10 \
  --model-arch resnet18 \
  --head-classes 0,3,4,6,9 \
  --tail-classes 1,2,5,7,8 \
  --seeds 10 \
  --warmup-steps 2000,5000,10000 \
  --head-train-per-class 500 \
  --tail-train-per-class 500 \
  --tail-eval-per-class 200 \
  --target-head-gain-fraction 0.005 \
  --jvp-epsilon 1e-4 \
  --output-dir results/e11_condition_score_v5_protocol/final_data_cifar10_cross \
  --figure-dir figures/e11_condition_score_v5_protocol/final_data_cifar10_cross \
  --discussion-path discussion/e11_condition_score_v5_data_cifar10_cross.md \
  --download \
  --progress
