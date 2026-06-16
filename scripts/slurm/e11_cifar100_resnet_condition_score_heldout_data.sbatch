#!/usr/bin/env bash
#SBATCH --job-name=e11-condscore-data
#SBATCH --partition=gpu
#SBATCH --gres=gpushare:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

cd /home/tyliu/ghworkspace/muonexperiment
mkdir -p outputs/slurm /tmp/matplotlib-e11-condscore-data

export MPLCONFIGDIR=/tmp/matplotlib-e11-condscore-data
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

/data/conda_envs/SpatialQuantization/bin/python scripts/e11_run_cifar100_resnet_layer_jvp_checkpoint_prediction.py \
  --device cuda \
  --dataset-name CIFAR10 \
  --model-arch resnet18 \
  --head-classes 0,1,2,3,4 \
  --tail-classes 5,6,7,8,9 \
  --seeds 10 \
  --warmup-steps 2000,5000,10000 \
  --head-train-per-class 500 \
  --tail-train-per-class 500 \
  --tail-eval-per-class 200 \
  --target-head-gain-fraction 0.005 \
  --jvp-epsilon 1e-4 \
  --output-dir results/e11_cifar100_resnet_condition_score_next/heldout_data \
  --figure-dir figures/e11_cifar100_resnet_condition_score_next/heldout_data \
  --discussion-path discussion/e11_cifar100_resnet_condition_score_next_heldout_data.md \
  --download \
  --progress
