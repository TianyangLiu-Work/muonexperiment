#!/usr/bin/env bash
#SBATCH --job-name=e11-condscore-arch
#SBATCH --partition=gpu
#SBATCH --gres=gpushare:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

cd /home/tyliu/ghworkspace/muonexperiment
mkdir -p outputs/slurm /tmp/matplotlib-e11-condscore-arch

export MPLCONFIGDIR=/tmp/matplotlib-e11-condscore-arch
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

/data/conda_envs/SpatialQuantization/bin/python scripts/e11_run_cifar100_resnet_layer_jvp_checkpoint_prediction.py \
  --device cuda \
  --dataset-name CIFAR100 \
  --model-arch resnet34 \
  --seeds 5 \
  --warmup-steps 2000,5000,10000 \
  --head-train-per-class 300 \
  --tail-train-per-class 300 \
  --target-head-gain-fraction 0.005 \
  --jvp-epsilon 1e-4 \
  --output-dir results/e11_cifar100_resnet_condition_score_next/heldout_architecture \
  --figure-dir figures/e11_cifar100_resnet_condition_score_next/heldout_architecture \
  --discussion-path discussion/e11_cifar100_resnet_condition_score_next_heldout_architecture.md \
  --no-download \
  --progress
