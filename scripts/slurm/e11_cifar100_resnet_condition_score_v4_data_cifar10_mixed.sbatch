#!/usr/bin/env bash
#SBATCH --job-name=e11-condscore-v4-c10mix
#SBATCH --partition=gpu
#SBATCH --gres=gpushare:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

cd /home/tyliu/ghworkspace/muonexperiment
mkdir -p outputs/slurm /tmp/matplotlib-e11-condscore-v4-c10mix

export MPLCONFIGDIR=/tmp/matplotlib-e11-condscore-v4-c10mix
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

/data/conda_envs/SpatialQuantization/bin/python scripts/e11_run_cifar100_resnet_layer_jvp_checkpoint_prediction.py \
  --device cuda \
  --dataset-name CIFAR10 \
  --model-arch resnet18 \
  --head-classes 0,1,4,7,8 \
  --tail-classes 2,3,5,6,9 \
  --seeds 10 \
  --warmup-steps 2000,5000,10000 \
  --head-train-per-class 500 \
  --tail-train-per-class 500 \
  --tail-eval-per-class 200 \
  --target-head-gain-fraction 0.005 \
  --jvp-epsilon 1e-4 \
  --output-dir results/e11_condition_score_v4_protocol/final_data_cifar10_mixed \
  --figure-dir figures/e11_condition_score_v4_protocol/final_data_cifar10_mixed \
  --discussion-path discussion/e11_condition_score_v4_data_cifar10_mixed.md \
  --download \
  --progress
