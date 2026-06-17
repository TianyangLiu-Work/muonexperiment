#!/usr/bin/env bash
#SBATCH --job-name=e11-condscore-r50
#SBATCH --partition=gpu
#SBATCH --gres=gpushare:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

cd /home/tyliu/ghworkspace/muonexperiment
mkdir -p outputs/slurm /tmp/matplotlib-e11-condscore-r50

export MPLCONFIGDIR=/tmp/matplotlib-e11-condscore-r50
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

/data/conda_envs/SpatialQuantization/bin/python scripts/e11_run_cifar100_resnet_layer_jvp_checkpoint_prediction.py \
  --device cuda \
  --dataset-name CIFAR100 \
  --model-arch resnet50 \
  --seeds 3 \
  --warmup-steps 2000,5000,10000 \
  --head-train-per-class 300 \
  --tail-train-per-class 300 \
  --target-head-gain-fraction 0.005 \
  --jvp-epsilon 1e-4 \
  --output-dir results/e11_condition_score_fresh_protocol/fresh_architecture_resnet50 \
  --figure-dir figures/e11_condition_score_fresh_protocol/fresh_architecture_resnet50 \
  --discussion-path discussion/e11_condition_score_fresh_architecture_resnet50.md \
  --no-download \
  --progress
