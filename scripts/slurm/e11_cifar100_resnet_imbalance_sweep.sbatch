#!/usr/bin/env bash
#SBATCH --job-name=e11-resnet-imb
#SBATCH --partition=gpu
#SBATCH --gres=gpushare:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

cd /home/tyliu/ghworkspace/muonexperiment
mkdir -p outputs/slurm /tmp/matplotlib-e11-resnet-imbalance

export MPLCONFIGDIR=/tmp/matplotlib-e11-resnet-imbalance
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

/data/conda_envs/SpatialQuantization/bin/python scripts/e11_run_cifar100_resnet_imbalance_sweep.py \
  --device cuda \
  --seeds 3 \
  --tail-train-per-class 10,30,100,300 \
  --head-train-per-class 300 \
  --tail-eval-per-class 40 \
  --warmup-steps 1000 \
  --target-head-gain-fraction 0.005 \
  --no-download \
  --progress
