#!/usr/bin/env bash
#SBATCH --job-name=e11-resnet-fc-cond
#SBATCH --partition=gpu
#SBATCH --gres=gpushare:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

cd /home/tyliu/ghworkspace/muonexperiment
mkdir -p outputs/slurm /tmp/matplotlib-e11-resnet-fc-cond

export MPLCONFIGDIR=/tmp/matplotlib-e11-resnet-fc-cond
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

/data/conda_envs/SpatialQuantization/bin/python scripts/e11_run_cifar100_resnet_fc_condition_scatter.py \
  --device cuda \
  --seeds 10 \
  --warmup-steps 250,500,1000,2000 \
  --target-head-gain-fraction 0.005 \
  --no-download \
  --progress
