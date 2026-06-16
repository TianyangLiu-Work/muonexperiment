#!/usr/bin/env bash
#SBATCH --job-name=e11-resnet-rho002
#SBATCH --partition=gpu
#SBATCH --gres=gpushare:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

cd /home/tyliu/ghworkspace/muonexperiment
mkdir -p outputs/slurm /tmp/matplotlib-e11-resnet-rho002

export MPLCONFIGDIR=/tmp/matplotlib-e11-resnet-rho002
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

/data/conda_envs/SpatialQuantization/bin/python scripts/e11_run_cifar100_resnet_one_step.py \
  --device cuda \
  --seeds 10 \
  --target-head-gain-fraction 0.002 \
  --output-dir results/e11_cifar100_resnet_one_step_rho002 \
  --figure-dir figures/e11_cifar100_resnet_one_step_rho002 \
  --discussion-path discussion/e11_cifar100_resnet_one_step_rho002.md \
  --no-download \
  --progress
