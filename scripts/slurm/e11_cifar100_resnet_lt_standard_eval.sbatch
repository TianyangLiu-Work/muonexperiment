#!/usr/bin/env bash
#SBATCH --job-name=e11-resnet-ltstd
#SBATCH --partition=gpu
#SBATCH --gres=gpushare:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

cd /home/tyliu/ghworkspace/muonexperiment
mkdir -p outputs/slurm /tmp/matplotlib-e11-resnet-ltstd

export MPLCONFIGDIR=/tmp/matplotlib-e11-resnet-ltstd
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

/data/conda_envs/SpatialQuantization/bin/python scripts/e11_run_cifar100_resnet_lt_standard_eval.py \
  --device cuda \
  --seeds 10 \
  --num-classes 100 \
  --max-train-count 500 \
  --imbalance-factor 100 \
  --many-threshold 100 \
  --medium-threshold 20 \
  --train-steps 10000 \
  --train-batch-size 256 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --eval-interval 1000 \
  --output-dir results/e11_cifar100_resnet_lt_standard_eval \
  --figure-dir figures/e11_cifar100_resnet_lt_standard_eval \
  --discussion-path discussion/e11_cifar100_resnet_lt_standard_eval.md \
  --no-download \
  --progress
