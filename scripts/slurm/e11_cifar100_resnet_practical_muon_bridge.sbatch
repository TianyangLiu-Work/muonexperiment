#!/usr/bin/env bash
#SBATCH --job-name=e11-resnet-muonbridge
#SBATCH --partition=gpu
#SBATCH --gres=gpushare:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

cd /home/tyliu/ghworkspace/muonexperiment
mkdir -p outputs/slurm /tmp/matplotlib-e11-resnet-muonbridge

export MPLCONFIGDIR=/tmp/matplotlib-e11-resnet-muonbridge
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

/data/conda_envs/SpatialQuantization/bin/python scripts/e11_run_cifar100_resnet_practical_muon_bridge.py \
  --device cuda \
  --seeds 10 \
  --warmup-steps 5000 \
  --trajectory-steps 3 \
  --head-train-per-class 300 \
  --tail-train-per-class 300 \
  --target-head-gain-fraction 0.005 \
  --adamw-trajectory-lr 3e-4 \
  --muon-trajectory-lr 1e-4 \
  --newton-schulz-steps 5 \
  --output-dir results/e11_cifar100_resnet_practical_muon_bridge \
  --figure-dir figures/e11_cifar100_resnet_practical_muon_bridge \
  --discussion-path discussion/e11_cifar100_resnet_practical_muon_bridge.md \
  --no-download \
  --progress
