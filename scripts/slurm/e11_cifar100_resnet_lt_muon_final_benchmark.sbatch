#!/usr/bin/env bash
#SBATCH --job-name=e11-resnet-ltmuon
#SBATCH --partition=gpu
#SBATCH --gres=gpushare:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

cd /home/tyliu/ghworkspace/muonexperiment
mkdir -p outputs/slurm /tmp/matplotlib-e11-resnet-ltmuon

export MPLCONFIGDIR=/tmp/matplotlib-e11-resnet-ltmuon
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

/data/conda_envs/SpatialQuantization/bin/python scripts/e11_run_cifar100_resnet_lt_recipe_benchmark.py \
  --device cuda \
  --seeds 3 \
  --recipes adamw_aug_ce ns_muon_aug_lr3e-5 ns_muon_aug_lr1e-4 \
  --num-classes 100 \
  --max-train-count 500 \
  --imbalance-factor 100 \
  --many-threshold 100 \
  --medium-threshold 20 \
  --train-steps 5000 \
  --train-batch-size 256 \
  --eval-interval 1000 \
  --output-dir results/e11_cifar100_resnet_lt_muon_final_benchmark \
  --figure-dir figures/e11_cifar100_resnet_lt_muon_final_benchmark \
  --discussion-path discussion/e11_cifar100_resnet_lt_muon_final_benchmark.md \
  --no-download \
  --progress
