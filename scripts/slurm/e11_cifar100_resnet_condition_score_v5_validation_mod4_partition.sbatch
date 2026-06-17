#!/usr/bin/env bash
#SBATCH --job-name=e11-condscore-v5-val
#SBATCH --partition=gpu
#SBATCH --gres=gpushare:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

cd /home/tyliu/ghworkspace/muonexperiment
mkdir -p outputs/slurm /tmp/matplotlib-e11-condscore-v5-val

export MPLCONFIGDIR=/tmp/matplotlib-e11-condscore-v5-val
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

/data/conda_envs/SpatialQuantization/bin/python scripts/e11_run_cifar100_resnet_layer_jvp_checkpoint_prediction.py \
  --device cuda \
  --dataset-name CIFAR100 \
  --model-arch resnet18 \
  --head-classes 0,1,4,5,8,9,12,13,16,17,20,21,24,25,28,29,32,33,36,37,40,41,44,45,48,49,52,53,56,57,60,61,64,65,68,69,72,73,76,77,80,81,84,85,88,89,92,93,96,97 \
  --tail-classes 2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31,34,35,38,39,42,43,46,47,50,51,54,55,58,59,62,63,66,67,70,71,74,75,78,79,82,83,86,87,90,91,94,95,98,99 \
  --seeds 10 \
  --warmup-steps 2000,5000,10000 \
  --head-train-per-class 300 \
  --tail-train-per-class 300 \
  --target-head-gain-fraction 0.005 \
  --jvp-epsilon 1e-4 \
  --output-dir results/e11_condition_score_v5_protocol/validation_cifar100_mod4_partition \
  --figure-dir figures/e11_condition_score_v5_protocol/validation_cifar100_mod4_partition \
  --discussion-path discussion/e11_condition_score_v5_validation_cifar100_mod4_partition.md \
  --no-download \
  --progress
