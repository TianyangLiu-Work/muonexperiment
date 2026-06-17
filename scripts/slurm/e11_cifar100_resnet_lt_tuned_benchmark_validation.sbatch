#!/usr/bin/env bash
#SBATCH --job-name=e11-lt-tuned-val
#SBATCH --partition=gpu
#SBATCH --gres=gpushare:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-163%1
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err

set -euo pipefail

cd /home/tyliu/ghworkspace/muonexperiment
mkdir -p outputs/slurm /tmp/matplotlib-e11-lt-tuned-benchmark

export MPLCONFIGDIR=/tmp/matplotlib-e11-lt-tuned-benchmark
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
EXTRA_ARGS=()
if [[ -n "${E11_TUNED_MAX_SETTINGS:-}" ]]; then
  EXTRA_ARGS+=(--max-settings "$E11_TUNED_MAX_SETTINGS")
fi

/data/conda_envs/SpatialQuantization/bin/python scripts/e11_run_cifar100_resnet_lt_tuned_benchmark.py \
  --array-index "$ARRAY_INDEX" \
  --device cuda \
  --no-download \
  --progress \
  "${EXTRA_ARGS[@]}"
