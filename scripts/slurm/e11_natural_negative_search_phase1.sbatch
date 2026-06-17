#!/usr/bin/env bash
#SBATCH --job-name=e11-nns-p1
#SBATCH --partition=gpu
#SBATCH --gres=gpushare:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --array=0-2
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err

set -euo pipefail

cd /home/tyliu/ghworkspace/muonexperiment
mkdir -p outputs/slurm /tmp/matplotlib-e11-natural-negative-phase1

export MPLCONFIGDIR=/tmp/matplotlib-e11-natural-negative-phase1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

SEARCH_IDS=(
  "NNS-P1-cifar100lt-resnet18-new-partitions"
  "NNS-P1-cifar10lt-resnet18-cross-partitions"
  "NNS-P1-tail-quality-controls"
)
ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
SEARCH_ID="${NNS_SEARCH_ID:-${SEARCH_IDS[$ARRAY_INDEX]}}"
SEEDS="${NNS_SEEDS:-5}"
EXTRA_ARGS=()
if [[ -n "${NNS_MAX_SETTINGS:-}" ]]; then
  EXTRA_ARGS+=(--max-settings "$NNS_MAX_SETTINGS")
fi

/data/conda_envs/SpatialQuantization/bin/python scripts/e11_run_natural_negative_search_phase1.py \
  --search-id "$SEARCH_ID" \
  --device cuda \
  --seeds "$SEEDS" \
  --no-download \
  --progress \
  "${EXTRA_ARGS[@]}"
