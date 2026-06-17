#!/usr/bin/env bash
#SBATCH --job-name=e11-nns-p2-arch
#SBATCH --partition=gpu
#SBATCH --gres=gpushare:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

cd /home/tyliu/ghworkspace/muonexperiment
mkdir -p outputs/slurm /tmp/matplotlib-e11-natural-negative-phase2

export MPLCONFIGDIR=/tmp/matplotlib-e11-natural-negative-phase2
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

SEEDS="${NNS_PHASE2_SEEDS:-0,1,2}"
EXTRA_ARGS=()
if [[ -n "${NNS_PHASE2_MAX_SETTINGS:-}" ]]; then
  EXTRA_ARGS+=(--max-settings "$NNS_PHASE2_MAX_SETTINGS")
fi

/data/conda_envs/SpatialQuantization/bin/python scripts/e11_run_natural_negative_search_phase2.py \
  --device cuda \
  --seeds "$SEEDS" \
  --no-download \
  --progress \
  "${EXTRA_ARGS[@]}"
