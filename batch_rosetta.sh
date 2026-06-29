#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=qtf_rosetta
#SBATCH --array=0-2
#SBATCH --output=logs/qtf_rosetta_%A_%a.out
#SBATCH --error=logs/qtf_rosetta_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=defq
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=puramv@ccf.org

# NOTE: SLURM opens the --output / --error files BEFORE this script runs,
# so the `logs/` directory MUST exist at submission time.  Run once before
# `sbatch`:    mkdir -p logs

set -euo pipefail

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate QTF

mkdir -p logs results/rosetta

python runner.py \
    --replica_id      $SLURM_ARRAY_TASK_ID \
    --energy_backend  rosetta \
    --max_iter        10 \
    --scout           50 \
    --strategy        random \
    --top_k           5 \
    --outdir          results
