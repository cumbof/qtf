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

# Make `conda activate` work inside a non-interactive SLURM shell.
for candidate in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [ -f "$candidate" ]; then
        # shellcheck source=/dev/null
        source "$candidate"
        break
    fi
done
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH; cannot activate QTF env." >&2
    exit 1
fi
conda activate QTF

mkdir -p logs results/rosetta

python runner.py \
    --replica_id      $SLURM_ARRAY_TASK_ID \
    --energy_backend  rosetta \
    --max_iter        500 \
    --scout           50 \
    --strategy        random \
    --top_k           5 \
    --outdir          results
