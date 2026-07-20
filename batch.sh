#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=qtf_custom
#SBATCH --output=logs/qtf_custom_%j.out
#SBATCH --error=logs/qtf_custom_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --partition=defq
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=puramv@ccf.org

# NOTE: SLURM opens the --output / --error files BEFORE this script runs,
# so the `logs/` directory MUST exist at submission time.  Run once before
# `sbatch`:    mkdir -p logs

set -euo pipefail

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate QTF

mkdir -p logs run_outputs/quantum_simulations

# Target sequence / reference (chignolin, matches previous runner.py defaults)
SEQUENCE="YYDPETGTWY"
REFERENCE_PDB_ID="5AWL"

qtf-fold \
    --predict             "$SEQUENCE" \
    --reference_structure "$REFERENCE_PDB_ID" \
    --energy_backend      custom \
    --maxiter             2000 \
    --top_k               1000 \
    --ensemble_size       400 \
    --output_root         run_outputs/quantum_simulations/custom
