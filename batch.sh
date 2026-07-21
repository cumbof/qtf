#!/bin/bash
#SBATCH --array=0-399%50
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=qtf_custom
#SBATCH --output=logs/qtf_custom_%A_%a.out
#SBATCH --error=logs/qtf_custom_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=defq
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=raubenb@ccf.org

# NOTE: SLURM opens the --output / --error files BEFORE this script runs,
# so the `logs/` directory MUST exist at submission time.  Run once before
# `sbatch`:    mkdir -p logs
#
# This script launches 400 qtf.cli.fold invocations as a SLURM job array
# (at most 50 concurrent). Each array task uses ensemble_size=1 / top_k=1
# and writes to its own output subdirectory (task_<SLURM_ARRAY_TASK_ID>)
# so the tasks do not clobber each other. The SLURM array task ID is also
# passed as --seed_offset, so each single-replica job gets a distinct
# deterministic scouting seed.

set -euo pipefail

#source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate qtf


# Target sequence / reference
# SEQUENCE="YYDPETGTWY"          # chignolin, REFERENCE_PDB_ID="5AWL"
SEQUENCE="DAYAQWLKDGGPSSGRPPPS"
REFERENCE_PDB_ID="references/2JOF.pdb"
ENERGY_BACKEND="custom"

# Set outdirs
OUTPUT_BASE="run_outputs/quantum_simulations/$REFERENCE_PDB_ID/$ENERGY_BACKEND"
mkdir -p logs "$OUTPUT_BASE"

srun python -m qtf.cli.fold \
    --predict             "$SEQUENCE" \
    --reference_pdb       "$REFERENCE_PDB_ID" \
    --mode                predict_and_compare \
    --ensemble_size       1 \
    --seed_offset         "$SLURM_ARRAY_TASK_ID" \
    --maxiter             2000 \
    --energy_backend      "$ENERGY_BACKEND" \
    --gromacs_minimize    1 \
    --gromacs_rerank      1 \
    --top_k               1 \
    --top_k_snapshots     2000  \
    --snapshot_energy_gap 0.1 \
    --snapshot_sort_by    rmsd \
    --output_root         "$OUTPUT_BASE/task_${SLURM_ARRAY_TASK_ID}"
