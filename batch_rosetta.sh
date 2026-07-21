#!/bin/bash
#SBATCH --ntasks=400
#SBATCH --job-name=qtf_rosetta
#SBATCH --output=logs/qtf_rosetta_%j.out
#SBATCH --error=logs/qtf_rosetta_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=1
#SBATCH --partition=defq
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=puramv@ccf.org

# NOTE: SLURM opens the --output / --error files BEFORE this script runs,
# so the `logs/` directory MUST exist at submission time.  Run once before
# `sbatch`:    mkdir -p logs
#
# This script launches 400 parallel qtf.cli.fold invocations via `srun`.
# Each task uses ensemble_size=1 / top_k=1 and writes to its own output
# subdirectory (task_<SLURM_PROCID>) so the tasks do not clobber each other.

set -euo pipefail

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate QTF

OUTPUT_BASE="run_outputs/quantum_simulations/rosetta"
mkdir -p logs "$OUTPUT_BASE"

# Target sequence / reference (chignolin, matches previous runner.py defaults)
SEQUENCE="YYDPETGTWY"
REFERENCE_PDB_ID="5AWL"

srun --ntasks=400 --cpus-per-task=1 bash -c '
    python -m qtf.cli.fold \
        --predict             "'"$SEQUENCE"'" \
        --reference_pdb       "'"$REFERENCE_PDB_ID"'" \
        --mode                predict_and_compare \
        --ensemble_size       1 \
        --maxiter             1000 \
        --energy_backend      rosetta \
        --gromacs_minimize    1 \
        --gromacs_rerank      1 \
        --top_k               1 \
        --top_k_snapshots     100 \
        --snapshot_energy_gap 0.1 \
        --snapshot_sort_by    rmsd \
        --output_root         "'"$OUTPUT_BASE"'/task_${SLURM_PROCID}"
'
