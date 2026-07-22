#!/bin/bash
#SBATCH --array=0-399%50
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=qtf_1prb_custom
#SBATCH --output=../logs/qtf_1prb_custom_%A_%a.out
#SBATCH --error=../logs/qtf_1prb_custom_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=defq
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=raubenb@ccf.org

set -eo pipefail

source ~/miniforge3/bin/activate
conda activate qtf

set -u

SEQUENCE="TIDQWLLKNAKEDAIAELKKAGITSDFYFNAINKAKTVEEVNALKNEILKAHA"
REFERENCE_PDB_ID="../references/1PRB.pdb"
ENERGY_BACKEND="custom"

REFERENCE_LABEL="$(basename "$REFERENCE_PDB_ID" .pdb)"
OUTPUT_BASE="../run_outputs/quantum_simulations/$REFERENCE_LABEL/$ENERGY_BACKEND"
JOB_OUTPUT_ROOT="$OUTPUT_BASE/task_${SLURM_ARRAY_TASK_ID}"
mkdir -p ../logs "$JOB_OUTPUT_ROOT"

exec > "$JOB_OUTPUT_ROOT/slurm_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1

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
    --top_k_snapshots     5000 \
    --snapshot_energy_gap 0.1 \
    --snapshot_sort_by    rmsd \
    --output_root         "$JOB_OUTPUT_ROOT"
