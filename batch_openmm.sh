#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=qtf_openmm
#SBATCH --array=0-2
#SBATCH --output=logs/openmm/qtf_%A_%a.out
#SBATCH --error=logs/openmm/qtf_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=defq
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=puramv@ccf.org

conda activate QTF

mkdir -p logs/openmm results/openmm

python runner.py \
    --replica_id      $SLURM_ARRAY_TASK_ID \
    --energy_backend  openmm \
    --max_iter        10 \
    --scout           50 \
    --strategy        random \
    --top_k           5 \
    --outdir          results
