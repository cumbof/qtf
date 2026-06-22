#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=qtf_custom
#SBATCH --array=0-2
#SBATCH --output=logs/custom/qtf_%A_%a.out
#SBATCH --error=logs/custom/qtf_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=defq
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=puramv@ccf.org

conda activate QTF

mkdir -p logs/custom results/custom

python runner.py \
    --replica_id      $SLURM_ARRAY_TASK_ID \
    --energy_backend  custom \
    --max_iter        10 \
    --scout           50 \
    --strategy        random \
    --top_k           5 \
    --outdir          results
