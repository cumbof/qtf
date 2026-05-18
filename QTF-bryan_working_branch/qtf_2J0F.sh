#!/bin/bash
#SBATCH --job-name=QTF_ensemble
#SBATCH --partition=defq
#SBATCH --array=0-999                  # 1000 replicas (0 to 999)
#SBATCH --ntasks=1                     # 1 task per replica
#SBATCH --cpus-per-task=4              # 4 CPUs per replica
#SBATCH --mem=16G                      # 16GB RAM per replica
#SBATCH --output=logs/qtf_%A_%a.out    # stdout: logs/qtf_<jobid>_<arrayid>.out
#SBATCH --error=logs/qtf_%A_%a.err     # stderr: logs/qtf_<jobid>_<arrayid>.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=puramv@ccf.org

# ── Environment ────────────────────────────────────────────────────────────────
module load python3
source /home/puramv/isilon/varun/QTF-bryan_working_branch/QTF-env/bin/activate

# ── Paths ──────────────────────────────────────────────────────────────────────
WORKDIR=/home/puramv/isilon/varun/QTF-bryan_working_branch
cd $WORKDIR

# ── Config ─────────────────────────────────────────────────────────────────────
SEQUENCE="DAYAQWLKDGGPSSGRPPPS"
REFERENCE="2JOF"
FORCEFIELD="amber"
MAXITER=2000
SHOTS=4096

# Each array task runs exactly 1 replica
REPLICA_ID=$SLURM_ARRAY_TASK_ID

# Output dir per replica
OUTDIR=outputs/reps_Brickwork_entanglement1/slurm_${SEQUENCE}_${FORCEFIELD}/replica_${REPLICA_ID}
mkdir -p $OUTDIR
mkdir -p logs

echo "============================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Array Task   : $SLURM_ARRAY_TASK_ID"
echo "Replica ID   : $REPLICA_ID"
echo "Node         : $SLURMD_NODENAME"
echo "Start time   : $(date)"
echo "============================================"

# ── Run single replica ─────────────────────────────────────────────────────────
python3 qtf_single_replica.py \
    --predict "$SEQUENCE" \
    --reference_structure "$REFERENCE" \
    --forcefield "$FORCEFIELD" \
    --replica_id $REPLICA_ID \
    --maxiter $MAXITER \
    --hw_backend aer \
    --hw_shots $SHOTS \
    --outdir $OUTDIR

echo "============================================"
echo "End time     : $(date)"
echo "Exit code    : $?"
echo "============================================"
