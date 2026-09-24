#!/bin/bash
# One fresh-USF case per array task.  Ranks, time, account, QOS and array range
# are set on the sbatch command line by jobs/submit_all.sh.
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
LIST=$1
FRESH=$(cd "$SLURM_SUBMIT_DIR/.." && pwd)
case=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$SLURM_SUBMIT_DIR/$LIST")
source "$SLURM_SUBMIT_DIR/modules.sh"
cd "$FRESH/$case" || exit 1
if grep -q "^DONE" log.lammps 2>/dev/null; then echo "$case already done"; exit 0; fi
echo "case $case  job $SLURM_JOB_ID  ranks $SLURM_NTASKS  node $(hostname)  start $(date)"
srun "$FRESH/bin/lmp_mpi" -in in.usf -log log.lammps -screen none
status=$?
echo "case $case end $(date) exit=$status"
grep -q "^DONE" log.lammps && echo "DONE" || echo "NOT FINISHED: see $FRESH/$case/log.lammps"
