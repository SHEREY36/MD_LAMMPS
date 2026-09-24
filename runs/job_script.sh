#!/bin/bash
#SBATCH -A morri353
#SBATCH -p cpu
#SBATCH --job-name=lmp_flat
#SBATCH --array=0-49               # 50 total jobs (5 e-folders × 10 phi-folders)
#SBATCH --nodes=1
#SBATCH --ntasks=8                 # 8 MPI ranks
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=96:00:00
#SBATCH -o logs/out_%A_%a.out
#SBATCH -e logs/err_%A_%a.err

# ------------------------------------------------------
# Folder definitions (from your screenshots)
# ------------------------------------------------------

E_FOLDERS=(e_070 e_080 e_090 e_095 e_099)

PHI_FOLDERS=(
    phi_01 phi_05 phi_10 phi_20 phi_30
    phi_40 phi_49 phi_50 phi_55 phi_60
)

# ------------------------------------------------------
# Compute mapping from array index → e-folder + phi-folder
# ------------------------------------------------------

eid=$(( SLURM_ARRAY_TASK_ID / 10 ))
pid=$(( SLURM_ARRAY_TASK_ID % 10 ))

E_DIR=${E_FOLDERS[$eid]}
PHI_DIR=${PHI_FOLDERS[$pid]}

echo "Selected case:"
echo "   E folder  = $E_DIR"
echo "   PHI folder = $PHI_DIR"

cd $SLURM_SUBMIT_DIR/$E_DIR/$PHI_DIR

# ------------------------------------------------------
# Run the simulation
# ------------------------------------------------------
export OMP_NUM_THREADS=1

if [[ -f in.chialvo ]]; then
    echo "Running LAMMPS in: $E_DIR/$PHI_DIR"
    srun $SLURM_SUBMIT_DIR/lmp_mpi < in.chialvo
else
    echo "ERROR: in.chialvo not found in $E_DIR/$PHI_DIR"
fi
