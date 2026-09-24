#!/bin/bash
#SBATCH -A morri353
#SBATCH -p cpu
#SBATCH --job-name=lmp_dil_array
#SBATCH --array=0-8
#SBATCH --nodes=1
#SBATCH --ntasks=8               # 8 MPI ranks per job
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=96:00:00
#SBATCH -o logs/out_%A_%a.out
#SBATCH -e logs/err_%A_%a.err

# List of case folders
cases=(e_060 e_065 e_070 e_075 e_080 e_085 e_090 e_095 e_100)

# Pick folder based on array index
case=${cases[$SLURM_ARRAY_TASK_ID]}

echo "Running case: $case"
cd $SLURM_SUBMIT_DIR/$case

# LAMMPS: pure MPI
export OMP_NUM_THREADS=1

# Run LAMMPS with 8 MPI ranks
srun $SLURM_SUBMIT_DIR/lmp_mpi < in.chialvo
