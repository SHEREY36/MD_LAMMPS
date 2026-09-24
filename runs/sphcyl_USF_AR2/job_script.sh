#!/bin/bash
#SBATCH -A morri353
#SBATCH -p cpu
#SBATCH --job-name=lmp_array
#SBATCH --array=0-12                   # 9 top-level folders e_060 ... e_100
#SBATCH --nodes=1
#SBATCH --ntasks=12                    # 8 MPI ranks per job
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=96:00:00
#SBATCH -o logs/out_%A_%a.out
#SBATCH -e logs/err_%A_%a.err

# --- CASES LIST ---
cases=(e_050 e_055 e_060 e_065 e_070 e_075 e_080 e_085 e_090 e_095 e_100)

case=${cases[$SLURM_ARRAY_TASK_ID]}

echo "=== Running top-level case: $case ==="
cd $SLURM_SUBMIT_DIR/$case

export OMP_NUM_THREADS=1

# Loop through each phi_* folder
for folder in phi_*; do
    echo "---- Running phi-folder: $folder ----"
    cd "$folder"

    if [[ -f in.sphcyl_usf ]]; then
        srun $SLURM_SUBMIT_DIR/lmp_mpi < in.sphcyl_usf
    else
        echo "WARNING: in.sphcyl_usf not found in $folder"
    fi

    cd ..
done