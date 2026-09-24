#!/bin/bash
#SBATCH -A morri353
#SBATCH -p cpu
#SBATCH --job-name=lmp_chialvo_base
#SBATCH --array=0-0
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=96:00:00
#SBATCH -o logs/out_%A_%a.out
#SBATCH -e logs/err_%A_%a.err

set -euo pipefail

# Usage:
#   sbatch --array=0-N hpc_array_lammps_cases.sh <cases_file> [lammps_bin] [input_script]
#
# cases_file: text file with one relative case directory per line.
#             each case directory must contain the selected input script
#
# lammps_bin: optional absolute path to LAMMPS MPI executable.
#             default: $SLURM_SUBMIT_DIR/lmp_mpi
# input_script: optional input filename inside each case directory.
#               default: in.chialvo

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <cases_file> [lammps_bin] [input_script]"
    exit 2
fi

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
CASES_FILE="$1"
LAMMPS_BIN="${2:-${SUBMIT_DIR}/lmp_mpi}"
INPUT_SCRIPT="${3:-in.chialvo}"

if [[ ! -f "$CASES_FILE" ]]; then
    echo "ERROR: cases file not found: $CASES_FILE"
    exit 2
fi

if [[ ! -x "$LAMMPS_BIN" ]]; then
    echo "ERROR: LAMMPS binary is not executable: $LAMMPS_BIN"
    exit 2
fi

mapfile -t RAW_CASES < "$CASES_FILE"
CASES=()
for line in "${RAW_CASES[@]}"; do
    s="$(echo "$line" | sed 's/[[:space:]]*$//')"
    [[ -z "$s" ]] && continue
    [[ "${s:0:1}" == "#" ]] && continue
    CASES+=("$s")
done

NCASES="${#CASES[@]}"
if [[ "$NCASES" -eq 0 ]]; then
    echo "ERROR: no runnable cases in $CASES_FILE"
    exit 2
fi

IDX="${SLURM_ARRAY_TASK_ID:-0}"
if (( IDX < 0 || IDX >= NCASES )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$IDX out of range [0,$((NCASES-1))]"
    exit 2
fi

CASE_DIR="${CASES[$IDX]}"
if [[ "${CASE_DIR:0:1}" == "/" ]]; then
    RUN_DIR="$CASE_DIR"
else
    RUN_DIR="${SUBMIT_DIR}/${CASE_DIR}"
fi

echo "Running case index ${IDX}/${NCASES}: ${CASE_DIR}"
echo "Run dir: $RUN_DIR"
echo "LAMMPS:  $LAMMPS_BIN"
echo "Input:   $INPUT_SCRIPT"

if [[ ! -d "$RUN_DIR" ]]; then
    echo "ERROR: case directory missing: $RUN_DIR"
    exit 2
fi

cd "$RUN_DIR"
if [[ ! -f "$INPUT_SCRIPT" ]]; then
    echo "ERROR: $INPUT_SCRIPT not found in $RUN_DIR"
    exit 2
fi

export OMP_NUM_THREADS=1
srun "$LAMMPS_BIN" -in "$INPUT_SCRIPT"
