#!/bin/bash
set -euo pipefail

# Submit an array job sized to the provided case list.
#
# Default case list preference:
#   1) runs/chialvo_baseline_chialvo_all_cases.txt
#   2) runs/chialvo_baseline_all_cases.txt (legacy)
#
# Examples:
#   runs/submit_hpc_sweep.sh
#   runs/submit_hpc_sweep.sh runs/chialvo_baseline_chialvo_all_cases.txt
#   runs/submit_hpc_sweep.sh runs/chialvo_baseline_usf_sllod_all_cases.txt
#   runs/submit_hpc_sweep.sh runs/chialvo_baseline_chialvo_dilute_cases.txt /path/to/lmp_mpi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CASES_NEW="${SCRIPT_DIR}/chialvo_baseline_chialvo_all_cases.txt"
DEFAULT_CASES_OLD="${SCRIPT_DIR}/chialvo_baseline_all_cases.txt"
if [[ $# -ge 1 ]]; then
    CASES_FILE="$1"
elif [[ -f "$DEFAULT_CASES_NEW" ]]; then
    CASES_FILE="$DEFAULT_CASES_NEW"
else
    CASES_FILE="$DEFAULT_CASES_OLD"
fi
LAMMPS_BIN="${2:-${SCRIPT_DIR}/lmp_mpi}"
JOB_SCRIPT="${SCRIPT_DIR}/hpc_array_lammps_cases.sh"

CASES_FILE="$(realpath "$CASES_FILE")"
if [[ -e "$LAMMPS_BIN" ]]; then
    LAMMPS_BIN="$(realpath "$LAMMPS_BIN")"
fi

if [[ ! -f "$CASES_FILE" ]]; then
    echo "ERROR: cases file not found: $CASES_FILE"
    exit 2
fi

if [[ ! -f "$JOB_SCRIPT" ]]; then
    echo "ERROR: job script not found: $JOB_SCRIPT"
    exit 2
fi

mkdir -p "${SCRIPT_DIR}/logs"

NCASES="$(grep -Ev '^[[:space:]]*(#|$)' "$CASES_FILE" | wc -l | awk '{print $1}')"
if [[ "$NCASES" -lt 1 ]]; then
    echo "ERROR: no cases found in $CASES_FILE"
    exit 2
fi

ARRAY_MAX=$((NCASES - 1))
echo "Submitting ${NCASES} cases as array 0-${ARRAY_MAX}"
echo "Case list : $CASES_FILE"
echo "LAMMPS bin: $LAMMPS_BIN"

sbatch \
  --chdir "$SCRIPT_DIR" \
  --array "0-${ARRAY_MAX}" \
  "$JOB_SCRIPT" \
  "$CASES_FILE" \
  "$LAMMPS_BIN"
