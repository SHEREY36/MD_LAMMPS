#!/bin/bash
set -euo pipefail

# Submit spherocylinder dilute USF cases as a SLURM array.
#
# Defaults:
#   case list   : runs/sphcyl_dilute_cases.txt
#   lammps bin  : runs/sphcyl/lmp
#   input script: in.sphcyl_usf
#
# Examples:
#   runs/submit_sphcyl_dilute_sweep.sh
#   runs/submit_sphcyl_dilute_sweep.sh runs/sphcyl_dilute_cases.txt
#   runs/submit_sphcyl_dilute_sweep.sh runs/sphcyl_dilute_cases.txt /path/to/lmp in.sphcyl_usf

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASES_FILE="${1:-${SCRIPT_DIR}/sphcyl_dilute_cases.txt}"
LAMMPS_BIN="${2:-${SCRIPT_DIR}/sphcyl/lmp}"
INPUT_SCRIPT="${3:-in.sphcyl_usf}"
JOB_SCRIPT="${SCRIPT_DIR}/hpc_array_lammps_cases.sh"

CASES_FILE="$(realpath "$CASES_FILE")"
if [[ -e "$LAMMPS_BIN" ]]; then
    LAMMPS_BIN="$(realpath "$LAMMPS_BIN")"
fi

if [[ ! -f "$CASES_FILE" ]]; then
    echo "ERROR: cases file not found: $CASES_FILE"
    exit 2
fi

if [[ ! -x "$LAMMPS_BIN" ]]; then
    echo "ERROR: LAMMPS binary not executable: $LAMMPS_BIN"
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
echo "Input file: $INPUT_SCRIPT"

sbatch \
  --chdir "$SCRIPT_DIR" \
  --array "0-${ARRAY_MAX}" \
  "$JOB_SCRIPT" \
  "$CASES_FILE" \
  "$LAMMPS_BIN" \
  "$INPUT_SCRIPT"
