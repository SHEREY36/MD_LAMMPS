#!/bin/bash
# ============================================================
# Run HCS sweep sequentially from generated case folders.
# Usage:
#   bash run_e_sweep.sh /path/to/lmp [cases_file] [input_name]
# ============================================================

set -euo pipefail

LMP="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lmp}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASES_FILE="${2:-${SCRIPT_DIR}/hcs_modeB_cases.txt}"
INPUT_NAME="${3:-in.hcs}"

if [[ ! -x "$LMP" ]]; then
    echo "ERROR: LAMMPS binary not executable: $LMP"
    exit 2
fi

if [[ ! -f "$CASES_FILE" ]]; then
    echo "ERROR: cases file not found: $CASES_FILE"
    exit 2
fi

while IFS= read -r case_rel; do
    [[ -z "${case_rel// }" ]] && continue
    [[ "${case_rel:0:1}" == "#" ]] && continue

    case_dir="${SCRIPT_DIR}/${case_rel}"
    if [[ ! -d "$case_dir" ]]; then
        echo "WARNING: skipping missing case dir: $case_dir"
        continue
    fi
    if [[ ! -f "${case_dir}/${INPUT_NAME}" ]]; then
        echo "WARNING: skipping (missing ${INPUT_NAME}): $case_dir"
        continue
    fi

    e_tag="$(basename "$case_dir")"
    echo "========================================="
    echo "Running HCS case: ${e_tag}"
    echo "Dir: ${case_dir}"
    echo "========================================="
    (
        cd "$case_dir"
        "$LMP" -in "$INPUT_NAME" -log log.lammps
    )
done

echo "All runs complete."
echo "Temperature files should be in each case folder."
