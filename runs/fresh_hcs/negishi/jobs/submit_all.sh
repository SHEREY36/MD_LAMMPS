#!/bin/bash
# Submit the fresh HCS sweep as one job array (4 ranks per case, ~5 min each on
# Negishi).  Run from anywhere after generate_fresh_hcs.py has written cases.txt.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs
n=$(grep -c . ../cases.txt)
sbatch -A morri353 -p cpu -J hcs_np4 --array=0-$((n-1))%64 --ntasks=4 --time=00:45:00 job_case.sbatch cases.txt
