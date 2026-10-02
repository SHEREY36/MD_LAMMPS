#!/bin/bash
# Submit the fresh HCS sweep (run from anywhere after generate_fresh_hcs.py).
#   main:        AR >= 1.5, 25 contacts/particle, ~5 min on 4 ranks
#   near-sphere: AR <= 1.25, 100 contacts/particle, 3500-4400 rods, ~40 min on 4 ranks
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs
n1=$(grep -c . ../cases_main.txt)
n2=$(grep -c . ../cases_near_sphere.txt)
sbatch -A morri353 -p cpu -J hcs_main --array=0-$((n1-1))%64 --ntasks=4 --time=00:45:00 job_case.sbatch cases_main.txt
sbatch -A morri353 -p cpu -J hcs_near --array=0-$((n2-1)) --ntasks=4 --time=03:00:00 job_case.sbatch cases_near_sphere.txt
