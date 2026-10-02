#!/bin/bash
# Generate the fresh HCS cases and submit them to Negishi's standby QOS.
#
#   bash negishi/submit_hcs.sh            # from runs/fresh_hcs (or anywhere)
#
# Standby (-q standby) runs on idle nodes anywhere in the cpu partition (446
# nodes x 128 cores), does not count against the morri353 allocation, and
# allows 4 h per job, so the whole sweep starts as soon as cores are idle and
# never competes with jobs queued on the group's normal QOS.
#   main         96 runs, AR >= 1.5, 1250-2900 rods, 4 ranks, ~3-5 min each
#   near-sphere  30 runs, AR <= 1.25, 3600-4400 rods, 100 contacts/particle,
#                8 ranks, ~15 min each
set -e
HCS=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$HCS"

# a python with numpy: $PYTHON if usable, else python3, else the conda module
PY=""
for cand in "${PYTHON:-}" python3 python; do
  if [ -n "$cand" ] && command -v "$cand" >/dev/null 2>&1 \
     && "$cand" -c "import numpy" >/dev/null 2>&1; then PY=$(command -v "$cand"); break; fi
done
if [ -z "$PY" ]; then
  type module >/dev/null 2>&1 || source /etc/profile.d/lmod.sh 2>/dev/null \
    || source /etc/profile.d/modules.sh 2>/dev/null || true
  module load conda >/dev/null 2>&1 || true
  for cand in python3 python; do
    if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import numpy" >/dev/null 2>&1; then
      PY=$(command -v "$cand"); break; fi
  done
fi
[ -n "$PY" ] || { echo "no python with numpy found: set PYTHON=/path/to/python"; exit 1; }
echo "python: $PY"

[ -x bin/lmp_mpi ] || { mkdir -p bin; cp ../fresh_usf/bin/lmp_mpi bin/; chmod +x bin/lmp_mpi; }
"$PY" generate_fresh_hcs.py --lmp "$HCS/bin/lmp_mpi" > generate.log
n1=$(grep -c . cases_main.txt)
n2=$(grep -c . cases_near_sphere.txt)
echo "cases: $n1 main, $n2 near-sphere"

cd negishi
mkdir -p logs
sbatch -A morri353 -p cpu -q standby -J hcs_main --array=0-$((n1-1)) \
       --ntasks=4 --time=00:40:00 job_case.sbatch cases_main.txt
sbatch -A morri353 -p cpu -q standby -J hcs_near --array=0-$((n2-1)) \
       --ntasks=8 --time=01:30:00 job_case.sbatch cases_near_sphere.txt
echo "submitted; watch with: squeue -u \$USER -q standby"
