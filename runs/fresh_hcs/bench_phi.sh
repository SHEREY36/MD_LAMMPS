#!/bin/bash
# Serial DEM wall time for the first collision per particle versus volume
# fraction (AR = 2, alpha = 0.8, N = 2003): free flights are integrated at the
# contact time step, so the DEM cost per collision grows like 1/phi.
# Usage: ./bench_phi.sh [LMP]     writes bench_phi/summary.txt
cd "$(dirname "$0")"
LMP=${1:-$(pwd)/bin/lmp_mpi}
PY=${PYTHON:-python3}
echo "# phi N steps_tau1 loop_seconds_1rank" > bench_phi/summary.txt 2>/dev/null || { mkdir -p bench_phi; echo "# phi N steps_tau1 loop_seconds_1rank" > bench_phi/summary.txt; }
for phi in 0.0025 0.005 0.01 0.02; do
  out=bench_phi/phi$phi
  $PY generate_fresh_hcs.py --ARs 2 --alphas 0.8 --seeds 1 --phi $phi --N 2003 --outdir $out > /dev/null
  c=$out/AR2/a0.80/s1
  cp templates/in.bench $c/
  ( cd $c && mpirun -np 1 $LMP -in in.bench -log log.bench -screen none < /dev/null )
  steps=$(grep -m1 "^BENCH START" $c/log.bench | sed 's/.*steps=\([0-9]*\).*/\1/')
  loop=$(grep "Loop time" $c/log.bench | tail -1 | awk '{print $4}')
  echo "$phi 2003 $steps $loop" | tee -a bench_phi/summary.txt
done
