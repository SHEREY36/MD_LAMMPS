#!/bin/bash
# Serial (1 MPI rank) wall time for the first collision per particle of the HCS,
# for the DEM-vs-DSMC cost comparison.  Usage: ./bench_serial.sh [LMP]
# Writes bench/summary.txt: AR alpha N steps loop_seconds
cd "$(dirname "$0")"
LMP=${1:-$(pwd)/bin/lmp_mpi}
PY=${PYTHON:-python3}
$PY generate_fresh_hcs.py --ARs 1.1 1.5 2 2.5 3 --alphas 0.5 0.8 0.95 --seeds 1 --outdir bench > /dev/null
echo "# AR alpha N steps_tau1 loop_seconds_1rank" > bench/summary.txt
while read c; do
  ( cd bench/$c && cp ../../../../templates/in.bench . ;
    mpirun -np 1 $LMP -in in.bench -log log.bench -screen none )
  N=$(grep -m1 "BENCH START" bench/$c/log.bench | sed 's/.*N=\([0-9]*\).*/\1/')
  steps=$(grep -m1 "BENCH START" bench/$c/log.bench | sed 's/.*steps=\([0-9]*\).*/\1/')
  loop=$(grep "Loop time" bench/$c/log.bench | tail -1 | awk '{print $4}')
  AR=$(echo $c | sed 's/AR\([^/]*\)\/a\([^/]*\)\/.*/\1/'); a=$(echo $c | sed 's/AR\([^/]*\)\/a\([^/]*\)\/.*/\2/')
  echo "$AR $a $N $steps $loop" | tee -a bench/summary.txt
done < bench/cases.txt
