#!/bin/bash
# Run every case in cases.txt on this workstation, JOBS cases at a time,
# NP MPI ranks each.  Finished cases (log contains DONE) are skipped, so the
# script can simply be restarted.  Usage: ./run_local_queue.sh [JOBS] [NP] [LMP]
cd "$(dirname "$0")"
JOBS=${1:-3}
NP=${2:-4}
LMP=${3:-/home/muhammed/Documents/Thesis/LAMMPS/src/lmp_mpi}
# Several runners may work on the same list: each case is claimed with a lock
# directory (.lock); delete stale locks after a crash before restarting.
run_case() {
  cd "$1" || exit 1
  if grep -q "^DONE" log.lammps 2>/dev/null; then echo "skip $1 (done)"; exit 0; fi
  mkdir .lock 2>/dev/null || { echo "skip $1 (claimed by another runner)"; exit 0; }
  echo "start $1 $(date)"
  mpirun --oversubscribe -np $NP $LMP -in in.hcs -log log.lammps -screen none
  echo "end   $1 $(date) exit=$?"
  grep -q "^DONE" log.lammps && rmdir .lock
}
export -f run_case
export NP LMP
xargs -a cases_main.txt -P $JOBS -I{} bash -c 'run_case {}' >> queue.log 2>&1
