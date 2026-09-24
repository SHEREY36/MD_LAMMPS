#!/bin/bash
# Build the patched LAMMPS on Negishi (run once, on a front-end node, from anywhere):
#   bash runs/fresh_usf/negishi/build_lammps.sh
# Result: runs/fresh_usf/bin/lmp_mpi  (frozen production binary used by all jobs)
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
source "$HERE/modules.sh"
module list 2>&1 | tee "$HERE/build_modules.txt"
cd "$REPO/src"
rm -rf Obj_mpi                      # never reuse object/dependency files from another machine
make yes-granular yes-asphere       # copies src/GRANULAR/*.cpp,*.h (incl. spherocylinder code) into src/
make -j 16 mpi
mkdir -p "$REPO/runs/fresh_usf/bin"
cp lmp_mpi "$REPO/runs/fresh_usf/bin/lmp_mpi"
"$REPO/runs/fresh_usf/bin/lmp_mpi" -h | grep -E "spherocyl|MPI v" | head -5
# 10-second functional check: single-contact virial must use the centre-of-mass branch
cd "$REPO/runs/regression/v2"
"$REPO/runs/fresh_usf/bin/lmp_mpi" -in in.t2_virial -log log.t2 -screen none
grep "T2 LAMMPS virial" log.t2
echo "expected: Pxy=0.001185854123 Pyy=0.003557562368"
