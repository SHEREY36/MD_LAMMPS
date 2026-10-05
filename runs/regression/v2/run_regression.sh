#!/bin/bash
# Regression suite v2 for the spherocylinder DEM code (run from anywhere).
#   ./run_regression.sh           all tests (T6 sphere USF takes ~30-60 min on 4 ranks)
#   ./run_regression.sh quick     skip T4 and T6
# Then: python3 check_regression.py
#   LMP=<binary>      default ../../bin/lmp_mpi (local); on Negishi ../../fresh_usf/bin/lmp_mpi
#   MPIRUN=<launcher> default "mpirun --oversubscribe" (also right inside sinteractive;
#                     the script passes -np N, so plain srun does not work here)
#   PYTHON=<python>   default python3 (needs numpy), e.g. .../.conda-v2/bin/python
#   LMP_R=<binary>    Model R binary (centre damping, no damp_velocity keyword) for T12;
#                     T12 is skipped when unset
# Every run, even on 1 rank, goes through $MPIRUN: a bare lmp_mpi inside an
# sinteractive/srun step fails in MPI_Init ("PMK_KVS_Barrier duplicate request").
cd "$(dirname "$0")"
LMP=$(readlink -f "${LMP:-../../bin/lmp_mpi}")
MPIRUN=${MPIRUN:-"mpirun --oversubscribe"}
MODE=${1:-all}
PYTHON=${PYTHON:-python3}
SERIAL="$MPIRUN -np 1"
set -x
$MPIRUN -np 2 $LMP -in in.t1_free_flight -log log.t1 -screen none
$SERIAL $LMP -in in.t2_virial -log log.t2 -screen none
for st in hertz hooke; do for a in 0.5 0.7 0.9; do
  $SERIAL $LMP -in in.t3_binary -var style $st -var alpha $a -log log.t3_${st}_$a -screen none
done; done
$SERIAL $LMP -in in.t3b_rods -log log.t3b -screen none
# T10 rotation-driven contacts, both damping definitions (contact = model C1, center = R1)
for c in a b; do for dv in contact center; do
  $SERIAL $LMP -in in.t10_rotation_contact -var case $c -var dv $dv -log log.t10${c}_$dv -screen none
done; done
# T11 spheres: contact and center damping must give identical trajectories
for dv in contact center; do $SERIAL $LMP -in in.t11_sphere_identity -var dv $dv -log log.t11_$dv -screen none; done
# T12 Model R reproduction (only when LMP_R points to a Model R binary, e.g. the archived one)
if [ -n "${LMP_R:-}" ]; then
  $SERIAL $LMP -in in.t12_modelR_repro -var dv center -log log.t12_center -screen none
  $SERIAL $(readlink -f "$LMP_R") -in in.t12_modelR_repro -var dv none -log log.t12_none -screen none
fi
# T7 must stop with an ERROR (rot_sllod was removed); check_regression.py looks for it in log.t7
echo "T7: the mpirun 'non-zero exit code / job aborted' message that follows is EXPECTED"
$SERIAL $LMP -in in.t7_rot_sllod -log log.t7 -screen none
$MPIRUN -np 2 $LMP -in in.t5_elastic_usf -log log.t5 -screen none
$SERIAL $LMP -in in.t8_decomp -var np 1 -log log.t8_np1 -screen none
$MPIRUN -np 4 $LMP -in in.t8_decomp -var np 4 -log log.t8_np4 -screen none
$MPIRUN -np 4 $LMP -in in.t9_flip -log log.t9 -screen none
if [ "$MODE" != "quick" ]; then
  $MPIRUN -np 4 $LMP -in in.t4_equipartition -log log.t4 -screen none
  # T6: sphere USF through the production template, compared with Boltzmann DSMC
  rm -rf t6_sphere_usf
  $PYTHON ../../fresh_usf/generate_fresh_usf.py --ARs 1 --alphas 0.7 --N 2000 --rho 1.0 \
      --nblocks 2 --strain_block 15 --outdir t6_sphere_usf --log_fraction 0.05 --lmp $LMP > /dev/null
  (cd t6_sphere_usf/AR1/a0.70 && $MPIRUN -np 4 $LMP -in in.usf -log log.lammps -screen none)
fi
