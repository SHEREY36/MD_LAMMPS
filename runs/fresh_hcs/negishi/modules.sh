# Module environment shared by the build and the jobs (edit once, used everywhere).
# Check what exists with:  module avail gcc openmpi
module purge 2>/dev/null
module load gcc openmpi
export OMP_NUM_THREADS=1
