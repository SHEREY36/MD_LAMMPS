# Fresh HCS runs (smooth spherocylinders, φ = 0.01)

Homogeneous cooling state with the corrected DEM of `runs/fresh_usf`: exact
Hertz damping η = √5·√(kn m*)·|ln α|/√(π²+ln²α), contact history kept across
neighbour rebuilds, and the hard-particle limit enforced by measurement. The old
`runs/HCS` sweeps predate these fixes (linear-spring damping on a Hertz contact
gave a realised restitution of 0.54 for a nominal 0.50) and must not be used.

## What one case does

1. **A: elastic equilibration** from T_tr = T_rot = T0, 5 contacts per particle
   (`diag_equil.*`: energy residual must stay ~0).
2. **Calibration**: kn from the measured mean overlap (target 0.4 % d) and dt
   from the median contact time (target 40 steps), as in fresh_usf.
3. **Exact start**: velocities and angular momenta rescaled to T_tr = T_rot = T0;
   time is reset to 0.
4. **B: cooling**, `nblocks` blocks of about `tau_block` contacts per particle.
   Between blocks kn ∝ T_tr and dt ∝ T_tr^(−1/2) (`harden.in`): for a Hertz
   contact this keeps the overlap and the steps per contact fixed, so the cost
   per collision does not grow as the gas cools.

Default sweep (`generate_fresh_hcs.py`): AR 1.1, 1.5, 2, 2.5, 3 × α 0.50–0.95
(step 0.05) plus 0.88 and 0.96, and AR 1.25 at 0.88/0.90/0.96, for the literature
check against Rubio-Largo et al. (Physica A 443, 477, 2016). Two seeds per case:
126 runs, N ≈ 1400–2900 rods in a 64 d box, 25 contacts per particle each.
Cost: about 3.5 min on 4 ranks per run on the workstation (≈ 30 core-hours in all).

## Outputs of one case

| file | content |
|---|---|
| `hcs.coll` | per window (0.05 contacts/particle): `N_start`, `nu` = 2 N_start/(N t_win), realised e_c and e_tr, multibody and same-partner fractions, overlaps, contact times |
| `hcs.stress` | window-averaged T_tr, T_rot, kinetic tensor, a2_tr, a2_rot |
| `hcs.energy` | energy bookkeeping (dissipation W_nc, residual) |
| `hcs.struct`, `hcs.sensors` | structure factor / clustering, sensor flags (see fresh_usf README) |
| `hcs_T.dat` | instantaneous T_tr, T_rot every window |
| `snapshots.lammpstrj.gz` | velocities, quaternions, angular momenta every 0.5 contacts/particle |

`python3 postprocess_fresh_hcs.py` writes `analysis/hcs_timeseries/*.csv` and
`analysis/summary.csv` (θ*, cooling rate γ = d ln T/dτ, α_eff = √(1+5γ),
effective cross-section σ_eff = ν/(n⟨g⟩), realised e_c, flags).

## Running on the workstation

```bash
python3 generate_fresh_hcs.py                 # writes AR*/a*/s*/ and cases.txt
./run_local_queue.sh 3 4 /path/to/lmp_mpi     # 3 cases at a time, 4 ranks each
python3 postprocess_fresh_hcs.py
./bench_serial.sh /path/to/lmp_mpi            # serial cost of the first collision/particle
```

## Running on Negishi (from `git pull`)

```bash
cd /scratch/negishi/$USER/MD_LAMMPS
git pull
cd runs/fresh_hcs
export PYTHON=/scratch/negishi/$USER/DSMC_V2/.conda-v2/bin/python   # any python with numpy
mkdir -p bin && cp ../fresh_usf/bin/lmp_mpi bin/                     # the frozen fresh_usf binary
$PYTHON generate_fresh_hcs.py                                       # 126 cases, cases.txt
bash negishi/jobs/submit_all.sh                                     # one array, 4 ranks/case, 45 min limit
squeue -u $USER
# when finished:
$PYTHON postprocess_fresh_hcs.py
# optional, serial timing on one compute node (about 45 min in all):
sinteractive -A morri353 -p cpu -n 1 -t 1:30:00
source negishi/modules.sh && ./bench_serial.sh $(pwd)/bin/lmp_mpi && exit
```

Copy back only the reduced data:
`rsync -av negishi:/scratch/negishi/$USER/MD_LAMMPS/runs/fresh_hcs/analysis/ analysis/`
(add `--include='*/' --include='hcs.*' --include='hcs_T.dat' --include='snapshots.lammpstrj.gz' --include='params.in' --include='log.lammps' --exclude='*'` on the case tree if the raw files are needed).
