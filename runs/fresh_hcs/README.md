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
check against Rubio-Largo et al. (Physica A 443, 477, 2016). Two seeds per case,
126 runs, N ≈ 1400–4400 rods in a 64 d box:

* `cases_main.txt` (96 runs, AR ≥ 1.5): 25 contacts per particle, about 3.5 min on
  4 ranks on the workstation, 2–3 min on Negishi;
* `cases_near_sphere.txt` (30 runs, AR ≤ 1.25): translation–rotation exchange
  needs about 20 encounters near the sphere, so θ settles only after ~50–100
  contacts per particle; these run 40 blocks (100 contacts), about 40 min on
  4 Negishi ranks.

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
python3 generate_fresh_hcs.py                 # writes AR*/a*/s*/ and cases*.txt
./run_local_queue.sh 6 1 /path/to/lmp_mpi     # one rank per case, one case per physical core
python3 postprocess_fresh_hcs.py
./bench_serial.sh /path/to/lmp_mpi            # serial cost of the first collision/particle
```

## Running on Negishi (from `git pull`)

Every run is far below the 4 h standby limit, so the sweep goes to the
**standby QOS**: idle nodes anywhere in the cpu partition (446 nodes × 128
cores), up to 14,272 cores per account, not counted against morri353 and not
competing with jobs queued on the group's normal QOS. All 126 runs are submitted
at once (4 ranks for the main cases, 8 for the near-spheres); with idle cores the
sweep finishes in about 20 minutes.

```bash
cd /scratch/negishi/$USER/MD_LAMMPS
git pull
cd runs/fresh_hcs
bash negishi/submit_hcs.sh            # finds python+numpy (or set PYTHON=...), generates, submits
squeue -u $USER -q standby            # watch
bash negishi/status.sh                # finished / running / not started
module load conda && python postprocess_fresh_hcs.py    # when finished: analysis/summary.csv
```

`submit_hcs.sh` copies the frozen fresh_usf binary to `bin/` if it is not there.
A killed or timed-out case restarts from the beginning when resubmitted; finished
cases (log contains `DONE`) are skipped.

Copy back only the reduced data:
`rsync -av negishi:/scratch/negishi/$USER/MD_LAMMPS/runs/fresh_hcs/analysis/ analysis/`
(add `--include='*/' --include='hcs.*' --include='hcs_T.dat' --include='snapshots.lammpstrj.gz' --include='params.in' --include='log.lammps' --exclude='*'` on the case tree if the raw files are needed).
