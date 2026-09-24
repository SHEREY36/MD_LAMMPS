# LAMMPS Spherocylinder DEM Project

## What This Is

Custom DEM (discrete element method) pair style for **spherocylinder** contact dynamics
in LAMMPS 22 Jul 2025 Update 3. Spherocylinder support is not native to LAMMPS; this
implements it by combining:

- **Contact geometry**: Closest-approach between two finite line segments (Vega-Lago algorithm)
- **Kinematics**: `atom_style ellipsoid` (ASPHERE package) for per-atom quaternion and angular momentum
- **Force model**: Hooke normal + tangential spring with shear history (GRANULAR package)
- **Integration**: `fix nve/asphere` (angular momentum DOF, not omega)

## Key Files

| File | Purpose |
|------|---------|
| `src/GRANULAR/pair_gran_spherocyl_history.cpp` | Base style (Hooke) + per-contact collision records for the diag fix |
| `src/GRANULAR/pair_gran_spherocyl_mfix_history.cpp` | Hertz + delta^1/4 damping (used for USF) |
| `src/GRANULAR/fix_nve_spherocyl.cpp` | Integrator (smooth rod, axial spin removed) |
| `src/GRANULAR/fix_spherocyl_diag.cpp` | Stress decomposition, collision stats, sensors, clustering |
| `src/fix_deform.cpp` | **Patched**: Lees-Edwards flip + remap v velocity bug |
| `runs/regression/v2/` | Regression suite (`./run_regression.sh; python3 check_regression.py`) |
| `runs/fresh_usf/` | Corrected USF production pipeline (README.md there) |
| `runs/bin/lmp_mpi` | Current binary |

## Build

Make-based build in `src/` (package files live in BOTH `src/GRANULAR/` and `src/`;
edit in `src/GRANULAR/` and copy to `src/`, they must stay identical):

```bash
cd src && make mpi -j12
cp lmp_mpi ../runs/bin/lmp_mpi.new && mv ../runs/bin/lmp_mpi.new ../runs/bin/lmp_mpi   # atomic: running jobs unaffected
```

## Run Test

```bash
cd runs/sphcyl
./lmp < in.test_spherocyl     # NOTE: < not >
```

## Critical Pitfalls

### Uniform shear flow physics (verified by runs/regression/v2)

- `fix deform ... remap v` stores LAB velocities; Newton (dv/dt = F/m) + Lees-Edwards
  IS the SLLOD dynamics. Do NOT add `fix addforce -gdot*m*vy` (counts the shear
  coupling twice: T comes out 4x too high, P* unchanged in the dilute limit).
  `rot_sllod` was removed (fictitious torque). LAMMPS `fix nvt/sllod` applies the
  same extra term to the stored velocity.
- `velocity ramp` must span the actual box: `ramp vx 0 $(v_gdot*ly) y $(ylo) $(yhi)`.
  `change_box ... scale` keeps the box centred (ylo != 0); a mismatch leaves a
  permanent uniform peculiar velocity (peculiar momentum is conserved) that inflates T.
- `src/fix_deform.cpp` is patched: at a box flip, unpatched LAMMPS wraps atoms that
  crossed y since the last rebuild WITHOUT the remap-v velocity shift (+-gdot*Ly kick).
- Hertz restitution: eta = sqrt(5)*sqrt(kn*m*)*|ln e|/sqrt(pi^2+ln^2 e) (exact);
  the linear-spring factor 2 (MFIX) gives e = 0.54 for 0.50.
- Virial must use the centre-of-mass branch x_i - x_j (not the contact-point
  separation); granular styles cannot use fdotr (no_virial_fdotr_compute = 1).
- Non-size neighbor lists never set the history bit: the pair style needs
  `beyond_contact = 1`, otherwise every contact in progress at a neighbor rebuild
  loses its history (touch flag, shear spring, collision record).
- Run with `lmp -in in.usf` (not `<`) when the input uses jump/label.
- Negishi: never start `lmp_mpi` bare inside `sinteractive` (itself an srun step):
  MPI_Init joins the step's PMI -> `srun: error: PMK_KVS_Barrier duplicate request`
  or a hang. Use `mpirun -np N` interactively, `srun` in batch jobs.
- LAMMPS variables have no two-argument max()/min(); use ternary().
- dt recalibration must use the MEDIAN contact time (f_diag[23]), never the mean:
  at low alpha ~7% rotation-driven contacts last >5x longer and inflate the mean
  (the first AR2 pilot at alpha<=0.65 ran with 90% under-resolved contacts).
- Git: the original .gitignore rule `dump.*` hid src/dump.cpp/.h and `Makefile`
  hid src/Makefile (both fixed). Commit with explicit paths, never `git add -A`
  (runs/ holds GB of outputs). Cluster workflow: runs/fresh_usf/negishi/README.md.

### atom_style ellipsoid — quaternion access

**WRONG** (atom->quat is NULL for atom_style ellipsoid):
```cpp
double **quat = atom->quat;
axis_from_quat(quat[i], u);   // segfault: NULL[i*8] = address i*8
```

**CORRECT** — quaternion lives in AtomVecEllipsoid bonus storage:
```cpp
auto *avec = dynamic_cast<AtomVecEllipsoid *>(atom->avec);
AtomVecEllipsoid::Bonus *bonus = avec->bonus;
int *ellipsoid = atom->ellipsoid;   // index into bonus array
axis_from_quat(bonus[ellipsoid[i]].quat, u);
```

### atom_style ellipsoid — angular velocity

**WRONG** (atom->omega is NULL for atom_style ellipsoid):
```cpp
double **omega = atom->omega;   // NULL
```

**CORRECT** — compute omega from angular momentum:
```cpp
MathExtra::mq_to_omega(angmom[i], quat_i, inertia_i, omegai);
```
Requires `fix nve/asphere` in the input script (not `fix nve`).

### Neighbor list: do NOT use REQ_SIZE

`atom_style ellipsoid` has **no `atom->radius` array** — it is never allocated.
`REQ_SIZE` makes `NPairBin::build` access `atom->radius[i]`, which crashes with
`address-not-mapped at 0x8` (= `null[1]` for a `double*`). The crash happens when
the neighbor list is rebuilt mid-run, not necessarily at step 0.

**WRONG** (in init_style):
```cpp
neighbor->add_request(this, NeighConst::REQ_SIZE | NeighConst::REQ_HISTORY);
```

**CORRECT** — use fixed geometric cutoff from init_one():
```cpp
neighbor->add_request(this, NeighConst::REQ_HISTORY);
```

### Input file redirect

```
./lmp < in.test_spherocyl   ← correct (reads input)
./lmp > in.test_spherocyl   ← WRONG (overwrites file with stdout!)
```

## Pair Style Syntax

```lammps
atom_style ellipsoid
pair_style gran/spherocyl/history Kn Kt gamman gammat xmu dampflag
pair_coeff * * R H
comm_modify vel yes
fix 1 all nve/asphere
```

Where `R` = capsule radius, `H` = half-length of cylindrical section.
Cutoff = `(Hi + Hj) + (Ri + Rj)`.

The `shape` in atom_style ellipsoid is used for moment-of-inertia computation.
A reasonable choice for a spherocylinder: `set type N shape R R (R+H)`.

## Architecture

`PairGranSpherocylHistory` inherits from `PairGranHookeHistory` and overrides:
- `init_style()` — skips parent's radius_flag check; sets up FixNeighHistory with same naming as parent
- `coeff()` — takes R and H instead of sphere radius
- `init_one()` — returns geometric cutoff
- `compute()` — replaces sphere geometry with segment closest-approach; uses angmom→omega

The shear history (3 doubles per neighbor pair) and FixNeighHistory wiring are
inherited from PairGranHookeHistory unchanged.

## Contact Visibility in OVITO

`kn` controls both contact duration and peak overlap:
```
T_c    = π / sqrt(kn/meff)         [contact half-period, steps = T_c/dt]
δ_max  = v_rel · sqrt(meff/kn)     [peak overlap, energy balance]
```

For OVITO-visible contact (δ_max ≥ 10% R, T_c ≥ 50 steps at dt=1e-7 s):

| kn (N/m) | v_each (m/s) | T_c (steps) | δ_max (% of R) |
|----------|-------------|-------------|----------------|
| 1e5      | 0.1         | 16          | 0.1%           | ← invisible
| 1e3      | 1.0         | 156         | 9.4%           | ← VISIBLE ✓

**Use kn=1e3, v≥1 m/s for regression/visualization tests.**
The contact detection proof and all force physics are correct at any kn.

## Damping Coefficient (gamman) — Correct Formula

LAMMPS `gran/hooke/history` expects `gamman` in **[1/s]** (a rate, not a viscous coefficient):
```
F_damp = meff × gamman × v_n        [code: damp = meff * gamman * vnnr * rsqinv]
```

To achieve restitution coefficient e:
```
eta    = |ln(e)| / sqrt(π² + ln(e)²)     ; critical damping ratio
gamman = 2 × eta × sqrt(kn / meff)        ; [1/s]  ← DIVIDE kn by meff
```

**WRONG** (gives e ≈ 1 because gamman is ≈ 0):
```lammps
variable gamman equal 2.0*${eta}*sqrt(${kn}*${meff})   # kn*meff = [kg²/s²]^0.5 = kg/s (wrong units)
```

**CORRECT**:
```lammps
variable gamman equal 2.0*${eta}*sqrt(${kn}/${meff})   # kn/meff = [1/s²]^0.5 = 1/s ✓
```

Example (kn=1e3, meff=2.479e-8 kg, e=0.9):
- eta = 0.03350, gamman = 13457 s⁻¹ → post-bounce |vx| = 0.9 m/s ✓
