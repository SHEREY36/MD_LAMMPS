# Regression suite v2 (spherocylinder DEM, USF physics)

```bash
./run_regression.sh          # ~1 h on 4 cores (T6 dominates); "quick" skips T4 and T6
python3 check_regression.py  # PASS/FAIL table -> regression_report.txt
```

Run it after any change to `src/GRANULAR/*spherocyl*`, `src/fix_deform.cpp` or the
fresh USF templates. Every test compares against an independent reference, not
against an earlier run of the same code.

| test | what | reference |
|---|---|---|
| T1 | collisionless AR=2 rods under Lees–Edwards shear | exact free flight: P_xy/nT0 = −γ̇t, (P_xx−P_yy)/nT0 = (γ̇t)², every L constant, dE = W_shear exactly. Fails if a SLLOD force or rotational SLLOD is present. |
| T2 | single static contact between crossed rods | hand-computed virial with the centre-of-mass branch (P_xy = 1.18585e-3; the old contact-point branch gave 0), in both LAMMPS `compute pressure` and the 9-component diag tensor |
| T3 | head-on sphere collisions, Hertz (√5 damping) and Hooke, α = 0.5, 0.7, 0.9 | realized restitution = α (Hertz ±0.4 %, Hooke ±2.5 % at 40 steps/contact), energy bookkeeping |
| T3b | oblique off-centre rod collision | linear and total (orbital + spin) angular momentum conserved to round-off; exactly one collision recorded; energy bookkeeping including rotation |
| T4 | elastic AR=2 rods, no shear, T_rot starts at 0.2 T_tr | equipartition T_rot/T_tr → 1; energy drift < 1e-3; momentum conserved |
| T5 | elastic AR=2 rods under shear | dE = shear work in every window; peculiar momentum conserved |
| T6 | dilute sphere USF (α = 0.7, φ = 0.01) through the full `fresh_usf` production template | Boltzmann DSMC (`dsmc_usf_spheres.py`): T/(mγ̇²d²) = 189.5, P* = (1.424, 0.766, 0.810, −0.501) plus the Enskog collisional pressure 2(1+α)φg0 on the diagonal; Enskog collision frequency; realized e in the gas; continuous vs collision-by-collision collisional stress; P_xy = P_yx; clean sensors |
| T7 | `fix nve/spherocyl rot_sllod` | must stop with an explanation |
| T8 | 1 vs 4 MPI ranks | identical stress and collision counts |
| T9 | fast shear with several box flips | peculiar momentum conserved through flips (unpatched `fix deform` gives jumps of m γ̇ L_y per atom) |

`dsmc_usf_spheres.py` is a homogeneous DSMC of the Boltzmann equation for inelastic
hard spheres in USF. Option k = 2 doubles the streaming term. It reproduces the
old SLLOD-force DEM: T is four times higher and P* is unchanged.
