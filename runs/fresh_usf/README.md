# Fresh USF production runs (smooth spherocylinders, φ = 0.01)

All cases use the corrected code and physics:

| item | old runs (`runs/USF`, `runs/sphcyl_USF_*`) | fresh runs |
|---|---|---|
| streaming | `fix addforce −γ̇ m v_y` (counts the shear coupling twice → T four times too high) | none: Lees–Edwards (`fix deform … remap v`) + Newton |
| rotation | `rot_sllod` fictitious torque | removed (hard error if requested) |
| restitution | η from the linear-spring formula (α = 0.50 gave 0.54) | η = √5·√(kn m*)·\|ln α\|/√(π²+ln²α), exact for Hertz |
| virial | contact-point branch c_i − c_j | centre-of-mass branch x_i − x_j |
| contact history | lost at every neighbour rebuild | kept (`beyond_contact = 1`) |
| box flips | ±γ̇L_y velocity kick to atoms crossing y since the last rebuild (LAMMPS `fix deform` bug) | patched in `src/fix_deform.cpp` |
| hardness | fixed kn, typical overlap 1–3 % d | kn, dt recalibrated at the steady T: mean overlap 0.4 % d, ≈40 steps per contact |
| production | `jump SELF` failed under `<`, only 1 of 3 blocks ran | explicit blocks, restart after each |

## Running

```bash
python3 generate_fresh_usf.py --ARs 2 1.5 2.5 3 1      # writes AR*/a*/ and cases.txt
./run_local_queue.sh 3 4                               # 3 cases at a time, 4 MPI ranks each
# cluster (Negishi): see negishi/README.md  (build, prepare_cases.py, jobs/submit_all.sh)
python3 postprocess_fresh_usf.py                       # summary.csv + diagnostics_report.txt
```

Always run LAMMPS with `-in in.usf` (not `< in.usf`).

Each case runs these stages in order:
1. **A: elastic equilibration**, no shear, 5 collisions per particle. This is a built-in energy-conservation test (`diag_equil.*`).
2. **B: shear transient 1**. Afterwards kn and dt are recalibrated from the measured mean overlap and contact duration.
3. **C: transient 2**, then a second recalibration.
4. **D: transient 3**, at fixed kn and dt.
5. **E: production**, `nblocks` × `strain_block` strain, with `restart_block_k.bin` written after each block.

Only the `prod.*` files are production data.

## Outputs of one case

| file | content (column names are in the `# 1:…` header line) |
|---|---|
| `prod.stress` | per window (0.5 strain): T_tr, T_rot, **kinetic K_ab (6)**, **collisional C_ab (9, continuous)**, **collisional CE_ab (9, from completed collisions)**, a2_tr, a2_rot |
| `prod.energy` | E_tr, E_rot, U_el, shear work (kinetic, collisional), dissipation W_nc, energy residual, peculiar-momentum drift, max\|c\|/c_rms, mean spin ⟨ω⟩, rotational tensor R_ab |
| `prod.coll` | collision frequency, restitution realised in the gas, contact duration and overlap, fraction binary/multi-body/long/gaining, energy transfer trans↔rot per collision, same-partner recollisions, end-cap contacts, ⟨\|u_i·u_j\|⟩, free-flight statistics, per-particle collision-count dispersion, coordination P(z), collision fabric ⟨nn⟩ |
| `prod.struct` | S(k) for 10 box modes, x-momentum (shear-band) and energy modes, density dispersion D(g³), nematic tensor Q_ab, λ_max and S2 = 1.5 λ_max |
| `prod.sensors` | one line per window with flags (see below) |
| `prod.events` | 5 % of particle pairs: every collision of those pairs (pre/post normal velocities, energies, overlap, lever arms, …) |
| `prof_y.dat`, `prof_z.dat` | profiles along the gradient (y) and vorticity (z): density, v, peculiar c_x, kinetic tensor, E_rot, orientation u_a u_b |
| `rdf.dat` | centre–centre g(r) |
| `snapshots.lammpstrj.gz` | every 5 strain: positions, velocities, quaternions, angular momenta, contacts, collision counts |
| `diag_equil/tr1/tr2/tr3.*` | same diagnostics for the non-production stages |
| `log.lammps` | includes the `RECALIBRATE:` lines (kn, η, dt, measured overlaps) |

### Stress definitions

```
K_ab = <Σ_i m c_ia c_ib>/V,       c = v − γ̇ (y − ylo) x̂
C_ab = <Σ_pairs (x_i − x_j)_a F_ij,b>/V    (branch index first)
P_ab = K_ab + C_ab ,   P*_ab = P_ab/(n T_tr) ,   T_tr = <Σ m c²>/(3N)
```

Every window value is an exact time average, accumulated every step, so soft contacts are never aliased. C is accumulated continuously at the force stage. CE is built from the integral of (x_i − x_j)⊗F over each collision, from first overlap to separation. The two estimators must agree up to collisions that straddle window boundaries: this is a built-in check of the collision bookkeeping.

In steady state the antisymmetric part C_xy − C_yx must average to zero, because ⟨Σ torques⟩ = d⟨spin⟩/dt = 0.

## Sensors (`prod.sensors`, and WARNING lines in the log)

| flag | meaning | why it matters |
|---|---|---|
| E | energy residual > 2 % of the window's work terms | dE = W_shear + W_nc exactly, for elastic or dissipative runs: a runaway, a missing force, or broken integration |
| M | peculiar momentum drift > 10⁻⁶ √(NmT) | conserved exactly under correct Lees–Edwards: flags broken boundary or velocity remapping |
| H | max overlap > 2 % of d | not in the hard-particle limit |
| R | > 1 % of contacts resolved by < 15 steps | time step too large |
| B | > 5 % multi-body collisions | DSMC is strictly binary |
| C | mean S(k) over 10 modes > 2.5, or D(4³) > 1.5 | density inhomogeneity (clustering, shear bands) |
| G | > 1 % of collisions gained pair energy (dissipative runs) | the translational-v_n damping can inject energy in rotation-driven contacts |
| V | a particle faster than 8 c_rms | runaway particle |

## If a case deviates from spheres or DSMC

| symptom | look at |
|---|---|
| P* or θ differ from DSMC | `prod.coll` e_c (restitution actually realised in the gas), frac_multibody, frac_long, frac_same_partner (correlated recollisions: DSMC assumes molecular chaos), dE_rel_tr/dE_rot per collision (energy exchange trans↔rot) |
| AR dependence | `prod.stress` collisional C vs kinetic K (share of P), `prod.coll` frac_endcap, ⟨\|u_i·u_j\|⟩, fabric ⟨nn⟩; `prod.struct` S2 and Q |
| T drifts or no steady state | `prod.energy` W_shear vs W_nc, resid; the first-vs-second-half drift in `diagnostics_report.txt` |
| clustering suspected | `prod.struct` S010, S001 (gradient and vorticity modes), D2/D4/D8, Jx010 (shear banding); `prod.coll` collcount_disp and freeflight_cv2 (> 1 means heterogeneous collision rates), P(z ≥ 2); `prof_y.dat`/`prof_z.dat` density; `rdf.dat` contact value; snapshots |
| non-Maxwellian velocities | `prod.stress` a2_tr, a2_rot (kurtosis) |
| mean spin | `prod.energy` wz vs −γ̇/2 |

The box is L = 64 d, only about 5 mean free paths. A box this small suppresses long-wavelength USF instabilities, so homogeneous USF is what DSMC also assumes. To test for clustering proper, rerun one case with `--N` ≈ 8× (2L).
