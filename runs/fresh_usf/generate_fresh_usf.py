#!/usr/bin/env python3
"""
Generate the fresh USF case directories:  <outdir>/AR<AR>/a<alpha>/

Each case gets
  params.in                 case parameters (included by in.usf)
  spherocyl.data            random non-overlapping rods, T_tr = T_rot = T_guess
  in.usf, recalibrate*.in   copied from templates/
  production_blocks.in      explicit production blocks with a restart after each

and the top level gets cases.txt and run_local_queue.sh (cluster: see negishi/README.md).

T_guess only sets the starting temperature and the initial stiffness; kn and
dt are recalibrated from measured overlaps during the transient.  It is built
from the dilute Grad solution for smooth spheres, corrected to the Boltzmann
DSMC values (runs/regression/v2/dsmc_usf_spheres.py), and scaled to rods with
the Minkowski mean collision cross-section (collision rate ~ n <S_ex>/4).
"""
import argparse
import math
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def grad_astar2(alpha):
    zeta = 5.0 / 12.0 * (1.0 - alpha**2)
    beta = (1.0 + alpha) * (3.0 - alpha) / 4.0
    return 1.5 * zeta * beta**2 / (beta - zeta)


def sphere_T_over_m(alpha, phi):
    """T/(m gdot^2 d^2) for dilute smooth spheres (Boltzmann), DSMC-corrected."""
    corr = 0.81 + (min(max(alpha, 0.5), 0.9) - 0.5) * (0.96 - 0.81) / 0.4
    return corr * math.pi / (grad_astar2(alpha) * (256.0 / 25.0) * 36.0 * phi**2)


def rod_geometry(AR, d=1.0):
    R = 0.5 * d
    Lc = (AR - 1.0) * d
    Vp = 4.0 / 3.0 * math.pi * R**3 + math.pi * R**2 * Lc
    S = 4.0 * math.pi * R**2 + 2.0 * math.pi * R * Lc
    M = 4.0 * math.pi * R + math.pi * Lc
    sigma_eff = (2.0 * S + M**2 / (2.0 * math.pi)) / 4.0    # mean cross-section
    return R, Lc, Vp, sigma_eff


def T_guess(AR, alpha, phi, gdot, rho, d=1.0):
    _, _, Vp, sig = rod_geometry(AR, d)
    _, _, Vs, sigs = rod_geometry(1.0, d)
    m = rho * Vp
    ratio = ((phi / Vs) * sigs / ((phi / Vp) * sig))**2
    rot = 1.15 if AR > 1.0 else 1.0
    return sphere_T_over_m(alpha, phi) * ratio * rot * m * (gdot * d)**2


def nu_guess(AR, T, phi, rho, d=1.0):
    _, _, Vp, sig = rod_geometry(AR, d)
    m = rho * Vp
    n = phi / Vp
    return 1.3 * n * sig * 4.0 * math.sqrt(T / (math.pi * m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ARs", type=float, nargs="+", default=[2.0])
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    ap.add_argument("--phi", type=float, default=0.01)
    ap.add_argument("--rho", type=float, default=0.763944)
    ap.add_argument("--gdot", type=float, default=1.8)
    ap.add_argument("--N", type=int, default=None,
                    help="particles (default: MFIX convention, box L=64 d)")
    ap.add_argument("--delta_target", type=float, default=0.004,
                    help="mean max overlap / d over collisions")
    ap.add_argument("--Nc_target", type=float, default=40.0,
                    help="mean steps per contact")
    ap.add_argument("--strain_tr", type=float, nargs=3, default=[15.0, 15.0, 10.0])
    ap.add_argument("--nblocks", type=int, default=4)
    ap.add_argument("--strain_block", type=float, default=30.0)
    ap.add_argument("--strain_window", type=float, default=0.5)
    ap.add_argument("--equil_collisions", type=float, default=5.0)
    ap.add_argument("--log_fraction", type=float, default=0.05)
    ap.add_argument("--skin", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--outdir", default=HERE)
    ap.add_argument("--lmp", default=os.path.join(HERE, "bin", "lmp_mpi"))
    ap.add_argument("--np", type=int, default=4, help="MPI ranks per case")
    args = ap.parse_args()

    tdir = os.path.join(HERE, "templates")
    # queue order: ARs in the order given, weakest dissipation (longest run) first
    alphas_sorted = sorted(args.alphas, reverse=True)
    gen = os.path.join(HERE, "tools", "gen_spherocyl_lammps.py")
    cases = []
    for ia, AR in enumerate(args.ARs):
        for alpha in alphas_sorted:
            ie = args.alphas.index(alpha)
            if alpha >= 1.0:
                print(f"skip alpha={alpha}: elastic USF has no steady state")
                continue
            cdir = os.path.join(args.outdir, f"AR{AR:g}", f"a{alpha:.2f}")
            os.makedirs(cdir, exist_ok=True)
            Tg = T_guess(AR, alpha, args.phi, args.gdot, args.rho)
            nug = nu_guess(AR, Tg, args.phi, args.rho)
            seed = args.seed + 1000 * ia + ie
            tr = list(args.strain_tr)
            if alpha >= 0.9:          # slower relaxation for weak dissipation
                tr = [max(tr[0], 20.0), max(tr[1], 20.0), tr[2]]

            cmd = [sys.executable, gen, "--AR", str(AR), "--phi", str(args.phi),
                   "--rho", str(args.rho), "--T_tr", f"{Tg:.8g}", "--T_rot", f"{Tg:.8g}",
                   "--seed", str(seed), "--outfile", os.path.join(cdir, "spherocyl.data")]
            if args.N:
                cmd += ["--N", str(args.N)]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

            with open(os.path.join(cdir, "params.in"), "w") as f:
                f.write(f"""# generated by generate_fresh_usf.py
variable        AR               equal {AR}
variable        alpha            equal {alpha}
variable        phi              equal {args.phi}
variable        d                equal 1.0
variable        rho_p            equal {args.rho}
variable        gdot             equal {args.gdot}
variable        T_guess          equal {Tg:.8g}
variable        nu_guess         equal {nug:.8g}
variable        equil_collisions equal {args.equil_collisions}
variable        delta_target     equal {args.delta_target}
variable        Nc_target        equal {args.Nc_target}
variable        strain_tr1       equal {tr[0]}
variable        strain_tr2       equal {tr[1]}
variable        strain_tr3       equal {tr[2]}
variable        nblocks          equal {args.nblocks}
variable        strain_block     equal {args.strain_block}
variable        strain_window    equal {args.strain_window}
variable        log_fraction     equal {args.log_fraction}
variable        skin             equal {args.skin}
variable        seed             equal {seed}
variable        datafile         string spherocyl.data
""")
            for fn in ("in.usf", "recalibrate.in", "recalibrate_apply.in"):
                shutil.copy(os.path.join(tdir, fn), cdir)
            with open(os.path.join(cdir, "production_blocks.in"), "w") as f:
                f.write("# production blocks (explicit; no jump/label needed)\n")
                for b in range(1, args.nblocks + 1):
                    f.write(f"print           \"PRODUCTION BLOCK {b}/{args.nblocks} start\"\n")
                    f.write("run             ${steps_block}\n")
                    f.write(f"write_restart   restart_block_{b}.bin\n")
                    f.write(f"print           \"PRODUCTION BLOCK {b}/{args.nblocks} done\"\n")
            rel = os.path.relpath(cdir, args.outdir)
            cases.append(rel)
            print(f"{rel:14s} T_guess={Tg:10.4g}  nu_guess={nug:8.4g}  seed={seed}")

    with open(os.path.join(args.outdir, "cases.txt"), "w") as f:
        f.write("\n".join(cases) + "\n")

    lmp = os.path.abspath(args.lmp)
    with open(os.path.join(args.outdir, "run_local_queue.sh"), "w") as f:
        f.write(f"""#!/bin/bash
# Run every case in cases.txt on this workstation, JOBS cases at a time,
# NP MPI ranks each.  Finished cases (log contains DONE) are skipped, so the
# script can simply be restarted.  Usage: ./run_local_queue.sh [JOBS] [NP]
cd "$(dirname "$0")"
JOBS=${{1:-3}}
NP=${{2:-{args.np}}}
LMP={lmp}
# Several runners may work on the same list: each case is claimed with a lock
# directory (.lock); delete stale locks after a crash before restarting.
run_case() {{
  cd "$1" || exit 1
  if grep -q "^DONE" log.lammps 2>/dev/null; then echo "skip $1 (done)"; exit 0; fi
  mkdir .lock 2>/dev/null || {{ echo "skip $1 (claimed by another runner)"; exit 0; }}
  echo "start $1 $(date)"
  mpirun --oversubscribe -np $NP $LMP -in in.usf -log log.lammps -screen none
  echo "end   $1 $(date) exit=$?"
  grep -q "^DONE" log.lammps && rmdir .lock
}}
export -f run_case
export NP LMP
xargs -a cases.txt -P $JOBS -I{{}} bash -c 'run_case {{}}' >> queue.log 2>&1
""")
    os.chmod(os.path.join(args.outdir, "run_local_queue.sh"), 0o755)

    print(f"\n{len(cases)} cases written to {args.outdir}")


if __name__ == "__main__":
    main()
