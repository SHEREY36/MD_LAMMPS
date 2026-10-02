#!/usr/bin/env python3
"""
Generate the fresh HCS case directories:  <outdir>/AR<AR>/a<alpha>/s<seed>/

The homogeneous cooling state (HCS) uses the corrected DEM of runs/fresh_usf
(exact Hertz damping, contact history kept, hard-particle recalibration), see
README.md.  Each case gets

  params.in          case parameters (included by in.hcs)
  spherocyl.data     random non-overlapping rods, T_tr = T_rot = T0
  in.hcs, recalibrate.in, recalibrate_apply.in, rescale_temperatures.in,
  harden.in          copied from templates/
  hcs_blocks.in      explicit cooling blocks (no jump/label)

and the top level gets cases.txt and run_local_queue.sh (cluster: negishi/README.md).

Default sweep: AR 1.1, 1.5, 2, 2.5, 3 x alpha 0.50-0.95 (step 0.05) plus the
Rubio-Largo et al. (Physica A 443, 2016) restitutions 0.88 and 0.96, and AR 1.25
at 0.88/0.90/0.96 for the same literature check; two seeds per case.
"""
import argparse
import math
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

CORE_ALPHAS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
RL_ALPHAS = [0.88, 0.9, 0.96]          # Rubio-Largo et al. (2016), Fig. 2(b)


def rod_geometry(AR, d=1.0):
    R = 0.5 * d
    Lc = (AR - 1.0) * d
    Vp = 4.0 / 3.0 * math.pi * R**3 + math.pi * R**2 * Lc
    S = 4.0 * math.pi * R**2 + 2.0 * math.pi * R * Lc
    M = 4.0 * math.pi * R + math.pi * Lc
    sigma_eff = (2.0 * S + M**2 / (2.0 * math.pi)) / 4.0    # mean cross-section
    return R, Lc, Vp, sigma_eff


def nu_guess(AR, T, phi, rho, d=1.0):
    """Contacts per particle per unit time in an equilibrium gas (rough)."""
    _, _, Vp, sig = rod_geometry(AR, d)
    m = rho * Vp
    n = phi / Vp
    return 1.3 * n * sig * 4.0 * math.sqrt(T / (math.pi * m))


def default_cases():
    cases = []
    for AR in (1.1, 1.5, 2.0, 2.5, 3.0):
        for a in sorted(set(CORE_ALPHAS + RL_ALPHAS), reverse=True):
            cases.append((AR, a))
    for a in RL_ALPHAS:
        cases.append((1.25, a))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ARs", type=float, nargs="+", default=None,
                    help="restrict to these aspect ratios (with --alphas: full product)")
    ap.add_argument("--alphas", type=float, nargs="+", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--phi", type=float, default=0.01)
    ap.add_argument("--rho", type=float, default=0.763944)
    ap.add_argument("--T0", type=float, default=1.0,
                    help="T_tr = T_rot at the start of cooling")
    ap.add_argument("--N", type=int, default=None,
                    help="particles (default: MFIX convention, box L = 64 d)")
    ap.add_argument("--delta_target", type=float, default=0.004,
                    help="mean max overlap / d over collisions")
    ap.add_argument("--Nc_target", type=float, default=40.0,
                    help="median steps per contact")
    ap.add_argument("--equil_collisions", type=float, default=5.0,
                    help="contacts per particle in the elastic stage A")
    ap.add_argument("--tau_block", type=float, default=2.5,
                    help="contacts per particle per cooling block (stiffness rescaled between blocks)")
    ap.add_argument("--nblocks", type=int, default=10)
    ap.add_argument("--tau_window", type=float, default=0.05,
                    help="contacts per particle per diagnostic window")
    ap.add_argument("--tau_snapshot", type=float, default=0.5,
                    help="contacts per particle between velocity snapshots")
    ap.add_argument("--log_fraction", type=float, default=0.0)
    ap.add_argument("--skin", type=float, default=0.5)
    ap.add_argument("--seed_base", type=int, default=20261002)
    ap.add_argument("--outdir", default=HERE)
    ap.add_argument("--lmp", default=os.path.join(HERE, "bin", "lmp_mpi"))
    ap.add_argument("--np", type=int, default=4, help="MPI ranks per case")
    args = ap.parse_args()

    if args.ARs or args.alphas:
        ARs = args.ARs or [1.1, 1.5, 2.0, 2.5, 3.0]
        alphas = args.alphas or sorted(set(CORE_ALPHAS + RL_ALPHAS), reverse=True)
        pairs = [(AR, a) for AR in ARs for a in sorted(alphas, reverse=True)]
    else:
        pairs = default_cases()

    tdir = os.path.join(HERE, "templates")
    gen = os.path.join(HERE, "tools", "gen_spherocyl_lammps.py")
    cases = []
    for AR, alpha in pairs:
        if alpha >= 1.0:
            print(f"skip alpha={alpha}: stage A already is the elastic run")
            continue
        for s in args.seeds:
            cdir = os.path.join(args.outdir, f"AR{AR:g}", f"a{alpha:.2f}", f"s{s}")
            os.makedirs(cdir, exist_ok=True)
            nug = nu_guess(AR, args.T0, args.phi, args.rho)
            # deterministic, distinct seed per (AR, alpha, replica)
            seed = args.seed_base + int(round(100 * AR)) * 10000 + int(round(100 * alpha)) * 10 + s
            cmd = [sys.executable, gen, "--AR", str(AR), "--phi", str(args.phi),
                   "--rho", str(args.rho), "--T_tr", f"{args.T0:.8g}", "--T_rot", f"{args.T0:.8g}",
                   "--seed", str(seed), "--outfile", os.path.join(cdir, "spherocyl.data")]
            if args.N:
                cmd += ["--N", str(args.N)]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

            with open(os.path.join(cdir, "params.in"), "w") as f:
                f.write(f"""# generated by generate_fresh_hcs.py
variable        AR               equal {AR}
variable        alpha            equal {alpha}
variable        phi              equal {args.phi}
variable        d                equal 1.0
variable        rho_p            equal {args.rho}
variable        T0               equal {args.T0}
variable        nu_guess         equal {nug:.8g}
variable        equil_collisions equal {args.equil_collisions}
variable        delta_target     equal {args.delta_target}
variable        Nc_target        equal {args.Nc_target}
variable        tau_block        equal {args.tau_block}
variable        nblocks          equal {args.nblocks}
variable        tau_window       equal {args.tau_window}
variable        tau_snapshot     equal {args.tau_snapshot}
variable        log_fraction     equal {args.log_fraction}
variable        skin             equal {args.skin}
variable        seed             equal {seed}
variable        datafile         string spherocyl.data
""")
            for fn in ("in.hcs", "recalibrate.in", "recalibrate_apply.in",
                       "rescale_temperatures.in", "harden.in"):
                shutil.copy(os.path.join(tdir, fn), cdir)
            with open(os.path.join(cdir, "hcs_blocks.in"), "w") as f:
                f.write("# cooling blocks (explicit; no jump/label needed).\n")
                f.write("# Between blocks kn ~ T_tr and dt ~ T_tr^-1/2 (harden.in) keep the\n")
                f.write("# mean overlap and the steps per contact at their calibrated values.\n")
                for b in range(1, args.nblocks + 1):
                    f.write(f"print           \"COOLING BLOCK {b}/{args.nblocks} start\"\n")
                    f.write("run             ${steps_block}\n")
                    if b < args.nblocks:
                        f.write("include         harden.in\n")
                    f.write(f"print           \"COOLING BLOCK {b}/{args.nblocks} done\"\n")
            rel = os.path.relpath(cdir, args.outdir)
            cases.append(rel)
            print(f"{rel:18s} nu_guess={nug:8.4g}  seed={seed}")

    with open(os.path.join(args.outdir, "cases.txt"), "w") as f:
        f.write("\n".join(cases) + "\n")

    lmp = os.path.abspath(args.lmp)
    with open(os.path.join(args.outdir, "run_local_queue.sh"), "w") as f:
        f.write(f"""#!/bin/bash
# Run every case in cases.txt on this workstation, JOBS cases at a time,
# NP MPI ranks each.  Finished cases (log contains DONE) are skipped, so the
# script can simply be restarted.  Usage: ./run_local_queue.sh [JOBS] [NP] [LMP]
cd "$(dirname "$0")"
JOBS=${{1:-3}}
NP=${{2:-{args.np}}}
LMP=${{3:-{lmp}}}
# Several runners may work on the same list: each case is claimed with a lock
# directory (.lock); delete stale locks after a crash before restarting.
run_case() {{
  cd "$1" || exit 1
  if grep -q "^DONE" log.lammps 2>/dev/null; then echo "skip $1 (done)"; exit 0; fi
  mkdir .lock 2>/dev/null || {{ echo "skip $1 (claimed by another runner)"; exit 0; }}
  echo "start $1 $(date)"
  mpirun --oversubscribe -np $NP $LMP -in in.hcs -log log.lammps -screen none
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
