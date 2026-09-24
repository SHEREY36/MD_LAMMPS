#!/usr/bin/env python3
"""
Prepare the fresh USF production for Negishi (Purdue RCAC).

  python3 prepare_cases.py                      # generate all 50 cases + job lists
  python3 prepare_cases.py --estimate-only      # only print the cost table
  python3 prepare_cases.py --ARs 2 --alphas 0.5 0.55 0.6 0.65   # a subset

It (1) runs ../generate_fresh_usf.py (deterministic seeds: identical data files
on every machine), (2) predicts the cost of every case, (3) chooses the number
of MPI ranks and the wall time per case, (4) groups the cases into a few job
arrays (jobs/*.txt) and writes jobs/submit_all.sh.

Cost model (calibrated on the finished local AR=2 runs, 23 Sep 2026):
  * the recalibrated time step satisfies dt ~= DT_COEF * sqrt(m / T), with the
    production temperature approximated by T_guess (conservative: gives more steps);
  * total steps = strain_total / (gdot * dt);
  * throughput RATE atom-steps/s per MPI rank at 4 ranks, parallel efficiency
    EFF[np] relative to 4 ranks (dilute gas, N = 1.3k-5k atoms, box 64 d).
The first finished jobs on Negishi give the true numbers: compare the
'Loop time' lines in their log.lammps with jobs/cost_table.txt and rerun this
script with --rate/--dt-coef if the predictions are off.
"""
import argparse
import math
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FRESH = os.path.dirname(HERE)

DT_COEF = 1.6e-4          # dt * sqrt(T/m) measured for well-resolved cases
RATE = 1.4e6              # atom-steps / s / rank, measured on a 6-core (12-thread) Intel
                          # workstation running 3 cases at once: conservative for Negishi
EFF = {4: 1.00, 8: 0.85, 16: 0.65}
TIME_BINS = [4, 8, 12, 24, 48, 72]


def read_params(case):
    p = {}
    for line in open(os.path.join(case, "params.in")):
        t = line.split()
        if len(t) >= 4 and t[0] == "variable" and t[2] == "equal":
            p[t[1]] = float(t[3])
    with open(os.path.join(case, "spherocyl.data")) as f:
        for line in f:
            m = re.match(r"\s*(\d+)\s+atoms", line)
            if m:
                p["N"] = int(m.group(1))
                break
    return p


def predict(p, dt_coef, rate):
    R, Lc = 0.5, p["AR"] - 1.0
    m = p["rho_p"] * (4.0 / 3.0 * math.pi * R**3 + math.pi * R**2 * Lc)
    dt = dt_coef * math.sqrt(m / p["T_guess"])
    strain = p["strain_tr1"] + p["strain_tr2"] + p["strain_tr3"] + p["nblocks"] * p["strain_block"] + 2.0
    steps = strain / (p["gdot"] * dt)
    work = steps * p["N"]
    wall = {np_: work / (rate * np_ * EFF[np_]) / 3600.0 for np_ in EFF}
    return steps, work, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ARs", type=float, nargs="+", default=[2.0, 1.5, 2.5, 3.0, 1.0])
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    ap.add_argument("--estimate-only", action="store_true")
    ap.add_argument("--max-wall", type=float, default=4.0,
                    help="target predicted wall hours per case; cases above it get 8 or 16 ranks")
    ap.add_argument("--cores", type=int, default=256, help="cores in the account (slist)")
    ap.add_argument("--safety", type=float, default=2.0, help="requested time / predicted time")
    ap.add_argument("--dt-coef", type=float, default=DT_COEF)
    ap.add_argument("--rate", type=float, default=RATE)
    ap.add_argument("--standby", action="store_true",
                    help="send arrays whose time limit is <= 4 h to the free standby QOS")
    ap.add_argument("--account", default="morri353")
    a = ap.parse_args()

    if not a.estimate_only:
        cmd = [sys.executable, os.path.join(FRESH, "generate_fresh_usf.py"), "--ARs"] + \
              [str(x) for x in a.ARs] + ["--alphas"] + [str(x) for x in a.alphas]
        print("generating cases:", " ".join(cmd[2:]))
        subprocess.run(cmd, check=True, cwd=FRESH, stdout=subprocess.DEVNULL)

    cases = []
    for AR in a.ARs:
        for al in a.alphas:
            c = os.path.join(FRESH, f"AR{AR:g}", f"a{al:.2f}")
            if os.path.exists(os.path.join(c, "params.in")):
                cases.append(os.path.relpath(c, FRESH))

    rows = []
    for c in cases:
        p = read_params(os.path.join(FRESH, c))
        steps, work, wall = predict(p, a.dt_coef, a.rate)
        np_ = next((n for n in (4, 8, 16) if wall[n] <= a.max_wall), 16)
        req = a.safety * wall[np_]
        tbin = next((t for t in TIME_BINS if t >= max(req, 2.0)), TIME_BINS[-1])
        rows.append(dict(case=c, N=p["N"], steps=steps, wall=wall[np_], np=np_, t=tbin,
                         core_h=wall[np_] * np_))

    os.makedirs(os.path.join(HERE, "jobs"), exist_ok=True)
    lines = [f"{'case':12s} {'N':>5s} {'steps':>10s} {'ranks':>5s} {'pred_wall_h':>11s} {'req_h':>5s} {'core_h':>7s}"]
    for r in rows:
        lines.append(f"{r['case']:12s} {r['N']:5d} {r['steps']:10.3g} {r['np']:5d} {r['wall']:11.2f} {r['t']:5d} {r['core_h']:7.1f}")
    tot = sum(r["core_h"] for r in rows)
    ncores = sum(r["np"] for r in rows)
    lines.append(f"\n{len(rows)} cases, predicted total {tot:.0f} core-hours; longest predicted case "
                 f"{max(r['wall'] for r in rows):.1f} h; all cases at once need {ncores} cores "
                 f"(account: {a.cores})")
    if ncores > a.cores:
        lines.append("  -> more than the account: the last jobs start as soon as the short cases finish")
    table = "\n".join(lines)
    print(table)
    open(os.path.join(HERE, "jobs", "cost_table.txt"), "w").write(table + "\n")
    if a.estimate_only:
        return

    # submit the widest and longest arrays first; inside an array, longest case first
    groups = {}
    for r in sorted(rows, key=lambda r: -r["wall"]):
        groups.setdefault((r["np"], r["t"]), []).append(r["case"])
    sub = ["#!/bin/bash",
           "# Submit every job array (generated by prepare_cases.py). Run from anywhere.",
           'cd "$(dirname "${BASH_SOURCE[0]}")/.."',
           "mkdir -p logs"]
    order = sorted(groups.items(), key=lambda kv: (-kv[0][0], -kv[0][1]))
    for (np_, t), cl in order:
        name = f"np{np_}_t{t}h"
        fn = os.path.join(HERE, "jobs", name + ".txt")
        open(fn, "w").write("\n".join(cl) + "\n")
        qos = "-q standby " if (a.standby and t <= 4) else ""
        sub.append(f"sbatch -A {a.account} -p cpu {qos}-J usf_{name} --array=0-{len(cl) - 1} "
                   f"--ntasks={np_} --time={t:02d}:00:00 job_case.sbatch jobs/{name}.txt")
    open(os.path.join(HERE, "jobs", "submit_all.sh"), "w").write("\n".join(sub) + "\n")
    os.chmod(os.path.join(HERE, "jobs", "submit_all.sh"), 0o755)
    print("\njob arrays:")
    for (np_, t), cl in order:
        print(f"  {len(cl):3d} cases x {np_:2d} ranks, time limit {t} h")
    print(f"\nsubmit with:  bash {os.path.relpath(os.path.join(HERE, 'jobs', 'submit_all.sh'))}")


if __name__ == "__main__":
    main()
