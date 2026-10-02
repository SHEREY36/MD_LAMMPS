#!/usr/bin/env python3
"""
Reduce the fresh HCS runs to compact time series and one summary table.

  python3 postprocess_fresh_hcs.py [--root .] [--tau-relax 8]

Writes
  analysis/hcs_timeseries/AR<AR>_a<alpha>_s<seed>.csv
      t, tau_c, T_tr, T_rot, theta, T, nu, a2_tr, a2_rot, e_c   (one row per window)
  analysis/summary.csv
      one row per finished case, see COLUMNS below

Definitions (window values come from fix spherocyl/diag):
  tau_c   cumulative contacts per particle, sum of 2 N_start / N
  T       (3 T_tr + 2 T_rot)/5
  gamma   d ln T / d tau_c, least squares over tau_c >= tau_relax
          (tau_relax = 8 for AR >= 1.5 and 60 for the near-spheres, AR <= 1.3)
  alpha_eff = sqrt(1 + 5 gamma): restitution of a gas with five quadratic
          degrees of freedom whose collisions remove (1 - alpha^2) of the
          normal-channel energy, cooling at the same rate per collision
  theta_star mean of T_tr/T_rot over tau_c >= tau_relax, error from 5 block means
  sigma_eff = nu / (n <g>), <g> = 4 sqrt(T_tr/(pi m)): contacts per unit
          relative flux, in units of pi d^2 (window average)
"""
import argparse
import csv
import glob
import math
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COLUMNS = ["AR", "alpha", "seed", "N", "phi", "m", "tau_end", "t_end",
           "theta_star", "theta_star_err", "gamma", "gamma_err", "alpha_eff",
           "sigma_eff", "sigma_eff_err", "e_c", "e_tr", "frac_multibody",
           "frac_same_partner", "frac_gain", "dmax_mean", "frac_windows_C",
           "a2_tr", "a2_rot", "max_resid_rel"]


def read_table(path):
    header = None
    with open(path) as f:
        for line in f:
            if line.startswith("# 1:"):
                header = [h.split(":", 1)[1] for h in line[2:].split()]
                break
    rows = [l.split() for l in open(path) if l.strip() and not l.startswith("#")]
    return header, rows


def column(header, rows, name, dtype=float):
    i = header.index(name)
    return np.array([dtype(r[i]) for r in rows])


def params(case):
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


def block_error(x, nblock=5):
    if len(x) < 2 * nblock:
        return float("nan")
    b = np.array([c.mean() for c in np.array_split(x, nblock)])
    return float(b.std(ddof=1) / math.sqrt(nblock))


def reduce_case(case, tau_relax):
    p = params(case)
    N, AR, phi, rho = p["N"], p["AR"], p["phi"], p["rho_p"]
    R, Lc = 0.5, AR - 1.0
    Vp = 4.0 / 3.0 * math.pi * R**3 + math.pi * R**2 * Lc
    m, n = rho * Vp, phi / Vp
    hc, rc = read_table(os.path.join(case, "hcs.coll"))
    hs, rs = read_table(os.path.join(case, "hcs.stress"))
    he, re_ = read_table(os.path.join(case, "hcs.energy"))
    hf, rf = read_table(os.path.join(case, "hcs.sensors"))
    k = min(len(rc), len(rs))
    t = column(hc, rc, "time")[:k]
    tau = np.cumsum(2.0 * column(hc, rc, "N_start") / N)[:k]
    nu = column(hc, rc, "nu")[:k]
    ttr, trot = column(hs, rs, "T_tr")[:k], column(hs, rs, "T_rot")[:k]
    a2t, a2r = column(hs, rs, "a2_tr")[:k], column(hs, rs, "a2_rot")[:k]
    ec, etr = column(hc, rc, "e_c")[:k], column(hc, rc, "e_tr")[:k]
    T = (3.0 * ttr + 2.0 * trot) / 5.0
    theta = ttr / trot
    g = 4.0 * np.sqrt(ttr / (math.pi * m))
    sig = nu / (n * g) / math.pi                     # units of pi d^2 (d = 1)

    os.makedirs(os.path.join(HERE, "analysis", "hcs_timeseries"), exist_ok=True)
    replica = os.path.basename(os.path.normpath(case))          # s1, s2, ...
    name = f"AR{AR:g}_a{p['alpha']:.2f}_{replica}"
    out = os.path.join(HERE, "analysis", "hcs_timeseries", name + ".csv")
    np.savetxt(out, np.column_stack([t, tau, ttr, trot, theta, T, nu, a2t, a2r, ec]),
               delimiter=",", header="t,tau_c,T_tr,T_rot,theta,T,nu,a2_tr,a2_rot,e_c",
               comments="", fmt="%.8g")

    late = tau >= tau_relax
    fit = np.polyfit(tau[late], np.log(T[late]), 1, cov=True)
    gamma, gamma_err = fit[0][0], math.sqrt(fit[1][0, 0])
    w = column(hc, rc, "N_end")[:k]
    wmean = lambda x: float(np.sum(w * x) / max(np.sum(w), 1.0))
    flags = column(hf, rf, "flags", str)
    resid = np.abs(column(he, re_, "resid_rel"))
    return {
        "AR": AR, "alpha": p["alpha"], "seed": replica, "N": N, "phi": phi, "m": m,
        "tau_end": tau[-1], "t_end": t[-1],
        "theta_star": float(theta[late].mean()), "theta_star_err": block_error(theta[late]),
        "gamma": gamma, "gamma_err": gamma_err,
        "alpha_eff": math.sqrt(max(1.0 + 5.0 * gamma, 0.0)),
        "sigma_eff": float(sig.mean()), "sigma_eff_err": block_error(sig),
        "e_c": wmean(ec), "e_tr": wmean(etr),
        "frac_multibody": wmean(column(hc, rc, "frac_multibody")[:k]),
        "frac_same_partner": wmean(column(hc, rc, "frac_same_partner")[:k]),
        "frac_gain": wmean(column(hc, rc, "frac_gain")[:k]),
        "dmax_mean": wmean(column(hc, rc, "dmax_mean")[:k]),
        "frac_windows_C": float(np.mean(["C" in f for f in flags])),
        "a2_tr": float(a2t[late].mean()), "a2_rot": float(a2r[late].mean()),
        "max_resid_rel": float(resid.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=HERE)
    ap.add_argument("--tau-relax", type=float, default=8.0,
                    help="contacts per particle after which theta is stationary (AR >= 1.5)")
    ap.add_argument("--tau-relax-near-sphere", type=float, default=60.0,
                    help="same for AR <= 1.3, where exchange takes ~20 encounters")
    args = ap.parse_args()
    rows = []
    for case in sorted(glob.glob(os.path.join(args.root, "AR*", "a*", "s*"))):
        log = os.path.join(case, "log.lammps")
        if not (os.path.exists(log) and re.search(r"^DONE", open(log).read(), re.M)):
            print(f"skip {os.path.relpath(case, args.root)} (not finished)")
            continue
        AR = float(os.path.relpath(case, args.root).split(os.sep)[0][2:])
        rows.append(reduce_case(case, args.tau_relax_near_sphere if AR <= 1.3 else args.tau_relax))
        r = rows[-1]
        print(f"AR={r['AR']:<4g} alpha={r['alpha']:.2f} {r['seed']}  tau={r['tau_end']:5.1f}  "
              f"theta*={r['theta_star']:.4f}+-{r['theta_star_err']:.4f}  gamma={r['gamma']:.5f}  "
              f"alpha_eff={r['alpha_eff']:.4f}  sigma_eff={r['sigma_eff']:.4f}  e_c={r['e_c']:.4f}")
    os.makedirs(os.path.join(HERE, "analysis"), exist_ok=True)
    with open(os.path.join(HERE, "analysis", "summary.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=COLUMNS)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r[k] for k in COLUMNS})
    print(f"{len(rows)} cases -> analysis/summary.csv")


if __name__ == "__main__":
    main()
