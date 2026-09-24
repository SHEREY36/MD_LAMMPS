#!/usr/bin/env python3
"""Check the v2 regression suite (run ./run_regression.sh first).

Every test compares against an independent reference: exact free-flight
kinematics, a hand-computed virial, the calibrated restitution, conservation
laws, equipartition, the energy bookkeeping identity, and a Boltzmann DSMC of
the same uniform shear flow (dsmc_usf_spheres.py)."""
import glob
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
results = []


skipped = []


def check(name, ok, detail):
    results.append((name, bool(ok), detail))


def failed(name, ex, optional=False):
    """An exception in a test block is a FAIL, except for the long tests that
    'run_regression.sh quick' does not run (T4, T6): missing outputs -> SKIP."""
    if optional and isinstance(ex, FileNotFoundError):
        skipped.append(name)
    else:
        check(name, False, f"error: {ex}")


def load(fn):
    return np.atleast_2d(np.loadtxt(fn, comments="#"))


def cols(fn):
    """column name -> 0-based index from the '# 1:name 2:name' header line"""
    for line in open(fn):
        if line.startswith("# 1:"):
            out = {}
            for tok in line[1:].split():
                k, v = tok.split(":", 1)
                out[v] = int(k) - 1
            return out
    raise RuntimeError(f"no column header in {fn}")


# ---------------------------------------------------------------- T1
try:
    d = load("t1_series.dat")
    s, pxy, pxx, pyy, corr = d[:, 0], d[:, 1], d[:, 2], d[:, 3], d[:, 4]
    i = np.argmin(abs(s - 0.9))
    r_xy = pxy[i] / pyy[i] - pxy[0] / pyy[0]
    r_xx = (pxx[i] - pxx[0]) / pyy[i]
    check("T1 free flight: P_xy/nT0 = -gdot t (Newton, no SLLOD double count)",
          abs(r_xy + s[i]) < 0.02 * s[i], f"strain {s[i]:.2f}: {r_xy:.4f} vs {-s[i]:.4f}")
    check("T1 free flight: (P_xx-P_yy)/nT0 grows as (gdot t)^2",
          abs(r_xx - s[i]**2) < 0.05 * s[i]**2, f"{r_xx:.4f} vs {s[i]**2:.4f}")
    check("T1 free flight: angular momenta constant (no rotational SLLOD)",
          corr.min() > 1 - 1e-10, f"min corr(L,L0) = {corr.min():.12f}")
    e = load("t1.energy")
    c = cols("t1.energy")
    check("T1 free flight: energy bookkeeping dE = W_shear", np.abs(e[:, c["resid_rel"]]).max() < 1e-3,
          f"max |resid_rel| = {np.abs(e[:, c['resid_rel']]).max():.2e}")
except Exception as ex:
    failed("T1", ex)

# ---------------------------------------------------------------- T2
try:
    txt = open("log.t2").read().split("T2 LAMMPS virial:")[-1].split()
    pxy = float(txt[0].split("=")[1])
    exact = (-0.3) * (-1000.0 * 0.1**1.5) / 8000.0
    check("T2 LAMMPS virial uses the centre-of-mass branch", abs(pxy - exact) < 1e-9 * 1e3,
          f"P_xy = {pxy:.8e}, exact {exact:.8e} (contact-point branch would give 0)")
    st = load("t2.stress")
    c = cols("t2.stress")
    ok = (abs(st[0, c["Cxy"]] - exact) < 1e-8 and abs(st[0, c["Cyx"]]) < 1e-12
          and abs(st[0, c["Cyy"]] - 3.0 * exact) < 1e-8)
    check("T2 fix spherocyl/diag 9-component collisional stress", ok,
          f"Cxy={st[0, c['Cxy']]:.6e} Cyx={st[0, c['Cyx']]:.2e} Cyy={st[0, c['Cyy']]:.6e}")
except Exception as ex:
    failed("T2", ex)

# ---------------------------------------------------------------- T3
for style, tol in (("hertz", 0.004), ("hooke", 0.025)):
    for a in (0.5, 0.7, 0.9):
        try:
            fn = f"t3_{style}_{a}.coll"
            c = cols(fn)
            r = load(fn)[-1]
            en = load(f"t3_{style}_{a}.energy")[-1]
            ce = cols(f"t3_{style}_{a}.energy")
            check(f"T3 {style} head-on restitution alpha={a}",
                  r[c["N_end"]] == 1 and abs(r[c["e_c"]] - a) < tol * a
                  and abs(en[ce["resid_rel"]]) < 0.005,
                  f"e={r[c['e_c']]:.4f} (tol {tol*100:.1f}%), {r[c['dur_mean_steps']]:.0f} steps/contact, "
                  f"energy resid {en[ce['resid_rel']]:.1e}")
        except Exception as ex:
            failed(f"T3 {style} {a}", ex)

# ---------------------------------------------------------------- T3b
try:
    d = load("t3b_conserved.dat")
    dP = np.abs(d[-1, 1:4] - d[0, 1:4]).max()
    dJ = np.abs(d[-1, 4:7] - d[0, 4:7]).max()
    c = cols("t3b.coll")
    r = load("t3b.coll")[-1]
    en = load("t3b.energy")[-1]
    ce = cols("t3b.energy")
    check("T3b off-centre rod collision: momentum + angular momentum conserved, one collision",
          dP < 1e-12 and dJ < 1e-10 and r[c["N_end"]] == 1,
          f"|dP|={dP:.1e} |dJ|={dJ:.1e} N_end={r[c['N_end']]:.0f} e_c={r[c['e_c']]:.3f}")
    check("T3b energy bookkeeping (rotation included)", abs(en[ce["resid_rel"]]) < 0.005,
          f"resid_rel={en[ce['resid_rel']]:.2e}")
except Exception as ex:
    failed("T3b", ex)

# ---------------------------------------------------------------- T4
try:
    e = load("t4.stress")
    c = cols("t4.stress")
    tail = e[int(0.7 * len(e)):]
    ratio = tail[:, c["T_rot"]].mean() / tail[:, c["T_tr"]].mean()
    check("T4 elastic rods: equipartition T_rot/T_tr -> 1", abs(ratio - 1) < 0.05,
          f"T_rot/T_tr = {ratio:.3f} over last 30% (start 0.2)")
    en = load("t4.energy")
    ce = cols("t4.energy")
    E = en[:, ce["E_tr"]] + en[:, ce["E_rot"]] + en[:, ce["U_el"]]
    drift = abs(E[-1] - E[0]) / E[0]
    check("T4 elastic rods: total energy conserved", drift < 1e-3 and np.abs(en[:, ce["dPc_x"]:ce["dPc_z"] + 1]).max() < 1e-6,
          f"relative energy drift {drift:.1e} over {len(E)} windows, max dPc {np.abs(en[:, ce['dPc_x']:ce['dPc_z']+1]).max():.1e}")
except Exception as ex:
    failed("T4", ex, optional=True)

# ---------------------------------------------------------------- T5
try:
    en = load("t5.energy")
    ce = cols("t5.energy")
    rmax = np.abs(en[:, ce["resid_rel"]]).max()
    check("T5 elastic USF: dE = shear work every window", rmax < 0.01,
          f"max |resid_rel| = {rmax:.1e}; E grew {en[-1, ce['E_tr']] / en[0, ce['E_tr']]:.3f}x")
    check("T5 elastic USF: peculiar momentum conserved",
          np.abs(en[:, ce["dPc_x"]:ce["dPc_z"] + 1]).max() < 1e-6,
          f"max |dPc| = {np.abs(en[:, ce['dPc_x']:ce['dPc_z'] + 1]).max():.1e}")
except Exception as ex:
    failed("T5", ex)

# ---------------------------------------------------------------- T6
try:
    base = "t6_sphere_usf/AR1/a0.70"
    st = load(f"{base}/prod.stress")
    c = cols(f"{base}/prod.stress")
    N, V = st[0, c["N"]], st[0, c["V"]]
    n = N / V
    T = st[:, c["T_tr"]].mean()
    m, gd = math.pi / 6.0, 1.8
    Tstar = T / (m * gd**2)
    P = {k: (st[:, c["K" + k]] + st[:, c["C" + k]]).mean() / (n * T) for k in ("xx", "yy", "zz", "xy")}
    Pyx = ((st[:, c["Kxy"]] + st[:, c["Cyx"]]).mean()) / (n * T)
    # Boltzmann DSMC (dsmc_usf_spheres.py, alpha=0.7): T*=189.5 ; P* = 1.424 0.766 0.810 -0.501
    # Enskog collisional pressure at phi=0.01 adds ~ 2(1+a) phi g0 = 0.035 to the diagonal
    ref = {"xx": 1.424 + 0.035, "yy": 0.766 + 0.035, "zz": 0.810 + 0.035, "xy": -0.501}
    check("T6 sphere USF: T/(m gdot^2 d^2) vs Boltzmann DSMC (189.5)", abs(Tstar / 189.5 - 1) < 0.06,
          f"T* = {Tstar:.1f}")
    for k in ref:
        check(f"T6 sphere USF: P*_{k} vs DSMC(+Enskog)", abs(P[k] - ref[k]) < 0.04,
              f"{P[k]:.4f} vs {ref[k]:.4f}")
    check("T6 sphere USF: P_xy = P_yx (no spurious antisymmetric stress)", abs(P["xy"] - Pyx) < 0.01,
          f"P*_xy={P['xy']:.4f} P*_yx={Pyx:.4f}")
    cc = cols(f"{base}/prod.coll")
    co = load(f"{base}/prod.coll")
    ec = (co[:, cc["e_c"]] * co[:, cc["N_end"]]).sum() / co[:, cc["N_end"]].sum()
    check("T6 sphere USF: realised restitution in the gas", abs(ec - 0.7) < 0.01, f"e_c = {ec:.4f}")
    # Enskog collision frequency with Maxwellian estimate nu = 4 g0 n sigma^2 sqrt(pi T/m)
    phi = 0.01
    g0 = (1 - phi / 2) / (1 - phi)**3
    nu_th = 4 * g0 * n * math.sqrt(math.pi * T / m)
    nu = co[:, cc["nu"]].mean()
    check("T6 sphere USF: collision frequency vs Enskog (Maxwellian, +-6%)", abs(nu / nu_th - 1) < 0.06,
          f"nu = {nu:.3f}, theory {nu_th:.3f}")
    Cc = np.array([st[:, c[k]].mean() for k in ("Cxx", "Cyy", "Czz", "Cxy")])
    Ce = np.array([st[:, c["CE" + k[1:]]].mean() for k in ("Cxx", "Cyy", "Czz", "Cxy")])
    check("T6 sphere USF: continuous vs collision-by-collision collisional stress", np.abs(Cc - Ce).max() < 0.03 * np.abs(Cc).max(),
          f"continuous {Cc} ; events {Ce}")
    se = open(f"{base}/prod.sensors").read().split("\n")
    flags = [l.split()[3] for l in se if l and not l.startswith("#")]
    check("T6 sphere USF: sensors clean in production", all(f == "-" for f in flags),
          f"{sum(f != '-' for f in flags)} flagged windows of {len(flags)}: {set(flags)}")
    er = load(f"{base}/prod.energy")
    ce = cols(f"{base}/prod.energy")
    check("T6 sphere USF: energy bookkeeping residual", np.abs(er[:, ce["resid_rel"]]).max() < 0.01,
          f"max |resid_rel| = {np.abs(er[:, ce['resid_rel']]).max():.1e}")
except Exception as ex:
    failed("T6", ex, optional=True)

# ---------------------------------------------------------------- T7
try:
    log = open("log.t7").read()
    check("T7 rot_sllod option removed (hard error)", "rot_sllod option has been removed" in log, "")
except Exception as ex:
    failed("T7", ex)

# ---------------------------------------------------------------- T8
try:
    a = load("t8_np1.stress")
    b = load("t8_np4.stress")
    rel = np.abs(a - b).max() / np.abs(a).max()
    ca = load("t8_np1.coll")
    cb = load("t8_np4.coll")
    c = cols("t8_np1.coll")
    same_n = np.array_equal(ca[:, c["N_end"]], cb[:, c["N_end"]])
    check("T8 1 vs 4 MPI ranks give the same physics", rel < 1e-6 and same_n,
          f"max relative stress difference {rel:.1e}; collision counts identical: {same_n}")
except Exception as ex:
    failed("T8", ex)

# ---------------------------------------------------------------- T9
try:
    en = load("t9.energy")
    ce = cols("t9.energy")
    strain = en[-1, ce["strain"]]
    dpc = np.abs(en[:, ce["dPc_x"]:ce["dPc_z"] + 1]).max()
    check("T9 Lees-Edwards box flips (patched fix deform): peculiar momentum conserved",
          strain > 2.0 and dpc < 1e-8,
          f"{int(strain + 0.5)} flips, max |dPc| = {dpc:.1e} (unpatched: jumps of ~0.08 per flip)")
    check("T9 energy bookkeeping through flips", np.abs(en[:, ce["resid_rel"]]).max() < 0.01,
          f"max |resid_rel| = {np.abs(en[:, ce['resid_rel']]).max():.1e}")
except Exception as ex:
    failed("T9", ex)

# ---------------------------------------------------------------- report
w = max(len(r[0]) for r in results)
lines = []
for name, ok, detail in results:
    lines.append(f"{'PASS' if ok else 'FAIL'}  {name:<{w}}  {detail}")
npass = sum(r[1] for r in results)
lines.append(f"\n{npass}/{len(results)} checks passed"
             + (f"  (not run, skipped: {', '.join(skipped)})" if skipped else ""))
report = "\n".join(lines)
print(report)
open("regression_report.txt", "w").write(report + "\n")
sys.exit(0 if npass == len(results) else 1)
