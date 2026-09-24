#!/usr/bin/env python3
"""
Summarise the fresh USF production runs (fix spherocyl/diag outputs).

For every case directory containing prod.stress it writes one row of
summary.csv with block-averaged means and standard errors (10 blocks), and a
plain-text diagnostics report (diagnostics_report.txt) that lists every
sensor that fired, stationarity checks and the clustering indicators.

Usage:  python3 postprocess_fresh_usf.py [root=.]  [--nblocks 10]
"""
import argparse
import glob
import math
import os

import numpy as np


def cols(fn):
    for line in open(fn):
        if line.startswith("# 1:"):
            return {tok.split(":", 1)[1]: int(tok.split(":", 1)[0]) - 1 for tok in line[1:].split()}
    raise RuntimeError(fn)


def load(fn):
    return np.atleast_2d(np.loadtxt(fn, comments="#"))


def block(x, nb):
    """mean and standard error from nb contiguous blocks"""
    x = np.asarray(x)
    nb = max(2, min(nb, len(x) // 2))
    b = np.array([c.mean() for c in np.array_split(x, nb)])
    return x.mean(), b.std(ddof=1) / math.sqrt(nb)


def params(case):
    p = {}
    for line in open(os.path.join(case, "params.in")):
        t = line.split()
        if len(t) >= 4 and t[0] == "variable" and t[2] == "equal":
            p[t[1]] = float(t[3])
    return p


def recal(case):
    out = []
    log = os.path.join(case, "log.lammps")
    if os.path.exists(log):
        for line in open(log):
            if line.startswith("RECALIBRATE: kn="):
                out.append(line.strip())
    return out


def summarise(case, nb):
    p = params(case)
    rows, notes = {}, []
    st = load(os.path.join(case, "prod.stress"))
    c = cols(os.path.join(case, "prod.stress"))
    N, V = st[0, c["N"]], st[0, c["V"]]
    n = N / V
    Ttr, Trot = st[:, c["T_tr"]], st[:, c["T_rot"]]
    nT = n * Ttr
    AR, alpha, gd = p["AR"], p["alpha"], p["gdot"]
    R, Lc = 0.5, AR - 1.0
    m = p["rho_p"] * (4.0 / 3.0 * math.pi * R**3 + math.pi * R**2 * Lc)
    rows.update(AR=AR, alpha=alpha, strain=st[-1, c["strain"]] - st[0, c["strain"]] + st[0, c["t_win"]] * gd,
                windows=len(st))
    for key, arr in (("Tstar", Ttr / (m * gd**2)), ("theta", Ttr / Trot),
                     ("a2_tr", st[:, c["a2_tr"]]), ("a2_rot", st[:, c["a2_rot"]])):
        rows[key], rows[key + "_se"] = block(arr, nb)
    # reduced stresses: kinetic, collisional, total (ratio of window values; T fluctuations ~1%)
    for ab in ("xx", "yy", "zz", "xy", "xz", "yz"):
        k = st[:, c["K" + ab]] / nT
        cc = st[:, c["C" + ab]] / nT
        rows["Pk_" + ab], rows["Pk_" + ab + "_se"] = block(k, nb)
        rows["Pc_" + ab], rows["Pc_" + ab + "_se"] = block(cc, nb)
        rows["P_" + ab], rows["P_" + ab + "_se"] = block(k + cc, nb)
    rows["Pc_yx"], rows["Pc_yx_se"] = block(st[:, c["Cyx"]] / nT, nb)
    rows["N1"], rows["N1_se"] = block((st[:, c["Kxx"]] + st[:, c["Cxx"]] - st[:, c["Kyy"]] - st[:, c["Cyy"]]) / nT, nb)
    rows["N2"], rows["N2_se"] = block((st[:, c["Kyy"]] + st[:, c["Cyy"]] - st[:, c["Kzz"]] - st[:, c["Czz"]]) / nT, nb)
    # consistency of the two collisional estimators
    ce_c = st[:, c["Cxy"]].mean()
    ce_e = st[:, c["CExy"]].mean()
    rows["Cxy_event_over_cont"] = ce_e / ce_c if ce_c != 0 else float("nan")
    # antisymmetric collisional stress (must vanish on average in steady state)
    rows["Pc_xy_minus_yx"] = rows["Pc_xy"] - rows["Pc_yx"]

    # stationarity: first vs second half of production
    h = len(st) // 2
    for key, arr in (("T", Ttr), ("Pxy", (st[:, c["Kxy"]] + st[:, c["Cxy"]]) / nT)):
        a, b = arr[:h].mean(), arr[h:].mean()
        rows["drift_" + key] = (b - a) / abs(a) if a != 0 else float("nan")

    # collisions
    co = load(os.path.join(case, "prod.coll"))
    cc = cols(os.path.join(case, "prod.coll"))
    w = co[:, cc["N_end"]]
    wsum = max(w.sum(), 1.0)
    def wavg(name):
        return float((co[:, cc[name]] * w).sum() / wsum)
    rows["nu_over_gdot"], rows["nu_over_gdot_se"] = block(co[:, cc["nu"]] / gd, nb)
    for name in ("e_c", "e_tr", "e_c_out", "e_tr_out", "frac_binary", "frac_multibody", "frac_long",
                 "frac_short", "frac_gain", "dmax_mean", "frac_same_partner", "frac_endcap",
                 "<|ui.uj|>", "dE/E_mean", "dE_rel_tr_per_coll", "dE_rot_per_coll", "dur_mean_steps"):
        rows[name] = wavg(name)
    rows["dmax_max"] = co[:, cc["dmax_max"]].max()
    rows["dur_min_steps"] = co[:, cc["dur_min_steps"]].min()
    rows["freeflight_cv2"] = float(np.nanmean(co[:, cc["freeflight_cv2"]]))
    rows["collcount_disp"] = float(np.nanmean(co[:, cc["collcount_disp"]]))
    for z in ("P(z=0)", "P(z=1)", "P(z=2)", "P(z>=3)"):
        rows[z] = float(co[:, cc[z]].mean())
    for ab in ("xx", "yy", "zz", "xy"):
        rows["fabric_" + ab] = wavg("nn_" + ab)

    # structure / clustering / orientation
    sr = load(os.path.join(case, "prod.struct"))
    cs = cols(os.path.join(case, "prod.struct"))
    smodes = [k for k in cs if k.startswith("S") and k[1:].replace("-", "").isdigit()]
    for k in smodes + ["Jx010", "Jx020", "Jx001", "E010", "E001", "D2", "D4", "D8"]:
        rows[k] = float(sr[:, cs[k]].mean())
    Q = np.array([[sr[:, cs["Qxx"]].mean(), sr[:, cs["Qxy"]].mean(), sr[:, cs["Qxz"]].mean()],
                  [sr[:, cs["Qxy"]].mean(), sr[:, cs["Qyy"]].mean(), sr[:, cs["Qyz"]].mean()],
                  [sr[:, cs["Qxz"]].mean(), sr[:, cs["Qyz"]].mean(), sr[:, cs["Qzz"]].mean()]])
    ev, evec = np.linalg.eigh(Q)
    rows["S2_nematic"] = 1.5 * ev[-1]            # standard nematic order of <Q> (0 iso, 1 aligned)
    d = evec[:, -1]
    rows["director_angle_xy_deg"] = math.degrees(math.atan2(d[1], d[0])) % 180.0
    if rows["director_angle_xy_deg"] > 90.0:
        rows["director_angle_xy_deg"] -= 180.0
    s2b = [1.5 * np.linalg.eigvalsh(np.array([[q[cs["Qxx"]], q[cs["Qxy"]], q[cs["Qxz"]]],
                                              [q[cs["Qxy"]], q[cs["Qyy"]], q[cs["Qyz"]]],
                                              [q[cs["Qxz"]], q[cs["Qyz"]], q[cs["Qzz"]]]]))[-1]
           for q in [b.mean(axis=0) for b in np.array_split(sr, nb)]]
    rows["S2_nematic_blockse"] = float(np.std(s2b, ddof=1) / math.sqrt(len(s2b)))
    rows["S2_isotropic_noise"] = 1.5 * 0.0119 * math.sqrt(2003.0 / N) / math.sqrt(len(sr))  # rough floor

    # energy / sensors
    en = load(os.path.join(case, "prod.energy"))
    ceg = cols(os.path.join(case, "prod.energy"))
    rows["resid_rel_max"] = float(np.abs(en[:, ceg["resid_rel"]]).max())
    rows["dPc_max"] = float(np.abs(en[:, ceg["dPc_x"]:ceg["dPc_z"] + 1]).max())
    rows["Wnc_over_Wshear"] = float(en[:, ceg["W_nc"]].sum() / (en[:, ceg["W_shear_kin"]] + en[:, ceg["W_shear_col"]]).sum())
    rows["spin_z_over_vort"] = float(en[:, ceg["wz"]].mean() / (-0.5 * gd))
    se = [l.split() for l in open(os.path.join(case, "prod.sensors")) if l.strip() and not l.startswith("#")]
    flags = {}
    for l in se:
        for ch in l[3]:
            if ch != "-":
                flags[ch] = flags.get(ch, 0) + 1
    rows["flag_windows"] = sum(1 for l in se if l[3] != "-")
    rows["flags"] = "".join(f"{k}{v}" for k, v in sorted(flags.items())) or "-"

    # notes for the report
    if rows["flag_windows"]:
        notes.append(f"sensor flags (letter+count of windows): {rows['flags']} of {len(se)} windows")
    if abs(rows["drift_T"]) > 0.03:
        notes.append(f"T differs by {100 * rows['drift_T']:.1f}% between production halves (not stationary?)")
    if rows["S010"] > 2 or rows["S001"] > 2 or rows["D4"] > 1.3:
        notes.append(f"density inhomogeneity: S010={rows['S010']:.2f} S001={rows['S001']:.2f} D4={rows['D4']:.2f}")
    if rows["frac_multibody"] > 0.02:
        notes.append(f"{100 * rows['frac_multibody']:.1f}% multi-body collisions (DSMC is strictly binary)")
    if rows["frac_long"] > 0.01:
        notes.append(f"{100 * rows['frac_long']:.1f}% long (rotation-driven) contacts")
    if rows["frac_gain"] > 0.01:
        notes.append(f"{100 * rows['frac_gain']:.1f}% of collisions gained pair energy")
    if rows["dmax_max"] > 0.02:
        notes.append(f"max overlap {100 * rows['dmax_max']:.2f}% of d")
    if abs(rows["Cxy_event_over_cont"] - 1) > 0.05:
        notes.append(f"event/continuous collisional stress ratio {rows['Cxy_event_over_cont']:.3f}")
    notes += recal(case)
    return rows, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--nblocks", type=int, default=10)
    a = ap.parse_args()
    cases = sorted(os.path.dirname(f) for f in glob.glob(os.path.join(a.root, "AR*", "a*", "prod.stress")))
    allrows, report = [], []
    for case in cases:
        try:
            rows, notes = summarise(case, a.nblocks)
        except Exception as ex:
            report.append(f"{os.path.relpath(case, a.root)}: could not summarise ({ex})")
            continue
        rows["case"] = os.path.relpath(case, a.root)
        allrows.append(rows)
        report.append(f"{rows['case']}: T*={rows['Tstar']:.1f} theta={rows['theta']:.4f} "
                      f"P*xy={rows['P_xy']:.4f}+-{rows['P_xy_se']:.4f} S2={rows['S2_nematic']:.4f} "
                      f"flags={rows['flags']}")
        report += ["    " + x for x in notes]
    if not allrows:
        print("no production output found")
        return
    keys = ["case"] + [k for k in allrows[0] if k != "case"]
    with open(os.path.join(a.root, "summary.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for r in allrows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    txt = "\n".join(report)
    open(os.path.join(a.root, "diagnostics_report.txt"), "w").write(txt + "\n")
    print(txt)
    print(f"\nwrote summary.csv ({len(allrows)} cases) and diagnostics_report.txt")


if __name__ == "__main__":
    main()
