#!/usr/bin/env python3
"""
Plots of the fresh USF results (fix spherocyl/diag outputs).

  python3 plot_fresh_usf.py [root=.] [--old ../USF]     -> <root>/analysis/*.png

Figures
  fig1_reduced_stress.png   P*_xx, P*_yy, P*_zz, P*_xy vs alpha (kinetic theory and
                            DSMC for smooth spheres as references; old DEM for comparison)
  fig2_temperatures.png     theta = T_tr/T_rot, T/(m gdot^2 d^2), N1*, N2*
  fig3_stress_split.png     collisional share of p and P_xy, antisymmetric stress
  fig4_collisions.png       realised restitution, long / multi-body / repeat collisions,
                            collision rate, contact resolution
  fig5_structure.png        nematic order S2, density modes S(k), D(4^3), collision-count
                            dispersion (clustering indicators)
  fig6_timeseries_<case>.png   stationarity of one case (T*, P*_xy, energy residual)

Cases whose contacts were under-resolved (>1% of contacts shorter than 15 steps;
the AR=2, alpha<=0.65 pilot runs) are drawn with hollow markers.
"""
import argparse
import glob
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from postprocess_fresh_usf import cols, load, summarise  # noqa: E402

# categorical slots (reference palette, fixed order): colour follows the aspect ratio
AR_STYLE = {1.0: ("#2a78d6", "o"), 1.5: ("#eb6834", "s"), 2.0: ("#1baf7a", "D"),
            2.5: ("#eda100", "^"), 3.0: ("#e87ba4", "v")}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 2.0, "lines.markersize": 7, "savefig.dpi": 160, "figure.facecolor": "white",
})


# ------------------------------------------------------------ references
def grad_spheres(alpha):
    """Grad/13-moment solution, dilute smooth spheres (the 'kinetic theory' line)."""
    z = 5.0 / 12.0 * (1 - alpha**2)
    b = (1 + alpha) * (3 - alpha) / 4.0
    pyy = 1 - z / b
    a = math.sqrt(1.5 * z * b**2 / (b - z)) if z > 0 else 0.0
    return dict(xx=3 - 2 * pyy, yy=pyy, zz=pyy, xy=-a * pyy / b)


DSMC_SPHERES = {  # runs/regression/v2/dsmc_usf_spheres.py (Boltzmann, phi -> 0)
    0.5: dict(xx=1.6644, yy=0.6373, zz=0.6983, xy=-0.5717),
    0.7: dict(xx=1.4238, yy=0.7661, zz=0.8101, xy=-0.5006),
    0.9: dict(xx=1.1507, yy=0.9150, zz=0.9343, xy=-0.3258),
}


def old_dem(old_root):
    """Previous DEM (SLLOD force, rot_sllod, input alpha != realised alpha) for comparison."""
    out = {}
    for d in sorted(glob.glob(os.path.join(old_root, "AR2", "e_*"))):
        f = os.path.join(d, "shear_stats.dat")
        if not os.path.exists(f):
            continue
        try:
            a = int(os.path.basename(d)[2:]) / 100.0
            s = np.loadtxt(f, comments="#")
            t = np.loadtxt(os.path.join(d, "temperature_stats.dat"), comments="#")
        except Exception:
            continue
        if a >= 1.0 or s.ndim != 2:
            continue
        out[a] = dict(xx=s[:, 1].mean(), yy=s[:, 2].mean(), zz=s[:, 3].mean(), xy=s[:, 4].mean(),
                      theta=(t[:, 1] / t[:, 2]).mean())
    return out


# ------------------------------------------------------------ helpers
def by_ar(rows):
    g = {}
    for r in rows:
        g.setdefault(r["AR"], []).append(r)
    for ar in g:
        g[ar].sort(key=lambda r: r["alpha"])
    return dict(sorted(g.items()))


def plot_series(ax, rows, key, se=None, scale=1.0, label=True):
    for ar, rs in by_ar(rows).items():
        col, mk = AR_STYLE.get(ar, ("#4a3aa7", "P"))
        x = np.array([r["alpha"] for r in rs])
        y = np.array([r[key] for r in rs]) * scale
        e = np.array([r.get(se, 0.0) for r in rs]) * scale if se else None
        ok = np.array([r["valid"] for r in rs])
        ax.plot(x, y, "-", color=col, lw=1.5, alpha=0.55, zorder=2)
        if e is not None:
            ax.errorbar(x, y, yerr=e, fmt="none", ecolor=col, elinewidth=1.2, capsize=2, zorder=3)
        ax.plot(x[ok], y[ok], mk, color=col, mec="white", mew=1.0, zorder=4,
                label=f"AR = {ar:g}" if label else None)
        if (~ok).any():
            ax.plot(x[~ok], y[~ok], mk, mfc="white", mec=col, mew=1.5, zorder=4)


def finish(fig, path, note=None):
    if note:
        fig.text(0.01, 0.005, note, fontsize=8, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03 if note else 0, 1, 1))
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


INVALID_NOTE = ("Hollow markers: contacts under-resolved (>1% shorter than 15 steps) - AR=2 pilot runs "
                "at alpha<=0.65, to be replaced by the Negishi runs.")


# ------------------------------------------------------------ figures
def fig_reduced_stress(rows, old, outdir):
    al = np.linspace(0.5, 1.0, 101)
    kt = [grad_spheres(a) for a in al]
    fig, axs = plt.subplots(2, 2, figsize=(10, 7.6), sharex=True)
    for ax, comp, lab in zip(axs.flat, ("xx", "yy", "zz", "xy"),
                             ("$P^*_{xx}$", "$P^*_{yy}$", "$P^*_{zz}$", "$P^*_{xy}$")):
        ax.plot(al, [k[comp] for k in kt], color=INK, lw=1.2, label="kinetic theory, spheres (Grad)")
        ax.plot(list(DSMC_SPHERES), [v[comp] for v in DSMC_SPHERES.values()], "x", color=INK,
                ms=8, mew=1.8, label="Boltzmann DSMC, spheres")
        if old:
            ax.plot(list(old), [v[comp] for v in old.values()], "o", mfc="none", mec="#a6a49d",
                    mew=1.2, ms=6, label="old DEM AR=2 (SLLOD force, input $\\alpha$)")
        plot_series(ax, rows, "P_" + comp, "P_" + comp + "_se")
        ax.set_ylabel(lab)
    for ax in axs[1]:
        ax.set_xlabel("coefficient of restitution $\\alpha$")
    axs[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Reduced stress $P^*_{ij} = (K_{ij}+C_{ij})/(n T_{tr})$  (kinetic + collisional, COM virial)",
                 fontsize=11)
    finish(fig, os.path.join(outdir, "fig1_reduced_stress.png"), INVALID_NOTE)


def fig_temperatures(rows, old, outdir):
    fig, axs = plt.subplots(2, 2, figsize=(10, 7.6), sharex=True)
    ax = axs[0, 0]
    if old:
        ax.plot(list(old), [v["theta"] for v in old.values()], "o", mfc="none", mec="#a6a49d",
                mew=1.2, ms=6, label="old DEM AR=2")
    plot_series(ax, rows, "theta", "theta_se")
    ax.axhline(1.0, color=INK2, lw=0.8, ls=":")
    ax.set_ylabel("$\\theta = T_{tr}/T_{rot}$")
    ax.legend(fontsize=8)
    ax = axs[0, 1]
    plot_series(ax, rows, "Tstar", "Tstar_se", label=False)
    ax.set_yscale("log")
    ax.set_ylabel("$T_{tr}/(m\\dot\\gamma^2 d^2)$")
    ax = axs[1, 0]
    plot_series(ax, rows, "N1", "N1_se", label=False)
    ax.plot(np.linspace(0.5, 1, 51), [grad_spheres(a)["xx"] - grad_spheres(a)["yy"] for a in np.linspace(0.5, 1, 51)],
            color=INK, lw=1.2)
    ax.set_ylabel("$N_1^* = P^*_{xx}-P^*_{yy}$")
    ax = axs[1, 1]
    plot_series(ax, rows, "N2", "N2_se", label=False)
    ax.axhline(0.0, color=INK, lw=1.2)
    ax.set_ylabel("$N_2^* = P^*_{yy}-P^*_{zz}$")
    for ax in axs[1]:
        ax.set_xlabel("coefficient of restitution $\\alpha$")
    finish(fig, os.path.join(outdir, "fig2_temperatures.png"), INVALID_NOTE)


def fig_stress_split(rows, outdir):
    for r in rows:
        p = (r["P_xx"] + r["P_yy"] + r["P_zz"]) / 3.0
        pc = (r["Pc_xx"] + r["Pc_yy"] + r["Pc_zz"]) / 3.0
        r["coll_share_p"] = pc / p
        r["coll_share_xy"] = r["Pc_xy"] / r["P_xy"] if r["P_xy"] else 0.0
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.0), sharex=True)
    plot_series(axs[0], rows, "coll_share_p", scale=100)
    axs[0].set_ylabel("collisional share of $p$ (%)")
    axs[0].legend(fontsize=8)
    plot_series(axs[1], rows, "coll_share_xy", scale=100, label=False)
    axs[1].set_ylabel("collisional share of $P_{xy}$ (%)")
    plot_series(axs[2], rows, "Pc_xy_minus_yx", label=False)
    axs[2].axhline(0, color=INK, lw=1.0)
    axs[2].set_ylabel("$(C_{xy}-C_{yx})/(nT)$  (must be ~0)")
    for ax in axs:
        ax.set_xlabel("$\\alpha$")
    finish(fig, os.path.join(outdir, "fig3_stress_split.png"), INVALID_NOTE)


def fig_collisions(rows, outdir):
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.2), sharex=True)
    ax = axs[0, 0]
    plot_series(ax, rows, "e_c")
    ax.plot([0.5, 1], [0.5, 1], color=INK, lw=1.0, ls="--", label="$e = \\alpha$")
    ax.set_ylabel("realised restitution $e_c$ (impulse axis)")
    ax.legend(fontsize=8)
    ax = axs[0, 1]
    plot_series(ax, rows, "frac_long", scale=100, label=False)
    ax.set_ylabel("long contacts, > 5x median (%)")
    ax = axs[0, 2]
    plot_series(ax, rows, "frac_same_partner", scale=100, label=False)
    ax.set_ylabel("collisions with previous partner (%)")
    ax = axs[1, 0]
    plot_series(ax, rows, "frac_multibody", scale=100, label=False)
    ax.set_ylabel("multi-body collisions (%)")
    ax = axs[1, 1]
    plot_series(ax, rows, "nu_over_gdot", "nu_over_gdot_se", label=False)
    ax.set_ylabel("collisions per particle per strain")
    ax = axs[1, 2]
    for r in rows:
        r["frac_short_plot"] = max(r["frac_short"], 1e-5)      # log axis: floor at 1e-3 %
    plot_series(ax, rows, "frac_short_plot", scale=100, label=False)
    ax.axhline(1.0, color="#e34948", lw=1.0, ls="--")
    ax.set_yscale("log")
    ax.set_ylim(5e-4, 300)
    ax.set_ylabel("contacts < 15 steps (%)  [sensor R at 1%]")
    for ax in axs[1]:
        ax.set_xlabel("$\\alpha$")
    finish(fig, os.path.join(outdir, "fig4_collisions.png"), INVALID_NOTE)


def fig_structure(rows, outdir):
    for r in rows:
        r["S_low"] = (r["S100"] + r["S010"] + r["S001"]) / 3.0
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.9), sharex=True)
    plot_series(axs[0], rows, "S2_nematic")
    axs[0].set_ylabel("nematic order $S_2 = 1.5\\,\\lambda_{max}(\\langle Q\\rangle)$")
    axs[0].legend(fontsize=8)
    plot_series(axs[1], rows, "S_low", label=False)
    axs[1].axhline(1.0, color=INK2, lw=0.8, ls=":")
    axs[1].set_ylabel("$S(k_{min})$, mean of x,y,z modes (ideal gas ~1)")
    plot_series(axs[2], rows, "D4", label=False)
    axs[2].axhline(1.0, color=INK2, lw=0.8, ls=":")
    axs[2].set_ylabel("density dispersion $D(4^3)$ (Poisson 1)")
    plot_series(axs[3], rows, "collcount_disp", label=False)
    axs[3].axhline(1.0, color=INK2, lw=0.8, ls=":")
    axs[3].set_ylabel("collision-count dispersion (Poisson 1)")
    for ax in axs:
        ax.set_xlabel("$\\alpha$")
    finish(fig, os.path.join(outdir, "fig5_structure.png"), INVALID_NOTE)


def fig_timeseries(case, root, outdir):
    st = load(os.path.join(case, "prod.stress"))
    c = cols(os.path.join(case, "prod.stress"))
    en = load(os.path.join(case, "prod.energy"))
    ce = cols(os.path.join(case, "prod.energy"))
    n = st[0, c["N"]] / st[0, c["V"]]
    nT = n * st[:, c["T_tr"]]
    strain = st[:, c["strain"]] - st[0, c["strain"]]
    pxy = (st[:, c["Kxy"]] + st[:, c["Cxy"]]) / nT
    fig, axs = plt.subplots(3, 1, figsize=(9, 7.2), sharex=True)
    axs[0].plot(strain, st[:, c["T_tr"]], color="#2a78d6", lw=1.2, label="$T_{tr}$")
    axs[0].plot(strain, st[:, c["T_rot"]], color="#eb6834", lw=1.2, label="$T_{rot}$")
    axs[0].set_ylabel("temperature")
    axs[0].legend(fontsize=8)
    axs[1].plot(strain, pxy, color="#1baf7a", lw=1.2)
    axs[1].axhline(pxy.mean(), color=INK, lw=0.8, ls="--")
    axs[1].set_ylabel("$P^*_{xy}$ per window")
    axs[2].plot(strain, en[:, ce["resid_rel"]] * 100, color="#4a3aa7", lw=1.0)
    axs[2].axhline(2, color="#e34948", lw=0.8, ls="--")
    axs[2].axhline(-2, color="#e34948", lw=0.8, ls="--")
    axs[2].set_ylabel("energy residual (% of work)")
    axs[2].set_xlabel("production strain")
    name = os.path.relpath(case, root).replace("/", "_")
    fig.suptitle(f"{os.path.relpath(case, root)}: stationarity and energy bookkeeping", fontsize=11)
    finish(fig, os.path.join(outdir, f"fig6_timeseries_{name}.png"))


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("root", nargs="?", default=here)
    ap.add_argument("--old", default=os.path.join(here, "..", "USF"))
    ap.add_argument("--nblocks", type=int, default=10)
    a = ap.parse_args()
    outdir = os.path.join(a.root, "analysis")
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for f in sorted(glob.glob(os.path.join(a.root, "AR*", "a*", "prod.stress"))):
        case = os.path.dirname(f)
        if not open(os.path.join(case, "log.lammps")).read().count("\nDONE"):
            continue
        r, _ = summarise(case, a.nblocks)
        r["case"] = os.path.relpath(case, a.root)
        r["valid"] = r["frac_short"] < 0.01
        rows.append(r)
    if not rows:
        print("no finished cases under", a.root)
        return
    print(f"{len(rows)} finished cases ({sum(r['valid'] for r in rows)} well resolved)")
    old = old_dem(a.old) if os.path.isdir(a.old) else {}
    fig_reduced_stress(rows, old, outdir)
    fig_temperatures(rows, old, outdir)
    fig_stress_split(rows, outdir)
    fig_collisions(rows, outdir)
    fig_structure(rows, outdir)
    for r in rows:
        if r["valid"] and abs(r["alpha"] - 0.7) < 1e-6:
            fig_timeseries(os.path.join(a.root, r["case"]), a.root, outdir)
            break


if __name__ == "__main__":
    main()
