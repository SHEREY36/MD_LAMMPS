"""
Overlay plot: LAMMPS DEM (dilute3a) + DSMC (Run_shear_NN) + Kinetic Theory

Components: Pxx*, Pyy*, Pzz*, Pxy*
- LAMMPS: filled markers
- DSMC:   open markers (same shape, facecolor='none')
- KT:     solid lines

Usage:
    python plot_dilute.py [--out-file PATH]
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from io import StringIO

# ──────────────────────────────────────────────────────────────
# Kinetic Theory (Rubio-Largo / Garzo-Dufty, smooth spheres)
# ──────────────────────────────────────────────────────────────
def beta_0(alpha):
    return ((1 + alpha) / 2) * (1 - (1 - alpha) / 3)

def zeta_0(alpha):
    return (1 - alpha**2) * (5 / 12)

def a_steady(beta, zeta):
    return np.sqrt(3 * zeta / (2 * beta)) * (beta + zeta)

alphas_kt = np.linspace(0.55, 1.0, 1000)
b_kt = beta_0(alphas_kt)
z_kt = zeta_0(alphas_kt)
Pxx_KT = (b_kt + 3 * z_kt) / (b_kt + z_kt)
Pyy_KT =  b_kt / (b_kt + z_kt)
Pxy_KT = -(b_kt / (b_kt + z_kt)**2) * a_steady(b_kt, z_kt)
# Pzz_KT = Pyy_KT for smooth spheres in simple shear

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def tail_mean(arr, frac=0.3):
    """Mean of the last `frac` fraction of an array."""
    i0 = int((1.0 - frac) * len(arr))
    return float(np.mean(arr[i0:]))

def load_fortran(path):
    """Load a Fortran-style text file (D exponent notation)."""
    with open(path) as fh:
        txt = fh.read().replace('D', 'E').replace('d', 'e')
    return np.loadtxt(StringIO(txt))

# ──────────────────────────────────────────────────────────────
# LAMMPS  (dilute3a/e_0XX/shear_stats.dat)
# Columns: step  Pxxs  Pyys  Pzzs  Pxys  gamma  tilt
# Pijs are already P* = Press_ij / (n·T)
# ──────────────────────────────────────────────────────────────
def load_lammps_case(folder):
    path = os.path.join(folder, "shear_stats.dat")
    if not os.path.exists(path):
        rep_files = sorted(glob(os.path.join(folder, "seed_*", "shear_stats.dat")))
        if not rep_files:
            raise FileNotFoundError(f"No shear_stats.dat in {folder}")
        reps = [load_lammps_case(os.path.dirname(f)) for f in rep_files]
        return {k: np.mean([r[k] for r in reps]) for k in ("Pxx", "Pyy", "Pzz", "Pxy")}

    d = np.atleast_2d(np.loadtxt(path))
    return {
        "Pxx": tail_mean(d[:, 1]),
        "Pyy": tail_mean(d[:, 2]),
        "Pzz": tail_mean(d[:, 3]),
        "Pxy": tail_mean(d[:, 4]),
    }

lammps_base = os.path.join(SCRIPT_DIR, "dilute3a")
lammps_folders = ["e_060","e_065","e_070","e_075","e_080","e_085","e_090","e_095","e_100"]

alpha_lmp, Pxx_lmp, Pyy_lmp, Pzz_lmp, Pxy_lmp = [], [], [], [], []
for f in lammps_folders:
    alpha = int(f.split("_")[1]) / 100.0
    r = load_lammps_case(os.path.join(lammps_base, f))
    alpha_lmp.append(alpha)
    Pxx_lmp.append(r["Pxx"]); Pyy_lmp.append(r["Pyy"])
    Pzz_lmp.append(r["Pzz"]); Pxy_lmp.append(r["Pxy"])

alpha_lmp = np.array(alpha_lmp)
Pxx_lmp = np.array(Pxx_lmp); Pyy_lmp = np.array(Pyy_lmp)
Pzz_lmp = np.array(Pzz_lmp); Pxy_lmp = np.array(Pxy_lmp)

# ──────────────────────────────────────────────────────────────
# DSMC  (Run_shear_NN/XX/)
# Pijk[:,0]=Pxx_k, [:,1]=Pyy_k, [:,2]=Pzz_k, [:,3]=Pxy_k
# Pijc[:,0]=Pxx_c, [:,1]=Pxy_c, [:,3]=Pyy_c, [:,5]=Pzz_c
# tg[:,2]  = granular temperature Tg
# P* = (Pijk + Pijc) / (density × Tg) × 1000
# where density = N/V = 1e5 / 0.1³ = 1e8
# ──────────────────────────────────────────────────────────────
DSMC_N       = 100_000
DSMC_VOL     = 0.1**3
DSMC_DENSITY = DSMC_N / DSMC_VOL   # 1e8

def load_dsmc_case(folder):
    pijk = load_fortran(os.path.join(folder, "Pijk.txt"))
    pijc = load_fortran(os.path.join(folder, "Pijc.txt"))
    tg   = load_fortran(os.path.join(folder, "tg.txt"))

    # Skip first row of pijk (length mismatch with pijc)
    pijk = pijk[1:] if pijk.shape[0] == pijc.shape[0] + 1 else pijk
    n    = min(pijk.shape[0], pijc.shape[0], tg.shape[0])
    pijk, pijc, tg = pijk[:n], pijc[:n], tg[:n]

    Pxx = pijk[:, 0] + pijc[:, 0]
    Pyy = pijk[:, 1] + pijc[:, 3]
    Pzz = pijk[:, 2] + pijc[:, 5]
    Pxy = pijk[:, 3] + pijc[:, 1]
    Tg  = tg[:, 2]

    scale = 1000.0 / (DSMC_DENSITY * Tg)
    return {
        "Pxx": tail_mean(Pxx * scale),
        "Pyy": tail_mean(Pyy * scale),
        "Pzz": tail_mean(Pzz * scale),
        "Pxy": tail_mean(Pxy * scale),
    }

dsmc_base    = os.path.join(SCRIPT_DIR, "Run_shear_NN")
dsmc_folders = ["60","65","70","75","80","85","90","95","100"]

alpha_dsmc, Pxx_dsmc, Pyy_dsmc, Pzz_dsmc, Pxy_dsmc = [], [], [], [], []
for f in dsmc_folders:
    alpha = int(f) / 100.0
    r = load_dsmc_case(os.path.join(dsmc_base, f))
    alpha_dsmc.append(alpha)
    Pxx_dsmc.append(r["Pxx"]); Pyy_dsmc.append(r["Pyy"])
    Pzz_dsmc.append(r["Pzz"]); Pxy_dsmc.append(r["Pxy"])

alpha_dsmc = np.array(alpha_dsmc)
Pxx_dsmc   = np.array(Pxx_dsmc); Pyy_dsmc = np.array(Pyy_dsmc)
Pzz_dsmc   = np.array(Pzz_dsmc); Pxy_dsmc = np.array(Pxy_dsmc)

# ──────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--out-file", default=os.path.join(SCRIPT_DIR, "dilute3a", "dilute_overlay.png"))
args, _ = parser.parse_known_args()

# One colour per stress component
C = {"xx": "#1f77b4",   # blue
     "yy": "#ff7f0e",   # orange
     "zz": "#2ca02c",   # green
     "xy": "#d62728"}   # red

ms = 7   # marker size

fig, ax = plt.subplots(figsize=(8, 5.5))

# KT lines
ax.plot(alphas_kt, Pxx_KT, color=C["xx"], lw=1.5, zorder=1)
ax.plot(alphas_kt, Pyy_KT, color=C["yy"], lw=1.5, zorder=1)
ax.plot(alphas_kt, Pyy_KT, color=C["zz"], lw=1.5, ls="--", zorder=1)  # Pzz_KT = Pyy_KT
ax.plot(alphas_kt, Pxy_KT, color=C["xy"], lw=1.5, zorder=1)

# LAMMPS — filled markers
kw_lmp = dict(s=ms**2, zorder=3, edgecolors="k", linewidths=0.4)
ax.scatter(alpha_lmp, Pxx_lmp, marker="s", color=C["xx"], label=r"LAMMPS $P_{xx}^*$", **kw_lmp)
ax.scatter(alpha_lmp, Pyy_lmp, marker="o", color=C["yy"], label=r"LAMMPS $P_{yy}^*$", **kw_lmp)
ax.scatter(alpha_lmp, Pzz_lmp, marker="^", color=C["zz"], label=r"LAMMPS $P_{zz}^*$", **kw_lmp)
ax.scatter(alpha_lmp, Pxy_lmp, marker="D", color=C["xy"], label=r"LAMMPS $P_{xy}^*$", **kw_lmp)

# DSMC — open markers (same colours, larger, no fill)
kw_dsmc = dict(s=(ms*1.4)**2, zorder=2, linewidths=1.2)
ax.scatter(alpha_dsmc, Pxx_dsmc, marker="s", facecolors="none", edgecolors=C["xx"],
           label=r"DSMC $P_{xx}^*$", **kw_dsmc)
ax.scatter(alpha_dsmc, Pyy_dsmc, marker="o", facecolors="none", edgecolors=C["yy"],
           label=r"DSMC $P_{yy}^*$", **kw_dsmc)
ax.scatter(alpha_dsmc, Pzz_dsmc, marker="^", facecolors="none", edgecolors=C["zz"],
           label=r"DSMC $P_{zz}^*$", **kw_dsmc)
ax.scatter(alpha_dsmc, Pxy_dsmc, marker="D", facecolors="none", edgecolors=C["xy"],
           label=r"DSMC $P_{xy}^*$", **kw_dsmc)

# Dummy handle for KT in legend
from matplotlib.lines import Line2D
ax.add_artist(ax.legend(
    [Line2D([0], [0], color="gray", lw=1.5)],
    ["KT (Garzo-Dufty)"],
    loc="upper left", fontsize=9, framealpha=0.7
))

ax.legend(ncol=2, fontsize=8.5, loc="upper right", framealpha=0.9)

ax.set_xlabel(r"Coefficient of restitution $\alpha$", fontsize=12)
ax.set_ylabel(r"Reduced stress $P_{ij}^* = P_{ij}/nT$", fontsize=12)
ax.set_xlim(0.57, 1.02)
ax.set_ylim(-0.6, 1.75)
ax.tick_params(axis="both", direction="in", which="both", right=True, top=True)
ax.minorticks_on()
ax.grid(True, alpha=0.2, lw=0.5)

# Square data window (notebook convention: aspect = xrange/yrange)
xval, yval = ax.get_xlim(), ax.get_ylim()
ax.set_aspect((xval[1] - xval[0]) / (yval[1] - yval[0]), adjustable="box")

fig.tight_layout()
fig.savefig(args.out_file, dpi=180, bbox_inches="tight")
print(f"Saved → {args.out_file}")
