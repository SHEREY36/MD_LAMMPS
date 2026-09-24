#!/usr/bin/env python3
import argparse
import os
import re
from glob import glob
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def average_last_fraction(arr: np.ndarray, frac: float) -> float:
    n = len(arr)
    if n == 0:
        return float("nan")
    i0 = int((1.0 - frac) * n)
    i0 = max(0, min(i0, n - 1))
    return float(np.mean(arr[i0:]))


def parse_e_from_folder(folder_name: str) -> float:
    return float(folder_name.split("_")[1]) / 100.0


def load_txt(path: str) -> np.ndarray:
    return np.atleast_2d(np.loadtxt(path, comments="#"))


def list_e_folders(base_dir: str) -> list[str]:
    out = []
    for name in os.listdir(base_dir):
        full = os.path.join(base_dir, name)
        if os.path.isdir(full) and re.match(r"^e_\d{3}$", name):
            out.append(name)
    return sorted(out)


def load_steady_shear_from_case(case_dir: str, frac: float) -> dict[str, float]:
    data = load_txt(os.path.join(case_dir, "shear_stats.dat"))
    return {
        "Pxx*": average_last_fraction(data[:, 1], frac),
        "Pyy*": average_last_fraction(data[:, 2], frac),
        "Pzz*": average_last_fraction(data[:, 3], frac),
        "Pxy*": average_last_fraction(data[:, 4], frac),
    }


def load_steady_shear_from_folder(folder: str, frac: float) -> dict[str, float]:
    single = os.path.join(folder, "shear_stats.dat")
    if os.path.exists(single):
        out = load_steady_shear_from_case(folder, frac)
        out["nrep"] = 1
        return out

    rep_files = sorted(glob(os.path.join(folder, "seed_*", "shear_stats.dat")))
    if not rep_files:
        raise FileNotFoundError(f"No shear_stats.dat in {folder} or {folder}/seed_*")

    vals = [load_txt(p) for p in rep_files]
    pxx = [average_last_fraction(v[:, 1], frac) for v in vals]
    pyy = [average_last_fraction(v[:, 2], frac) for v in vals]
    pzz = [average_last_fraction(v[:, 3], frac) for v in vals]
    pxy = [average_last_fraction(v[:, 4], frac) for v in vals]
    return {
        "Pxx*": float(np.mean(pxx)),
        "Pyy*": float(np.mean(pyy)),
        "Pzz*": float(np.mean(pzz)),
        "Pxy*": float(np.mean(pxy)),
        "nrep": len(rep_files),
    }


def collect_stress_series(base_dir: str, frac: float) -> dict[str, np.ndarray]:
    es, pxx, pyy, pzz, pxy = [], [], [], [], []
    for folder in list_e_folders(base_dir):
        vals = load_steady_shear_from_folder(os.path.join(base_dir, folder), frac)
        es.append(parse_e_from_folder(folder))
        pxx.append(vals["Pxx*"])
        pyy.append(vals["Pyy*"])
        pzz.append(vals["Pzz*"])
        pxy.append(vals["Pxy*"])
    idx = np.argsort(np.array(es))
    return {
        "e": np.array(es)[idx],
        "Pxx*": np.array(pxx)[idx],
        "Pyy*": np.array(pyy)[idx],
        "Pzz*": np.array(pzz)[idx],
        "Pxy*": np.array(pxy)[idx],
    }


def load_steady_temp_ratio(case_dir: str, frac: float) -> float:
    data = load_txt(os.path.join(case_dir, "temperature_stats.dat"))
    ttr = average_last_fraction(data[:, 1], frac)
    trot = average_last_fraction(data[:, 2], frac)
    return ttr / (trot + 1.0e-30)


def collect_temp_ratio_series(base_dir: str, frac: float) -> tuple[np.ndarray, np.ndarray]:
    es, ratios = [], []
    for folder in list_e_folders(base_dir):
        case_dir = os.path.join(base_dir, folder)
        temp_path = os.path.join(case_dir, "temperature_stats.dat")
        if not os.path.exists(temp_path):
            continue
        es.append(parse_e_from_folder(folder))
        ratios.append(load_steady_temp_ratio(case_dir, frac))
    if not es:
        return np.array([]), np.array([])
    idx = np.argsort(np.array(es))
    return np.array(es)[idx], np.array(ratios)[idx]


def e_to_folder_name(e: float) -> str:
    return f"e_{int(round(e * 100.0)):03d}"


def parse_velocity_profile_snapshots(path: str) -> list[dict[str, np.ndarray]]:
    snaps: list[dict[str, np.ndarray]] = []
    with open(path, "r", encoding="utf-8") as fh:
        lines = iter(fh)
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 3:
                continue

            try:
                _ = int(float(parts[0]))        # timestep
                nchunk = int(float(parts[1]))   # number of chunks
            except ValueError:
                continue

            coord = np.zeros(nchunk, dtype=float)
            ncount = np.zeros(nchunk, dtype=float)
            vx = np.zeros(nchunk, dtype=float)
            for i in range(nchunk):
                cparts = next(lines).split()
                coord[i] = float(cparts[1])
                ncount[i] = float(cparts[2])
                vx[i] = float(cparts[3])

            snaps.append({"coord": coord, "ncount": ncount, "vx": vx})

    if not snaps:
        raise ValueError(f"No velocity-profile snapshots parsed from {path}")
    return snaps


def load_steady_velocity_profile(case_dir: str, frac: float) -> tuple[np.ndarray, np.ndarray]:
    path = os.path.join(case_dir, "velocity_profile.dat")
    snaps = parse_velocity_profile_snapshots(path)

    ns = len(snaps)
    i0 = int((1.0 - frac) * ns)
    i0 = max(0, min(i0, ns - 1))
    tail = snaps[i0:]

    coord = tail[0]["coord"]
    sum_w = np.zeros_like(coord)
    sum_v = np.zeros_like(coord)

    for s in tail:
        w = s["ncount"]
        sum_w += w
        sum_v += s["vx"] * w

    vx_mean = np.full_like(coord, np.nan)
    nz = sum_w > 0.0
    vx_mean[nz] = sum_v[nz] / sum_w[nz]

    return coord[nz], vx_mean[nz]


def parse_e_list(text: str) -> List[float]:
    vals = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(float(tok))
    if not vals:
        raise ValueError("No valid values in --vel-cases")
    return vals


def find_input_script(case_dir: str) -> str | None:
    preferred = [
        "in.sphcyl_usf",
        "in.chialvo",
    ]
    for name in preferred:
        p = os.path.join(case_dir, name)
        if os.path.exists(p):
            return p
    candidates = sorted(glob(os.path.join(case_dir, "in.*")))
    return candidates[0] if candidates else None


def parse_gdot_from_input(case_dir: str) -> float | None:
    in_path = find_input_script(case_dir)
    if in_path is None:
        return None
    with open(in_path, "r", encoding="utf-8", errors="ignore") as fh:
        txt = fh.read()
    m = re.search(r"^\s*variable\s+gdot\s+equal\s+([0-9eE+\-.]+)\s*$", txt, flags=re.MULTILINE)
    return float(m.group(1)) if m else None


def parse_ly_from_log(case_dir: str) -> float | None:
    log_path = os.path.join(case_dir, "log.lammps")
    if not os.path.exists(log_path):
        return None

    pat = re.compile(r"(?:orthogonal|triclinic)\s+box\s*=\s*\(([^)]+)\)\s+to\s+\(([^)]+)\)")
    ylo = None
    yhi = None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = pat.search(line)
            if not m:
                continue
            lo = [float(v) for v in m.group(1).split()]
            hi = [float(v) for v in m.group(2).split()]
            if len(lo) >= 2 and len(hi) >= 2:
                ylo = lo[1]
                yhi = hi[1]
    if ylo is None or yhi is None:
        return None
    return yhi - ylo


def infer_wall_velocity(case_dir: str, vx_profile: np.ndarray) -> float:
    gdot = parse_gdot_from_input(case_dir)
    ly = parse_ly_from_log(case_dir)
    if gdot is not None and ly is not None:
        uw = gdot * ly
        if abs(uw) > 0.0:
            return uw

    # Fallback if metadata is missing.
    vmax = float(np.nanmax(np.abs(vx_profile)))
    return vmax if vmax > 0.0 else 1.0


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Compare reduced stresses: spheres (dilute3) vs spherocylinders (AR=1.5 and AR=2)."
    )
    parser.add_argument("--sphere-dir", default=os.path.join(script_dir, "dilute3"))
    parser.add_argument("--sphcyl-ar15-dir", default=os.path.join(script_dir, "sphcyl_USF_AR15"))
    parser.add_argument("--sphcyl-ar2-dir", default=os.path.join(script_dir, "sphcyl_USF_AR2"))
    parser.add_argument("--avg-frac", type=float, default=0.3, help="Tail fraction used for steady averaging.")
    parser.add_argument(
        "--out",
        default=os.path.join(script_dir, "sphere_vs_sphcyl_AR15_AR2_reduced_stress.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--temp-ratio-out",
        default=os.path.join(script_dir, "sphcyl_AR15_AR2_Ttr_over_Trot_vs_e.png"),
        help="Output PNG path for steady T_tr/T_rot vs e overlay.",
    )
    parser.add_argument(
        "--vel-cases",
        default="0.6,0.8,1.0",
        help="Comma-separated restitution values for velocity-profile overlays.",
    )
    parser.add_argument(
        "--vel-out-prefix",
        default=os.path.join(script_dir, "sphere_vs_sphcyl_velocity_profile"),
        help="Output prefix for velocity-profile PNGs; _eXXX.png is appended.",
    )
    args = parser.parse_args()

    sphere = collect_stress_series(args.sphere_dir, args.avg_frac)
    ar15 = collect_stress_series(args.sphcyl_ar15_dir, args.avg_frac)
    ar2 = collect_stress_series(args.sphcyl_ar2_dir, args.avg_frac)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), sharex=True)
    comps = ["Pxx*", "Pyy*", "Pzz*", "Pxy*"]
    titles = [r"$P_{xx}^*$", r"$P_{yy}^*$", r"$P_{zz}^*$", r"$P_{xy}^*$"]
    for ax, comp, ttl in zip(axes.ravel(), comps, titles):
        ax.plot(sphere["e"], sphere[comp], "o-", lw=1.8, ms=5, label="Sphere")
        ax.plot(ar15["e"], ar15[comp], "^-", lw=1.8, ms=5, label="Spherocyl AR=1.5")
        ax.plot(ar2["e"], ar2[comp], "s-", lw=1.8, ms=5, label="Spherocyl AR=2.0")
        ax.set_title(ttl)
        ax.set_ylabel("Reduced Stress")
        ax.grid(alpha=0.3)
    axes[1, 0].set_xlabel("Coefficient of Restitution e")
    axes[1, 1].set_xlabel("Coefficient of Restitution e")
    axes[0, 0].legend(loc="best", fontsize=9)
    fig.suptitle("Reduced Stress Tensor: Sphere vs Spherocylinder (AR=1.5, 2.0)", y=0.98)
    fig.tight_layout()
    fig.savefig(args.out, dpi=180)

    e15, r15 = collect_temp_ratio_series(args.sphcyl_ar15_dir, args.avg_frac)
    e2, r2 = collect_temp_ratio_series(args.sphcyl_ar2_dir, args.avg_frac)

    if len(e15) == 0 and len(e2) == 0:
        raise FileNotFoundError(
            "No temperature_stats.dat found in either spherocylinder directory."
        )

    fig2, ax = plt.subplots(1, 1, figsize=(7.0, 5.0))
    if len(e15) > 0:
        ax.plot(e15, r15, "^-", lw=1.8, ms=6, label="Spherocyl AR=1.5")
    if len(e2) > 0:
        ax.plot(e2, r2, "s-", lw=1.8, ms=6, label="Spherocyl AR=2.0")
    ax.set_xlabel("Coefficient of Restitution e")
    ax.set_ylabel(r"Steady $T_{tr}/T_{rot}$")
    ax.set_title(r"Spherocylinders: Steady $T_{tr}/T_{rot}$ vs e")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    fig2.tight_layout()
    fig2.savefig(args.temp_ratio_out, dpi=180)

    print("Saved:")
    print(f"  {args.out}")
    print(f"  {args.temp_ratio_out}")

    vel_cases = parse_e_list(args.vel_cases)
    vel_saved = []
    for e_val in vel_cases:
        folder = e_to_folder_name(e_val)
        case_sphere = os.path.join(args.sphere_dir, folder)
        case_ar15 = os.path.join(args.sphcyl_ar15_dir, folder)
        case_ar2 = os.path.join(args.sphcyl_ar2_dir, folder)

        y_sph, vx_sph = load_steady_velocity_profile(case_sphere, args.avg_frac)
        y_15, vx_15 = load_steady_velocity_profile(case_ar15, args.avg_frac)
        y_2, vx_2 = load_steady_velocity_profile(case_ar2, args.avg_frac)

        uw_sph = infer_wall_velocity(case_sphere, vx_sph)
        uw_15 = infer_wall_velocity(case_ar15, vx_15)
        uw_2 = infer_wall_velocity(case_ar2, vx_2)

        un_sph = vx_sph / uw_sph
        un_15 = vx_15 / uw_15
        un_2 = vx_2 / uw_2

        figv, axv = plt.subplots(1, 1, figsize=(7.2, 5.0))
        axv.plot(un_sph, y_sph, "o-", lw=1.7, ms=4.5, label="Sphere")
        axv.plot(un_15, y_15, "^-", lw=1.7, ms=4.5, label="Spherocyl AR=1.5")
        axv.plot(un_2, y_2, "s-", lw=1.7, ms=4.5, label="Spherocyl AR=2.0")
        axv.set_xlabel(r"Normalized velocity $u_x/U_w$")
        axv.set_ylabel(r"Reduced coordinate $y/L_y$")
        axv.set_title(f"Normalized Velocity Profile Overlay at e={e_val:.1f}")
        axv.grid(alpha=0.3)
        axv.legend(loc="best", fontsize=9)
        figv.tight_layout()

        out_path = f"{args.vel_out_prefix}_{folder}.png"
        figv.savefig(out_path, dpi=180)
        plt.close(figv)
        vel_saved.append(out_path)

    for p in vel_saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
