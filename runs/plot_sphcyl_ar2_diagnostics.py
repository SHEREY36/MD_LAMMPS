#!/usr/bin/env python3
import argparse
import csv
import os
import re
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def average_last_fraction(arr: np.ndarray, frac: float) -> float:
    n = len(arr)
    if n == 0:
        return float("nan")
    i0 = int((1.0 - frac) * n)
    i0 = max(0, min(i0, n - 1))
    return float(np.mean(arr[i0:]))


def tail_slice(arr: np.ndarray, frac: float) -> np.ndarray:
    n = len(arr)
    if n == 0:
        return arr
    i0 = int((1.0 - frac) * n)
    i0 = max(0, min(i0, n - 1))
    return arr[i0:]


def load_txt(path: str) -> np.ndarray:
    return np.atleast_2d(np.loadtxt(path, comments="#"))


def parse_e_from_folder(name: str) -> float:
    return float(name.split("_")[1]) / 100.0


def e_to_folder_name(e: float) -> str:
    return f"e_{int(round(e * 100.0)):03d}"


def list_e_folders(base_dir: str) -> list[str]:
    out = []
    for name in os.listdir(base_dir):
        full = os.path.join(base_dir, name)
        if os.path.isdir(full) and re.match(r"^e_\d{3}$", name):
            out.append(name)
    return sorted(out)


def find_input_script(case_dir: str) -> str | None:
    preferred = ["in.sphcyl_usf", "in.chialvo"]
    for name in preferred:
        p = os.path.join(case_dir, name)
        if os.path.exists(p):
            return p
    for name in sorted(os.listdir(case_dir)):
        if name.startswith("in."):
            return os.path.join(case_dir, name)
    return None


def parse_gdot_from_input(case_dir: str) -> float | None:
    in_path = find_input_script(case_dir)
    if in_path is None:
        return None
    with open(in_path, "r", encoding="utf-8", errors="ignore") as fh:
        text = fh.read()
    match = re.search(
        r"^\s*variable\s+gdot\s+equal\s+([0-9eE+\-.]+)\s*$",
        text,
        flags=re.MULTILINE,
    )
    return float(match.group(1)) if match else None


def parse_ly_from_log(case_dir: str) -> float | None:
    log_path = os.path.join(case_dir, "log.lammps")
    if not os.path.exists(log_path):
        return None
    pat = re.compile(r"(?:orthogonal|triclinic)\s+box\s*=\s*\(([^)]+)\)\s+to\s+\(([^)]+)\)")
    ylo = None
    yhi = None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            match = pat.search(line)
            if not match:
                continue
            lo = [float(v) for v in match.group(1).split()]
            hi = [float(v) for v in match.group(2).split()]
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
    vmax = float(np.nanmax(np.abs(vx_profile)))
    return vmax if vmax > 0.0 else 1.0


def parse_numeric_variable_from_input(case_dir: str, var_name: str) -> float | None:
    in_path = find_input_script(case_dir)
    if in_path is None:
        return None
    with open(in_path, "r", encoding="utf-8", errors="ignore") as fh:
        text = fh.read()
    match = re.search(
        rf"^\s*variable\s+{re.escape(var_name)}\s+equal\s+([0-9eE+\-.]+)\s*$",
        text,
        flags=re.MULTILINE,
    )
    return float(match.group(1)) if match else None


def parse_stage_plan(case_dir: str) -> dict[str, float]:
    gdot = parse_numeric_variable_from_input(case_dir, "gdot")
    strain_warmup = parse_numeric_variable_from_input(case_dir, "strain_warmup")
    strain_transient = parse_numeric_variable_from_input(case_dir, "strain_transient")
    strain_prod = parse_numeric_variable_from_input(case_dir, "strain_prod")
    nprod_blocks = parse_numeric_variable_from_input(case_dir, "Nprod_blocks")
    if None in (gdot, strain_warmup, strain_transient, strain_prod, nprod_blocks):
        raise ValueError(f"Could not parse stage plan from {case_dir}")

    warmup_end = strain_warmup / gdot
    transient_each = strain_transient / gdot
    transient_end = warmup_end + 2.0 * transient_each
    production_end = transient_end + nprod_blocks * strain_prod / gdot
    return {
        "gdot": gdot,
        "warmup_end": warmup_end,
        "transient_start": warmup_end,
        "transient_mid": warmup_end + transient_each,
        "transient_end": transient_end,
        "production_start": transient_end,
        "production_end": production_end,
    }


def parse_thermo_history(case_dir: str) -> dict[str, np.ndarray]:
    log_path = os.path.join(case_dir, "log.lammps")
    if not os.path.exists(log_path):
        raise FileNotFoundError(log_path)

    columns = None
    parsed_columns = None
    rows = []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if parts[:2] == ["Step", "Time"] and "c_Ttrans" in parts and "v_Trot_pec" in parts:
                columns = parts
                parsed_columns = parts
                continue
            if columns is None:
                continue
            if len(parts) != len(columns):
                columns = None
                continue
            try:
                rows.append([float(tok) for tok in parts])
            except ValueError:
                columns = None

    if not rows:
        raise ValueError(f"No thermo history parsed from {log_path}")

    data = np.asarray(rows, dtype=float)
    if parsed_columns is None:
        raise ValueError(f"No thermo header parsed from {log_path}")
    out = {name: data[:, i] for i, name in enumerate(parsed_columns)}

    time = out["Time"]
    continuous = np.zeros_like(time)
    offset = 0.0
    prev_t = time[0]
    continuous[0] = time[0]
    for i in range(1, len(time)):
        if time[i] < prev_t - 1.0e-12:
            offset += prev_t
        continuous[i] = time[i] + offset
        prev_t = time[i]
    out["time_cont"] = continuous
    return out


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
                int(float(parts[0]))
                nchunk = int(float(parts[1]))
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
    snaps = parse_velocity_profile_snapshots(os.path.join(case_dir, "velocity_profile.dat"))
    ns = len(snaps)
    i0 = int((1.0 - frac) * ns)
    i0 = max(0, min(i0, ns - 1))
    tail = snaps[i0:]

    coord = tail[0]["coord"]
    sum_w = np.zeros_like(coord)
    sum_v = np.zeros_like(coord)
    for snap in tail:
        w = snap["ncount"]
        sum_w += w
        sum_v += snap["vx"] * w

    valid = sum_w > 0.0
    vx_mean = np.full_like(coord, np.nan)
    vx_mean[valid] = sum_v[valid] / sum_w[valid]
    return coord[valid], vx_mean[valid]


def linear_fit_stats(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    coeff = np.polyfit(x, y, 1)
    fit = np.polyval(coeff, x)
    resid = y - fit
    ss_res = float(np.sum(resid * resid))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return fit, float(coeff[0]), float(r2)


def thin_series(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_points:
        return x, y
    step = max(1, int(np.ceil(len(x) / max_points)))
    return x[::step], y[::step]


def tail_metrics(series: np.ndarray, frac: float) -> dict[str, float]:
    tail = tail_slice(series, frac)
    mean = float(np.mean(tail))
    std = float(np.std(tail))
    cv_pct = 100.0 * std / (abs(mean) + 1.0e-30)
    drift_pct = 100.0 * (tail[-1] - tail[0]) / (abs(mean) + 1.0e-30)
    return {
        "mean": mean,
        "std": std,
        "cv_pct": cv_pct,
        "drift_pct": drift_pct,
    }


def collect_summary(base_dir: str, frac: float) -> list[dict[str, float]]:
    rows = []
    for folder in list_e_folders(base_dir):
        case_dir = os.path.join(base_dir, folder)
        e = parse_e_from_folder(folder)
        shear = load_txt(os.path.join(case_dir, "shear_stats.dat"))
        temp = load_txt(os.path.join(case_dir, "temperature_stats.dat"))

        ttr = temp[:, 1]
        trot = temp[:, 2]
        tmix = (3.0 * ttr + 2.0 * trot) / 5.0

        row = {
            "e": e,
            "Pxx*": average_last_fraction(shear[:, 1], frac),
            "Pyy*": average_last_fraction(shear[:, 2], frac),
            "Pzz*": average_last_fraction(shear[:, 3], frac),
            "Pxy*": average_last_fraction(shear[:, 4], frac),
            "Ttr_mean": average_last_fraction(ttr, frac),
            "Trot_mean": average_last_fraction(trot, frac),
            "Tmix_mean": average_last_fraction(tmix, frac),
        }
        row["Ttr_over_Trot"] = row["Ttr_mean"] / (row["Trot_mean"] + 1.0e-30)

        for prefix, series in (
            ("Pxx", shear[:, 1]),
            ("Pyy", shear[:, 2]),
            ("Pzz", shear[:, 3]),
            ("Pxy", shear[:, 4]),
        ):
            metrics = tail_metrics(series, frac)
            row[f"{prefix}_std"] = metrics["std"]
            row[f"{prefix}_cv_pct"] = metrics["cv_pct"]
            row[f"{prefix}_drift_pct"] = metrics["drift_pct"]

        for prefix, series in (("Ttr", ttr), ("Trot", trot), ("Tmix", tmix)):
            metrics = tail_metrics(series, frac)
            row[f"{prefix}_std"] = metrics["std"]
            row[f"{prefix}_cv_pct"] = metrics["cv_pct"]
            row[f"{prefix}_drift_pct"] = metrics["drift_pct"]

        rows.append(row)

    rows.sort(key=lambda x: x["e"])
    return rows


def write_summary_csv(rows: Iterable[dict[str, float]], out_path: str) -> None:
    rows = list(rows)
    fieldnames = [
        "e",
        "Pxx*",
        "Pyy*",
        "Pzz*",
        "Pxy*",
        "Pxx_std",
        "Pxx_cv_pct",
        "Pxx_drift_pct",
        "Pyy_std",
        "Pyy_cv_pct",
        "Pyy_drift_pct",
        "Pzz_std",
        "Pzz_cv_pct",
        "Pzz_drift_pct",
        "Pxy_std",
        "Pxy_cv_pct",
        "Pxy_drift_pct",
        "Ttr_mean",
        "Trot_mean",
        "Tmix_mean",
        "Ttr_over_Trot",
        "Ttr_std",
        "Ttr_cv_pct",
        "Ttr_drift_pct",
        "Trot_std",
        "Trot_cv_pct",
        "Trot_drift_pct",
        "Tmix_std",
        "Tmix_cv_pct",
        "Tmix_drift_pct",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_stress(rows: list[dict[str, float]], out_path: str) -> None:
    e = np.array([r["e"] for r in rows])
    comps = ["Pxx*", "Pyy*", "Pzz*", "Pxy*"]
    titles = [r"$P_{xx}^*$", r"$P_{yy}^*$", r"$P_{zz}^*$", r"$P_{xy}^*$"]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), sharex=True)
    for ax, comp, title in zip(axes.ravel(), comps, titles):
        y = np.array([r[comp] for r in rows])
        ax.plot(e, y, "s-", lw=1.9, ms=5.5, color="tab:green")
        ax.set_title(title)
        ax.set_ylabel("Reduced Stress")
        ax.grid(alpha=0.3)
    axes[1, 0].set_xlabel("Coefficient of Restitution e")
    axes[1, 1].set_xlabel("Coefficient of Restitution e")
    fig.suptitle("Reduced Stress Tensor - Spherocylinder USF (AR=2.0)", y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_temperature_diagnostics(
    base_dir: str,
    rows: list[dict[str, float]],
    frac: float,
    rep_e: float,
    out_path: str,
) -> None:
    e = np.array([r["e"] for r in rows])
    ratio = np.array([r["Ttr_over_Trot"] for r in rows])

    case_dir = os.path.join(base_dir, e_to_folder_name(rep_e))
    stage = parse_stage_plan(case_dir)
    thermo = parse_thermo_history(case_dir)
    time_full = thermo["time_cont"]
    ttr_full = thermo["c_Ttrans"]
    trot_full = thermo["v_Trot_pec"]
    tmix_full = (3.0 * ttr_full + 2.0 * trot_full) / 5.0

    prod_mask = time_full >= stage["production_start"]
    time_prod = time_full[prod_mask]
    ttr_prod = ttr_full[prod_mask]
    trot_prod = trot_full[prod_mask]
    tmix_prod = tmix_full[prod_mask]

    nprod = len(time_prod)
    i0 = int((1.0 - frac) * nprod)
    i0 = max(0, min(i0, nprod - 1))
    tail_start = time_prod[i0]

    fig, axes = plt.subplots(1, 3, figsize=(19.0, 5.4))

    ax = axes[0]
    ax.plot(e, ratio, "s-", lw=1.9, ms=5.5, color="tab:orange")
    ax.set_xlabel("Coefficient of Restitution e")
    ax.set_ylabel(r"Steady $T_{tr}/T_{rot}$")
    ax.set_title(r"Asymptotic Temperature Ratio - AR=2.0")
    ax.grid(alpha=0.3)

    ax = axes[1]
    for lo, hi, color, label in (
        (0.0, stage["warmup_end"], "#d9edf7", "e=1 warmup"),
        (stage["transient_start"], stage["transient_end"], "#fcf8e3", "dissipative transient"),
        (stage["production_start"], stage["production_end"], "#eef7ea", "production"),
    ):
        ax.axvspan(lo, hi, color=color, alpha=0.65, lw=0.0, label=label)
    ax.axvspan(tail_start, stage["production_end"], color="#ccebc5", alpha=0.85, lw=0.0, label=f"tail avg ({frac:.0%})")

    tx, yy = thin_series(time_full, ttr_full, 1600)
    ax.plot(tx, yy, lw=1.3, color="tab:blue", label=r"$T_{tr}$")
    tx, yy = thin_series(time_full, trot_full, 1600)
    ax.plot(tx, yy, lw=1.3, color="tab:orange", label=r"$T_{rot}$")
    tx, yy = thin_series(time_full, tmix_full, 1600)
    ax.plot(tx, yy, "--", lw=1.8, color="tab:green", label=r"$(3T_{tr}+2T_{rot})/5$")
    ax.axvline(stage["warmup_end"], color="0.35", lw=1.0)
    ax.axvline(stage["transient_end"], color="0.35", lw=1.0)
    ax.set_xlabel("Physical time")
    ax.set_ylabel("Temperature")
    ax.set_title(f"Temperature Evolution From Onset ({e_to_folder_name(rep_e)})")
    ax.grid(alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    dedup_h = []
    dedup_l = []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        dedup_h.append(h)
        dedup_l.append(l)
    ax.legend(dedup_h, dedup_l, loc="best", fontsize=8.4)

    ax = axes[2]
    tail_time = time_prod[i0:]
    offset = tail_time - tail_time[0]
    tail_ttr = ttr_prod[i0:]
    tail_trot = trot_prod[i0:]
    tail_tmix = tmix_prod[i0:]
    tx, yy = thin_series(offset, tail_ttr, 900)
    ax.plot(tx, yy, lw=1.5, color="tab:blue", label=r"$T_{tr}$")
    tx, yy = thin_series(offset, tail_trot, 900)
    ax.plot(tx, yy, lw=1.5, color="tab:orange", label=r"$T_{rot}$")
    tx, yy = thin_series(offset, tail_tmix, 900)
    ax.plot(tx, yy, "--", lw=1.8, color="tab:green", label=r"$(3T_{tr}+2T_{rot})/5$")
    ax.axhline(np.mean(tail_ttr), color="tab:blue", lw=1.0, alpha=0.45)
    ax.axhline(np.mean(tail_trot), color="tab:orange", lw=1.0, alpha=0.45)
    ax.axhline(np.mean(tail_tmix), color="tab:green", lw=1.0, alpha=0.45)
    ax.set_xlabel("Physical time offset within tail")
    ax.set_ylabel("Temperature")
    ax.set_title(f"Tail Window Used For Averages ({frac:.0%} of production)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8.5)

    fig.suptitle("Temperature Diagnostics - Spherocylinder USF (AR=2.0)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_velocity_profiles(base_dir: str, frac: float, e_vals: list[float], out_path: str) -> None:
    n = len(e_vals)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.8), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, e_val in zip(axes, e_vals):
        case_dir = os.path.join(base_dir, e_to_folder_name(e_val))
        y, vx = load_steady_velocity_profile(case_dir, frac)
        uw = infer_wall_velocity(case_dir, vx)
        un = vx / uw
        fit, slope, r2 = linear_fit_stats(y, un)

        ax.plot(un, y, "s-", lw=1.5, ms=4.2, color="tab:green", label="AR=2.0")
        ax.plot(fit, y, "--", lw=1.4, color="black", alpha=0.75, label=f"linear fit, m={slope:.3f}, $R^2$={r2:.4f}")
        ax.set_title(f"e={e_val:.2f}")
        ax.set_xlabel(r"Normalized velocity $u_x/U_w$")
        ax.set_ylabel(r"Reduced coordinate $y/L_y$")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8.5)

    fig.suptitle("Steady Velocity Profiles - Spherocylinder USF (AR=2.0)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="AR=2.0 diagnostics for spherocylinder USF runs.")
    parser.add_argument("--base-dir", default=os.path.join(script_dir, "sphcyl_USF_AR2"))
    parser.add_argument("--avg-frac", type=float, default=0.50, help="Tail fraction used for steady averages.")
    parser.add_argument("--rep-e", type=float, default=0.95, help="Representative restitution used for the temperature-evolution panel.")
    parser.add_argument("--vel-cases", default="0.60,0.80,1.00", help="Comma-separated restitution values used for velocity-profile panels.")
    parser.add_argument("--stress-out", default=os.path.join(script_dir, "sphcyl_AR2_reduced_stress.png"))
    parser.add_argument("--temp-out", default=os.path.join(script_dir, "sphcyl_AR2_temperature_diagnostics.png"))
    parser.add_argument("--velocity-out", default=os.path.join(script_dir, "sphcyl_AR2_velocity_profiles.png"))
    parser.add_argument("--summary-out", default=os.path.join(script_dir, "sphcyl_AR2_steady_summary.csv"))
    args = parser.parse_args()

    e_vals = [float(tok.strip()) for tok in args.vel_cases.split(",") if tok.strip()]
    rows = collect_summary(args.base_dir, args.avg_frac)
    if not rows:
        raise FileNotFoundError(f"No e_* cases found under {args.base_dir}")

    write_summary_csv(rows, args.summary_out)
    plot_stress(rows, args.stress_out)
    plot_temperature_diagnostics(args.base_dir, rows, args.avg_frac, args.rep_e, args.temp_out)
    plot_velocity_profiles(args.base_dir, args.avg_frac, e_vals, args.velocity_out)

    print("Saved:")
    print(f"  {args.summary_out}")
    print(f"  {args.stress_out}")
    print(f"  {args.temp_out}")
    print(f"  {args.velocity_out}")


if __name__ == "__main__":
    main()
