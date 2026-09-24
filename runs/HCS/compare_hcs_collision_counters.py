#!/usr/bin/env python3
"""Compare old/new LAMMPS HCS collision counters against calibrated DSMC."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


DEFAULT_ALPHAS = (0.9, 0.8, 0.7, 0.6)
DEFAULT_LAMMPS_ROOT = Path(
    "/home/muhammed/Documents/LAMMPS/runs/HCS/Equal/calibrate_cpp/modeB_e_sweep2"
)
DEFAULT_DSMC_ROOT = Path("/home/muhammed/Documents/Thesis/DSMC_0D")
DEFAULT_OUT = Path(
    "/home/muhammed/Documents/LAMMPS/runs/HCS/Equal/calibrate_cpp/"
    "counter_validation_AR2_e090_080_070_060_t300.png"
)
DEFAULT_SUMMARY = Path(
    "/home/muhammed/Documents/LAMMPS/runs/HCS/Equal/calibrate_cpp/"
    "counter_validation_AR2_e090_080_070_060_t300.csv"
)
N_PART = 2003
DSMC_EQUIL_TIME = 2.0
DEFAULT_TMAX = 300.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lammps-root",
        type=Path,
        default=DEFAULT_LAMMPS_ROOT,
        help="LAMMPS HCS sweep root containing e_* folders",
    )
    parser.add_argument(
        "--dsmc-root",
        type=Path,
        default=DEFAULT_DSMC_ROOT,
        help="DSMC project root",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=DEFAULT_ALPHAS,
        help="Restitution values to compare",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output figure path")
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="Output CSV summary path",
    )
    parser.add_argument(
        "--tmax",
        type=float,
        default=DEFAULT_TMAX,
        help="Maximum physical time shown and summarized",
    )
    return parser.parse_args()


def read_dt_from_log(log_path: Path) -> float:
    pattern = re.compile(r"dt=([0-9eE+\-.]+)")
    with log_path.open() as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                return float(match.group(1))
    raise RuntimeError(f"Could not find dt in {log_path}")


def load_table(path: Path) -> dict[str, float]:
    with path.open() as handle:
        return json.load(handle)


def load_array(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, comments="#")
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def downsample(t: np.ndarray, y: np.ndarray, max_points: int = 1800) -> tuple[np.ndarray, np.ndarray]:
    if len(t) <= max_points:
        return t, y
    stride = max(1, math.ceil(len(t) / max_points))
    return t[::stride], y[::stride]


def clip_series(t: np.ndarray, y: np.ndarray, tmax: float) -> tuple[np.ndarray, np.ndarray]:
    mask = t <= tmax
    if not np.any(mask):
        return np.array([0.0]), np.array([0.0])
    t_clip = t[mask]
    y_clip = y[mask]
    if t_clip[-1] < tmax and t[-1] > tmax:
        t_clip = np.append(t_clip, tmax)
        y_clip = np.append(y_clip, np.interp(tmax, t, y))
    return t_clip, y_clip


def align_old_counter(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    step = arr[:, 0]
    resets = np.where(np.diff(step) <= 0)[0] + 1
    start = int(resets[-1]) if len(resets) else 0
    stage = arr[start:].copy()
    stage[:, 0] -= stage[0, 0]
    stage[:, 2] -= stage[0, 2]
    t = stage[:, 0]
    cum = stage[:, 2]
    if t[0] != 0.0 or cum[0] != 0.0:
        t = np.insert(t, 0, 0.0)
        cum = np.insert(cum, 0, 0.0)
    return t, cum


def align_new_counter(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = arr[:, 0].copy()
    cum = arr[:, 2].copy()
    if t[0] != 0.0 or cum[0] != 0.0:
        t = np.insert(t, 0, 0.0)
        cum = np.insert(cum, 0, 0.0)
    return t, cum


def load_lammps_case(case_dir: Path) -> dict[str, np.ndarray]:
    dt = read_dt_from_log(case_dir / "log.lammps")
    old_t_step, old_cum = align_old_counter(load_array(case_dir / "collision_events.dat"))
    new_t_step, new_cum = align_new_counter(load_array(case_dir / "collision_count.dat"))
    old_t = old_t_step * dt
    new_t = new_t_step * dt
    return {
        "dt": np.array([dt]),
        "old_t": old_t,
        "old_raw": old_cum / N_PART,
        "old_x2": 2.0 * old_cum / N_PART,
        "new_t": new_t,
        "new_raw": new_cum / N_PART,
        "new_x2": 2.0 * new_cum / N_PART,
    }


def load_dsmc_case(alpha: float, dsmc_root: Path, table: dict[str, float]) -> dict[str, np.ndarray] | None:
    key = f"({alpha:.3f}, 2.0)"
    if key not in table:
        return None
    c_opt = table[key]
    alpha_int = int(round(alpha * 100))
    calib_dir = dsmc_root / "runs" / "calib_C_alpha"
    path = calib_dir / f"calib_C{c_opt:.5f}_a{alpha_int:03d}_s42.txt"
    if not path.exists():
        candidates = sorted(calib_dir.glob(f"calib_C*_a{alpha_int:03d}_s42.txt"))
        if not candidates:
            return None
        def candidate_c_value(candidate: Path) -> float:
            match = re.search(r"calib_C([0-9.]+)_a", candidate.name)
            if not match:
                return float("inf")
            return float(match.group(1))
        path = min(candidates, key=lambda candidate: abs(candidate_c_value(candidate) - c_opt))
    data = np.loadtxt(path)
    idx = int(np.searchsorted(data[:, 0], DSMC_EQUIL_TIME))
    idx = min(idx, len(data) - 1)
    t = data[idx:, 0] - data[idx, 0]
    tau = data[idx:, 1] - data[idx, 1]
    return {"t": t, "tau": tau}


def interp_last_value(t: np.ndarray, y: np.ndarray, query_t: float) -> float:
    return float(np.interp(query_t, t, y))


def build_summary_row(
    alpha: float,
    lammps: dict[str, np.ndarray],
    dsmc: dict[str, np.ndarray] | None,
    tmax: float,
) -> dict[str, float]:
    common_end = min(lammps["old_t"][-1], lammps["new_t"][-1], tmax)
    if dsmc is not None:
        common_end = min(common_end, dsmc["t"][-1])
    row = {
        "e": alpha,
        "t_common_end": common_end,
        "old_raw": interp_last_value(lammps["old_t"], lammps["old_raw"], common_end),
        "old_x2": interp_last_value(lammps["old_t"], lammps["old_x2"], common_end),
        "new_raw": interp_last_value(lammps["new_t"], lammps["new_raw"], common_end),
        "new_x2": interp_last_value(lammps["new_t"], lammps["new_x2"], common_end),
    }
    if dsmc is None:
        row["dsmc_tau"] = float("nan")
        row["old_x2_over_dsmc"] = float("nan")
        row["new_x2_over_dsmc"] = float("nan")
    else:
        row["dsmc_tau"] = interp_last_value(dsmc["t"], dsmc["tau"], common_end)
        row["old_x2_over_dsmc"] = row["old_x2"] / row["dsmc_tau"] if row["dsmc_tau"] else float("nan")
        row["new_x2_over_dsmc"] = row["new_x2"] / row["dsmc_tau"] if row["dsmc_tau"] else float("nan")
    return row


def save_summary(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "e",
        "t_common_end",
        "dsmc_tau",
        "old_raw",
        "old_x2",
        "new_raw",
        "new_x2",
        "old_x2_over_dsmc",
        "new_x2_over_dsmc",
    ]
    with path.open("w") as handle:
        handle.write(",".join(headers) + "\n")
        for row in rows:
            handle.write(",".join(f"{row[h]:.12g}" for h in headers) + "\n")


def main() -> None:
    args = parse_args()
    table = load_table(args.dsmc_root / "models" / "C_alpha_table_AR20.json")
    rows: list[dict[str, float]] = []

    fig, (ax_raw, ax_x2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    cmap = plt.get_cmap("plasma")
    colors = {alpha: cmap(0.08 + 0.84 * i / max(len(args.alphas) - 1, 1)) for i, alpha in enumerate(args.alphas)}

    for alpha in args.alphas:
        alpha_int = int(round(alpha * 100))
        case_dir = args.lammps_root / f"e_{alpha_int:03d}"
        lammps = load_lammps_case(case_dir)
        dsmc = load_dsmc_case(alpha, args.dsmc_root, table)
        rows.append(build_summary_row(alpha, lammps, dsmc, args.tmax))
        color = colors[alpha]

        if dsmc is not None:
            t_dsmc, tau_dsmc = clip_series(dsmc["t"], dsmc["tau"], args.tmax)
            t_dsmc, tau_dsmc = downsample(t_dsmc, tau_dsmc)
            ax_raw.plot(t_dsmc, tau_dsmc, color=color, linewidth=1.8)
            ax_x2.plot(t_dsmc, tau_dsmc, color=color, linewidth=1.8)

        t_old, old_raw = clip_series(lammps["old_t"], lammps["old_raw"], args.tmax)
        t_old, old_raw = downsample(t_old, old_raw)
        t_old_x2, old_x2 = clip_series(lammps["old_t"], lammps["old_x2"], args.tmax)
        t_old_x2, old_x2 = downsample(t_old_x2, old_x2)
        t_new, new_raw = clip_series(lammps["new_t"], lammps["new_raw"], args.tmax)
        t_new, new_raw = downsample(t_new, new_raw)
        t_new_x2, new_x2 = clip_series(lammps["new_t"], lammps["new_x2"], args.tmax)
        t_new_x2, new_x2 = downsample(t_new_x2, new_x2)

        ax_raw.plot(t_old, old_raw, color=color, linestyle="--", linewidth=1.4)
        ax_raw.plot(t_new, new_raw, color=color, linestyle="-.", linewidth=1.4)
        ax_x2.plot(t_old_x2, old_x2, color=color, linestyle="--", linewidth=1.4)
        ax_x2.plot(t_new_x2, new_x2, color=color, linestyle="-.", linewidth=1.4)

    ax_raw.set_title("Raw cumulative collisions per particle")
    ax_x2.set_title(r"Cumulative collisions per particle ($\times 2$)")
    for ax in (ax_raw, ax_x2):
        ax.set_xlabel("Physical time")
        ax.set_xlim(0.0, args.tmax)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.tick_params(axis="both", direction="in", top=True, right=True)
    ax_raw.set_ylabel(r"Cumulative collisions per particle $\tau$")

    color_handles = [Line2D([0], [0], color=colors[a], linewidth=2, label=f"e={a:.2f}") for a in args.alphas]
    style_handles = [
        Line2D([0], [0], color="k", linewidth=1.8, linestyle="-", label="DSMC"),
        Line2D([0], [0], color="k", linewidth=1.4, linestyle="--", label="LAMMPS old counter"),
        Line2D([0], [0], color="k", linewidth=1.4, linestyle="-.", label="LAMMPS new counter"),
    ]
    ax_raw.legend(handles=color_handles, loc="upper left", fontsize=9, ncol=1)
    ax_x2.legend(handles=style_handles, loc="upper left", fontsize=9)

    fig.suptitle(
        "AR=2.0 HCS collision-counter validation vs DSMC\n"
        "Solid: DSMC, dashed: old LAMMPS counter, dash-dot: new LAMMPS counter",
        fontsize=13,
    )
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    save_summary(rows, args.summary)
    print(f"Saved figure: {args.out}")
    print(f"Saved summary: {args.summary}")


if __name__ == "__main__":
    main()
