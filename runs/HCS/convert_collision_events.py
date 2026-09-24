#!/usr/bin/env python3
"""
Convert LAMMPS collision event logs to collisions-per-particle time series.

Usage:
  python3 convert_collision_events.py collision_events.dat output.dat N_part dt
  python3 convert_collision_events.py collision_events.dat output.dat N_part dt --contact-factor 2.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("events_file", type=Path)
    parser.add_argument("output_file", type=Path)
    parser.add_argument("n_part", type=float)
    parser.add_argument("dt", type=float)
    parser.add_argument("--contact-factor", type=float, default=1.0)
    args = parser.parse_args()

    data = np.loadtxt(args.events_file, comments="#")
    data = np.atleast_2d(data)

    with args.output_file.open("w", encoding="utf-8") as fh:
        fh.write("# time cpp coll_rate_per_particle\n")
        if data.shape[1] < 3 or len(data) < 2 or args.n_part <= 0.0:
            return

        prev_t = None
        prev_cpp = None
        for row in data[1:]:
            step = float(row[0])
            cumulative_events = float(row[2])
            t = step * args.dt
            cpp = args.contact_factor * cumulative_events / args.n_part
            if prev_t is None:
                rate = 0.0
            else:
                rate = (cpp - prev_cpp) / (t - prev_t + 1.0e-30)
            fh.write(f"{t:.12g} {cpp:.12g} {rate:.12g}\n")
            prev_t = t
            prev_cpp = cpp


if __name__ == "__main__":
    main()
