#!/usr/bin/env python3
import argparse
import sys
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate collision_events.dat for single-collision test.")
    parser.add_argument("--file", default="collision_events.dat", help="Path to collision_events.dat")
    parser.add_argument("--expected", type=int, default=1, help="Expected final cumulative events")
    args = parser.parse_args()

    try:
        data = np.loadtxt(args.file, comments="#")
    except OSError as exc:
        print(f"ERROR: could not read {args.file}: {exc}")
        return 2

    data = np.atleast_2d(data)
    if data.shape[1] < 3:
        print(f"ERROR: {args.file} has unexpected format (need 3 columns)")
        return 2

    step = data[:, 0]
    new_events = data[:, 1]
    cumulative = data[:, 2]

    final_cum = int(round(cumulative[-1]))
    nonzero_rows = int(np.count_nonzero(new_events > 0.5))
    sum_new = int(round(np.sum(new_events)))

    print(f"File           : {args.file}")
    print(f"Rows           : {len(step)}")
    print(f"Final cumulative events : {final_cum}")
    print(f"Sum(new_events): {sum_new}")
    print(f"Rows with new_events>0 : {nonzero_rows}")

    if final_cum != args.expected:
        print(f"FAIL: expected final cumulative = {args.expected}, got {final_cum}")
        return 1
    if sum_new != args.expected:
        print(f"FAIL: expected sum(new_events) = {args.expected}, got {sum_new}")
        return 1

    print("PASS: collision-event counter matches expected single-collision behavior.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
