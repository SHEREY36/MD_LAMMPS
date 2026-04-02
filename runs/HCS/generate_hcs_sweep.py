#!/usr/bin/env python3
"""
Generate HCS Mode-B sweep folders for spherocylinders.

Default sweep:
  e = 0.50, 0.55, ..., 1.00
with one case folder per restitution value:
  modeB_e_sweep/e_050, ..., modeB_e_sweep/e_100
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def set_equal_var(text: str, name: str, value: str) -> str:
    pat = re.compile(rf"^(\s*variable\s+{re.escape(name)}\s+equal\s+)(\S+)(.*)$", re.MULTILINE)
    out, n = pat.subn(rf"\g<1>{value}\g<3>", text, count=1)
    if n != 1:
        raise ValueError(f"Could not set variable '{name}'")
    return out


def set_string_var(text: str, name: str, value: str) -> str:
    pat = re.compile(rf"^(\s*variable\s+{re.escape(name)}\s+string\s+)(\S+)(.*)$", re.MULTILINE)
    out, n = pat.subn(rf"\g<1>{value}\g<3>", text, count=1)
    if n != 1:
        raise ValueError(f"Could not set string variable '{name}'")
    return out


def e_tag(e: float) -> str:
    return f"e_{int(round(e * 100)):03d}"


def build_e_values(e_start: float, e_stop: float, e_step: float) -> list[float]:
    vals = []
    e = e_start
    eps = 1.0e-12
    while e <= e_stop + eps:
        vals.append(round(e, 6))
        e += e_step
    return vals


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Generate HCS Mode-B restitution sweep folders.")
    parser.add_argument("--template", type=Path, default=script_dir / "hcs_spherocyl.in")
    parser.add_argument("--output-dir", type=Path, default=script_dir / "modeB_e_sweep")
    parser.add_argument("--cases-file", type=Path, default=script_dir / "hcs_modeB_cases.txt")
    parser.add_argument("--input-name", default="in.hcs")
    parser.add_argument("--mode", choices=("A", "B"), default="B")
    parser.add_argument("--e-start", type=float, default=0.50)
    parser.add_argument("--e-stop", type=float, default=1.00)
    parser.add_argument("--e-step", type=float, default=0.05)
    parser.add_argument("--seed-base", type=int, default=12349)
    args = parser.parse_args()

    template = args.template.resolve().read_text()
    out_dir = args.output_dir.resolve()
    cases_file = args.cases_file.resolve()
    e_values = build_e_values(args.e_start, args.e_stop, args.e_step)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    case_paths: list[str] = []
    for i, e in enumerate(e_values):
        text = template
        text = set_string_var(text, "mode", args.mode)
        text = set_equal_var(text, "e", f"{e:.6g}")
        text = set_equal_var(text, "seed", str(args.seed_base + i))

        case_dir = out_dir / e_tag(e)
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / args.input_name).write_text(text)

        try:
            rel = case_dir.relative_to(script_dir)
        except ValueError:
            rel = case_dir
        case_paths.append(str(rel))

    cases_file.write_text("\n".join(case_paths) + "\n")

    print(f"Generated {len(case_paths)} HCS cases in: {out_dir}")
    print(f"Mode: {args.mode}")
    print(f"Input per case: {args.input_name}")
    print(f"Case list: {cases_file}")


if __name__ == "__main__":
    main()
