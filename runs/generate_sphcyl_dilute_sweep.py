#!/usr/bin/env python3
"""
Generate dilute USF spherocylinder sweep folders from runs/in.sphcyl_dilute_baseline.

Default sweep matches sphere dilute3-style e-scan at fixed phi:
  e in [0.60, 0.65, ..., 1.00], phi=0.01, AR=2.0, gdot=1.8, N=2000.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


DEFAULT_E = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]


def set_equal_var(text: str, name: str, value: str) -> str:
    pat = re.compile(rf"^(\s*variable\s+{re.escape(name)}\s+equal\s+)(\S+)(.*)$", re.MULTILINE)
    out, n = pat.subn(rf"\g<1>{value}\g<3>", text, count=1)
    if n != 1:
        raise ValueError(f"Could not set variable '{name}' in template")
    return out


def e_tag(e: float) -> str:
    return f"e_{int(round(e * 100)):03d}"


def parse_e_list(s: str) -> list[float]:
    vals = []
    for part in s.split(","):
        p = part.strip()
        if not p:
            continue
        vals.append(float(p))
    if not vals:
        raise ValueError("Empty e-list")
    return vals


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Generate dilute USF spherocylinder sweep cases.")
    parser.add_argument(
        "--template",
        type=Path,
        default=script_dir / "in.sphcyl_dilute_baseline",
        help="Template input file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "sphcyl_dilute3",
        help="Directory that will contain e_* case folders.",
    )
    parser.add_argument(
        "--cases-file",
        type=Path,
        default=script_dir / "sphcyl_dilute_cases.txt",
        help="Output case-list file for array submission.",
    )
    parser.add_argument(
        "--input-name",
        default="in.sphcyl_usf",
        help="LAMMPS input filename written in each case folder.",
    )
    parser.add_argument("--n-part", type=int, default=2000, help="Number of particles.")
    parser.add_argument("--phi", type=float, default=0.01, help="Volume fraction.")
    parser.add_argument("--ar", type=float, default=2.0, help="Aspect ratio AR=(L+d)/d.")
    parser.add_argument("--gdot", type=float, default=1.8, help="Shear rate.")
    parser.add_argument("--seed-base", type=int, default=12345, help="Base random seed.")
    parser.add_argument(
        "--e-list",
        default=",".join(f"{x:.2f}" for x in DEFAULT_E),
        help="Comma-separated restitution list (e.g. 0.60,0.70,0.80).",
    )
    args = parser.parse_args()

    template = args.template.resolve().read_text()
    out_dir = args.output_dir.resolve()
    cases_file = args.cases_file.resolve()
    e_values = parse_e_list(args.e_list)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    case_rel_paths: list[str] = []
    for i, e in enumerate(e_values):
        text = template
        text = set_equal_var(text, "N_part", str(args.n_part))
        text = set_equal_var(text, "phi", f"{args.phi:.4g}")
        text = set_equal_var(text, "AR", f"{args.ar:.6g}")
        text = set_equal_var(text, "e", f"{e:.6g}")
        text = set_equal_var(text, "gdot", f"{args.gdot:.6g}")
        text = set_equal_var(text, "seed", str(args.seed_base + i))

        case_dir = out_dir / e_tag(e)
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / args.input_name).write_text(text)

        try:
            case_rel = case_dir.relative_to(script_dir)
        except ValueError:
            case_rel = case_dir
        case_rel_paths.append(str(case_rel))

    cases_file.write_text("\n".join(case_rel_paths) + "\n")

    print(f"Generated {len(case_rel_paths)} cases in: {out_dir}")
    print(f"Input filename per case: {args.input_name}")
    print(f"Case list: {cases_file}")


if __name__ == "__main__":
    main()
