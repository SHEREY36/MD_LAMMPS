#!/usr/bin/env python3
"""
Generate LAMMPS sweep folders for:
1) dilute-limit tensor plot (single phi, varying e)
2) Chialvo-style 3 plots (varying e and phi)

Uses runs/in.chialvo_baseline as template and rewrites:
  variable phi
  variable e
  variable seed
  variable gdot   (optional override)
  variable use_sllod (mode switch)

Modes:
  mode=chialvo   -> use_sllod=0
  mode=usf_sllod -> use_sllod=1
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable


DILUTE_E = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
CHIALVO_E = [0.70, 0.80, 0.90, 0.95, 0.99]
CHIALVO_PHI = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.49, 0.50, 0.55, 0.60]
MODE_TO_USE_SLLOD = {"chialvo": 0, "usf_sllod": 1}


def set_equal_var(text: str, name: str, value: str) -> str:
    pattern = re.compile(rf"^(\s*variable\s+{re.escape(name)}\s+equal\s+)(\S+)(.*)$", re.MULTILINE)
    updated, count = pattern.subn(rf"\g<1>{value}\g<3>", text, count=1)
    if count != 1:
        raise ValueError(f"Could not uniquely set variable '{name}' in template")
    return updated


def e_tag(e: float) -> str:
    return f"e_{int(round(e * 100)):03d}"


def phi_tag(phi: float) -> str:
    return f"phi_{int(round(phi * 100)):02d}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_case(case_dir: Path, text: str) -> None:
    ensure_dir(case_dir)
    (case_dir / "in.chialvo").write_text(text)


def noncomment_lines(lines: Iterable[str]) -> list[str]:
    out = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Generate Chialvo baseline sweeps from a template input file.")
    parser.add_argument(
        "--template",
        type=Path,
        default=script_dir / "in.chialvo_baseline",
        help="Template LAMMPS input file",
    )
    parser.add_argument(
        "--dilute-dir",
        type=Path,
        default=None,
        help="Output directory for dilute sweep (default: runs/dilute_baseline_<mode>)",
    )
    parser.add_argument(
        "--chialvo-dir",
        type=Path,
        default=None,
        help="Output directory for Chialvo sweep (default: runs/jobsubmit_baseline_<mode>)",
    )
    parser.add_argument(
        "--dilute-phi",
        type=float,
        default=0.01,
        help="Volume fraction used for dilute sweep",
    )
    parser.add_argument(
        "--gdot",
        type=float,
        default=1.8,
        help="Shear rate to set in generated inputs",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=12345,
        help="Base seed used to generate unique per-case seeds",
    )
    parser.add_argument(
        "--mode",
        choices=("chialvo", "usf_sllod", "both"),
        default="both",
        help="Shear mode to generate",
    )
    parser.add_argument(
        "--low-e-threshold",
        type=float,
        default=0.80,
        help="In mode=chialvo, apply replicate seeding in dilute cases for e <= threshold",
    )
    parser.add_argument(
        "--low-e-seeds",
        type=int,
        default=3,
        help="In mode=chialvo, number of seeds for low-e dilute cases",
    )
    parser.add_argument(
        "--cases-file-prefix",
        type=Path,
        default=script_dir / "chialvo_baseline",
        help="Prefix for generated case-list files; mode suffix is added automatically",
    )
    args = parser.parse_args()

    template_path = args.template.resolve()
    cases_prefix = args.cases_file_prefix.resolve()

    template_text = template_path.read_text()
    modes = ["chialvo", "usf_sllod"] if args.mode == "both" else [args.mode]
    if args.low_e_seeds < 1:
        raise ValueError("--low-e-seeds must be >= 1")

    total_cases = 0
    for mode in modes:
        use_sllod = MODE_TO_USE_SLLOD[mode]
        if args.dilute_dir is None:
            dilute_dir = script_dir / f"dilute_baseline_{mode}"
        else:
            dilute_base = args.dilute_dir.resolve()
            dilute_dir = dilute_base / mode if len(modes) > 1 else dilute_base

        if args.chialvo_dir is None:
            chialvo_dir = script_dir / f"jobsubmit_baseline_{mode}"
        else:
            chialvo_base = args.chialvo_dir.resolve()
            chialvo_dir = chialvo_base / mode if len(modes) > 1 else chialvo_base

        dilute_rel: list[str] = []
        chialvo_rel: list[str] = []
        case_counter = 0

        ensure_dir(dilute_dir)
        ensure_dir(chialvo_dir)
        ensure_dir(dilute_dir / "logs")
        ensure_dir(chialvo_dir / "logs")

        # Dilute sweep: e only, fixed phi.
        for e in DILUTE_E:
            nrep = args.low_e_seeds if (mode == "chialvo" and e <= args.low_e_threshold + 1.0e-12) else 1
            for rep in range(1, nrep + 1):
                text = template_text
                text = set_equal_var(text, "phi", f"{args.dilute_phi:.2f}")
                text = set_equal_var(text, "e", f"{e:.2f}")
                text = set_equal_var(text, "gdot", f"{args.gdot:g}")
                text = set_equal_var(text, "use_sllod", str(use_sllod))
                text = set_equal_var(text, "seed", str(args.seed_base + case_counter))
                case_counter += 1

                if nrep == 1:
                    case_dir = dilute_dir / e_tag(e)
                else:
                    case_dir = dilute_dir / e_tag(e) / f"seed_{rep:02d}"
                write_case(case_dir, text)
                try:
                    ref = case_dir.relative_to(script_dir)
                except ValueError:
                    ref = case_dir
                dilute_rel.append(str(ref))

        # Chialvo sweep: e x phi grid.
        for e in CHIALVO_E:
            for phi in CHIALVO_PHI:
                text = template_text
                text = set_equal_var(text, "phi", f"{phi:.2f}")
                text = set_equal_var(text, "e", f"{e:.2f}")
                text = set_equal_var(text, "gdot", f"{args.gdot:g}")
                text = set_equal_var(text, "use_sllod", str(use_sllod))
                text = set_equal_var(text, "seed", str(args.seed_base + case_counter))
                case_counter += 1

                case_dir = chialvo_dir / e_tag(e) / phi_tag(phi)
                write_case(case_dir, text)
                try:
                    ref = case_dir.relative_to(script_dir)
                except ValueError:
                    ref = case_dir
                chialvo_rel.append(str(ref))

        all_rel = dilute_rel + chialvo_rel
        total_cases += len(all_rel)

        dilute_file = cases_prefix.with_name(f"{cases_prefix.name}_{mode}_dilute_cases.txt")
        chialvo_file = cases_prefix.with_name(f"{cases_prefix.name}_{mode}_chialvo_cases.txt")
        all_file = cases_prefix.with_name(f"{cases_prefix.name}_{mode}_all_cases.txt")

        dilute_file.write_text("\n".join(noncomment_lines(dilute_rel)) + "\n")
        chialvo_file.write_text("\n".join(noncomment_lines(chialvo_rel)) + "\n")
        all_file.write_text("\n".join(noncomment_lines(all_rel)) + "\n")

        print(f"\nGenerated sweep inputs for mode={mode}:")
        print(f"  dilute dir : {dilute_dir} ({len(dilute_rel)} cases)")
        print(f"  chialvo dir: {chialvo_dir} ({len(chialvo_rel)} cases)")
        print("Generated case lists:")
        print(f"  {dilute_file}")
        print(f"  {chialvo_file}")
        print(f"  {all_file}")
        print(f"Total cases (mode={mode}): {len(all_rel)}")

    if len(modes) > 1:
        print(f"\nGrand total cases across modes: {total_cases}")


if __name__ == "__main__":
    main()
