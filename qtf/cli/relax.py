#!/usr/bin/env python3
"""
Relax (energy-minimize) a PDB structure using GROMACS.

Runs pdb2gmx + steepest-descent minimization and outputs the minimized structure.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

from qtf.utils import gromacs as qtf_gromacs
from qtf.utils.paths import relativize_absolute_paths


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="qtf relax",
        description="Energy-minimize a PDB structure using GROMACS."
    )
    ap.add_argument("--input_pdb", required=True, help="Path to input PDB file")
    ap.add_argument("--outdir", default=None, help="Output directory for minimized structure (default: next to input)")
    ap.add_argument("--forcefield", default="amber99sb-ildn")
    ap.add_argument("--water", default="tip3p")
    ap.add_argument("--nsteps", type=int, default=5000, help="Maximum minimization steps")
    ap.add_argument("--emtol", type=float, default=100.0, help="Force tolerance (kJ/mol/nm)")
    ap.add_argument("--maxwarn", type=int, default=2)
    args = ap.parse_args(argv)

    input_pdb = Path(args.input_pdb).resolve()
    if not input_pdb.is_file():
        raise FileNotFoundError(f"Input PDB not found: {input_pdb}")

    outdir = Path(args.outdir or input_pdb.parent / f"{input_pdb.stem}_gromacs_minimized")
    outdir = outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[relax] minimizing {input_pdb} with GROMACS ({args.forcefield}/{args.water})")
    print(f"[relax] output dir: {outdir}")

    result = qtf_gromacs.minimize_pdb_with_gromacs(
        str(input_pdb),
        str(outdir),
        forcefield=args.forcefield,
        water=args.water,
        nsteps=args.nsteps,
        emtol=args.emtol,
        maxwarn=args.maxwarn,
    )

    gmx_status = result.get("gromacs_status", "unknown")
    print(f"[relax] GROMACS status: {gmx_status}")

    min_pdb = result.get("gromacs_minimized_full_pdb_path", "")
    if min_pdb and Path(min_pdb).is_file():
        print(f"[relax] minimized PDB: {min_pdb}")

    result_path = outdir / "relax_result.json"
    result_json = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in result.items()}
    with open(result_path, "w") as f:
        json.dump(relativize_absolute_paths(result_json), f, indent=2)
    print(f"[relax] result JSON: {result_path}")


if __name__ == "__main__":
    main()
