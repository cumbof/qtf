#!/usr/bin/env python3
"""
Score experimental/native PDB structures with the QTF energy function.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from qtf.utils.paths import relativize_absolute_paths, write_portable_csv
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from qtf.utils import workflow as utils


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="qtf eval",
        description="Score experimental/native structures with the QTF energy function."
    )
    ap.add_argument("--panel", help="JSON or CSV with columns/name,pdb_path,chain,residue_start,residue_end")
    ap.add_argument("--name")
    ap.add_argument("--pdb_path")
    ap.add_argument("--chain", default=None)
    ap.add_argument("--residue_start", type=int, default=None)
    ap.add_argument("--residue_end", type=int, default=None)
    ap.add_argument("--chi_mode", default="selective", choices=["beam", "selective", "all"])
    ap.add_argument("--rmsd_mode", default="ca", choices=["ca", "heavy"],
                    help="RMSD atom selection: all CA atoms or all heavy atoms")
    ap.add_argument("--rmsd_residue_scope", default="core", choices=["core", "all"],
                    help="Residue range used for RMSD; core excludes the first and last residues")
    ap.add_argument("--energy_backend", default="custom", choices=["custom", "openmm"])
    ap.add_argument("--use_e2e_constraint", type=int, default=1)
    ap.add_argument("--e2e_scale", type=float, default=1.0)
    ap.add_argument("--gromacs_minimize", type=int, default=None,
                    help="1 to add hydrogens/topology and minimize the rebuilt native structure with GROMACS")
    ap.add_argument("--gromacs_forcefield", default="amber99sb-ildn")
    ap.add_argument("--gromacs_water", default="tip3p")
    ap.add_argument("--gromacs_nsteps", type=int, default=5000)
    ap.add_argument("--gromacs_emtol", type=float, default=100.0)
    ap.add_argument("--gromacs_maxwarn", type=int, default=2)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args(argv)
    if args.gromacs_minimize is None:
        args.gromacs_minimize = 1

    base_output_dir = Path(args.out_json).parent if args.out_json else Path(args.out_csv).parent
    base_output_dir.mkdir(parents=True, exist_ok=True)
    pdb_output_dir = base_output_dir / "raw_pdbs"
    pdb_output_dir.mkdir(parents=True, exist_ok=True)
    gromacs_output_dir = base_output_dir / "gromacs_pdbs"
    gromacs_output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    if args.panel:
        specs = utils.load_panel(args.panel)
        for spec in specs:
            start = spec.get("residue_start")
            end = spec.get("residue_end")
            rebuilt_ca_pdb_path, rebuilt_ca_centroid_pdb_path, rebuilt_full_pdb_path = utils.make_rebuilt_output_paths(
                pdb_output_dir, spec["name"], start, end
            )
            rows.append(
                utils.score_native_structure(
                    name=spec["name"],
                    pdb_path=spec["pdb_path"],
                    chain=spec.get("chain") or None,
                    start=start,
                    end=end,
                    chi_mode=spec.get("chi_mode", args.chi_mode),
                    rmsd_mode=args.rmsd_mode,
                    rmsd_residue_scope=args.rmsd_residue_scope,
                    energy_backend=args.energy_backend,
                    use_e2e_constraint=bool(args.use_e2e_constraint),
                    e2e_scale=args.e2e_scale,
                    gromacs_minimize=bool(args.gromacs_minimize),
                    gromacs_forcefield=args.gromacs_forcefield,
                    gromacs_water=args.gromacs_water,
                    gromacs_nsteps=args.gromacs_nsteps,
                    gromacs_emtol=args.gromacs_emtol,
                    gromacs_maxwarn=args.gromacs_maxwarn,
                    rebuilt_ca_pdb_path=str(rebuilt_ca_pdb_path),
                    rebuilt_ca_centroid_pdb_path=str(rebuilt_ca_centroid_pdb_path),
                    rebuilt_full_pdb_path=str(rebuilt_full_pdb_path),
                )
            )
    else:
        if not args.name or not args.pdb_path:
            raise SystemExit("single-structure mode requires --name and --pdb_path")
        rebuilt_ca_pdb_path, rebuilt_ca_centroid_pdb_path, rebuilt_full_pdb_path = utils.make_rebuilt_output_paths(
            pdb_output_dir, args.name, args.residue_start, args.residue_end
        )
        rows.append(
            utils.score_native_structure(
                name=args.name,
                pdb_path=args.pdb_path,
                chain=args.chain,
                start=args.residue_start,
                end=args.residue_end,
                chi_mode=args.chi_mode,
                rmsd_mode=args.rmsd_mode,
                rmsd_residue_scope=args.rmsd_residue_scope,
                energy_backend=args.energy_backend,
                use_e2e_constraint=bool(args.use_e2e_constraint),
                e2e_scale=args.e2e_scale,
                gromacs_minimize=bool(args.gromacs_minimize),
                gromacs_forcefield=args.gromacs_forcefield,
                gromacs_water=args.gromacs_water,
                gromacs_nsteps=args.gromacs_nsteps,
                gromacs_emtol=args.gromacs_emtol,
                gromacs_maxwarn=args.gromacs_maxwarn,
                rebuilt_ca_pdb_path=str(rebuilt_ca_pdb_path),
                rebuilt_ca_centroid_pdb_path=str(rebuilt_ca_centroid_pdb_path),
                rebuilt_full_pdb_path=str(rebuilt_full_pdb_path),
            )
        )

    df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    write_portable_csv(df, args.out_csv)

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(relativize_absolute_paths(rows), indent=2))

    print(df.to_string(index=False))
    print(f"\nWrote {args.out_csv}")
    if args.out_json:
        print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
