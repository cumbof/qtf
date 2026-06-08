#!/usr/bin/env python3
"""
[deprecated] Unified QTF dispatcher.

This script is kept for backward compatibility. Please use the individual
subcommand entry points instead:

    qtf-fold          -> quantum folding prediction
    qtf-bench         -> beam-search benchmark
    qtf-eval          -> score experimental/native structures
    qtf-grid-search   -> grid search over energy-scale parameters
    qtf-relax         -> GROMACS relaxation of PDB structures
"""

from __future__ import annotations

import argparse
import csv
import json
import itertools
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List

from qtf.utils import workflow as utils
from qtf.cli import bench as bench_mod
from qtf.cli import eval as eval_mod
from qtf.cli import grid_search as grid_mod


DEFAULT_PANEL_CSV = "experimental_structures/panel_csvs/protein_panel.csv"
DEFAULT_GRID_JSON = None
DEFAULT_RUN_ROOT = "run_outputs"
BEAM_WIDTH = 1000
MAX_SIDECHAIN_OPTS = 9
RANDOM_SEED = 42
ENERGY_BACKEND = "custom"
USE_E2E_CONSTRAINT = 1
E2E_SCALE = 1.0
ROSETTA_REPACK = 0
ROSETTA_FA_MIN = 0
ROSETTA_CEN_MIN = 0

HBOND_SCALE = [0.55]
SASA_SCALE = [0.85]
VDW_REP_SCALE = [0.01]
VDW_ATTR_SCALE = [0.10]
ROTAMER_SCALE = [1.00]
PI_STACK_SCALE = [0.10]


@contextmanager
def _patched_environ(updates: Dict[str, str]):
    old_values = {}
    missing = []
    for key, value in updates.items():
        if key in os.environ:
            old_values[key] = os.environ[key]
        else:
            missing.append(key)
        os.environ[key] = value
    try:
        yield
    finally:
        for key in missing:
            os.environ.pop(key, None)
        for key, value in old_values.items():
            os.environ[key] = value


def _run_beam(argv: List[str]) -> None:
    bench_mod.main(argv=argv)


def _run_predict(argv: List[str]) -> None:
    from qtf.cli import fold as fold_mod
    fold_mod.main(argv=argv)


def _run_score_native(argv: List[str]) -> None:
    eval_mod.main(argv=argv)


def _run_grid(args: argparse.Namespace) -> None:
    grid_args = [
        "--panel_csv", args.panel_csv,
    ]
    if args.grid_json:
        grid_args.extend(["--grid_json", args.grid_json])
    grid_args.extend([
        "--outsubdir", args.outsubdir,
        "--window_deg", str(args.window_deg),
        "--step_deg", str(args.step_deg),
        "--beam_width", str(args.beam_width),
        "--max_sidechain_opts_per_residue", str(args.max_sidechain_opts_per_residue),
        "--rmsd_mode", args.rmsd_mode,
        "--rmsd_residue_scope", args.rmsd_residue_scope,
        "--energy_backend", args.energy_backend,
        "--use_e2e_constraint", str(args.use_e2e_constraint),
        "--e2e_scale", str(args.e2e_scale),
        "--hard_clash_reject_A", str(args.hard_clash_reject_A),
        "--rosetta_repack", str(args.rosetta_repack),
        "--rosetta_fa_min", str(args.rosetta_fa_min),
        "--rosetta_cen_min", str(args.rosetta_cen_min),
    ])
    if args.gromacs_minimize is not None:
        grid_args.extend(["--gromacs_minimize", str(args.gromacs_minimize)])
    if args.only_proteins:
        grid_args.extend(["--only_proteins"] + list(args.only_proteins))
    if not args.skip_existing:
        grid_args.append("--no-skip_existing")
    grid_mod.main(argv=grid_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified QTF dispatcher. [deprecated]")
    sub = parser.add_subparsers(dest="mode", required=True)

    beam = sub.add_parser("beam", help="Run beam search")
    beam.add_argument("args", nargs=argparse.REMAINDER, help="Forwarded to qtf-bench")

    predict = sub.add_parser("predict", help="Run predictor")
    predict.add_argument("args", nargs=argparse.REMAINDER, help="Forwarded to qtf-fold")

    native = sub.add_parser("score-native", help="Score experimental/native structures")
    native.add_argument("args", nargs=argparse.REMAINDER, help="Forwarded to qtf-eval")

    grid = sub.add_parser("grid", help="Run panel/grid workflow")
    grid.add_argument("--panel_csv", default=DEFAULT_PANEL_CSV)
    grid.add_argument("--grid_json", default=DEFAULT_GRID_JSON,
                      help="Optional JSON file with parameter lists for hbond/sasa/vdw/rotamer/pi.")
    grid.add_argument("--outsubdir", required=True)
    grid.add_argument("--window_deg", type=int, required=True)
    grid.add_argument("--step_deg", type=int, required=True)
    grid.add_argument("--beam_width", type=int, default=BEAM_WIDTH)
    grid.add_argument("--max_sidechain_opts_per_residue", type=int, default=MAX_SIDECHAIN_OPTS)
    grid.add_argument("--rmsd_mode", default="ca", choices=["ca", "heavy"])
    grid.add_argument("--rmsd_residue_scope", default="core", choices=["core", "all"])
    grid.add_argument("--energy_backend", default=ENERGY_BACKEND, choices=["custom", "rosetta", "openmm"])
    grid.add_argument("--use_e2e_constraint", type=int, default=USE_E2E_CONSTRAINT)
    grid.add_argument("--e2e_scale", type=float, default=E2E_SCALE)
    grid.add_argument("--gromacs_minimize", type=int, default=None,
                     help="Override the default GROMACS postprocess behavior; when omitted, GROMACS minimization is enabled for all backends.")
    grid.add_argument("--hard_clash_reject_A", type=float, default=0.75)
    grid.add_argument("--rosetta_repack", type=int, default=ROSETTA_REPACK)
    grid.add_argument("--rosetta_fa_min", type=int, default=ROSETTA_FA_MIN)
    grid.add_argument("--rosetta_cen_min", type=int, default=ROSETTA_CEN_MIN)
    grid.add_argument("--only_proteins", nargs="*", default=[])
    grid.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    print("[warn] scripts/qtf_run.py is deprecated; use the individual entry points:", file=sys.stderr)
    print("[warn]   qtf-fold, qtf-bench, qtf-eval, qtf-grid-search, qtf-relax", file=sys.stderr)
    args = build_parser().parse_args()
    if args.mode == "beam":
        _run_beam(args.args)
    elif args.mode == "predict":
        _run_predict(args.args)
    elif args.mode == "score-native":
        _run_score_native(args.args)
    elif args.mode == "grid":
        _run_grid(args)
    else:
        raise SystemExit(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
