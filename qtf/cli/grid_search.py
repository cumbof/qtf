#!/usr/bin/env python3
"""
Grid-search workflow for QTF energy function parameter tuning.

Iterates over all Cartesian-product combinations of energy-scale parameters
(hbond, sasa, vdw_rep, vdw_attr, rotamer, pi_stack), runs beam search and
native-structure scoring for each combination, then runs panel analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import itertools
import sys
from pathlib import Path
from typing import Dict, Iterable, List

from qtf.utils import workflow as utils
from qtf.cli import bench as bench_mod
from qtf.cli import eval as eval_mod


DEFAULT_PANEL_CSV = "experimental_structures/panel_csvs/protein_panel.csv"
DEFAULT_RUN_ROOT = "run_outputs"
BEAM_WIDTH = 1000
MAX_SIDECHAIN_OPTS = 9
RANDOM_SEED = 42
ENERGY_BACKEND = "custom"
USE_E2E_CONSTRAINT = 1
E2E_SCALE = 1.0

HBOND_SCALE = [0.75]
SASA_SCALE = [0.7]
VDW_REP_SCALE = [0.01]
VDW_ATTR_SCALE = [0.1]
ROTAMER_SCALE = [1.0]
PI_STACK_SCALE = [1.0]


def safe_name(s: str) -> str:
    return s.strip().replace(" ", "_")


def _as_list(value):
    if isinstance(value, list):
        return value
    return [value]


def load_grid_spec(path: str | None) -> Dict[str, List[float]]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("grid JSON must contain an object at the top level")
    grid = {}
    for key in ["hbond_scale", "sasa_scale", "vdw_rep_scale", "vdw_attr_scale", "rotamer_scale", "pi_stack_scale"]:
        if key in data:
            grid[key] = [float(v) for v in _as_list(data[key])]
    return grid


def iter_param_grid(grid_spec: Dict[str, List[float]] | None = None) -> Iterable[Dict[str, float]]:
    spec = grid_spec or {}
    hbond = spec.get("hbond_scale", HBOND_SCALE)
    sasa = spec.get("sasa_scale", SASA_SCALE)
    rep = spec.get("vdw_rep_scale", VDW_REP_SCALE)
    attr = spec.get("vdw_attr_scale", VDW_ATTR_SCALE)
    rot = spec.get("rotamer_scale", ROTAMER_SCALE)
    pi = spec.get("pi_stack_scale", PI_STACK_SCALE)
    for hb, sasa_v, rep_v, attr_v, rot_v, pi_v in itertools.product(hbond, sasa, rep, attr, rot, pi):
        yield {
            "hbond_scale": hb,
            "sasa_scale": sasa_v,
            "vdw_rep_scale": rep_v,
            "vdw_attr_scale": attr_v,
            "rotamer_scale": rot_v,
            "pi_stack_scale": pi_v,
        }


def write_run_settings(path: Path, settings: Dict[str, str | float]) -> None:
    lines = [f"{k}={v}" for k, v in settings.items()]
    path.write_text("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="qtf grid-search",
        description="Run grid search over energy scale parameters for peptide panels."
    )
    ap.add_argument("--panel_csv", default=DEFAULT_PANEL_CSV)
    ap.add_argument("--grid_json", default=None,
                    help="Optional JSON file with parameter lists for hbond/sasa/vdw/rotamer/pi.")
    ap.add_argument("--outsubdir", required=True)
    ap.add_argument("--window_deg", type=int, required=True)
    ap.add_argument("--step_deg", type=int, required=True)
    ap.add_argument("--beam_width", type=int, default=BEAM_WIDTH)
    ap.add_argument("--max_sidechain_opts_per_residue", type=int, default=MAX_SIDECHAIN_OPTS)
    ap.add_argument("--rmsd_mode", default="ca", choices=["ca", "heavy"])
    ap.add_argument("--rmsd_residue_scope", default="core", choices=["core", "all"])
    ap.add_argument("--energy_backend", default=ENERGY_BACKEND, choices=["custom", "openmm"])
    ap.add_argument("--use_e2e_constraint", type=int, default=USE_E2E_CONSTRAINT)
    ap.add_argument("--e2e_scale", type=float, default=E2E_SCALE)
    ap.add_argument("--gromacs_minimize", type=int, default=None,
                    help="Override the default GROMACS postprocess behavior; when omitted, GROMACS minimization is enabled for all backends.")
    ap.add_argument("--hard_clash_reject_A", type=float, default=0.75)
    ap.add_argument("--only_proteins", nargs="*", default=[])
    ap.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args(argv)

    panel = utils.load_panel(args.panel_csv)
    grid_spec = load_grid_spec(args.grid_json)
    outroot = Path(DEFAULT_RUN_ROOT) / args.outsubdir
    outroot.mkdir(parents=True, exist_ok=True)
    analysis_dir = outroot / "analysis"
    resolved_gromacs_minimize = (
        int(args.gromacs_minimize)
        if args.gromacs_minimize is not None
        else 1
    )

    manifest_rows = []
    only_proteins = {safe_name(x) for x in args.only_proteins} if args.only_proteins else set()

    for params in iter_param_grid(grid_spec):
        for row in panel:
            name = safe_name(str(row["name"]))
            if only_proteins and name not in only_proteins:
                continue

            seq = str(row["sequence"]).strip().upper()
            pdb_path = str(row["pdb_path"]).strip()
            if not Path(pdb_path).is_file():
                candidate = Path("experimental_structures/pdb_files") / Path(pdb_path).name
                if candidate.is_file():
                    pdb_path = str(candidate)
                else:
                    pdb_path = str(Path(pdb_path).resolve())
            chain = str(row.get("chain", "")).strip()
            chi_mode = str(row.get("chi_mode", "selective")).strip() or "selective"

            exp_id = (
                f"{name}_chi-{chi_mode}"
                f"_win-{args.window_deg}_step-{args.step_deg}"
                f"_bw-{args.beam_width}_scopts-{args.max_sidechain_opts_per_residue}"
                f"_rmsd-{args.rmsd_mode}_scope-{args.rmsd_residue_scope}"
                f"_backend-{args.energy_backend}_e2e-{args.use_e2e_constraint}"
                f"_hardclash-{args.hard_clash_reject_A}"
                f"_hb-{params['hbond_scale']}"
                f"_sasa-{params['sasa_scale']}"
                f"_vdwr-{params['vdw_rep_scale']}"
                f"_vdwa-{params['vdw_attr_scale']}"
                f"_rot-{params['rotamer_scale']}"
                f"_pi-{params['pi_stack_scale']}"
            )

            run_dir = outroot / exp_id
            beam_dir = run_dir / "beam"
            native_dir = run_dir / "native"
            beam_dir.mkdir(parents=True, exist_ok=True)
            native_dir.mkdir(parents=True, exist_ok=True)

            beam_csv = beam_dir / "beamsearch_ranked.csv"
            native_csv = native_dir / f"{name}_native_score.csv"
            if args.skip_existing and beam_csv.exists() and native_csv.exists():
                print(f"[skip] {exp_id}")
                continue

            settings = {
                "name": name,
                "sequence": seq,
                "pdb_path": pdb_path,
                "chain": chain,
                "chi_mode": chi_mode,
                "beam_width": args.beam_width,
                "window_deg": args.window_deg,
                "step_deg": args.step_deg,
                "max_sidechain_opts_per_residue": args.max_sidechain_opts_per_residue,
                "rmsd_mode": args.rmsd_mode,
                "rmsd_residue_scope": args.rmsd_residue_scope,
                "random_seed": RANDOM_SEED,
                "energy_backend": args.energy_backend,
                "use_e2e_constraint": args.use_e2e_constraint,
                "e2e_scale": args.e2e_scale,
                "gromacs_minimize": resolved_gromacs_minimize,
                "hard_clash_reject_A": args.hard_clash_reject_A,
                "grid_json": args.grid_json or "",
                **params,
            }
            write_run_settings(run_dir / "run_settings.txt", settings)

            print("=" * 80)
            print(f"[run] {exp_id}")
            print("=" * 80)

            try:
                beam_argv = [
                    "--protein_name", name,
                    "--sequence", seq,
                    "--beam_width", str(args.beam_width),
                    "--window_deg", str(args.window_deg),
                    "--step_deg", str(args.step_deg),
                    "--chi_mode", chi_mode,
                    "--rmsd_mode", args.rmsd_mode,
                    "--rmsd_residue_scope", args.rmsd_residue_scope,
                    "--max_sidechain_opts_per_residue", str(args.max_sidechain_opts_per_residue),
                    "--save_partial",
                    "--random_seed", str(RANDOM_SEED),
                    "--energy_backend", args.energy_backend,
                    "--use_e2e_constraint", str(args.use_e2e_constraint),
                    "--e2e_scale", str(args.e2e_scale),
                    "--gromacs_minimize", str(resolved_gromacs_minimize),
                    "--hard_clash_reject_A", str(args.hard_clash_reject_A),
                    "--reference_pdb", pdb_path,
                    "--outdir", str(beam_dir),
                ]
                bench_mod.main(argv=beam_argv)

                native_argv = [
                    "--name", name,
                    "--pdb_path", pdb_path,
                    "--chi_mode", chi_mode,
                    "--rmsd_mode", args.rmsd_mode,
                    "--rmsd_residue_scope", args.rmsd_residue_scope,
                    "--energy_backend", args.energy_backend,
                    "--use_e2e_constraint", str(args.use_e2e_constraint),
                    "--e2e_scale", str(args.e2e_scale),
                    "--gromacs_minimize", str(resolved_gromacs_minimize),
                    "--out_csv", str(native_csv),
                    "--out_json", str(native_dir / f"{name}_native_score.json"),
                ]
                if chain:
                    native_argv.extend(["--chain", chain])
                eval_mod.main(argv=native_argv)

                status = "ok"
                error = ""
            except SystemExit as e:
                status = "failed"
                error = f"SystemExit: {e}"
                print(f"[error] {exp_id}: {error}")
            except Exception as e:
                status = "failed"
                error = f"{type(e).__name__}: {e}"
                print(f"[error] {exp_id}: {error}")

            manifest_rows.append({
                "experiment_id": exp_id,
                "protein_name": name,
                "reference_pdb_path": pdb_path,
                "reference_pdb_id": Path(pdb_path).stem.upper(),
                "sequence": seq,
                "chi_mode": chi_mode,
                "window_deg": args.window_deg,
                "step_deg": args.step_deg,
                "beam_width": args.beam_width,
                "max_sidechain_opts_per_residue": args.max_sidechain_opts_per_residue,
                "rmsd_mode": args.rmsd_mode,
                "rmsd_residue_scope": args.rmsd_residue_scope,
                "energy_backend": args.energy_backend,
                "use_e2e_constraint": args.use_e2e_constraint,
                "e2e_scale": args.e2e_scale,
                "gromacs_minimize": resolved_gromacs_minimize,
                "hard_clash_reject_A": args.hard_clash_reject_A,
                "grid_json": args.grid_json or "",
                **params,
                "status": status,
                "error": error,
                "run_dir": str(run_dir),
            })

    # Write manifest once after all iterations
    manifest = outroot / "grid_manifest.csv"
    if manifest_rows:
        with open(manifest, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)

    try:
        from qtf.analysis import panel as panel_analysis
        panel_analysis.run_panel_analysis(outroot, analysis_dir)
    except Exception as e:
        print(f"[warn] analysis failed for {outroot}: {type(e).__name__}: {e}")

    print(f"[done] wrote manifest: {outroot / 'grid_manifest.csv'}")


if __name__ == "__main__":
    main()
