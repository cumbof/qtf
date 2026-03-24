#!/usr/bin/env python3
from __future__ import annotations

import csv
import itertools
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List

# -----------------------------
# User-editable settings
# -----------------------------
PANEL_CSV = "protein_panel.csv"
OUTROOT = Path("grid_runs/grid_v3_rot_pi_largegrid")
PYTHON = "python"

BEAM_SCRIPT = "qtf_beamsearch_benchmark.py"
NATIVE_SCRIPT = "qtf_score_experimental.py"

BEAM_WIDTH = 200
WINDOW_DEG = 30
STEP_DEG = 15
MAX_SIDECHAIN_OPTS = 9
RANDOM_SEED = 42

# Grid combinations
HBOND_SCALE = [0.55, 0.65, 0.75]
SASA_SCALE = [0.65, 0.75, 0.85]
VDW_REP_SCALE = [0.03, 0.05, 0.07]
VDW_ATTR_SCALE = [0.10, 0.15, 0.20]
ROTAMER_SCALE = [0.25, 0.5, 0.75, 1.0]
PI_STACK_SCALE = [0.10, 0.25, 0.5, 0.75]

# Optional filters
ONLY_PROTEINS: list[str] = ["chignolin", "trp_cage", "MBH12"]   # e.g. ["chignolin", "trp_cage"]
SKIP_EXISTING = True


def safe_name(s: str) -> str:
    return s.strip().replace(" ", "_")


def load_panel(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    cleaned = []
    for row in rows:
        cleaned.append({k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()})
    return cleaned


def iter_param_grid() -> Iterable[Dict[str, float]]:
    for hb, sasa, rep, attr, rot, pi in itertools.product(
        HBOND_SCALE,
        SASA_SCALE,
        VDW_REP_SCALE,
        VDW_ATTR_SCALE,
        ROTAMER_SCALE,
        PI_STACK_SCALE,
    ):
        yield {
            "hbond_scale": hb,
            "sasa_scale": sasa,
            "vdw_rep_scale": rep,
            "vdw_attr_scale": attr,
            "rotamer_scale": rot,
            "pi_stack_scale": pi,
        }


def write_run_settings(path: Path, settings: Dict[str, str | float]) -> None:
    lines = [f"{k}={v}" for k, v in settings.items()]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    panel = load_panel(PANEL_CSV)
    OUTROOT.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    for params in iter_param_grid():
        for row in panel:
            name = safe_name(row["name"])
            if ONLY_PROTEINS and name not in ONLY_PROTEINS:
                continue

            seq = row["sequence"].strip().upper()
            pdb_path = str(Path(row["pdb_path"]).resolve())
            chain = row.get("chain", "A").strip() or "A"
            forcefield = row.get("forcefield", "amber").strip() or "amber"
            chi_mode = row.get("chi_mode", "selective").strip() or "selective"

            exp_id = (
                f"{name}_ff-{forcefield}_chi-{chi_mode}"
                f"_hb-{params['hbond_scale']}"
                f"_sasa-{params['sasa_scale']}"
                f"_vdwr-{params['vdw_rep_scale']}"
                f"_vdwa-{params['vdw_attr_scale']}"
                f"_rot-{params['rotamer_scale']}"
                f"_pi-{params['pi_stack_scale']}"
            )

            run_dir = OUTROOT / exp_id
            beam_dir = run_dir / "beam"
            native_dir = run_dir / "native"
            beam_dir.mkdir(parents=True, exist_ok=True)
            native_dir.mkdir(parents=True, exist_ok=True)

            beam_csv = beam_dir / "beamsearch_ranked.csv"
            native_csv = native_dir / f"{name}_native_score.csv"

            if SKIP_EXISTING and beam_csv.exists() and native_csv.exists():
                print(f"[skip] {exp_id}")
                continue

            env = os.environ.copy()
            env["QTF_HBOND_SCALE"] = str(params["hbond_scale"])
            env["QTF_SASA_SCALE"] = str(params["sasa_scale"])
            env["QTF_VDW_REP_SCALE"] = str(params["vdw_rep_scale"])
            env["QTF_VDW_ATTR_SCALE"] = str(params["vdw_attr_scale"])
            env["QTF_ROTAMER_SCALE"] = str(params["rotamer_scale"])
            env["QTF_PI_STACK_SCALE"] = str(params["pi_stack_scale"])

            settings = {
                "name": name,
                "sequence": seq,
                "pdb_path": pdb_path,
                "chain": chain,
                "forcefield": forcefield,
                "chi_mode": chi_mode,
                "beam_width": BEAM_WIDTH,
                "window_deg": WINDOW_DEG,
                "step_deg": STEP_DEG,
                "max_sidechain_opts_per_residue": MAX_SIDECHAIN_OPTS,
                "random_seed": RANDOM_SEED,
                **params,
            }
            write_run_settings(run_dir / "run_settings.txt", settings)

            print("=" * 80)
            print(f"[run] {exp_id}")
            print("=" * 80)

            try:
                subprocess.run([
                    PYTHON, BEAM_SCRIPT,
                    "--protein_name", name,
                    "--sequence", seq,
                    "--forcefield", forcefield,
                    "--beam_width", str(BEAM_WIDTH),
                    "--window_deg", str(WINDOW_DEG),
                    "--step_deg", str(STEP_DEG),
                    "--chi_mode", chi_mode,
                    "--max_sidechain_opts_per_residue", str(MAX_SIDECHAIN_OPTS),
                    "--random_seed", str(RANDOM_SEED),
                    "--reference_pdb", pdb_path,
                    "--outdir", str(beam_dir),
                ], check=True, env=env)

                subprocess.run([
                    PYTHON, NATIVE_SCRIPT,
                    "--name", name,
                    "--pdb_path", pdb_path,
                    "--chain", chain,
                    "--forcefield", forcefield,
                    "--chi_mode", chi_mode,
                    "--out_csv", str(native_csv),
                    "--out_json", str(native_dir / f"{name}_native_score.json"),
                ], check=True, env=env)

                status = "ok"
                error = ""
            except subprocess.CalledProcessError as e:
                status = "failed"
                error = f"{type(e).__name__}: returncode={e.returncode}"
                print(f"[error] {exp_id}: {error}")

            manifest_rows.append({
                "experiment_id": exp_id,
                "protein_name": name,
                "reference_pdb_path": pdb_path,
                "reference_pdb_id": Path(pdb_path).stem.upper(),
                "sequence": seq,
                "forcefield": forcefield,
                "chi_mode": chi_mode,
                **params,
                "status": status,
                "error": error,
                "run_dir": str(run_dir),
            })

            manifest = OUTROOT / "grid_manifest.csv"
            with open(manifest, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
                writer.writeheader()
                writer.writerows(manifest_rows)

    print(f"[done] wrote manifest: {OUTROOT / 'grid_manifest.csv'}")


if __name__ == "__main__":
    main()
