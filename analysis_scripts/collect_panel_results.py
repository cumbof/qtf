#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[warn] failed to read CSV {path}: {e}")
        return None


def _protein_from_experiment_id(experiment_id: Optional[str]) -> Optional[str]:
    if not experiment_id:
        return None
    return str(experiment_id).split("_ff-")[0]


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _read_run_settings(run_dir: Path) -> Dict[str, Optional[str]]:
    txt = run_dir / "run_settings.txt"
    out: Dict[str, Optional[str]] = {}
    if not txt.exists():
        return out
    for line in txt.read_text().splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _pdb_id_from_path(pathlike: Optional[str]) -> Optional[str]:
    if pathlike is None:
        return None
    s = str(pathlike).strip()
    return Path(s).stem.upper() if s else None


def _make_protein_label(protein_name: Optional[str], reference_pdb_id: Optional[str]) -> Optional[str]:
    pname = None if protein_name is None else str(protein_name).strip()
    pid = None if reference_pdb_id is None else str(reference_pdb_id).strip()
    if pname and pid:
        return f"{pname} ({pid})"
    return pname or pid


def _coalesce_str(primary: Optional[str], secondary: Optional[str]) -> Optional[str]:
    if primary is not None and str(primary).strip() and str(primary).strip().lower() != "nan":
        return str(primary).strip()
    if secondary is not None and str(secondary).strip() and str(secondary).strip().lower() != "nan":
        return str(secondary).strip()
    return None


def _discover_run_metadata(path: Path, kind: str) -> Dict[str, Optional[str]]:
    run_dir = path.parent.parent
    beam_dir = run_dir / "beam" if kind == "native" else path.parent
    native_dir = run_dir / "native" if kind == "beam" else path.parent

    run_name = run_dir.name
    settings = _read_run_settings(run_dir)

    protein_name = settings.get("name") or None
    reference_pdb_path = settings.get("pdb_path") or None
    reference_pdb_id = _pdb_id_from_path(reference_pdb_path)
    experiment_id = run_name
    sequence = settings.get("sequence") or None
    forcefield = settings.get("forcefield") or None
    chi_mode = settings.get("chi_mode") or None
    window_deg = settings.get("window_deg") or None
    step_deg = settings.get("step_deg") or None

    best_json = beam_dir / "beamsearch_best.json"
    best_obj = _read_json(best_json) if best_json.exists() else None
    if isinstance(best_obj, dict):
        protein_name = _coalesce_str(protein_name, best_obj.get("protein_name"))
        reference_pdb_id = _coalesce_str(reference_pdb_id, best_obj.get("reference_pdb_id"))
        reference_pdb_path = _coalesce_str(reference_pdb_path, best_obj.get("reference_pdb_path"))
        if reference_pdb_id is None:
            reference_pdb_id = _pdb_id_from_path(best_obj.get("reference_pdb"))
        if reference_pdb_path is None:
            reference_pdb_path = _coalesce_str(reference_pdb_path, best_obj.get("reference_pdb"))
        experiment_id = _coalesce_str(best_obj.get("experiment_id"), experiment_id)
        sequence = _coalesce_str(sequence, best_obj.get("sequence"))
        forcefield = _coalesce_str(forcefield, best_obj.get("forcefield"))
        chi_mode = _coalesce_str(chi_mode, best_obj.get("chi_mode"))

    native_csvs = sorted(native_dir.glob("*_native_score.csv"))
    if native_csvs:
        native_df = safe_read_csv(native_csvs[0])
        if native_df is not None and not native_df.empty:
            row0 = native_df.iloc[0]
            protein_name = _coalesce_str(protein_name, row0.get("protein_name"))
            protein_name = _coalesce_str(protein_name, row0.get("name"))
            reference_pdb_id = _coalesce_str(reference_pdb_id, row0.get("reference_pdb_id"))
            reference_pdb_path = _coalesce_str(reference_pdb_path, row0.get("reference_pdb_path"))
            if reference_pdb_id is None:
                reference_pdb_id = _pdb_id_from_path(row0.get("pdb_path"))
            if reference_pdb_path is None:
                reference_pdb_path = _coalesce_str(reference_pdb_path, row0.get("pdb_path"))
            experiment_id = _coalesce_str(row0.get("experiment_id"), experiment_id)
            sequence = _coalesce_str(sequence, row0.get("sequence"))
            forcefield = _coalesce_str(forcefield, row0.get("forcefield"))
            chi_mode = _coalesce_str(chi_mode, row0.get("chi_mode"))

    if not protein_name:
        protein_name = _protein_from_experiment_id(experiment_id) or run_name

    return {
        "protein_name": protein_name,
        "protein_label": _make_protein_label(protein_name, reference_pdb_id),
        "reference_pdb_id": reference_pdb_id,
        "reference_pdb_path": reference_pdb_path,
        "experiment_id": experiment_id,
        "sequence": sequence,
        "forcefield": forcefield,
        "chi_mode": chi_mode,
        "window_deg": window_deg,
        "step_deg": step_deg,
        "run_dir": str(path.parent),
    }


def collect_beam_rows(root: Path) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    for f in sorted(root.rglob("beamsearch_ranked.csv")):
        df = safe_read_csv(f)
        if df is None or df.empty:
            continue
        meta = _discover_run_metadata(f, kind="beam")
        df = df.copy()
        df["source_file"] = str(f)
        df["run_dir"] = str(f.parent)
        for col, val in meta.items():
            if col not in df.columns:
                df[col] = val
            else:
                df[col] = df[col].replace({"nan": None}).fillna(val)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True, sort=False) if dfs else pd.DataFrame()


def collect_native_rows(root: Path) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    for f in sorted(root.rglob("*_native_score.csv")):
        df = safe_read_csv(f)
        if df is None or df.empty:
            continue
        meta = _discover_run_metadata(f, kind="native")
        df = df.copy()
        df["source_file"] = str(f)
        df["run_dir"] = str(f.parent)
        if "protein_name" not in df.columns and "name" in df.columns:
            df["protein_name"] = df["name"]
        for col, val in meta.items():
            if col not in df.columns:
                df[col] = val
            else:
                df[col] = df[col].replace({"nan": None}).fillna(val)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True, sort=False) if dfs else pd.DataFrame()


def collect_manifests(root: Path) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    for f in sorted(root.rglob("grid_manifest.csv")):
        df = safe_read_csv(f)
        if df is None or df.empty:
            continue
        df = df.copy()
        df["source_manifest"] = str(f)
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True, sort=False)
    subset_cols = [c for c in ["experiment_id", "run_dir", "protein_name", "reference_pdb_path", "reference_pdb_id",
                               "sequence", "forcefield", "chi_mode", "window_deg", "step_deg",
                               "hbond_scale", "sasa_scale", "vdw_rep_scale",
                               "vdw_attr_scale", "rotamer_scale", "pi_stack_scale", "status", "error"]
                   if c in out.columns]
    return out.drop_duplicates(subset=subset_cols, keep="first") if subset_cols else out.drop_duplicates()


def build_summary(beam_df: pd.DataFrame, native_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    if not beam_df.empty:
        for experiment_id, g in beam_df.groupby("experiment_id", dropna=False):
            g_energy = g[pd.notnull(g["energy"])] if "energy" in g.columns else g
            best_energy_row = g_energy.sort_values("energy", ascending=True).iloc[0] if not g_energy.empty else None
            best_rmsd_row = None
            if "rmsd_to_reference_A" in g.columns:
                g_rmsd = g[pd.notnull(g["rmsd_to_reference_A"])].copy()
                if not g_rmsd.empty:
                    best_rmsd_row = g_rmsd.sort_values("rmsd_to_reference_A", ascending=True).iloc[0]
            row = {
                "experiment_id": experiment_id,
                "protein_name": g["protein_name"].iloc[0] if "protein_name" in g.columns else None,
                "protein_label": g["protein_label"].iloc[0] if "protein_label" in g.columns else None,
                "reference_pdb_id": g["reference_pdb_id"].iloc[0] if "reference_pdb_id" in g.columns else None,
                "reference_pdb_path": g["reference_pdb_path"].iloc[0] if "reference_pdb_path" in g.columns else None,
                "sequence": g["sequence"].iloc[0] if "sequence" in g.columns else None,
                "forcefield": g["forcefield"].iloc[0] if "forcefield" in g.columns else None,
                "chi_mode": g["chi_mode"].iloc[0] if "chi_mode" in g.columns else None,
                "window_deg": g["window_deg"].iloc[0] if "window_deg" in g.columns else None,
                "step_deg": g["step_deg"].iloc[0] if "step_deg" in g.columns else None,
                "hbond_scale": g["hbond_scale"].iloc[0] if "hbond_scale" in g.columns else None,
                "sasa_scale": g["sasa_scale"].iloc[0] if "sasa_scale" in g.columns else None,
                "vdw_rep_scale": g["vdw_rep_scale"].iloc[0] if "vdw_rep_scale" in g.columns else None,
                "vdw_attr_scale": g["vdw_attr_scale"].iloc[0] if "vdw_attr_scale" in g.columns else None,
                "rotamer_scale": g["rotamer_scale"].iloc[0] if "rotamer_scale" in g.columns else None,
                "pi_stack_scale": g["pi_stack_scale"].iloc[0] if "pi_stack_scale" in g.columns else None,
                "n_beam_rows": int(len(g)),
            }
            if best_energy_row is not None:
                row["best_energy"] = best_energy_row.get("energy")
                row["best_energy_rmsd"] = best_energy_row.get("rmsd_to_reference_A")
            if best_rmsd_row is not None:
                row["best_rmsd"] = best_rmsd_row.get("rmsd_to_reference_A")
                row["best_rmsd_energy"] = best_rmsd_row.get("energy")
            summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    if not native_df.empty:
        keep_cols = [c for c in ["experiment_id", "protein_name", "protein_label", "reference_pdb_id", "reference_pdb_path",
                                 "window_deg", "step_deg",
                                 "total_energy", "rebuilt_vs_native_ca_rmsd", "rebuilt_end_to_end", "rebuilt_rg",
                                 "native_end_to_end", "native_rg"] if c in native_df.columns]
        native_small = native_df[keep_cols].copy().rename(columns={
            "total_energy": "native_energy",
            "rebuilt_vs_native_ca_rmsd": "native_rebuilt_rmsd",
            "rebuilt_end_to_end": "native_rebuilt_e2e",
            "rebuilt_rg": "native_rebuilt_rg",
        })
        summary_df = native_small if summary_df.empty else summary_df.merge(native_small, on=["experiment_id"], how="outer", suffixes=("", "_native"))
    return summary_df


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Collect panel outputs into master dataframes.")
    ap.add_argument("--root", required=True, help="Root folder containing run outputs")
    ap.add_argument("--outdir", required=True, help="Output directory for master CSVs")
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    beam_df = collect_beam_rows(root)
    native_df = collect_native_rows(root)
    manifest_df = collect_manifests(root)
    summary_df = build_summary(beam_df, native_df)

    if not beam_df.empty:
        beam_df.to_csv(outdir / "master_beam_rows.csv", index=False)
        print(f"Wrote {outdir / 'master_beam_rows.csv'} ({len(beam_df)} rows)")
    else:
        print("[warn] no beam rows found")

    if not native_df.empty:
        native_df.to_csv(outdir / "master_native_rows.csv", index=False)
        print(f"Wrote {outdir / 'master_native_rows.csv'} ({len(native_df)} rows)")
    else:
        print("[warn] no native rows found")

    if not manifest_df.empty:
        manifest_df.to_csv(outdir / "master_grid_manifest.csv", index=False)
        print(f"Wrote {outdir / 'master_grid_manifest.csv'} ({len(manifest_df)} rows)")
    else:
        print("[warn] no manifest rows found")

    if not summary_df.empty:
        summary_df.to_csv(outdir / "master_experiment_summary.csv", index=False)
        print(f"Wrote {outdir / 'master_experiment_summary.csv'} ({len(summary_df)} rows)")
    else:
        print("[warn] no summary rows built")


if __name__ == "__main__":
    main()
