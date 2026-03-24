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
    s = str(experiment_id)
    return s.split("_ff-")[0]


def _read_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            return json.load(f)
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
    if not s:
        return None
    return Path(s).stem.upper()


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
    """
    Collect best-effort metadata from:
      - run_settings.txt
      - beamsearch_best.json
      - sibling native score files
      - path-derived run name
    """
    if kind == "beam":
        run_dir = path.parent.parent
        beam_dir = path.parent
        native_dir = run_dir / "native"
    else:
        run_dir = path.parent.parent
        beam_dir = run_dir / "beam"
        native_dir = path.parent

    run_name = run_dir.name
    settings = _read_run_settings(run_dir)

    protein_name = settings.get("name") or None
    reference_pdb_path = settings.get("pdb_path") or None
    reference_pdb_id = _pdb_id_from_path(reference_pdb_path)
    experiment_id = run_name
    sequence = settings.get("sequence") or None
    forcefield = settings.get("forcefield") or None
    chi_mode = settings.get("chi_mode") or None

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
                reference_pdb_id = _pdb_id_from_path(row0.get("reference_pdb"))
            if reference_pdb_path is None:
                reference_pdb_path = _coalesce_str(reference_pdb_path, row0.get("reference_pdb"))
            if reference_pdb_path is None:
                reference_pdb_path = _coalesce_str(reference_pdb_path, row0.get("pdb_path"))
            if reference_pdb_id is None:
                reference_pdb_id = _pdb_id_from_path(row0.get("pdb_path"))
            experiment_id = _coalesce_str(row0.get("experiment_id"), experiment_id)
            sequence = _coalesce_str(sequence, row0.get("sequence"))
            forcefield = _coalesce_str(forcefield, row0.get("forcefield"))
            chi_mode = _coalesce_str(chi_mode, row0.get("chi_mode"))

    if not protein_name:
        protein_name = _protein_from_experiment_id(experiment_id) or _protein_from_experiment_id(run_name) or run_name

    protein_label = _make_protein_label(protein_name, reference_pdb_id)

    return {
        "protein_name": protein_name,
        "protein_label": protein_label,
        "reference_pdb_id": reference_pdb_id,
        "reference_pdb_path": reference_pdb_path,
        "experiment_id": experiment_id,
        "run_name": run_name,
        "sequence": sequence,
        "forcefield": forcefield,
        "chi_mode": chi_mode,
        "run_dir": str(path.parent),
    }


def collect_beam_rows(root: Path) -> pd.DataFrame:
    beam_files = sorted(root.rglob("beamsearch_ranked.csv"))
    dfs: List[pd.DataFrame] = []

    for f in beam_files:
        df = safe_read_csv(f)
        if df is None or df.empty:
            continue

        meta = _discover_run_metadata(f, kind="beam")

        df = df.copy()
        df["source_file"] = str(f)
        df["run_dir"] = str(f.parent)

        if "experiment_id" not in df.columns:
            df["experiment_id"] = meta["experiment_id"]
        else:
            df["experiment_id"] = df["experiment_id"].fillna(meta["experiment_id"])

        if "protein_name" not in df.columns:
            df["protein_name"] = meta["protein_name"]
        else:
            fixed = df["protein_name"].astype(str)
            fixed = fixed.where(~fixed.str.contains(r"_ff-", na=False), fixed.str.split("_ff-").str[0])
            df["protein_name"] = fixed.replace({"nan": None}).fillna(meta["protein_name"])

        if "protein_label" not in df.columns:
            df["protein_label"] = meta["protein_label"]
        else:
            df["protein_label"] = df["protein_label"].replace({"nan": None}).fillna(meta["protein_label"])

        if "reference_pdb_id" not in df.columns:
            df["reference_pdb_id"] = meta["reference_pdb_id"]
        else:
            df["reference_pdb_id"] = df["reference_pdb_id"].fillna(meta["reference_pdb_id"])

        if "reference_pdb_path" not in df.columns:
            df["reference_pdb_path"] = meta["reference_pdb_path"]
        else:
            df["reference_pdb_path"] = df["reference_pdb_path"].fillna(meta["reference_pdb_path"])

        if "reference_pdb" in df.columns:
            missing_id = pd.isna(df["reference_pdb_id"])
            df.loc[missing_id, "reference_pdb_id"] = df.loc[missing_id, "reference_pdb"].astype(str).map(_pdb_id_from_path)
            missing_path = pd.isna(df["reference_pdb_path"])
            df.loc[missing_path, "reference_pdb_path"] = df.loc[missing_path, "reference_pdb"]
            df = df.drop(columns=["reference_pdb"])

        if "sequence" not in df.columns:
            df["sequence"] = meta["sequence"]
        if "forcefield" not in df.columns:
            df["forcefield"] = meta["forcefield"]
        if "chi_mode" not in df.columns:
            df["chi_mode"] = meta["chi_mode"]

        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    out = pd.concat(dfs, ignore_index=True, sort=False)
    out["protein_name"] = out["protein_name"].astype(str).str.replace(r"_ff-.*$", "", regex=True)
    out["protein_name"] = out["protein_name"].replace({"nan": None})
    out["protein_label"] = out.apply(
        lambda r: _make_protein_label(r.get("protein_name"), r.get("reference_pdb_id")),
        axis=1
    )
    return out


def collect_native_rows(root: Path) -> pd.DataFrame:
    native_files = sorted(root.rglob("*_native_score.csv"))
    dfs: List[pd.DataFrame] = []

    for f in native_files:
        df = safe_read_csv(f)
        if df is None or df.empty:
            continue

        meta = _discover_run_metadata(f, kind="native")

        df = df.copy()
        df["source_file"] = str(f)
        df["run_dir"] = str(f.parent)

        if "protein_name" not in df.columns:
            if "name" in df.columns:
                df["protein_name"] = df["name"]
            else:
                df["protein_name"] = meta["protein_name"]
        else:
            df["protein_name"] = df["protein_name"].fillna(meta["protein_name"])

        if "protein_label" not in df.columns:
            df["protein_label"] = meta["protein_label"]
        else:
            df["protein_label"] = df["protein_label"].replace({"nan": None}).fillna(meta["protein_label"])

        if "reference_pdb_id" not in df.columns:
            df["reference_pdb_id"] = meta["reference_pdb_id"]
        else:
            df["reference_pdb_id"] = df["reference_pdb_id"].fillna(meta["reference_pdb_id"])

        if "reference_pdb_path" not in df.columns:
            df["reference_pdb_path"] = meta["reference_pdb_path"]
        else:
            df["reference_pdb_path"] = df["reference_pdb_path"].fillna(meta["reference_pdb_path"])

        if "reference_pdb" in df.columns:
            missing_id = pd.isna(df["reference_pdb_id"])
            df.loc[missing_id, "reference_pdb_id"] = df.loc[missing_id, "reference_pdb"].astype(str).map(_pdb_id_from_path)
            missing_path = pd.isna(df["reference_pdb_path"])
            df.loc[missing_path, "reference_pdb_path"] = df.loc[missing_path, "reference_pdb"]
            df = df.drop(columns=["reference_pdb"])

        if "experiment_id" not in df.columns:
            df["experiment_id"] = meta["experiment_id"]
        else:
            df["experiment_id"] = df["experiment_id"].fillna(meta["experiment_id"])

        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    out = pd.concat(dfs, ignore_index=True, sort=False)
    out["protein_name"] = out["protein_name"].astype(str).str.replace(r"_ff-.*$", "", regex=True)
    out["protein_name"] = out["protein_name"].replace({"nan": None})
    out["protein_label"] = out.apply(
        lambda r: _make_protein_label(r.get("protein_name"), r.get("reference_pdb_id")),
        axis=1
    )
    return out


def build_summary(beam_df: pd.DataFrame, native_df: pd.DataFrame, root: Path) -> pd.DataFrame:
    summary_rows = []

    if beam_df.empty and native_df.empty:
        return pd.DataFrame()

    native_like_map = {}
    for counts_path in sorted(root.rglob("native_like_counts.json")):
        try:
            with open(counts_path, "r") as f:
                obj = json.load(f)
            run_dir = str(counts_path.parent)
            native_like_map[run_dir] = {
                "native_like_count": obj.get("native_like"),
                "non_native_count": obj.get("non_native"),
                "native_like_thresh": obj.get("native_thresh_A"),
            }
        except Exception as e:
            print(f"[warn] failed to read {counts_path}: {e}")

    if not beam_df.empty:
        for experiment_id, g in beam_df.groupby("experiment_id", dropna=False):
            g = g.copy()

            protein_name = g["protein_name"].iloc[0] if "protein_name" in g.columns else _protein_from_experiment_id(experiment_id)
            reference_pdb_id = g["reference_pdb_id"].iloc[0] if "reference_pdb_id" in g.columns else None
            reference_pdb_path = g["reference_pdb_path"].iloc[0] if "reference_pdb_path" in g.columns else None
            protein_label = _make_protein_label(protein_name, reference_pdb_id)

            g_energy = g[pd.notnull(g["energy"])] if "energy" in g.columns else g
            best_energy_row = g_energy.sort_values("energy", ascending=True).iloc[0] if not g_energy.empty else None

            best_rmsd_row = None
            if "rmsd_to_reference_A" in g.columns:
                g_rmsd = g[pd.notnull(g["rmsd_to_reference_A"])].copy()
                if not g_rmsd.empty:
                    best_rmsd_row = g_rmsd.sort_values("rmsd_to_reference_A", ascending=True).iloc[0]

            run_dir = str(g["run_dir"].iloc[0]) if "run_dir" in g.columns and not g.empty else None
            nl = native_like_map.get(run_dir, {})

            row = {
                "experiment_id": experiment_id,
                "protein_name": protein_name,
                "protein_label": protein_label,
                "reference_pdb_id": reference_pdb_id,
                "reference_pdb_path": reference_pdb_path,
                "sequence": g["sequence"].iloc[0] if "sequence" in g.columns else None,
                "forcefield": g["forcefield"].iloc[0] if "forcefield" in g.columns else None,
                "chi_mode": g["chi_mode"].iloc[0] if "chi_mode" in g.columns else None,
                "hbond_scale": g["hbond_scale"].iloc[0] if "hbond_scale" in g.columns else None,
                "sasa_scale": g["sasa_scale"].iloc[0] if "sasa_scale" in g.columns else None,
                "vdw_rep_scale": g["vdw_rep_scale"].iloc[0] if "vdw_rep_scale" in g.columns else None,
                "vdw_attr_scale": g["vdw_attr_scale"].iloc[0] if "vdw_attr_scale" in g.columns else None,
                "rotamer_scale": g["rotamer_scale"].iloc[0] if "rotamer_scale" in g.columns else None,
                "pi_stack_scale": g["pi_stack_scale"].iloc[0] if "pi_stack_scale" in g.columns else None,
                "n_beam_rows": int(len(g)),
                "native_like_count": nl.get("native_like_count"),
                "non_native_count": nl.get("non_native_count"),
                "native_like_thresh": nl.get("native_like_thresh"),
            }

            if best_energy_row is not None:
                row.update({
                    "best_energy": best_energy_row.get("energy"),
                    "best_energy_rmsd": best_energy_row.get("rmsd_to_reference_A"),
                    "best_energy_rank": best_energy_row.get("energy_rank"),
                    "best_energy_e2e": best_energy_row.get("pred_e2e_A"),
                    "best_energy_rg": best_energy_row.get("pred_rg_A"),
                })

            if best_rmsd_row is not None:
                row.update({
                    "best_rmsd": best_rmsd_row.get("rmsd_to_reference_A"),
                    "best_rmsd_energy": best_rmsd_row.get("energy"),
                    "best_rmsd_rank": best_rmsd_row.get("energy_rank"),
                    "best_rmsd_e2e": best_rmsd_row.get("pred_e2e_A"),
                    "best_rmsd_rg": best_rmsd_row.get("pred_rg_A"),
                })

            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    if not native_df.empty:
        keep_cols = [
            "experiment_id",
            "protein_name",
            "protein_label",
            "reference_pdb_id",
            "reference_pdb_path",
            "total_energy",
            "rebuilt_vs_native_ca_rmsd",
            "rebuilt_end_to_end",
            "rebuilt_rg",
            "native_end_to_end",
            "native_rg",
        ]
        keep_cols = [c for c in keep_cols if c in native_df.columns]

        native_small = native_df[keep_cols].copy()
        native_small = native_small.rename(columns={
            "total_energy": "native_energy",
            "rebuilt_vs_native_ca_rmsd": "native_rebuilt_rmsd",
            "rebuilt_end_to_end": "native_rebuilt_e2e",
            "rebuilt_rg": "native_rebuilt_rg",
        })

        if summary_df.empty:
            summary_df = native_small.copy()
        else:
            summary_df = summary_df.merge(
                native_small,
                on=["experiment_id"],
                how="outer",
                suffixes=("", "_native"),
            )

            for base in ["protein_name", "protein_label", "reference_pdb_id", "reference_pdb_path"]:
                alt = f"{base}_native"
                if alt in summary_df.columns:
                    if base not in summary_df.columns:
                        summary_df[base] = summary_df[alt]
                    else:
                        summary_df[base] = summary_df[base].fillna(summary_df[alt])
                    summary_df = summary_df.drop(columns=[alt])

    if not summary_df.empty:
        if "protein_name" in summary_df.columns:
            summary_df["protein_name"] = (
                summary_df["protein_name"]
                .astype(str)
                .str.replace(r"_ff-.*$", "", regex=True)
                .replace({"nan": None})
            )
        summary_df["protein_label"] = summary_df.apply(
            lambda r: _make_protein_label(r.get("protein_name"), r.get("reference_pdb_id")),
            axis=1
        )

    return summary_df


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Collect panel outputs into master dataframes.")
    ap.add_argument("--root", default="panel_runs", help="Root folder containing panel run outputs")
    ap.add_argument("--outdir", default="panel_analysis", help="Output directory for master CSVs")
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    beam_df = collect_beam_rows(root)
    native_df = collect_native_rows(root)
    summary_df = build_summary(beam_df, native_df, root)

    beam_out = outdir / "master_beam_rows.csv"
    native_out = outdir / "master_native_rows.csv"
    summary_out = outdir / "master_experiment_summary.csv"

    if not beam_df.empty:
        beam_df.to_csv(beam_out, index=False)
        print(f"Wrote {beam_out} ({len(beam_df)} rows)")
    else:
        print("[warn] no beam rows found")

    if not native_df.empty:
        native_df.to_csv(native_out, index=False)
        print(f"Wrote {native_out} ({len(native_df)} rows)")
    else:
        print("[warn] no native rows found")

    if not summary_df.empty:
        summary_df.to_csv(summary_out, index=False)
        print(f"Wrote {summary_out} ({len(summary_df)} rows)")
    else:
        print("[warn] no summary rows built")


if __name__ == "__main__":
    main()
