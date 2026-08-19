#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from qtf.utils.paths import relativize_absolute_paths, write_portable_csv


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy proxies for the heavy optional dependencies.
#
# ``pandas`` is a default QTF dependency but is heavy; ``matplotlib``
# is in the ``[workflows]`` extra and may not be installed at all.
# Importing either at module load time forces every user of
# ``qtf.analysis.panel`` to pay the cost and to surface a bare
# ``ModuleNotFoundError`` if the install is broken. The two proxy
# classes below resolve the real module on first attribute access
# and cache it in ``globals()`` so subsequent attribute access is
# the normal fast path.
# ---------------------------------------------------------------------------


class _LazyPandas:
    """Proxy module that defers the :mod:`pandas` import to first use."""

    def __getattr__(self, name: str):
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required by qtf.analysis.panel but could "
                "not be imported. pandas is a default QTF dependency; "
                "install it with `pip install pandas`. The original "
                "ImportError is chained below."
            ) from exc
        globals()["pd"] = pd
        return getattr(pd, name)


class _LazyPyplot:
    """Proxy module that defers the :mod:`matplotlib.pyplot` import to
    first use. ``matplotlib`` is in the ``[workflows]`` extra, so this
    is the *expected* failure mode on a minimal install."""

    def __getattr__(self, name: str):
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required by qtf.analysis.panel but "
                "could not be imported. matplotlib lives in the "
                "`[workflows]` extra; install it with "
                "`pip install \"qtf[workflows]\"` (or "
                "`conda install -c conda-forge matplotlib`). The "
                "original ImportError is chained below."
            ) from exc
        globals()["plt"] = plt
        return getattr(plt, name)


pd = _LazyPandas()
plt = _LazyPyplot()

PARAM_COLS: List[str] = [
    "hbond_scale", "sasa_scale", "vdw_rep_scale", "vdw_attr_scale", "rotamer_scale", "pi_stack_scale",
]
RUN_KEY_COLS: List[str] = [
    "protein_name", "reference_pdb_id", "reference_pdb_path", "sequence", "forcefield", "chi_mode",
    "window_deg", "step_deg", *PARAM_COLS,
]
GOOD_REBUILD_THRESH = 1.5


def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception as e:
        logger.warning("failed to read CSV %s: %s", path, e)
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


def collect_panel_results(root: Path, outdir: Path) -> Dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    beam_df = collect_beam_rows(root)
    native_df = collect_native_rows(root)
    manifest_df = collect_manifests(root)
    summary_df = build_summary(beam_df, native_df)

    written = {}
    if not beam_df.empty:
        written["master_beam_rows"] = outdir / "master_beam_rows.csv"
        write_portable_csv(beam_df, written["master_beam_rows"])
        logger.info("Wrote %s (%d rows)", written["master_beam_rows"], len(beam_df))
    else:
        logger.warning("no beam rows found")

    if not native_df.empty:
        written["master_native_rows"] = outdir / "master_native_rows.csv"
        write_portable_csv(native_df, written["master_native_rows"])
        logger.info("Wrote %s (%d rows)", written["master_native_rows"], len(native_df))
    else:
        logger.warning("no native rows found")

    if not manifest_df.empty:
        written["master_grid_manifest"] = outdir / "master_grid_manifest.csv"
        write_portable_csv(manifest_df, written["master_grid_manifest"])
        logger.info("Wrote %s (%d rows)", written["master_grid_manifest"], len(manifest_df))
    else:
        logger.warning("no manifest rows found")

    if not summary_df.empty:
        written["master_experiment_summary"] = outdir / "master_experiment_summary.csv"
        write_portable_csv(summary_df, written["master_experiment_summary"])
        logger.info("Wrote %s (%d rows)", written["master_experiment_summary"], len(summary_df))
    else:
        logger.warning("no summary rows built")

    return written


def summarize_beam_group(g: pd.DataFrame) -> pd.Series:
    g = g.sort_values(["energy", "energy_rank"], kind="stable").reset_index(drop=True)
    best_energy_row = g.iloc[0]
    best_rmsd_idx = g["rmsd_to_reference_A"].astype(float).idxmin()
    best_rmsd_row = g.loc[best_rmsd_idx]
    native_like_count = int(g["native_like"].fillna(False).astype(bool).sum()) if "native_like" in g.columns else 0
    return pd.Series({
        "n_ranked_rows": int(len(g)),
        "best_energy": float(best_energy_row["energy"]),
        "best_energy_rmsd": float(best_energy_row["rmsd_to_reference_A"]),
        "best_energy_rank": int(best_energy_row["energy_rank"]) if "energy_rank" in best_energy_row else 1,
        "best_rmsd": float(best_rmsd_row["rmsd_to_reference_A"]),
        "best_rmsd_energy": float(best_rmsd_row["energy"]),
        "best_rmsd_rank": int(best_rmsd_row["energy_rank"]) if "energy_rank" in best_rmsd_row else -1,
        "ranking_gap": float(best_energy_row["rmsd_to_reference_A"] - best_rmsd_row["rmsd_to_reference_A"]),
        "native_like_count": native_like_count,
        "native_like_fraction": float(native_like_count / len(g)) if len(g) else 0.0,
        "mean_rmsd_topk": float(g["rmsd_to_reference_A"].mean()),
        "median_rmsd_topk": float(g["rmsd_to_reference_A"].median()),
        "mean_energy_topk": float(g["energy"].mean()),
    })


def build_corrected_summary(beam_csv: Path, native_csv: Path, manifest_csv: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    beam = pd.read_csv(beam_csv)
    native = pd.read_csv(native_csv)
    manifest = pd.read_csv(manifest_csv)
    if "status" in manifest.columns:
        manifest = manifest.loc[manifest["status"].fillna("").eq("ok")].copy()

    for raw in (beam, native):
        for col in ("window_deg", "step_deg"):
            if col not in raw.columns:
                raw[col] = pd.NA

    if {"experiment_id", "window_deg", "step_deg"}.issubset(manifest.columns):
        lookup = manifest[["experiment_id", "window_deg", "step_deg"]].drop_duplicates()
        for raw_name, raw in [("beam", beam), ("native", native)]:
            if "experiment_id" in raw.columns:
                raw2 = raw.merge(lookup, on="experiment_id", how="left", suffixes=("", "_mf"))
                for col in ("window_deg", "step_deg"):
                    raw2[col] = raw2[col].fillna(raw2[f"{col}_mf"])
                    raw2 = raw2.drop(columns=[f"{col}_mf"])
                if raw_name == "beam":
                    beam = raw2
                else:
                    native = raw2

    beam_summary = (
        beam.groupby(RUN_KEY_COLS, dropna=False)
        .apply(summarize_beam_group, include_groups=False)
        .reset_index()
    )

    native_keep_cols = [
        c for c in native.columns if c in RUN_KEY_COLS or c in {
            "name", "pdb_path", "chain", "residue_start", "residue_end", "n_residues", "total_energy",
            "rebuilt_vs_native_ca_rmsd", "native_end_to_end", "native_rg", "rebuilt_end_to_end", "rebuilt_rg",
            "rebuilt_ca_pdb_path", "rebuilt_ca_centroid_pdb_path"
        } or c.startswith("term_")
    ]
    native2 = native[native_keep_cols].copy().rename(columns={
        "total_energy": "native_total_energy",
        "rebuilt_vs_native_ca_rmsd": "native_rebuild_ca_rmsd",
        "rebuilt_ca_pdb_path": "native_rebuilt_ca_pdb_path",
        "rebuilt_ca_centroid_pdb_path": "native_rebuilt_ca_centroid_pdb_path",
    })
    native2 = native2.drop_duplicates(subset=[c for c in RUN_KEY_COLS if c in native2.columns], keep="first")

    merged = beam_summary.merge(
        native2,
        on=[c for c in RUN_KEY_COLS if c in native2.columns],
        how="left",
        validate="one_to_one",
    )

    manifest_keep = [c for c in manifest.columns if c in RUN_KEY_COLS or c in {"experiment_id", "run_dir", "status", "error"}]
    merged = merged.merge(
        manifest[manifest_keep].drop_duplicates(),
        on=[c for c in RUN_KEY_COLS if c in manifest_keep],
        how="left",
        validate="one_to_one",
    )

    merged["native_beats_best_energy"] = merged["native_total_energy"] < merged["best_energy"]
    merged["native_beats_best_rmsd_energy"] = merged["native_total_energy"] < merged["best_rmsd_energy"]
    write_portable_csv(merged, outdir / "master_experiment_summary_corrected.csv")

    protein_summary = (
        merged.groupby("protein_name", dropna=False)
        .agg(
            n_runs=("protein_name", "size"),
            mean_best_rmsd=("best_rmsd", "mean"),
            median_best_rmsd=("best_rmsd", "median"),
            min_best_rmsd=("best_rmsd", "min"),
            mean_best_energy_rmsd=("best_energy_rmsd", "mean"),
            median_best_energy_rmsd=("best_energy_rmsd", "median"),
            mean_ranking_gap=("ranking_gap", "mean"),
            median_ranking_gap=("ranking_gap", "median"),
            min_ranking_gap=("ranking_gap", "min"),
            total_native_like=("native_like_count", "sum"),
            mean_native_rebuild_ca_rmsd=("native_rebuild_ca_rmsd", "mean"),
        )
        .reset_index()
    )
    write_portable_csv(protein_summary, outdir / "protein_level_summary_corrected.csv")
    return beam, native, merged


def _bar(df: pd.DataFrame, x: str, y: str, title: str, ylabel: str, outpath: Path):
    if df.empty:
        return
    plt.figure(figsize=(8, 4.8))
    plt.bar(df[x].astype(str), df[y].astype(float))
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def _save_term_correlations(beam: pd.DataFrame, summary: pd.DataFrame, outdir: Path):
    term_cols = [c for c in beam.columns if c.startswith("term_")]
    if "energy" in beam.columns:
        term_cols = term_cols + ["energy"]

    if "rmsd_to_reference_A" in beam.columns and term_cols:
        corr_rows = []
        for c in term_cols:
            sub = beam[[c, "rmsd_to_reference_A"]].dropna()
            if len(sub) < 3:
                continue
            corr_rows.append({
                "feature": c,
                "pearson_r": sub[c].corr(sub["rmsd_to_reference_A"], method="pearson"),
                "spearman_r": sub[c].corr(sub["rmsd_to_reference_A"], method="spearman"),
            })
        corr_df = pd.DataFrame(corr_rows).sort_values("spearman_r")
        write_portable_csv(corr_df, outdir / "term_correlations_overall.csv")

        if not corr_df.empty:
            plt.figure(figsize=(8, max(4, 0.28 * len(corr_df))))
            plt.barh(corr_df["feature"], corr_df["spearman_r"])
            plt.xlabel("Spearman correlation with RMSD")
            plt.title("Term correlations vs RMSD (all beam rows)")
            plt.tight_layout()
            plt.savefig(outdir / "term_correlations_overall.png", dpi=200)
            plt.close()

    good_proteins = set()
    if "native_rebuild_ca_rmsd" in summary.columns:
        good_proteins = set(
            summary.loc[summary["native_rebuild_ca_rmsd"] <= GOOD_REBUILD_THRESH, "protein_name"].dropna().tolist()
        )

    if good_proteins and "rmsd_to_reference_A" in beam.columns and term_cols:
        beam_good = beam[beam["protein_name"].isin(good_proteins)].copy()
        corr_rows = []
        for c in term_cols:
            sub = beam_good[[c, "rmsd_to_reference_A"]].dropna()
            if len(sub) < 3:
                continue
            corr_rows.append({
                "feature": c,
                "pearson_r": sub[c].corr(sub["rmsd_to_reference_A"], method="pearson"),
                "spearman_r": sub[c].corr(sub["rmsd_to_reference_A"], method="spearman"),
            })
        corr_good = pd.DataFrame(corr_rows).sort_values("spearman_r")
        write_portable_csv(corr_good, outdir / "term_correlations_good_rebuild_only.csv")

        if not corr_good.empty:
            plt.figure(figsize=(8, max(4, 0.28 * len(corr_good))))
            plt.barh(corr_good["feature"], corr_good["spearman_r"])
            plt.xlabel("Spearman correlation with RMSD")
            plt.title("Term correlations vs RMSD (good native rebuilds only)")
            plt.tight_layout()
            plt.savefig(outdir / "term_correlations_good_rebuild_only.png", dpi=200)
            plt.close()


def save_ranked_parameter_sets(summary: pd.DataFrame, outdir: Path):
    needed = {"best_rmsd", "best_energy_rmsd", *PARAM_COLS}
    if summary.empty or not needed.issubset(summary.columns):
        return

    df = summary.copy()
    if "ranking_gap" not in df.columns:
        df["ranking_gap"] = df["best_energy_rmsd"] - df["best_rmsd"]

    grouped = df.groupby(PARAM_COLS, dropna=False).agg(
        n_proteins=("protein_name", "nunique"),
        mean_best_rmsd=("best_rmsd", "mean"),
        median_best_rmsd=("best_rmsd", "median"),
        mean_best_energy_rmsd=("best_energy_rmsd", "mean"),
        median_best_energy_rmsd=("best_energy_rmsd", "median"),
        mean_ranking_gap=("ranking_gap", "mean"),
        median_ranking_gap=("ranking_gap", "median"),
        mean_native_rebuild=("native_rebuild_ca_rmsd", "mean"),
        min_best_rmsd=("best_rmsd", "min"),
        min_ranking_gap=("ranking_gap", "min"),
        total_native_like=("native_like_count", "sum"),
    ).reset_index()

    grouped["score"] = grouped["mean_ranking_gap"] * 2.0 + grouped["mean_best_rmsd"]
    grouped = grouped.sort_values(
        ["score", "mean_ranking_gap", "mean_best_rmsd", "min_best_rmsd"]
    ).reset_index(drop=True)

    write_portable_csv(grouped, outdir / "grid_ranked_parameter_sets.csv")


def make_plots(beam: pd.DataFrame, summary: pd.DataFrame, outdir: Path):
    if not summary.empty and "native_rebuild_ca_rmsd" in summary.columns:
        rebuild = summary[["protein_name", "reference_pdb_id", "native_rebuild_ca_rmsd"]].drop_duplicates().sort_values("protein_name")
        write_portable_csv(rebuild, outdir / "native_rebuild_quality.csv")
        plt.figure(figsize=(8, 4.5))
        plt.bar(rebuild["protein_name"], rebuild["native_rebuild_ca_rmsd"])
        plt.axhline(GOOD_REBUILD_THRESH, linestyle="--")
        plt.ylabel("Native rebuild RMSD (Å)")
        plt.title("Rebuild quality by protein")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(outdir / "native_rebuild_quality.png", dpi=200)
        plt.close()

    if not summary.empty:
        _bar(
            summary.groupby("protein_name", dropna=False)["best_rmsd"].min().reset_index().sort_values("protein_name"),
            "protein_name", "best_rmsd", "Best sampled RMSD by protein", "Best RMSD (Å)",
            outdir / "best_rmsd_by_protein.png"
        )
        _bar(
            summary.groupby("protein_name", dropna=False)["best_energy_rmsd"].min().reset_index().sort_values("protein_name"),
            "protein_name", "best_energy_rmsd", "Best-energy RMSD by protein", "Best-energy RMSD (Å)",
            outdir / "best_energy_rmsd_by_protein.png"
        )
        _bar(
            summary.groupby("protein_name", dropna=False)["ranking_gap"].min().reset_index().sort_values("protein_name"),
            "protein_name", "ranking_gap", "Best ranking gap by protein", "Ranking gap (Å)",
            outdir / "ranking_gap_by_protein.png"
        )

        gap_cols = [c for c in [
            "protein_name", "reference_pdb_id", "chi_mode", "window_deg", "step_deg",
            "hbond_scale", "sasa_scale", "vdw_rep_scale", "vdw_attr_scale",
            "rotamer_scale", "pi_stack_scale", "best_rmsd", "best_energy_rmsd", "ranking_gap"
        ] if c in summary.columns]
        write_portable_csv(summary[gap_cols].copy(), outdir / "ranking_gap_summary.csv")
        write_portable_csv(
            summary.sort_values(["protein_name", "best_rmsd", "ranking_gap"]).groupby("protein_name", dropna=False).head(5),
            outdir / "slide_top5_by_best_rmsd.csv",
        )
        write_portable_csv(
            summary.sort_values(["protein_name", "ranking_gap", "best_rmsd"]).groupby("protein_name", dropna=False).head(5),
            outdir / "slide_top5_by_ranking_gap.csv",
        )

    if not beam.empty and "rmsd_to_reference_A" in beam.columns and "energy" in beam.columns:
        for protein, g in beam.groupby("protein_name", dropna=False):
            g = g[pd.notnull(g["rmsd_to_reference_A"]) & pd.notnull(g["energy"])].copy()
            if g.empty:
                continue
            plt.figure(figsize=(6.5, 5))
            plt.scatter(g["rmsd_to_reference_A"], g["energy"], alpha=0.3)
            plt.xlabel("RMSD to reference (Å)")
            plt.ylabel("Energy")
            plt.title(f"Energy funnel — {protein}")
            plt.tight_layout()
            plt.savefig(outdir / f"funnel_{protein}.png", dpi=200)
            plt.close()

    _save_term_correlations(beam, summary, outdir)


def analyze_collected_results(indir: Path, outdir: Path):
    beam_csv = indir / "master_beam_rows.csv"
    native_csv = indir / "master_native_rows.csv"
    manifest_csv = indir / "master_grid_manifest.csv"
    for p in [beam_csv, native_csv, manifest_csv]:
        if not p.exists():
            raise FileNotFoundError(f"Required input file not found: {p}")
    outdir.mkdir(parents=True, exist_ok=True)
    beam, native, summary = build_corrected_summary(beam_csv, native_csv, manifest_csv, outdir)
    make_plots(beam, summary, outdir)
    save_ranked_parameter_sets(summary, outdir)
    (outdir / "analysis_run_info.json").write_text(json.dumps(relativize_absolute_paths({
        "indir": str(indir),
        "outdir": str(outdir),
        "files_written": sorted([p.name for p in outdir.iterdir() if p.is_file()]),
    }), indent=2))


def run_panel_analysis(root: Path, outdir: Path):
    collect_panel_results(root, outdir)
    analyze_collected_results(outdir, outdir)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Collect and analyze panel/grid results.")
    sub = ap.add_subparsers(dest="mode", required=True)

    collect = sub.add_parser("collect", help="Collect master CSVs from a grid run root")
    collect.add_argument("--root", required=True, help="Root folder containing run outputs")
    collect.add_argument("--outdir", required=True, help="Output directory for master CSVs")

    analyze = sub.add_parser("analyze", help="Analyze previously collected master CSVs")
    analyze.add_argument("--indir", required=True, help="Directory containing master_beam_rows/native/manifest CSVs")
    analyze.add_argument("--outdir", required=True, help="Output directory for analysis products")

    allp = sub.add_parser("all", help="Collect and analyze in one step")
    allp.add_argument("--root", required=True, help="Root folder containing run outputs")
    allp.add_argument("--outdir", required=True, help="Output directory for analysis products")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "collect":
        collect_panel_results(Path(args.root), Path(args.outdir))
    elif args.mode == "analyze":
        analyze_collected_results(Path(args.indir), Path(args.outdir))
    elif args.mode == "all":
        run_panel_analysis(Path(args.root), Path(args.outdir))
    else:
        raise SystemExit(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
