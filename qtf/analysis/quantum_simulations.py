#!/usr/bin/env python3
"""Analyze QTF ensemble-fold quantum simulation output trees.

Expected input layout::

    run_outputs/quantum_simulations/
      5AWL/
        custom/task_0/<timestamped_run>/ensemble_ranked.csv
        custom/task_0/<timestamped_run>/snapshot_ranked.csv
        ...
      2JOF/
        openmm/task_0/<timestamped_run>/...

For tasks with multiple timestamped runs, the newest run directory containing
``ensemble_ranked.csv`` is selected. Snapshot files are streamed one task at a
time and reduced to compact tables plus sampled funnel-plot points.
"""

from __future__ import annotations

import argparse
import heapq
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from qtf.utils.paths import relativize_absolute_paths, write_portable_csv

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

BACKENDS = ("custom", "openmm", "rosetta")
PANEL_BACKENDS = ("custom", "rosetta", "openmm")
PANEL_PROTEINS = ("5AWL", "2JOF")
BACKEND_LABELS = {
    "custom": "Custom Energy Function",
    "rosetta": "Rosetta Energy Function",
    "openmm": "OpenMM Energy Function",
}
BACKEND_COLORS = {
    "custom": "#0072B2",
    "rosetta": "#009E73",
    "openmm": "#D55E00",
}
BACKEND_RAW_ENERGY_UNITS = {
    "custom": "kcal/mol",
    "rosetta": "REU",
    "openmm": "kJ/mol",
}
PROTEIN_COLORS = {
    "5AWL": "#009E73",
    "2JOF": "#0072B2",
}
SNAPSHOT_CORR_FEATURE_LABELS = {
    "snapshot_energy": "raw energy",
    "snapshot_gromacs_potential_kj_mol": "GROMACS potential (amber99sb-ildn)",
}
SNAPSHOT_SAMPLE_COLOR = "#999999"
NATIVE_SNAPSHOT_COLOR = "#E69F00"
FINAL_MODEL_COLOR = "#0072B2"
SUMMARY_SNAPSHOT_COLOR = "#009E73"
NATIVE_FINAL_COLOR = "#CC79A7"
THRESHOLD_COLOR = "#D55E00"
RUN_TS_RE = re.compile(r"_(custom|openmm|rosetta)_(\d{8}_\d{6})$")


def _raw_energy_ylabel(backend: object) -> str:
    unit = BACKEND_RAW_ENERGY_UNITS.get(str(backend), "units")
    return f"Energy ({unit})"


def _panel_letter(index: int) -> str:
    letters = []
    while True:
        index, remainder = divmod(index, 26)
        letters.append(chr(ord("A") + remainder))
        if index == 0:
            return "".join(reversed(letters))
        index -= 1


def _label_panel(ax, index: int) -> None:
    ax.text(
        -0.12,
        1.08,
        _panel_letter(index),
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _label_panel_inside(ax, index: int) -> None:
    ax.text(
        0.02,
        0.95,
        _panel_letter(index),
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
    )


def _correlation_title(protein: str, backend: str) -> str:
    return f"Energy Function Component Correlations:\n{protein} with {BACKEND_LABELS.get(backend, backend)}"


def _snapshot_corr_feature_labels(features: pd.Series) -> pd.Series:
    return features.map(lambda feature: SNAPSHOT_CORR_FEATURE_LABELS.get(str(feature), str(feature)))

FINAL_USECOLS = [
    "ensemble_id",
    "init_type",
    "energy_rank",
    "energy",
    "rmsd_to_reference_A",
    "raw_rmsd_to_reference_A",
    "gromacs_rmsd_to_reference_A",
    "pred_e2e_A",
    "pred_rg_A",
    "ref_e2e_A",
    "ref_rg_A",
    "rebuilt_full_pdb_path",
    "gromacs_status",
    "gromacs_minimized_full_pdb_path",
    "gromacs_potential_kj_mol",
    "gromacs_potential_kcal_mol",
    "gromacs_converged_fmax_lt_100",
    "gromacs_final_max_force",
    "clash_min_heavy_dist",
    "clash_flag",
    "local_clash_min_heavy_dist",
    "local_clash_flag",
    "ring_penetration_min_dist_A",
    "ring_penetration_flag",
    "gromacs_energy_rank",
]

SNAPSHOT_USECOLS = [
    "ensemble_id",
    "ensemble_rank",
    "snapshot_rank_within_replica",
    "snapshot_energy",
    "snapshot_rmsd_to_reference_A",
    "snapshot_effective_rmsd_to_reference_A",
    "snapshot_gromacs_status",
    "snapshot_gromacs_potential_kj_mol",
    "snapshot_gromacs_potential_kcal_mol",
    "snapshot_gromacs_rmsd_to_reference_A",
    "snapshot_gromacs_final_max_force",
    "snapshot_gromacs_converged_fmax_lt_100",
    "snapshot_gromacs_minimized_full_pdb_path",
    "snapshot_raw_e2e_A",
    "snapshot_raw_rg_A",
    "snapshot_raw_n_ca",
    "snapshot_gromacs_e2e_A",
    "snapshot_gromacs_rg_A",
    "snapshot_gromacs_n_ca",
    "snapshot_effective_e2e_A",
    "snapshot_effective_rg_A",
    "snapshot_effective_n_ca",
    "snapshot_e2e_A",
    "snapshot_rg_A",
    "snapshot_n_ca",
    "snapshot_pdb_path",
    "snapshot_raw_pdb_retained",
    "replica_final_energy",
    "replica_final_rmsd_to_reference_A",
    "snapshot_global_rank",
]


@dataclass(frozen=True)
class SelectedRun:
    protein: str
    backend: str
    task: str
    task_id: int
    run_dir: Path
    run_name: str
    timestamp: str
    n_runs_in_task: int
    ensemble_csv: Path
    snapshot_csv: Optional[Path]
    summary_csv: Optional[Path]


def _as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _float_or_nan(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _safe_read_csv(path: Path, *, usecols: Optional[list[str]] = None) -> pd.DataFrame:
    if usecols is None:
        return pd.read_csv(path)
    header = pd.read_csv(path, nrows=0)
    cols = [c for c in usecols if c in header.columns]
    return pd.read_csv(path, usecols=cols)


def _run_timestamp(path: Path) -> str:
    match = RUN_TS_RE.search(path.name)
    return match.group(2) if match else path.name


def iter_selected_runs(
    root: Path,
    proteins: Optional[Iterable[str]] = None,
    *,
    expected_tasks_per_backend: Optional[int] = 400,
) -> tuple[list[SelectedRun], list[dict]]:
    proteins_set = set(proteins or [])
    selected: list[SelectedRun] = []
    missing: list[dict] = []

    for protein_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        protein = protein_dir.name
        if protein == "older_tests":
            continue
        if not any((protein_dir / backend).is_dir() for backend in BACKENDS):
            logger.info("skipping non-result directory under root: %s", protein_dir)
            continue
        if proteins_set and protein not in proteins_set:
            continue
        for backend in BACKENDS:
            backend_dir = protein_dir / backend
            if not backend_dir.is_dir():
                missing.append({"protein": protein, "backend": backend, "task": "", "reason": "missing_backend"})
                continue
            task_dirs: dict[int, Path] = {}
            extra_task_dirs: list[Path] = []
            for p in backend_dir.iterdir():
                if not (p.is_dir() and p.name.startswith("task_")):
                    continue
                suffix = p.name.split("_", 1)[1]
                if suffix.isdigit():
                    task_dirs[int(suffix)] = p
                else:
                    extra_task_dirs.append(p)

            task_ids = set(task_dirs)
            if expected_tasks_per_backend is not None:
                task_ids.update(range(int(expected_tasks_per_backend)))
            ordered_task_ids = sorted(task_ids)

            for task_id in ordered_task_ids:
                task_dir = task_dirs.get(task_id)
                if task_dir is None:
                    missing.append({
                        "protein": protein,
                        "backend": backend,
                        "task": f"task_{task_id}",
                        "task_id": task_id,
                        "reason": "missing_task_dir",
                    })
                    continue
                candidates: list[tuple[str, Path]] = []
                for run_dir in task_dir.iterdir():
                    if not run_dir.is_dir():
                        continue
                    ensemble_csv = run_dir / "ensemble_ranked.csv"
                    if ensemble_csv.exists():
                        candidates.append((_run_timestamp(run_dir), run_dir))
                if not candidates:
                    missing.append({
                        "protein": protein,
                        "backend": backend,
                        "task": task_dir.name,
                        "task_id": task_id,
                        "reason": "no_ensemble_ranked_csv",
                    })
                    continue
                timestamp, run_dir = sorted(candidates, key=lambda item: item[0])[-1]
                snapshot_csv = run_dir / "snapshot_ranked.csv"
                summary_csv = run_dir / "summary_results.csv"
                selected.append(SelectedRun(
                    protein=protein,
                    backend=backend,
                    task=task_dir.name,
                    task_id=task_id,
                    run_dir=run_dir,
                    run_name=run_dir.name,
                    timestamp=timestamp,
                    n_runs_in_task=len(candidates),
                    ensemble_csv=run_dir / "ensemble_ranked.csv",
                    snapshot_csv=snapshot_csv if snapshot_csv.exists() else None,
                    summary_csv=summary_csv if summary_csv.exists() else None,
                ))

            for task_dir in sorted(extra_task_dirs):
                task_id = -1
                candidates = []
                for run_dir in task_dir.iterdir():
                    if run_dir.is_dir() and (run_dir / "ensemble_ranked.csv").exists():
                        candidates.append((_run_timestamp(run_dir), run_dir))
                if not candidates:
                    missing.append({
                        "protein": protein,
                        "backend": backend,
                        "task": task_dir.name,
                        "task_id": task_id,
                        "reason": "no_ensemble_ranked_csv",
                    })
                    continue
                timestamp, run_dir = sorted(candidates, key=lambda item: item[0])[-1]
                snapshot_csv = run_dir / "snapshot_ranked.csv"
                summary_csv = run_dir / "summary_results.csv"
                selected.append(SelectedRun(
                    protein=protein,
                    backend=backend,
                    task=task_dir.name,
                    task_id=task_id,
                    run_dir=run_dir,
                    run_name=run_dir.name,
                    timestamp=timestamp,
                    n_runs_in_task=len(candidates),
                    ensemble_csv=run_dir / "ensemble_ranked.csv",
                    snapshot_csv=snapshot_csv if snapshot_csv.exists() else None,
                    summary_csv=summary_csv if summary_csv.exists() else None,
                ))

    return selected, missing


def selected_runs_to_frame(selected: list[SelectedRun], missing: list[dict]) -> pd.DataFrame:
    rows = [
        {
            "protein": r.protein,
            "backend": r.backend,
            "task": r.task,
            "task_id": r.task_id,
            "selected_run": r.run_name,
            "timestamp": r.timestamp,
            "n_runs_in_task": r.n_runs_in_task,
            "selected_run_dir": str(r.run_dir),
            "ensemble_ranked_csv": str(r.ensemble_csv),
            "snapshot_ranked_csv": str(r.snapshot_csv) if r.snapshot_csv else "",
            "summary_results_csv": str(r.summary_csv) if r.summary_csv else "",
            "status": "selected",
            "reason": "",
        }
        for r in selected
    ]
    rows.extend({**m, "status": "missing"} for m in missing)
    return pd.DataFrame(rows)


def collect_final_models(selected: list[SelectedRun]) -> pd.DataFrame:
    rows: list[dict] = []
    for run in selected:
        try:
            # Final files are tiny compared with snapshots; keep backend term columns
            # so term-vs-RMSD correlations can be computed without guessing names.
            df = _safe_read_csv(run.ensemble_csv)
        except Exception as exc:
            logger.warning("failed to read %s: %s", run.ensemble_csv, exc)
            continue
        if df.empty:
            continue
        sort_cols = [c for c in ("gromacs_energy_rank", "energy_rank", "energy") if c in df.columns]
        row = df.sort_values(sort_cols, na_position="last").iloc[0].to_dict() if sort_cols else df.iloc[0].to_dict()
        row.update({
            "protein": run.protein,
            "backend": run.backend,
            "task": run.task,
            "task_id": run.task_id,
            "selected_run": run.run_name,
            "selected_run_dir": str(run.run_dir),
            "n_runs_in_task": run.n_runs_in_task,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _best_row(df: pd.DataFrame, col: str) -> Optional[dict]:
    if col not in df.columns:
        return None
    values = _as_numeric(df[col])
    if values.notna().sum() == 0:
        return None
    return df.loc[int(values.idxmin())].to_dict()


def _clean_snapshot_frame(df: pd.DataFrame) -> pd.DataFrame:
    for col in [
        "snapshot_energy",
        "snapshot_rmsd_to_reference_A",
        "snapshot_effective_rmsd_to_reference_A",
        "snapshot_gromacs_potential_kj_mol",
        "snapshot_gromacs_potential_kcal_mol",
        "snapshot_gromacs_rmsd_to_reference_A",
        "snapshot_global_rank",
        "snapshot_rank_within_replica",
        "replica_final_rmsd_to_reference_A",
        "replica_final_energy",
    ]:
        if col in df.columns:
            df[col] = _as_numeric(df[col])
    return df


def _annotate_snapshot_row(row: dict, run: SelectedRun, selector: str) -> dict:
    row = dict(row)
    row.update({
        "protein": run.protein,
        "backend": run.backend,
        "task": run.task,
        "task_id": run.task_id,
        "selected_run": run.run_name,
        "selected_run_dir": str(run.run_dir),
        "selector": selector,
    })
    return row


def _push_top(heap: list[tuple], row: dict, key: float, limit: int, counter: int) -> None:
    if not np.isfinite(key):
        return
    item = (-float(key), counter, row)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif key < -heap[0][0]:
        heapq.heapreplace(heap, item)


def reduce_snapshots(
    selected: list[SelectedRun],
    *,
    native_threshold: float,
    global_top_n: int,
    max_funnel_points_per_group: int,
    rng_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    best_rows: list[dict] = []
    snapshot_summary_rows: list[dict] = []
    native_rows: list[dict] = []
    sample_rows_by_group: dict[tuple[str, str], list[dict]] = {}
    sample_seen_by_group: dict[tuple[str, str], int] = {}
    global_heap_rmsd: list[tuple] = []
    global_heap_gmx: list[tuple] = []
    rng = np.random.default_rng(rng_seed)

    top_counter = 0

    for idx, run in enumerate(selected, start=1):
        if run.snapshot_csv is None:
            continue
        try:
            df = _safe_read_csv(run.snapshot_csv, usecols=SNAPSHOT_USECOLS)
        except Exception as exc:
            logger.warning("failed to read %s: %s", run.snapshot_csv, exc)
            continue
        if df.empty:
            continue
        df = _clean_snapshot_frame(df)

        selectors = {
            "best_effective_rmsd": "snapshot_effective_rmsd_to_reference_A",
            "best_raw_snapshot_rmsd": "snapshot_rmsd_to_reference_A",
            "best_gromacs_snapshot_rmsd": "snapshot_gromacs_rmsd_to_reference_A",
            "best_snapshot_energy": "snapshot_energy",
            "best_snapshot_gromacs_energy": "snapshot_gromacs_potential_kj_mol",
        }
        for selector, col in selectors.items():
            row = _best_row(df, col)
            if row is not None:
                best_rows.append(_annotate_snapshot_row(row, run, selector))

        eff = df.get("snapshot_effective_rmsd_to_reference_A")
        gmx = df.get("snapshot_gromacs_potential_kj_mol")
        if eff is not None:
            eff_num = _as_numeric(eff)
            native_mask = eff_num <= native_threshold
            native_count = int(native_mask.sum())
        else:
            eff_num = pd.Series(dtype=float)
            native_mask = pd.Series(False, index=df.index)
            native_count = 0
        summary = {
            "protein": run.protein,
            "backend": run.backend,
            "task": run.task,
            "task_id": run.task_id,
            "selected_run": run.run_name,
            "n_snapshots": int(len(df)),
            "native_like_snapshot_count": native_count,
            "native_like_snapshot_fraction": float(native_count / len(df)) if len(df) else 0.0,
            "best_snapshot_effective_rmsd_A": float(eff_num.min()) if eff_num.notna().any() else np.nan,
        }
        if gmx is not None:
            gmx_num = _as_numeric(gmx)
            summary["best_snapshot_gromacs_potential_kj_mol"] = float(gmx_num.min()) if gmx_num.notna().any() else np.nan
        snapshot_summary_rows.append(summary)

        if native_count:
            native_keep = df.loc[native_mask].copy()
            native_keep["protein"] = run.protein
            native_keep["backend"] = run.backend
            native_keep["task"] = run.task
            native_keep["task_id"] = run.task_id
            native_keep["selected_run"] = run.run_name
            native_keep["point_type"] = "native_like_snapshot"
            native_rows.extend(native_keep.to_dict(orient="records"))

        group = (run.protein, run.backend)
        group_sample = sample_rows_by_group.setdefault(group, [])
        seen = sample_seen_by_group.get(group, 0)

        for _, raw in df.iterrows():
            raw_row = raw.to_dict()
            row = _annotate_snapshot_row(raw_row, run, "global_best_rmsd")
            _push_top(
                global_heap_rmsd,
                row,
                float(row.get("snapshot_effective_rmsd_to_reference_A", np.nan)),
                global_top_n,
                top_counter,
            )
            top_counter += 1
            row2 = _annotate_snapshot_row(raw_row, run, "global_best_gromacs_energy")
            _push_top(
                global_heap_gmx,
                row2,
                float(row2.get("snapshot_gromacs_potential_kj_mol", np.nan)),
                global_top_n,
                top_counter,
            )
            top_counter += 1

            # Uniform reservoir sample per protein/backend for funnel background points.
            # Exact/native-like summaries above still inspect every snapshot.
            seen += 1
            sample_row = dict(raw_row)
            sample_row.update({
                "protein": run.protein,
                "backend": run.backend,
                "task": run.task,
                "task_id": run.task_id,
                "selected_run": run.run_name,
                "point_type": (
                    "native_like_snapshot"
                    if _float_or_nan(sample_row.get("snapshot_effective_rmsd_to_reference_A", np.nan)) <= native_threshold
                    else "snapshot_sample"
                ),
            })
            if len(group_sample) < max_funnel_points_per_group:
                group_sample.append(sample_row)
            else:
                replace_idx = int(rng.integers(0, seen))
                if replace_idx < max_funnel_points_per_group:
                    group_sample[replace_idx] = sample_row
            sample_seen_by_group[group] = seen

        if idx % 100 == 0:
            logger.info("processed snapshots for %d/%d selected runs", idx, len(selected))

    global_top = [item[2] for item in sorted(global_heap_rmsd, key=lambda x: -x[0])]
    global_top.extend(item[2] for item in sorted(global_heap_gmx, key=lambda x: -x[0]))
    global_top_df = pd.DataFrame(global_top)
    global_top_df = global_top_df.drop_duplicates(
        subset=[c for c in ["protein", "backend", "task", "snapshot_rank_within_replica", "selector"] if c in global_top_df.columns],
        keep="first",
    ) if not global_top_df.empty else global_top_df

    sample_rows = [row for rows in sample_rows_by_group.values() for row in rows]
    sample_df = pd.DataFrame(sample_rows)
    if native_rows:
        native_df = pd.DataFrame(native_rows)
        sample_df = pd.concat([sample_df, native_df], ignore_index=True, sort=False)
        sample_df = sample_df.drop_duplicates(
            subset=[c for c in ["protein", "backend", "task", "snapshot_rank_within_replica"] if c in sample_df.columns],
            keep="first",
        )

    return (
        pd.DataFrame(best_rows),
        pd.DataFrame(snapshot_summary_rows),
        global_top_df,
        sample_df,
    )


def build_final_vs_best(final_df: pd.DataFrame, best_snapshots: pd.DataFrame) -> pd.DataFrame:
    if final_df.empty or best_snapshots.empty:
        return pd.DataFrame()
    best = best_snapshots.loc[best_snapshots["selector"].eq("best_effective_rmsd")].copy()
    keep = [
        "protein", "backend", "task", "task_id", "selected_run",
        "snapshot_rank_within_replica", "snapshot_global_rank",
        "snapshot_effective_rmsd_to_reference_A",
        "snapshot_rmsd_to_reference_A",
        "snapshot_gromacs_rmsd_to_reference_A",
        "snapshot_gromacs_potential_kj_mol",
        "snapshot_gromacs_minimized_full_pdb_path",
        "snapshot_pdb_path",
    ]
    best = best[[c for c in keep if c in best.columns]].rename(columns={
        "snapshot_rank_within_replica": "best_snapshot_rank_within_replica",
        "snapshot_global_rank": "best_snapshot_global_rank",
        "snapshot_effective_rmsd_to_reference_A": "best_snapshot_effective_rmsd_A",
        "snapshot_rmsd_to_reference_A": "best_snapshot_raw_rmsd_A",
        "snapshot_gromacs_rmsd_to_reference_A": "best_snapshot_gromacs_rmsd_A",
        "snapshot_gromacs_potential_kj_mol": "best_snapshot_gromacs_potential_kj_mol",
        "snapshot_gromacs_minimized_full_pdb_path": "best_snapshot_gromacs_minimized_full_pdb_path",
        "snapshot_pdb_path": "best_snapshot_pdb_path",
    })
    final_keep = [
        "protein", "backend", "task", "task_id", "selected_run",
        "rmsd_to_reference_A", "raw_rmsd_to_reference_A", "gromacs_rmsd_to_reference_A",
        "energy", "gromacs_potential_kj_mol", "rebuilt_full_pdb_path", "gromacs_minimized_full_pdb_path",
        "clash_flag", "local_clash_flag", "ring_penetration_flag", "gromacs_converged_fmax_lt_100",
    ]
    out = final_df[[c for c in final_keep if c in final_df.columns]].merge(
        best,
        on=["protein", "backend", "task", "task_id", "selected_run"],
        how="left",
    )
    out["snapshot_rmsd_improvement_A"] = (
        _as_numeric(out["rmsd_to_reference_A"]) - _as_numeric(out["best_snapshot_effective_rmsd_A"])
    )
    return out


def _native_rate(series: pd.Series, threshold: float) -> float:
    vals = _as_numeric(series)
    denom = int(vals.notna().sum())
    if denom == 0:
        return np.nan
    return float((vals <= threshold).sum() / denom)


def build_backend_summary(final_df: pd.DataFrame, final_vs_best: pd.DataFrame, snapshot_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (protein, backend), g in final_df.groupby(["protein", "backend"], dropna=False):
        if {"protein", "backend"}.issubset(final_vs_best.columns):
            vb = final_vs_best.loc[(final_vs_best["protein"] == protein) & (final_vs_best["backend"] == backend)]
        else:
            vb = pd.DataFrame()
        if {"protein", "backend"}.issubset(snapshot_summary.columns):
            ss = snapshot_summary.loc[(snapshot_summary["protein"] == protein) & (snapshot_summary["backend"] == backend)]
        else:
            ss = pd.DataFrame()
        row = {
            "protein": protein,
            "backend": backend,
            "n_final_models": int(len(g)),
            "n_tasks_with_snapshots": int(len(ss)),
            "n_snapshots": int(ss["n_snapshots"].sum()) if "n_snapshots" in ss.columns and not ss.empty else 0,
            "final_best_rmsd_A": float(_as_numeric(g["rmsd_to_reference_A"]).min()) if "rmsd_to_reference_A" in g else np.nan,
            "final_median_rmsd_A": float(_as_numeric(g["rmsd_to_reference_A"]).median()) if "rmsd_to_reference_A" in g else np.nan,
            "snapshot_best_rmsd_A": float(_as_numeric(vb["best_snapshot_effective_rmsd_A"]).min()) if "best_snapshot_effective_rmsd_A" in vb else np.nan,
            "snapshot_median_best_rmsd_A": float(_as_numeric(vb["best_snapshot_effective_rmsd_A"]).median()) if "best_snapshot_effective_rmsd_A" in vb else np.nan,
            "median_snapshot_improvement_A": float(_as_numeric(vb["snapshot_rmsd_improvement_A"]).median()) if "snapshot_rmsd_improvement_A" in vb else np.nan,
            "mean_snapshot_improvement_A": float(_as_numeric(vb["snapshot_rmsd_improvement_A"]).mean()) if "snapshot_rmsd_improvement_A" in vb else np.nan,
        }
        for threshold in (1.5, 2.0, 3.0, 4.0):
            row[f"final_frac_le_{threshold:.1f}A"] = _native_rate(g.get("rmsd_to_reference_A", pd.Series(dtype=float)), threshold)
            row[f"snapshot_task_frac_le_{threshold:.1f}A"] = _native_rate(vb.get("best_snapshot_effective_rmsd_A", pd.Series(dtype=float)), threshold)
        for col in ["clash_flag", "local_clash_flag", "ring_penetration_flag", "gromacs_converged_fmax_lt_100"]:
            if col in g.columns:
                row[f"{col}_fraction"] = float(g[col].fillna(False).astype(bool).mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["protein", "backend"]).reset_index(drop=True) if rows else pd.DataFrame()


def _snapshot_metric_column(df: pd.DataFrame, metric: str) -> str | None:
    for col in (f"snapshot_effective_{metric}_A", f"snapshot_{metric}_A"):
        if col in df.columns:
            return col
    return None


def build_snapshot_physics_summary(outdir: Path) -> pd.DataFrame:
    manifest_path = outdir / "selected_run_manifest.csv"
    if not manifest_path.exists():
        return pd.DataFrame()
    manifest = pd.read_csv(manifest_path)
    if "status" in manifest.columns:
        manifest = manifest.loc[manifest["status"].fillna("") == "selected"]

    rows: list[dict] = []
    for _, run in manifest.iterrows():
        snapshot_csv = str(run.get("snapshot_ranked_csv", "") or "")
        if not snapshot_csv:
            continue
        source_path = Path(snapshot_csv)
        usecols = [
            "snapshot_effective_e2e_A", "snapshot_effective_rg_A", "snapshot_effective_n_ca",
            "snapshot_e2e_A", "snapshot_rg_A", "snapshot_n_ca",
        ]
        df = pd.read_csv(source_path, usecols=lambda c: c in set(usecols))
        if _snapshot_metric_column(df, "e2e") is None or _snapshot_metric_column(df, "rg") is None:
            continue
        if df.empty:
            continue
        row = {
            "protein": run.get("protein"),
            "backend": run.get("backend"),
            "task": run.get("task"),
            "task_id": run.get("task_id"),
            "selected_run": run.get("selected_run"),
            "snapshot_ranked_csv": snapshot_csv,
            "n_snapshot_physics": int(len(df)),
        }
        for prefix in ("e2e", "rg"):
            metric_col = _snapshot_metric_column(df, prefix)
            values = _as_numeric(df.get(metric_col, pd.Series(dtype=float))).dropna() if metric_col else pd.Series(dtype=float)
            row[f"snapshot_{prefix}_n"] = int(len(values))
            row[f"snapshot_{prefix}_mean_A"] = float(values.mean()) if not values.empty else np.nan
            row[f"snapshot_{prefix}_median_A"] = float(values.median()) if not values.empty else np.nan
            row[f"snapshot_{prefix}_q25_A"] = float(values.quantile(0.25)) if not values.empty else np.nan
            row[f"snapshot_{prefix}_q75_A"] = float(values.quantile(0.75)) if not values.empty else np.nan
        rows.append(row)

    task_summary = pd.DataFrame(rows)
    if task_summary.empty:
        return task_summary

    grouped_rows: list[dict] = []
    for (protein, backend), g in task_summary.groupby(["protein", "backend"], dropna=False):
        grouped = {
            "protein": protein,
            "backend": backend,
            "n_snapshot_physics_tasks": int(len(g)),
            "n_snapshot_physics": int(g["n_snapshot_physics"].sum()),
        }
        for metric in ("e2e", "rg"):
            expanded = []
            # Re-read the compact source metric columns once per group so group medians are weighted
            # by snapshots, not by tasks.
            for path in g["snapshot_ranked_csv"].dropna().astype(str):
                vals_df = pd.read_csv(
                    Path(path),
                    usecols=lambda c, m=metric: c in {f"snapshot_effective_{m}_A", f"snapshot_{m}_A"},
                )
                metric_col = _snapshot_metric_column(vals_df, metric)
                if metric_col is None:
                    continue
                vals = _as_numeric(vals_df[metric_col]).dropna()
                if not vals.empty:
                    expanded.append(vals)
            values = pd.concat(expanded, ignore_index=True) if expanded else pd.Series(dtype=float)
            grouped[f"snapshot_{metric}_mean_A"] = float(values.mean()) if not values.empty else np.nan
            grouped[f"snapshot_{metric}_median_A"] = float(values.median()) if not values.empty else np.nan
            grouped[f"snapshot_{metric}_q25_A"] = float(values.quantile(0.25)) if not values.empty else np.nan
            grouped[f"snapshot_{metric}_q75_A"] = float(values.quantile(0.75)) if not values.empty else np.nan
        grouped_rows.append(grouped)
    return pd.DataFrame(grouped_rows).sort_values(["protein", "backend"]).reset_index(drop=True)


def _ca_coords_from_pdb(path: Path) -> np.ndarray:
    coords: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(("ATOM  ", "HETATM")) and line[12:16].strip() == "CA":
                try:
                    coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
                except ValueError:
                    continue
    return np.asarray(coords, dtype=float)


def _compactness_from_pdb(path: Path) -> tuple[float, float, int]:
    coords = _ca_coords_from_pdb(path)
    n_ca = int(len(coords))
    if n_ca == 0:
        return np.nan, np.nan, 0
    e2e = float(np.linalg.norm(coords[-1] - coords[0])) if n_ca >= 2 else np.nan
    centered = coords - coords.mean(axis=0)
    rg = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    return e2e, rg, n_ca


def _first_existing_path(*values: object) -> Path | None:
    for value in values:
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if not text:
            continue
        path = Path(text)
        if path.exists():
            return path
        marker = "run_outputs/quantum_simulations/"
        if marker in text:
            local_path = Path(marker + text.split(marker, 1)[1])
            if local_path.exists():
                return local_path
    return None


def build_best_snapshot_physics_summary(final_vs_best: pd.DataFrame) -> pd.DataFrame:
    if final_vs_best.empty or not {"protein", "backend"}.issubset(final_vs_best.columns):
        return pd.DataFrame()
    rows: list[dict] = []
    for _, row in final_vs_best.iterrows():
        path = _first_existing_path(
            row.get("best_snapshot_gromacs_minimized_full_pdb_path"),
            row.get("best_snapshot_pdb_path"),
        )
        if path is None:
            continue
        e2e, rg, n_ca = _compactness_from_pdb(path)
        rows.append({
            "protein": row.get("protein"),
            "backend": row.get("backend"),
            "task": row.get("task"),
            "task_id": row.get("task_id"),
            "selected_run": row.get("selected_run"),
            "best_snapshot_pdb_path": str(path),
            "best_snapshot_e2e_A": e2e,
            "best_snapshot_rg_A": rg,
            "best_snapshot_n_ca": n_ca,
        })

    per_replica = pd.DataFrame(rows)
    if per_replica.empty:
        return per_replica

    grouped_rows: list[dict] = []
    for (protein, backend), g in per_replica.groupby(["protein", "backend"], dropna=False):
        grouped = {
            "protein": protein,
            "backend": backend,
            "n_best_snapshot_physics": int(len(g)),
        }
        for metric in ("e2e", "rg"):
            values = _as_numeric(g[f"best_snapshot_{metric}_A"]).dropna()
            grouped[f"best_snapshot_{metric}_mean_A"] = float(values.mean()) if not values.empty else np.nan
            grouped[f"best_snapshot_{metric}_median_A"] = float(values.median()) if not values.empty else np.nan
            grouped[f"best_snapshot_{metric}_q25_A"] = float(values.quantile(0.25)) if not values.empty else np.nan
            grouped[f"best_snapshot_{metric}_q75_A"] = float(values.quantile(0.75)) if not values.empty else np.nan
        grouped_rows.append(grouped)
    return pd.DataFrame(grouped_rows).sort_values(["protein", "backend"]).reset_index(drop=True)


def add_physics_to_backend_summary(
    backend_summary: pd.DataFrame,
    final_df: pd.DataFrame,
    snapshot_physics_summary: pd.DataFrame,
    best_snapshot_physics_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if backend_summary.empty:
        return backend_summary
    out = backend_summary.copy()
    physics_prefixes = (
        "final_e2e_", "final_rg_", "ref_e2e_", "ref_rg_",
        "snapshot_e2e_", "snapshot_rg_", "best_snapshot_e2e_", "best_snapshot_rg_",
    )
    physics_exact = {"n_snapshot_physics", "n_best_snapshot_physics"}
    drop_cols = [
        c for c in out.columns
        if c in physics_exact
        or c.startswith(physics_prefixes)
        or any(c.startswith(prefix) and (c.endswith("_x") or c.endswith("_y")) for prefix in physics_prefixes)
        or c in {"n_snapshot_physics_x", "n_snapshot_physics_y"}
    ]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    final_rows = []
    for (protein, backend), g in final_df.groupby(["protein", "backend"], dropna=False):
        row = {"protein": protein, "backend": backend}
        for source, col in (
            ("final_e2e", "pred_e2e_A"),
            ("final_rg", "pred_rg_A"),
            ("ref_e2e", "ref_e2e_A"),
            ("ref_rg", "ref_rg_A"),
        ):
            vals = _as_numeric(g.get(col, pd.Series(dtype=float))).dropna()
            row[f"{source}_mean_A"] = float(vals.mean()) if not vals.empty else np.nan
            row[f"{source}_median_A"] = float(vals.median()) if not vals.empty else np.nan
            row[f"{source}_q25_A"] = float(vals.quantile(0.25)) if not vals.empty else np.nan
            row[f"{source}_q75_A"] = float(vals.quantile(0.75)) if not vals.empty else np.nan
        final_rows.append(row)
    final_summary = pd.DataFrame(final_rows)
    if not final_summary.empty:
        out = out.merge(final_summary, on=["protein", "backend"], how="left")
    if not snapshot_physics_summary.empty:
        keep = [
            "protein", "backend", "n_snapshot_physics",
            "snapshot_e2e_mean_A", "snapshot_e2e_median_A", "snapshot_e2e_q25_A", "snapshot_e2e_q75_A",
            "snapshot_rg_mean_A", "snapshot_rg_median_A", "snapshot_rg_q25_A", "snapshot_rg_q75_A",
        ]
        out = out.merge(snapshot_physics_summary[[c for c in keep if c in snapshot_physics_summary.columns]], on=["protein", "backend"], how="left")
    if best_snapshot_physics_summary is not None and not best_snapshot_physics_summary.empty:
        keep = [
            "protein", "backend", "n_best_snapshot_physics",
            "best_snapshot_e2e_mean_A", "best_snapshot_e2e_median_A", "best_snapshot_e2e_q25_A", "best_snapshot_e2e_q75_A",
            "best_snapshot_rg_mean_A", "best_snapshot_rg_median_A", "best_snapshot_rg_q25_A", "best_snapshot_rg_q75_A",
        ]
        out = out.merge(
            best_snapshot_physics_summary[[c for c in keep if c in best_snapshot_physics_summary.columns]],
            on=["protein", "backend"],
            how="left",
        )
    return out


def compute_term_correlations(final_df: pd.DataFrame, snapshot_sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    final_rows: list[dict] = []
    for (protein, backend), g in final_df.groupby(["protein", "backend"], dropna=False):
        term_cols = [c for c in g.columns if c.startswith("term_")]
        if "energy" in g.columns:
            term_cols.append("energy")
        if "gromacs_potential_kj_mol" in g.columns:
            term_cols.append("gromacs_potential_kj_mol")
        y = _as_numeric(g.get("rmsd_to_reference_A", pd.Series(dtype=float)))
        for col in sorted(set(term_cols)):
            x = _as_numeric(g[col])
            sub = pd.DataFrame({"x": x, "y": y}).dropna()
            if len(sub) < 3:
                continue
            if sub["x"].nunique() < 2 or sub["y"].nunique() < 2:
                continue
            final_rows.append({
                "protein": protein,
                "backend": backend,
                "feature": col,
                "n": int(len(sub)),
                "spearman_r": float(sub["x"].corr(sub["y"], method="spearman")),
                "pearson_r": float(sub["x"].corr(sub["y"], method="pearson")),
            })

    snap_rows: list[dict] = []
    if not snapshot_sample.empty:
        for (protein, backend), g in snapshot_sample.groupby(["protein", "backend"], dropna=False):
            y = _as_numeric(g.get("snapshot_effective_rmsd_to_reference_A", pd.Series(dtype=float)))
            for col in ["snapshot_energy", "snapshot_gromacs_potential_kj_mol"]:
                if col not in g.columns:
                    continue
                x = _as_numeric(g[col])
                sub = pd.DataFrame({"x": x, "y": y}).dropna()
                if len(sub) < 3:
                    continue
                if sub["x"].nunique() < 2 or sub["y"].nunique() < 2:
                    continue
                snap_rows.append({
                    "protein": protein,
                    "backend": backend,
                    "feature": col,
                    "n": int(len(sub)),
                    "spearman_r": float(sub["x"].corr(sub["y"], method="spearman")),
                    "pearson_r": float(sub["x"].corr(sub["y"], method="pearson")),
                })

    return pd.DataFrame(final_rows), pd.DataFrame(snap_rows)


def _require_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qtf_matplotlib"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    return plt


def _plot_funnel_points(
    ax,
    snaps: pd.DataFrame,
    finals: pd.DataFrame,
    native_threshold: float,
    *,
    snapshot_energy_col: str = "snapshot_gromacs_potential_kj_mol",
    final_energy_col: str = "gromacs_potential_kj_mol",
) -> None:
    if not snaps.empty:
        x = _as_numeric(snaps["snapshot_effective_rmsd_to_reference_A"])
        y = _as_numeric(snaps[snapshot_energy_col])
        native = x <= native_threshold
        other = ~native
        ax.scatter(x[other], y[other], s=4, alpha=0.14, c=SNAPSHOT_SAMPLE_COLOR, linewidths=0, label="snapshot sample")
        ax.scatter(
            x[native],
            y[native],
            s=24,
            alpha=0.85,
            c=NATIVE_SNAPSHOT_COLOR,
            edgecolors="none",
            label=f"native-like snapshots <= {native_threshold:g} Å",
        )
    if not finals.empty:
        fx = _as_numeric(finals["rmsd_to_reference_A"])
        fy = _as_numeric(finals[final_energy_col])
        fnative = fx <= native_threshold
        ax.scatter(fx[~fnative], fy[~fnative], s=54, marker="X", alpha=0.9, c=FINAL_MODEL_COLOR, label="final models")
        ax.scatter(
            fx[fnative],
            fy[fnative],
            s=92,
            marker="*",
            alpha=0.95,
            c=NATIVE_FINAL_COLOR,
            label=f"native-like final <= {native_threshold:g} Å",
        )


def make_panel_plots(
    outdir: Path,
    final_df: pd.DataFrame,
    final_vs_best: pd.DataFrame,
    snapshot_sample: pd.DataFrame,
    backend_summary: pd.DataFrame,
    term_corr: pd.DataFrame,
    snapshot_corr: pd.DataFrame,
    physics_summary: pd.DataFrame,
    *,
    native_threshold: float,
) -> None:
    plt = _require_matplotlib()
    plots_dir = outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    proteins = [p for p in PANEL_PROTEINS if p in set(final_df.get("protein", pd.Series(dtype=str)))]
    if not proteins and "protein" in final_df.columns:
        proteins = sorted(final_df["protein"].dropna().astype(str).unique().tolist())
    backends = [b for b in PANEL_BACKENDS if b in set(final_df.get("backend", pd.Series(dtype=str)))]

    if proteins and backends and {"protein", "backend"}.issubset(snapshot_sample.columns):
        fig, axes = plt.subplots(len(backends), len(proteins), figsize=(5.9 * len(proteins), 3.4 * len(backends)), squeeze=False)
        for r, backend in enumerate(backends):
            for c, protein in enumerate(proteins):
                ax = axes[r][c]
                _label_panel_inside(ax, r * len(proteins) + c)
                snaps = snapshot_sample.loc[(snapshot_sample["protein"] == protein) & (snapshot_sample["backend"] == backend)]
                finals = final_df.loc[(final_df["protein"] == protein) & (final_df["backend"] == backend)]
                _plot_funnel_points(ax, snaps, finals, native_threshold)
                y = _as_numeric(snaps.get("snapshot_gromacs_potential_kj_mol", pd.Series(dtype=float))).dropna()
                if not y.empty:
                    lo, hi = y.quantile([0.01, 0.95]).to_list()
                    if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                        ax.set_ylim(float(lo), float(hi))
                ax.set_title(f"{protein} {BACKEND_LABELS.get(backend, backend)}")
                if r == len(backends) - 1:
                    ax.set_xlabel("RMSD to reference (Å)")
                if c == 0:
                    ax.set_ylabel("GROMACS Potential Energy (kJ/mol)")
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), fontsize=8)
        fig.suptitle("Folding Funnels by Protein and Energy Function", y=0.995)
        fig.tight_layout(rect=(0, 0.055, 1, 0.97))
        fig.savefig(plots_dir / "panel_folding_funnels_zoom.png", dpi=240)
        plt.close(fig)

        if {"snapshot_energy"}.issubset(snapshot_sample.columns) and {"energy"}.issubset(final_df.columns):
            fig, axes = plt.subplots(len(backends), len(proteins), figsize=(5.9 * len(proteins), 3.4 * len(backends)), squeeze=False)
            for r, backend in enumerate(backends):
                for c, protein in enumerate(proteins):
                    ax = axes[r][c]
                    _label_panel_inside(ax, r * len(proteins) + c)
                    snaps = snapshot_sample.loc[(snapshot_sample["protein"] == protein) & (snapshot_sample["backend"] == backend)]
                    finals = final_df.loc[(final_df["protein"] == protein) & (final_df["backend"] == backend)]
                    _plot_funnel_points(
                        ax,
                        snaps,
                        finals,
                        native_threshold,
                        snapshot_energy_col="snapshot_energy",
                        final_energy_col="energy",
                    )
                    y = _as_numeric(snaps.get("snapshot_energy", pd.Series(dtype=float))).dropna()
                    if not y.empty:
                        lo, hi = y.quantile([0.01, 0.95]).to_list()
                        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                            ax.set_ylim(float(lo), float(hi))
                    ax.set_title(f"{protein} {BACKEND_LABELS.get(backend, backend)}")
                    if r == len(backends) - 1:
                        ax.set_xlabel("RMSD to reference (Å)")
                    if c == 0:
                        ax.set_ylabel(_raw_energy_ylabel(backend))
            handles, labels = axes[0][0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), fontsize=8)
            fig.suptitle("Folding Funnels by Protein and Energy Function", y=0.995)
            fig.tight_layout(rect=(0, 0.055, 1, 0.97))
            fig.savefig(plots_dir / "panel_folding_funnels_raw_energy_zoom.png", dpi=240)
            fig.savefig(plots_dir / "panel_folding_funnels_raw_energy_zoom.pdf")
            plt.close(fig)

    if proteins and not final_vs_best.empty and {"protein", "backend"}.issubset(final_vs_best.columns):
        fig, axes = plt.subplots(1, len(proteins), figsize=(5.8 * len(proteins), 5.0), squeeze=False)
        for c, protein in enumerate(proteins):
            ax = axes[0][c]
            _label_panel(ax, c)
            g0 = final_vs_best.loc[final_vs_best["protein"] == protein]
            for backend in backends:
                g = g0.loc[g0["backend"] == backend]
                if g.empty:
                    continue
                ax.scatter(
                    _as_numeric(g["rmsd_to_reference_A"]),
                    _as_numeric(g["best_snapshot_effective_rmsd_A"]),
                    s=18,
                    alpha=0.55,
                    color=BACKEND_COLORS.get(backend),
                    label=BACKEND_LABELS.get(backend, backend),
                )
            vals = pd.concat([
                _as_numeric(g0["rmsd_to_reference_A"]),
                _as_numeric(g0["best_snapshot_effective_rmsd_A"]),
            ]).dropna()
            if not vals.empty:
                lo, hi = float(vals.min()), float(vals.max())
                ax.plot([lo, hi], [lo, hi], color="black", linewidth=1, linestyle="--")
            ax.set_title(protein)
            ax.set_xlabel("Final model RMSD (Å)")
            if c == 0:
                ax.set_ylabel("Best snapshot RMSD per replica (Å)")
            ax.legend(fontsize=8)
        fig.suptitle("Final models versus best snapshots")
        fig.tight_layout()
        fig.savefig(plots_dir / "panel_final_vs_best_snapshot.png", dpi=240)
        plt.close(fig)

        fig, axes = plt.subplots(1, len(proteins), figsize=(6.0 * len(proteins), 5.0), squeeze=False)
        for c, protein in enumerate(proteins):
            ax = axes[0][c]
            _label_panel(ax, c)
            g0 = final_vs_best.loc[final_vs_best["protein"] == protein]
            data = []
            labels = []
            for backend in backends:
                g = g0.loc[g0["backend"] == backend]
                if g.empty:
                    continue
                data.append(_as_numeric(g["rmsd_to_reference_A"]).dropna().to_numpy())
                labels.append(f"{BACKEND_LABELS.get(backend, backend).split()[0]}\nfinal")
                data.append(_as_numeric(g["best_snapshot_effective_rmsd_A"]).dropna().to_numpy())
                labels.append(f"{BACKEND_LABELS.get(backend, backend).split()[0]}\nsnapshot")
            if data:
                ax.boxplot(data, tick_labels=labels, showfliers=False)
            ax.axhline(native_threshold, color=THRESHOLD_COLOR, linestyle="--", linewidth=1)
            ax.set_title(protein)
            if c == 0:
                ax.set_ylabel("RMSD to reference (Å)")
            ax.tick_params(axis="x", labelrotation=30)
        fig.suptitle("RMSD distributions for final models and best snapshots")
        fig.tight_layout()
        fig.savefig(plots_dir / "panel_rmsd_distributions.png", dpi=240)
        plt.close(fig)

        fig, axes = plt.subplots(1, len(proteins), figsize=(5.8 * len(proteins), 4.6), squeeze=False)
        for c, protein in enumerate(proteins):
            ax = axes[0][c]
            _label_panel(ax, c)
            g0 = final_vs_best.loc[final_vs_best["protein"] == protein]
            for backend in backends:
                g = g0.loc[g0["backend"] == backend]
                vals = _as_numeric(g.get("snapshot_rmsd_improvement_A", pd.Series(dtype=float))).dropna()
                if not vals.empty:
                    ax.hist(vals, bins=40, alpha=0.45, color=BACKEND_COLORS.get(backend), label=BACKEND_LABELS.get(backend, backend))
            ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
            ax.set_title(protein)
            ax.set_xlabel("Final RMSD - best snapshot RMSD (Å)")
            if c == 0:
                ax.set_ylabel("Replica count")
            ax.legend(fontsize=8)
        fig.suptitle("RMSD improvement from snapshot mining")
        fig.tight_layout()
        fig.savefig(plots_dir / "panel_snapshot_improvement.png", dpi=240)
        plt.close(fig)

        fig, axes = plt.subplots(3, len(proteins), figsize=(5.9 * len(proteins), 13.6), squeeze=False)
        for c, protein in enumerate(proteins):
            g0 = final_vs_best.loc[final_vs_best["protein"] == protein]

            ax = axes[0][c]
            _label_panel(ax, c)
            for backend in backends:
                g = g0.loc[g0["backend"] == backend]
                if g.empty:
                    continue
                ax.scatter(
                    _as_numeric(g["rmsd_to_reference_A"]),
                    _as_numeric(g["best_snapshot_effective_rmsd_A"]),
                    s=18,
                    alpha=0.55,
                    color=BACKEND_COLORS.get(backend),
                    label=BACKEND_LABELS.get(backend, backend),
                )
            vals = pd.concat([
                _as_numeric(g0["rmsd_to_reference_A"]),
                _as_numeric(g0["best_snapshot_effective_rmsd_A"]),
            ]).dropna()
            if not vals.empty:
                lo, hi = float(vals.min()), float(vals.max())
                ax.plot([lo, hi], [lo, hi], color="black", linewidth=1, linestyle="--")
            ax.set_title(protein)
            ax.set_xlabel("Final model RMSD (Å)")
            if c == 0:
                ax.set_ylabel("Best snapshot RMSD per replica (Å)")
            ax.legend(fontsize=8)

            ax = axes[1][c]
            _label_panel(ax, len(proteins) + c)
            data = []
            labels = []
            for backend in backends:
                g = g0.loc[g0["backend"] == backend]
                if g.empty:
                    continue
                data.append(_as_numeric(g["rmsd_to_reference_A"]).dropna().to_numpy())
                labels.append(f"{BACKEND_LABELS.get(backend, backend).split()[0]}\nfinal")
                data.append(_as_numeric(g["best_snapshot_effective_rmsd_A"]).dropna().to_numpy())
                labels.append(f"{BACKEND_LABELS.get(backend, backend).split()[0]}\nsnapshot")
            if data:
                ax.boxplot(data, tick_labels=labels, showfliers=False)
            ax.axhline(native_threshold, color=THRESHOLD_COLOR, linestyle="--", linewidth=1)
            ax.set_title(protein)
            if c == 0:
                ax.set_ylabel("RMSD to reference (Å)")
            ax.tick_params(axis="x", labelrotation=30)

            ax = axes[2][c]
            _label_panel(ax, 2 * len(proteins) + c)
            for backend in backends:
                g = g0.loc[g0["backend"] == backend]
                vals = _as_numeric(g.get("snapshot_rmsd_improvement_A", pd.Series(dtype=float))).dropna()
                if not vals.empty:
                    ax.hist(vals, bins=40, alpha=0.45, color=BACKEND_COLORS.get(backend), label=BACKEND_LABELS.get(backend, backend))
            ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
            ax.set_title(protein)
            ax.set_xlabel("Final RMSD - best snapshot RMSD (Å)")
            if c == 0:
                ax.set_ylabel("Replica count")
            ax.legend(fontsize=8)

        fig.suptitle("Final replicas and best retained snapshots", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.975))
        fig.savefig(plots_dir / "panel_rmsd_snapshot_summary.png", dpi=240)
        fig.savefig(plots_dir / "panel_rmsd_snapshot_summary.pdf")
        plt.close(fig)

    if proteins and not backend_summary.empty:
        fig, axes = plt.subplots(1, len(proteins), figsize=(5.4 * len(proteins), 4.6), squeeze=False)
        for c, protein in enumerate(proteins):
            ax = axes[0][c]
            _label_panel(ax, c)
            g = backend_summary.loc[backend_summary["protein"] == protein].copy()
            g["backend"] = pd.Categorical(g["backend"], categories=backends, ordered=True)
            g = g.sort_values("backend")
            labels = [BACKEND_LABELS.get(str(b), str(b)).split()[0] for b in g["backend"].astype(str)]
            x = np.arange(len(labels))
            width = 0.36
            ax.bar(x - width / 2, g["final_frac_le_2.0A"], width, color=FINAL_MODEL_COLOR, label="final <= 2 Å")
            ax.bar(x + width / 2, g["snapshot_task_frac_le_2.0A"], width, color=PROTEIN_COLORS.get(protein, "#009E73"), label="snapshot <= 2 Å")
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylim(0, 1)
            ax.set_title(protein)
            if c == 0:
                ax.set_ylabel("Fraction of replicas")
            ax.legend(fontsize=8)
        fig.suptitle("Native-like hit rates")
        fig.tight_layout()
        fig.savefig(plots_dir / "panel_native_like_hit_rate.png", dpi=240)
        plt.close(fig)

    if proteins and backends and not snapshot_corr.empty:
        fig, axes = plt.subplots(len(backends), len(proteins), figsize=(5.5 * len(proteins), 2.7 * len(backends)), squeeze=False)
        for r, backend in enumerate(backends):
            for c, protein in enumerate(proteins):
                ax = axes[r][c]
                _label_panel_inside(ax, r * len(proteins) + c)
                g = snapshot_corr.loc[(snapshot_corr["protein"] == protein) & (snapshot_corr["backend"] == backend)]
                if not g.empty:
                    ax.barh(
                        _snapshot_corr_feature_labels(g["feature"]),
                        g["spearman_r"],
                        color=PROTEIN_COLORS.get(protein, "#0072B2"),
                    )
                ax.axvline(0.0, color="black", linewidth=1)
                ax.set_xlim(-0.5, 0.5)
                ax.set_title(_correlation_title(protein, backend), fontsize=8)
                if c == 0:
                    ax.set_ylabel("Snapshot energy")
                if r == len(backends) - 1:
                    ax.set_xlabel("Spearman rho with RMSD")
        fig.suptitle("Snapshot energy-RMSD correlations")
        fig.tight_layout()
        fig.savefig(plots_dir / "panel_snapshot_energy_spearman.png", dpi=240)
        plt.close(fig)

    if proteins and backends and not term_corr.empty:
        fig, axes = plt.subplots(len(backends), len(proteins), figsize=(6.2 * len(proteins), 3.2 * len(backends)), squeeze=False)
        for r, backend in enumerate(backends):
            for c, protein in enumerate(proteins):
                ax = axes[r][c]
                _label_panel(ax, r * len(proteins) + c)
                g = term_corr.loc[(term_corr["protein"] == protein) & (term_corr["backend"] == backend)].dropna(subset=["spearman_r"]).copy()
                if not g.empty:
                    g["abs_r"] = g["spearman_r"].abs()
                    g = g.sort_values("abs_r", ascending=False).head(8).sort_values("spearman_r")
                    ax.barh(g["feature"], g["spearman_r"], color=PROTEIN_COLORS.get(protein, "#009E73"))
                ax.axvline(0.0, color="black", linewidth=1)
                ax.set_xlim(-0.7, 0.7)
                ax.set_title(_correlation_title(protein, backend), fontsize=8)
                if r == len(backends) - 1:
                    ax.set_xlabel("Spearman rho with final RMSD")
        fig.suptitle("Final-model energy term correlations")
        fig.tight_layout()
        fig.savefig(plots_dir / "panel_final_term_spearman.png", dpi=240)
        plt.close(fig)

    physics_plot_df = backend_summary if not backend_summary.empty else pd.DataFrame()
    if proteins and backends and not physics_summary.empty and not physics_plot_df.empty:
        for metric, title, ylabel in (
            ("e2e", "End-to-end distance", "End-to-end distance (Å)"),
            ("rg", "Radius of gyration", "Radius of gyration (Å)"),
        ):
            required_cols = {
                f"ref_{metric}_median_A",
                f"final_{metric}_median_A",
                f"final_{metric}_q25_A",
                f"final_{metric}_q75_A",
                f"snapshot_{metric}_median_A",
                f"snapshot_{metric}_q25_A",
                f"snapshot_{metric}_q75_A",
            }
            if not required_cols.issubset(physics_plot_df.columns):
                continue
            fig, axes = plt.subplots(1, len(proteins), figsize=(6.4 * len(proteins), 5.0), squeeze=False)
            for c, protein in enumerate(proteins):
                ax = axes[0][c]
                _label_panel(ax, c)
                g = physics_plot_df.loc[physics_plot_df["protein"] == protein].copy()
                g["backend"] = pd.Categorical(g["backend"], categories=backends, ordered=True)
                g = g.sort_values("backend")
                labels = [BACKEND_LABELS.get(str(b), str(b)).split()[0] for b in g["backend"].astype(str)]
                x = np.arange(len(labels))
                width = 0.24
                ref = _as_numeric(g[f"ref_{metric}_median_A"])
                final = _as_numeric(g[f"final_{metric}_median_A"])
                snap = _as_numeric(g[f"snapshot_{metric}_median_A"])
                final_err = np.vstack([
                    (final - _as_numeric(g[f"final_{metric}_q25_A"])).clip(lower=0),
                    (_as_numeric(g[f"final_{metric}_q75_A"]) - final).clip(lower=0),
                ])
                snap_err = np.vstack([
                    (snap - _as_numeric(g[f"snapshot_{metric}_q25_A"])).clip(lower=0),
                    (_as_numeric(g[f"snapshot_{metric}_q75_A"]) - snap).clip(lower=0),
                ])
                ax.bar(x - width, ref, width, color="#999999", label="experimental")
                ax.bar(x, final, width, yerr=final_err, capsize=3, color=FINAL_MODEL_COLOR, label="final replicas")
                ax.bar(x + width, snap, width, yerr=snap_err, capsize=3, color=SUMMARY_SNAPSHOT_COLOR, label="snapshots")
                ax.set_xticks(x)
                ax.set_xticklabels(labels)
                ax.set_title(protein)
                if c == 0:
                    ax.set_ylabel(ylabel)
                ax.legend(fontsize=8)
            fig.suptitle(f"{title}: experimental, final models, and saved snapshots")
            fig.tight_layout()
            fig.savefig(plots_dir / f"panel_{metric}_summary_bars.png", dpi=240)
            fig.savefig(plots_dir / f"panel_{metric}_summary_bars.pdf")
            plt.close(fig)

        rmsd_summary_path = outdir / "all_snapshot_vs_final_rmsd_summary.csv"
        if rmsd_summary_path.exists():
            rmsd_summary = pd.read_csv(rmsd_summary_path)
        else:
            rmsd_summary = pd.DataFrame()
        if not rmsd_summary.empty:
            fig, axes = plt.subplots(3, len(proteins), figsize=(6.4 * len(proteins), 12.0), squeeze=False)
            metric_specs = [
                ("rmsd", "RMSD to reference (Å)"),
                ("rg", "Radius of gyration (Å)"),
                ("e2e", "End-to-end distance (Å)"),
            ]
            for r, (metric, ylabel) in enumerate(metric_specs):
                for c, protein in enumerate(proteins):
                    ax = axes[r][c]
                    _label_panel(ax, r * len(proteins) + c)
                    if metric == "rmsd":
                        g = rmsd_summary.loc[rmsd_summary["protein"].astype(str) == protein].copy()
                        g["backend"] = pd.Categorical(g["backend"], categories=backends, ordered=True)
                        g = g.sort_values(["backend", "model_class"])
                        labels = [BACKEND_LABELS.get(str(b), str(b)).split()[0] for b in backends]
                        x = np.arange(len(labels))
                        width = 0.34
                        for offset, model_class, color, label in (
                            (-width / 2, "Final models", FINAL_MODEL_COLOR, "final replicas"),
                            (width / 2, "All snapshots", SUMMARY_SNAPSHOT_COLOR, "snapshots"),
                        ):
                            rows = (
                                g.loc[g["model_class"] == model_class]
                                .set_index("backend")
                                .reindex(backends)
                                .reset_index()
                            )
                            vals = _as_numeric(rows["median_rmsd_A"])
                            err = np.vstack([
                                (vals - _as_numeric(rows["q1_rmsd_A"])).clip(lower=0),
                                (_as_numeric(rows["q3_rmsd_A"]) - vals).clip(lower=0),
                            ])
                            ax.bar(x + offset, vals, width, yerr=err, capsize=3, color=color, label=label)
                    else:
                        g = physics_plot_df.loc[physics_plot_df["protein"].astype(str) == protein].copy()
                        g["backend"] = pd.Categorical(g["backend"], categories=backends, ordered=True)
                        g = g.sort_values("backend")
                        labels = [BACKEND_LABELS.get(str(b), str(b)).split()[0] for b in g["backend"].astype(str)]
                        x = np.arange(len(labels))
                        width = 0.24
                        ref = _as_numeric(g[f"ref_{metric}_median_A"])
                        final = _as_numeric(g[f"final_{metric}_median_A"])
                        snap = _as_numeric(g[f"snapshot_{metric}_median_A"])
                        final_err = np.vstack([
                            (final - _as_numeric(g[f"final_{metric}_q25_A"])).clip(lower=0),
                            (_as_numeric(g[f"final_{metric}_q75_A"]) - final).clip(lower=0),
                        ])
                        snap_err = np.vstack([
                            (snap - _as_numeric(g[f"snapshot_{metric}_q25_A"])).clip(lower=0),
                            (_as_numeric(g[f"snapshot_{metric}_q75_A"]) - snap).clip(lower=0),
                        ])
                        ax.bar(x - width, ref, width, color="#999999", label="experimental")
                        ax.bar(x, final, width, yerr=final_err, capsize=3, color=FINAL_MODEL_COLOR, label="final replicas")
                        ax.bar(x + width, snap, width, yerr=snap_err, capsize=3, color=SUMMARY_SNAPSHOT_COLOR, label="snapshots")
                    ax.set_xticks(x)
                    ax.set_xticklabels(labels)
                    if r == 0:
                        ax.set_title(protein)
                    if c == 0:
                        ax.set_ylabel(ylabel)
                    if r == 0:
                        ax.axhline(native_threshold, color=THRESHOLD_COLOR, linestyle="--", linewidth=1)
            handles, labels = axes[1][0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10)
            fig.suptitle("Final replicas and saved snapshots: RMSD, radius of gyration, and end-to-end distance", y=0.995)
            fig.tight_layout(rect=(0, 0.035, 1, 0.975))
            fig.savefig(plots_dir / "panel_snapshot_final_rmsd_rg_e2e_bars.png", dpi=240)
            fig.savefig(plots_dir / "panel_snapshot_final_rmsd_rg_e2e_bars.pdf")
            plt.close(fig)


def write_native_like_summary_table(outdir: Path, backend_summary: pd.DataFrame) -> None:
    if backend_summary.empty:
        return
    plt = _require_matplotlib()
    plots_dir = outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for protein in [p for p in PANEL_PROTEINS if p in set(backend_summary["protein"].astype(str))]:
        g0 = backend_summary.loc[backend_summary["protein"].astype(str) == protein].copy()
        g0["final_rank"] = _as_numeric(g0["final_median_rmsd_A"]).rank(method="min")
        g0["snapshot_rank"] = _as_numeric(g0["snapshot_median_best_rmsd_A"]).rank(method="min")
        g0["backend"] = pd.Categorical(g0["backend"], categories=PANEL_BACKENDS, ordered=True)
        g0 = g0.sort_values("backend")
        for _, row in g0.iterrows():
            rows.append({
                "Protein": protein,
                "Energy function": BACKEND_LABELS.get(str(row["backend"]), str(row["backend"])),
                "Median final RMSD (Å)": row.get("final_median_rmsd_A", np.nan),
                "Best final RMSD (Å)": row.get("final_best_rmsd_A", np.nan),
                "Median top-snapshot RMSD (Å)": row.get("snapshot_median_best_rmsd_A", np.nan),
                "Best snapshot RMSD (Å)": row.get("snapshot_best_rmsd_A", np.nan),
                "Experimental E2E (Å)": row.get("ref_e2e_median_A", np.nan),
                "Final E2E median (Å)": row.get("final_e2e_median_A", np.nan),
                "Best-snapshot E2E median (Å)": row.get("best_snapshot_e2e_median_A", np.nan),
                "Experimental Rg (Å)": row.get("ref_rg_median_A", np.nan),
                "Final Rg median (Å)": row.get("final_rg_median_A", np.nan),
                "Best-snapshot Rg median (Å)": row.get("best_snapshot_rg_median_A", np.nan),
                "Final RMSD rank": int(row["final_rank"]) if pd.notna(row.get("final_rank")) else np.nan,
                "Snapshot RMSD rank": int(row["snapshot_rank"]) if pd.notna(row.get("snapshot_rank")) else np.nan,
            })
    table_df = pd.DataFrame(rows)
    if table_df.empty:
        return
    write_portable_csv(table_df, outdir / "energy_function_native_like_summary_table.csv")

    display_cols = [
        "Protein",
        "Energy function",
        "Median final RMSD (Å)",
        "Best final RMSD (Å)",
        "Median top-snapshot RMSD (Å)",
        "Best snapshot RMSD (Å)",
        "Experimental E2E (Å)",
        "Final E2E median (Å)",
        "Best-snapshot E2E median (Å)",
        "Experimental Rg (Å)",
        "Final Rg median (Å)",
        "Best-snapshot Rg median (Å)",
    ]
    display = table_df[[c for c in display_cols if c in table_df.columns]].copy()
    numeric_cols = [c for c in display.columns if c not in {"Protein", "Energy function"}]
    for col in numeric_cols:
        display[col] = _as_numeric(display[col]).map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    display["Energy function"] = display["Energy function"].astype(str).str.replace(" Energy Function", "", regex=False)

    column_labels = [
        "Protein", "Energy", "Final\nRMSD\nmed.", "Final\nRMSD\nbest",
        "Snapshot\nRMSD\nmed.", "Snapshot\nRMSD\nbest",
        "Exp.\nE2E", "Final\nE2E\nmed.", "Best snap.\nE2E\nmed.",
        "Exp.\nRg", "Final\nRg\nmed.", "Best snap.\nRg\nmed.",
    ]
    col_widths = [0.06, 0.085, 0.085, 0.08, 0.095, 0.085, 0.075, 0.085, 0.095, 0.07, 0.08, 0.085]
    fig, ax = plt.subplots(figsize=(10.8, 3.15))
    ax.axis("off")
    table = ax.table(
        cellText=display.values,
        colLabels=column_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.1)
    table.scale(1.0, 1.55)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D0D0")
        cell.get_text().set_wrap(True)
        if row == 0:
            cell.set_facecolor("#EAEAEA")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F7F7F7")
    fig.tight_layout()
    fig.savefig(plots_dir / "table_energy_function_native_like_summary.png", dpi=240, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(plots_dir / "table_energy_function_native_like_summary.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def make_plots(
    outdir: Path,
    final_df: pd.DataFrame,
    final_vs_best: pd.DataFrame,
    snapshot_sample: pd.DataFrame,
    backend_summary: pd.DataFrame,
    term_corr: pd.DataFrame,
    snapshot_corr: pd.DataFrame,
    physics_summary: pd.DataFrame,
    *,
    native_threshold: float,
) -> None:
    plt = _require_matplotlib()
    plots_dir = outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Folding funnels: all sampled snapshots, all native-like snapshots, and all final models.
    if {"protein", "backend"}.issubset(snapshot_sample.columns):
        funnel_groups = snapshot_sample.groupby(["protein", "backend"], dropna=False)
    else:
        funnel_groups = []
    for (protein, backend), snaps in funnel_groups:
        finals = final_df.loc[(final_df["protein"] == protein) & (final_df["backend"] == backend)]
        fig, ax = plt.subplots(figsize=(7.2, 5.4))
        if not snaps.empty:
            x = _as_numeric(snaps["snapshot_effective_rmsd_to_reference_A"])
            y = _as_numeric(snaps["snapshot_gromacs_potential_kj_mol"])
            native = x <= native_threshold
            other = ~native
            ax.scatter(x[other], y[other], s=4, alpha=0.14, c=SNAPSHOT_SAMPLE_COLOR, linewidths=0, label="snapshot sample")
            ax.scatter(x[native], y[native], s=24, alpha=0.85, c=NATIVE_SNAPSHOT_COLOR, edgecolors="none", label=f"native-like snapshots <= {native_threshold:g} Å")
        if not finals.empty:
            fx = _as_numeric(finals["rmsd_to_reference_A"])
            fy = _as_numeric(finals["gromacs_potential_kj_mol"])
            fnative = fx <= native_threshold
            ax.scatter(fx[~fnative], fy[~fnative], s=54, marker="X", alpha=0.9, c=FINAL_MODEL_COLOR, label="final models")
            ax.scatter(fx[fnative], fy[fnative], s=92, marker="*", alpha=0.95, c=NATIVE_FINAL_COLOR, label=f"native-like final <= {native_threshold:g} Å")
        ax.set_xlabel("RMSD to reference (Å)")
        ax.set_ylabel("GROMACS Potential Energy (kJ/mol)")
        ax.set_title(f"{protein} {BACKEND_LABELS.get(backend, backend)}")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(plots_dir / f"funnel_{protein}_{backend}.png", dpi=220)
        plt.close(fig)

        if {"snapshot_energy"}.issubset(snaps.columns) and {"energy"}.issubset(finals.columns):
            fig, ax = plt.subplots(figsize=(7.2, 5.4))
            _plot_funnel_points(
                ax,
                snaps,
                finals,
                native_threshold,
                snapshot_energy_col="snapshot_energy",
                final_energy_col="energy",
            )
            ax.set_xlabel("RMSD to reference (Å)")
            ax.set_ylabel(_raw_energy_ylabel(backend))
            ax.set_title(f"{protein} {BACKEND_LABELS.get(backend, backend)}")
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            fig.savefig(plots_dir / f"funnel_raw_energy_{protein}_{backend}.png", dpi=220)
            plt.close(fig)

    # Final vs best snapshot.
    final_vs_best_groups = final_vs_best.groupby("protein", dropna=False) if "protein" in final_vs_best.columns else []
    for protein, g0 in final_vs_best_groups:
        fig, ax = plt.subplots(figsize=(6.2, 5.8))
        for backend, g in g0.groupby("backend", dropna=False):
            ax.scatter(
                _as_numeric(g["rmsd_to_reference_A"]),
                _as_numeric(g["best_snapshot_effective_rmsd_A"]),
                s=18,
                alpha=0.55,
                color=BACKEND_COLORS.get(str(backend)),
                label=BACKEND_LABELS.get(str(backend), str(backend)),
            )
        vals = pd.concat([
            _as_numeric(g0["rmsd_to_reference_A"]),
            _as_numeric(g0["best_snapshot_effective_rmsd_A"]),
        ]).dropna()
        if not vals.empty:
            lo, hi = float(vals.min()), float(vals.max())
            ax.plot([lo, hi], [lo, hi], color="black", linewidth=1, linestyle="--")
        ax.set_xlabel("Final model RMSD (Å)")
        ax.set_ylabel("Best snapshot RMSD per replica (Å)")
        ax.set_title(f"Final vs best snapshot: {protein}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / f"final_vs_best_snapshot_{protein}.png", dpi=220)
        plt.close(fig)

    # RMSD distributions: final and per-task best snapshot.
    final_vs_best_groups = final_vs_best.groupby("protein", dropna=False) if "protein" in final_vs_best.columns else []
    for protein, g0 in final_vs_best_groups:
        backends = [b for b in BACKENDS if b in set(g0["backend"])]
        data = []
        labels = []
        for backend in backends:
            g = g0.loc[g0["backend"] == backend]
            data.append(_as_numeric(g["rmsd_to_reference_A"]).dropna().to_numpy())
            labels.append(f"{BACKEND_LABELS.get(backend, backend).split()[0]}\nfinal")
            data.append(_as_numeric(g["best_snapshot_effective_rmsd_A"]).dropna().to_numpy())
            labels.append(f"{BACKEND_LABELS.get(backend, backend).split()[0]}\nsnapshot")
        if not data:
            continue
        fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(data)), 5.2))
        ax.boxplot(data, tick_labels=labels, showfliers=False)
        ax.axhline(native_threshold, color=THRESHOLD_COLOR, linestyle="--", linewidth=1)
        ax.set_ylabel("RMSD to reference (Å)")
        ax.set_title(f"RMSD distributions: {protein}")
        fig.tight_layout()
        fig.savefig(plots_dir / f"rmsd_distributions_{protein}.png", dpi=220)
        plt.close(fig)

    # Native-like hit rates.
    if not backend_summary.empty:
        for protein, g in backend_summary.groupby("protein", dropna=False):
            labels = [BACKEND_LABELS.get(str(b), str(b)).split()[0] for b in g["backend"].astype(str)]
            x = np.arange(len(labels))
            width = 0.36
            fig, ax = plt.subplots(figsize=(7.2, 4.8))
            ax.bar(x - width / 2, g["final_frac_le_2.0A"], width, color=FINAL_MODEL_COLOR, label="final <= 2 Å")
            ax.bar(x + width / 2, g["snapshot_task_frac_le_2.0A"], width, color=PROTEIN_COLORS.get(str(protein), "#009E73"), label="any snapshot <= 2 Å")
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Fraction of replicas")
            ax.set_title(f"Native-like hit rate: {protein}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(plots_dir / f"native_like_hit_rate_{protein}.png", dpi=220)
            plt.close(fig)

    # Snapshot improvement histogram.
    final_vs_best_groups = final_vs_best.groupby("protein", dropna=False) if "protein" in final_vs_best.columns else []
    for protein, g0 in final_vs_best_groups:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for backend, g in g0.groupby("backend", dropna=False):
            vals = _as_numeric(g["snapshot_rmsd_improvement_A"]).dropna()
            if vals.empty:
                continue
            ax.hist(vals, bins=40, alpha=0.45, color=BACKEND_COLORS.get(str(backend)), label=BACKEND_LABELS.get(str(backend), str(backend)))
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Final RMSD - best snapshot RMSD (Å)")
        ax.set_ylabel("Replica count")
        ax.set_title(f"Snapshot improvement distribution: {protein}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / f"snapshot_improvement_{protein}.png", dpi=220)
        plt.close(fig)

    # Correlation bars.
    for name, corr_df in [("final_term", term_corr), ("snapshot_energy", snapshot_corr)]:
        if corr_df.empty:
            continue
        for (protein, backend), g in corr_df.groupby(["protein", "backend"], dropna=False):
            g = g.dropna(subset=["spearman_r"]).sort_values("spearman_r")
            if g.empty:
                continue
            fig, ax = plt.subplots(figsize=(8.5, max(4.0, 0.22 * len(g))))
            feature_labels = _snapshot_corr_feature_labels(g["feature"]) if name == "snapshot_energy" else g["feature"]
            ax.barh(feature_labels, g["spearman_r"], color=PROTEIN_COLORS.get(str(protein), "#0072B2"))
            ax.axvline(0.0, color="black", linewidth=1)
            ax.set_xlabel("Spearman rho with RMSD")
            ax.set_title(_correlation_title(str(protein), str(backend)), fontsize=10)
            fig.tight_layout()
            fig.savefig(plots_dir / f"{name}_spearman_{protein}_{backend}.png", dpi=220)
            plt.close(fig)

    make_panel_plots(
        outdir,
        final_df,
        final_vs_best,
        snapshot_sample,
        backend_summary,
        term_corr,
        snapshot_corr,
        physics_summary,
        native_threshold=native_threshold,
    )


def regenerate_plots_from_existing(outdir: Path, *, native_threshold: float) -> dict:
    final_df = pd.read_csv(outdir / "final_models.csv")
    final_vs_best = pd.read_csv(outdir / "final_vs_best_snapshot.csv") if (outdir / "final_vs_best_snapshot.csv").exists() else pd.DataFrame()
    snapshot_sample = pd.read_csv(outdir / "funnel_snapshot_points.csv", low_memory=False) if (outdir / "funnel_snapshot_points.csv").exists() else pd.DataFrame()
    backend_summary = pd.read_csv(outdir / "backend_summary.csv") if (outdir / "backend_summary.csv").exists() else pd.DataFrame()
    term_corr = pd.read_csv(outdir / "term_correlations_final.csv") if (outdir / "term_correlations_final.csv").exists() else pd.DataFrame()
    snapshot_corr = pd.read_csv(outdir / "snapshot_energy_correlations.csv") if (outdir / "snapshot_energy_correlations.csv").exists() else pd.DataFrame()
    physics_summary_path = outdir / "snapshot_physics_summary.csv"
    physics_summary = pd.read_csv(physics_summary_path) if physics_summary_path.exists() else build_snapshot_physics_summary(outdir)
    if not physics_summary.empty:
        write_portable_csv(physics_summary, physics_summary_path)
    best_snapshot_physics_path = outdir / "best_snapshot_physics_summary.csv"
    best_snapshot_physics = (
        pd.read_csv(best_snapshot_physics_path)
        if best_snapshot_physics_path.exists()
        else build_best_snapshot_physics_summary(final_vs_best)
    )
    if not best_snapshot_physics.empty:
        write_portable_csv(best_snapshot_physics, best_snapshot_physics_path)
    backend_summary = add_physics_to_backend_summary(backend_summary, final_df, physics_summary, best_snapshot_physics)
    write_portable_csv(backend_summary, outdir / "backend_summary.csv")
    write_native_like_summary_table(outdir, backend_summary)
    make_plots(
        outdir,
        final_df,
        final_vs_best,
        snapshot_sample,
        backend_summary,
        term_corr,
        snapshot_corr,
        physics_summary,
        native_threshold=native_threshold,
    )
    info = {
        "outdir": str(outdir),
        "native_threshold": native_threshold,
        "plots_written": sorted(p.name for p in (outdir / "plots").iterdir()) if (outdir / "plots").exists() else [],
    }
    (outdir / "plot_regeneration_info.json").write_text(
        json.dumps(relativize_absolute_paths(info), indent=2)
    )
    return info


def run_analysis(
    *,
    root: Path,
    outdir: Path,
    proteins: Optional[list[str]],
    native_threshold: float,
    global_top_n: int,
    max_funnel_points_per_group: int,
    rng_seed: int,
    limit_tasks_per_backend: Optional[int],
    expected_tasks_per_backend: Optional[int],
    skip_snapshots: bool,
) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    selected, missing = iter_selected_runs(root, proteins, expected_tasks_per_backend=expected_tasks_per_backend)
    if limit_tasks_per_backend is not None:
        selected = [
            run for run in selected
            if run.task_id >= 0 and run.task_id < int(limit_tasks_per_backend)
        ]

    manifest = selected_runs_to_frame(selected, missing)
    write_portable_csv(manifest, outdir / "selected_run_manifest.csv")

    final_df = collect_final_models(selected)
    write_portable_csv(final_df, outdir / "final_models.csv")

    if skip_snapshots:
        best_snapshots = pd.DataFrame()
        snapshot_summary = pd.DataFrame()
        global_top = pd.DataFrame()
        snapshot_sample = pd.DataFrame()
        final_vs_best = pd.DataFrame()
    else:
        best_snapshots, snapshot_summary, global_top, snapshot_sample = reduce_snapshots(
            selected,
            native_threshold=native_threshold,
            global_top_n=global_top_n,
            max_funnel_points_per_group=max_funnel_points_per_group,
            rng_seed=rng_seed,
        )
        write_portable_csv(best_snapshots, outdir / "best_snapshots_by_task.csv")
        write_portable_csv(snapshot_summary, outdir / "snapshot_task_summary.csv")
        write_portable_csv(global_top, outdir / "global_top_snapshots.csv")
        write_portable_csv(snapshot_sample, outdir / "funnel_snapshot_points.csv")
        final_vs_best = build_final_vs_best(final_df, best_snapshots)
        write_portable_csv(final_vs_best, outdir / "final_vs_best_snapshot.csv")

    snapshot_physics_summary = build_snapshot_physics_summary(outdir)
    if not snapshot_physics_summary.empty:
        write_portable_csv(snapshot_physics_summary, outdir / "snapshot_physics_summary.csv")
    best_snapshot_physics_summary = build_best_snapshot_physics_summary(final_vs_best)
    if not best_snapshot_physics_summary.empty:
        write_portable_csv(best_snapshot_physics_summary, outdir / "best_snapshot_physics_summary.csv")

    backend_summary = build_backend_summary(final_df, final_vs_best, snapshot_summary)
    backend_summary = add_physics_to_backend_summary(
        backend_summary,
        final_df,
        snapshot_physics_summary,
        best_snapshot_physics_summary,
    )
    write_portable_csv(backend_summary, outdir / "backend_summary.csv")

    term_corr, snapshot_corr = compute_term_correlations(final_df, snapshot_sample)
    write_portable_csv(term_corr, outdir / "term_correlations_final.csv")
    write_portable_csv(snapshot_corr, outdir / "snapshot_energy_correlations.csv")

    make_plots(
        outdir,
        final_df,
        final_vs_best,
        snapshot_sample,
        backend_summary,
        term_corr,
        snapshot_corr,
        snapshot_physics_summary,
        native_threshold=native_threshold,
    )
    write_native_like_summary_table(outdir, backend_summary)

    info = {
        "root": str(root),
        "outdir": str(outdir),
        "proteins": proteins or "all",
        "native_threshold": native_threshold,
        "n_selected_runs": len(selected),
        "n_missing": len(missing),
        "expected_tasks_per_backend": expected_tasks_per_backend,
        "skip_snapshots": skip_snapshots,
        "files_written": sorted(p.name for p in outdir.iterdir() if p.is_file()),
        "plots_written": sorted(p.name for p in (outdir / "plots").iterdir()) if (outdir / "plots").exists() else [],
    }
    (outdir / "analysis_run_info.json").write_text(
        json.dumps(relativize_absolute_paths(info), indent=2)
    )
    return info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze QTF quantum_simulations ensemble outputs.")
    parser.add_argument("--root", default="run_outputs/quantum_simulations")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--proteins", nargs="*", default=None, help="Optional protein directory names, e.g. 5AWL 2JOF")
    parser.add_argument("--native-threshold", type=float, default=2.0)
    parser.add_argument("--global-top-n", type=int, default=200)
    parser.add_argument("--max-funnel-points-per-group", type=int, default=75000)
    parser.add_argument("--rng-seed", type=int, default=123)
    parser.add_argument("--limit-tasks-per-backend", type=int, default=None, help="Smoke-test limit by task_id")
    parser.add_argument("--expected-tasks-per-backend", type=int, default=400)
    parser.add_argument("--skip-snapshots", action="store_true", help="Only collect final model tables and final term plots.")
    parser.add_argument("--plots-from-existing", action="store_true", help="Regenerate plots from CSVs already present in --outdir.")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    if args.plots_from_existing:
        info = regenerate_plots_from_existing(Path(args.outdir), native_threshold=args.native_threshold)
        print(json.dumps(info, indent=2))
        return
    info = run_analysis(
        root=Path(args.root),
        outdir=Path(args.outdir),
        proteins=args.proteins,
        native_threshold=args.native_threshold,
        global_top_n=args.global_top_n,
        max_funnel_points_per_group=args.max_funnel_points_per_group,
        rng_seed=args.rng_seed,
        limit_tasks_per_backend=args.limit_tasks_per_backend,
        expected_tasks_per_backend=args.expected_tasks_per_backend,
        skip_snapshots=bool(args.skip_snapshots),
    )
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
