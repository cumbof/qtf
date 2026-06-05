#!/usr/bin/env python3
"""Render compact reports from ranked QTF result CSVs.

This module is intentionally usable both as a library and as a command:

    python -m qtf.analysis.ranked_results_report --csv ensemble_ranked.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def _first_present(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _fmt_float(value, ndigits: int = 3) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{ndigits}f}"


def _fmt_bool(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, str):
        return value
    return "True" if bool(value) else "False"


def _render_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    widths = {c: max(len(c), *(len(str(v)) for v in df[c].tolist())) for c in cols}

    def row(items):
        return "| " + " | ".join(f"{str(v):<{widths[c]}}" for c, v in zip(cols, items)) + " |"

    header = row(cols)
    divider = "| " + " | ".join("-" * widths[c] for c in cols) + " |"
    body = [row(r) for r in df.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *body])


def _ensure_report_rank(
    df: pd.DataFrame,
    gromacs_energy_col: Optional[str],
    qtf_energy_col: str,
    raw_rank_col: str,
) -> pd.DataFrame:
    work = df.copy()
    if gromacs_energy_col is None:
        work = work.sort_values([raw_rank_col, qtf_energy_col], na_position="last").reset_index(drop=True)
        work["report_rank"] = range(1, len(work) + 1)
    elif "gromacs_energy_rank" not in work.columns:
        work = work.sort_values(
            [gromacs_energy_col, qtf_energy_col],
            na_position="last",
        ).reset_index(drop=True)
        work["gromacs_energy_rank"] = range(1, len(work) + 1)
        work["report_rank"] = work["gromacs_energy_rank"]
    else:
        work["report_rank"] = work["gromacs_energy_rank"]
    return work


def build_top_table(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Build a normalized top-N table ranked by GROMACS energy.

    The returned columns compare the original QTF ranking/energy against the
    GROMACS reranking when a GROMACS energy column is available. Older result
    tables sometimes include GROMACS-minimized PDBs and RMSDs but no parsed
    GROMACS potential; those are reported in original rank order. If separate
    raw and GROMACS RMSD columns exist, both are reported; otherwise the best
    available RMSD column is used.
    """

    raw_rank_col = _first_present(df, ["energy_rank", "rank", "model_rank", "ensemble_id"])
    gmx_energy_col = _first_present(
        df,
        ["gromacs_potential_kj_mol", "gromacs_energy_kj_mol", "gromacs_energy", "gmx_energy"],
    )
    qtf_energy_col = _first_present(df, ["energy", "total_energy"])
    raw_rmsd_col = _first_present(
        df,
        ["raw_rmsd_to_reference_A", "rebuilt_vs_native_rmsd_A", "rmsd_A", "rmsd"],
    )
    gmx_rmsd_col = _first_present(
        df,
        ["gromacs_rmsd_to_reference_A", "rmsd_to_reference_A"],
    )
    fallback_rmsd_col = _first_present(
        df,
        ["rmsd_to_reference_A", "rebuilt_vs_native_rmsd_A", "rmsd_A", "rmsd"],
    )
    converge_col = _first_present(
        df,
        ["gromacs_converged_fmax_lt_100", "gromacs_converged", "openmm_converged", "converged"],
    )
    gromacs_status_col = _first_present(df, ["gromacs_status", "gmx_status"])

    work = df.copy()
    if raw_rank_col is None:
        work["raw_rank"] = range(1, len(work) + 1)
        raw_rank_col = "raw_rank"

    if qtf_energy_col is None:
        raise ValueError("Could not find a QTF energy column in the input table")

    work = _ensure_report_rank(work, gmx_energy_col, qtf_energy_col, raw_rank_col)
    table = work.nsmallest(top_n, "report_rank").copy()

    out_cols = {
        "QTF rank": table[raw_rank_col].astype(int),
        "QTF E": table[qtf_energy_col].map(_fmt_float),
    }
    if gmx_energy_col is not None:
        out_cols = {
            "gmx rank": table["gromacs_energy_rank"].astype(int),
            **out_cols,
            "GROMACS kJ/mol": table[gmx_energy_col].map(_fmt_float),
        }
    else:
        out_cols["GROMACS kJ/mol"] = ""
    out = pd.DataFrame(out_cols)

    if raw_rmsd_col and gmx_rmsd_col and raw_rmsd_col != gmx_rmsd_col:
        out["raw RMSD A"] = table[raw_rmsd_col].map(_fmt_float)
        out["GROMACS RMSD A"] = table[gmx_rmsd_col].map(_fmt_float)
        raw_vals = pd.to_numeric(table[raw_rmsd_col], errors="coerce")
        gmx_vals = pd.to_numeric(table[gmx_rmsd_col], errors="coerce")
        out["RMSD delta A"] = (gmx_vals - raw_vals).map(_fmt_float)
    elif fallback_rmsd_col:
        out["RMSD A"] = table[fallback_rmsd_col].map(_fmt_float)

    if gromacs_status_col:
        out["gromacs status"] = table[gromacs_status_col].fillna("").astype(str)
    if converge_col:
        out["converged"] = table[converge_col].map(_fmt_bool)

    return out


def render_top_report(csv_path: str | Path, top_n: int = 20) -> str:
    """Return the markdown top-N report for a ranked CSV."""

    df = pd.read_csv(csv_path)
    top = build_top_table(df, top_n=top_n)
    return f"Source: {csv_path}\n\n{_render_markdown(top)}"


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Render a compact top-N report from a ranked QTF CSV.")
    ap.add_argument("--csv", required=True, help="Ranked QTF CSV to summarize.")
    ap.add_argument("--top_n", type=int, default=20, help="How many rows to include in the report.")
    ap.add_argument("--out_md", default=None, help="Optional markdown file to write.")
    ap.add_argument("--out_csv", default=None, help="Optional normalized CSV to write.")
    args = ap.parse_args(argv)

    df = pd.read_csv(args.csv)
    top = build_top_table(df, top_n=args.top_n)
    md = _render_markdown(top)

    print(f"Source: {args.csv}")
    print(md)

    if args.out_md:
        Path(args.out_md).write_text(f"Source: {args.csv}\n\n{md}\n")
    if args.out_csv:
        top.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()
