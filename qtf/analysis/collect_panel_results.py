#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from qtf.analysis.panel import collect_panel_results


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect master CSVs from a grid run root.")
    ap.add_argument("--root", required=True, help="Root folder containing run outputs")
    ap.add_argument("--outdir", required=True, help="Output directory for master CSVs")
    args = ap.parse_args()
    collect_panel_results(Path(args.root), Path(args.outdir))


if __name__ == "__main__":
    main()
