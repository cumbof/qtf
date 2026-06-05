#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from qtf.analysis.panel import analyze_collected_results


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze previously collected grid outputs.")
    ap.add_argument("--indir", required=True, help="Directory containing master_beam_rows/native/manifest CSVs")
    ap.add_argument("--outdir", required=True, help="Output directory for analysis products")
    args = ap.parse_args()
    analyze_collected_results(Path(args.indir), Path(args.outdir))


if __name__ == "__main__":
    main()
