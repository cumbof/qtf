#!/usr/bin/env python3
"""
Unified QTF entry point. Dispatches to the focused subcommands:

    qtf-run fold        -> qtf-fold
    qtf-run bench       -> qtf-bench
    qtf-run eval        -> qtf-eval
    qtf-run grid-search -> qtf-grid-search
    qtf-run relax       -> qtf-relax
"""

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="qtf-run",
        description="Unified QTF entry point. Delegates to focused subcommands.",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    fold_parser = sub.add_parser("fold", help="Run quantum folding prediction")
    fold_parser.add_argument("args", nargs=argparse.REMAINDER,
                             help="Arguments forwarded to qtf-fold")

    bench_parser = sub.add_parser("bench", help="Run beam-search benchmark")
    bench_parser.add_argument("args", nargs=argparse.REMAINDER,
                              help="Arguments forwarded to qtf-bench")

    eval_parser = sub.add_parser("eval", help="Score experimental/native structures")
    eval_parser.add_argument("args", nargs=argparse.REMAINDER,
                             help="Arguments forwarded to qtf-eval")

    grid_parser = sub.add_parser("grid-search", help="Run parameter grid sweep")
    grid_parser.add_argument("args", nargs=argparse.REMAINDER,
                             help="Arguments forwarded to qtf-grid-search")

    relax_parser = sub.add_parser("relax", help="Run GROMACS relaxation")
    relax_parser.add_argument("args", nargs=argparse.REMAINDER,
                              help="Arguments forwarded to qtf-relax")

    parsed = parser.parse_args()

    if parsed.mode == "fold":
        from qtf.cli import fold as fold_mod
        fold_mod.main(argv=parsed.args)

    elif parsed.mode == "bench":
        from qtf.cli import bench as bench_mod
        bench_mod.main(argv=parsed.args)

    elif parsed.mode == "eval":
        from qtf.cli import eval as eval_mod
        eval_mod.main(argv=parsed.args)

    elif parsed.mode == "grid-search":
        from qtf.cli import grid_search as grid_mod
        grid_mod.main(argv=parsed.args)

    elif parsed.mode == "relax":
        from qtf.cli import relax as relax_mod
        relax_mod.main(argv=parsed.args)

    else:
        raise SystemExit(f"Unknown mode: {parsed.mode}")


if __name__ == "__main__":
    main()
