#!/usr/bin/env python3
"""
Unified QTF entry point. Dispatches to the focused subcommands:

    qtf-run fold        -> qtf-fold
    qtf-run bench       -> qtf-bench
    qtf-run eval        -> qtf-eval
    qtf-run grid-search -> qtf-grid-search
    qtf-run relax       -> qtf-relax
    qtf-run vmd-trajectory -> qtf-make-vmd-trajectory
"""

import argparse
import sys


_COMMANDS = {
    "fold": ("Run quantum folding prediction", "qtf-fold"),
    "bench": ("Run beam-search benchmark", "qtf-bench"),
    "eval": ("Score experimental/native structures", "qtf-eval"),
    "grid-search": ("Run parameter grid sweep", "qtf-grid-search"),
    "relax": ("Run GROMACS relaxation", "qtf-relax"),
    "vmd-trajectory": ("Create a VMD-compatible multi-model PDB", "qtf-make-vmd-trajectory"),
}


def _help_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qtf-run",
        description="Unified QTF entry point. Delegates to focused subcommands.",
    )
    sub = parser.add_subparsers(dest="mode", metavar="{" + ",".join(_COMMANDS) + "}")
    for command, (help_text, delegate) in _COMMANDS.items():
        command_parser = sub.add_parser(command, help=help_text)
        command_parser.add_argument("args", nargs=argparse.REMAINDER, help=f"Arguments forwarded to {delegate}")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        _help_parser().parse_args(args)
        return

    mode = args[0]
    forwarded_args = args[1:]
    if mode not in _COMMANDS:
        _help_parser().error(f"Unknown mode: {mode}")

    if mode == "fold":
        from qtf.cli import fold as fold_mod
        fold_mod.main(argv=forwarded_args)

    elif mode == "bench":
        from qtf.cli import bench as bench_mod
        bench_mod.main(argv=forwarded_args)

    elif mode == "eval":
        from qtf.cli import eval as eval_mod
        eval_mod.main(argv=forwarded_args)

    elif mode == "grid-search":
        from qtf.cli import grid_search as grid_mod
        grid_mod.main(argv=forwarded_args)

    elif mode == "relax":
        from qtf.cli import relax as relax_mod
        relax_mod.main(argv=forwarded_args)

    elif mode == "vmd-trajectory":
        from qtf.cli import make_vmd_trajectory as vmd_mod
        vmd_mod.main(argv=forwarded_args)


if __name__ == "__main__":
    main()
