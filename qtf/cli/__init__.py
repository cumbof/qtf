"""Command line interface for QTF."""

from __future__ import annotations

import argparse
import copy
import importlib
import sys
from typing import Optional, Sequence

from qtf.recipes import load_recipes, resolve_recipe


_PASSTHROUGH_COMMANDS = {
    "bench": "qtf.cli.bench",
    "eval": "qtf.cli.eval",
    "grid-search": "qtf.cli.grid_search",
    "relax": "qtf.cli.relax",
    "vmd-trajectory": "qtf.cli.make_vmd_trajectory",
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in _PASSTHROUGH_COMMANDS:
        return _run_passthrough(argv[0], argv[1:])
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "fold":
        return _fold(args)
    if args.command in _PASSTHROUGH_COMMANDS:
        return _run_passthrough(args.command, getattr(args, "args", []))
    parser.print_help()
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qtf")
    sub = parser.add_subparsers(dest="command")
    fold = sub.add_parser("fold", help="Run a QTF fold recipe.")
    fold.add_argument("recipe_name", nargs="?")
    fold.add_argument("--recipe-file", default=None, help="YAML file containing additional or overriding recipes.")
    fold.add_argument("--list-recipes", action="store_true")
    fold.add_argument("--show-recipe", default=None, metavar="NAME")

    fold.add_argument("--sequence", dest="sequence", default=None)
    fold.add_argument("--reference-structure", dest="reference_structure", default=None)
    fold.add_argument("--metric-atom-sets", default=None)
    fold.add_argument("--rmsd-alignment-atom-set", default=None)
    fold.add_argument("--report-structure-domain", default=None)
    fold.add_argument("--outdir", default=None)
    fold.add_argument("--run-id", default=None)
    fold.add_argument("--run-label", default=None)
    fold.add_argument("--n-runs", type=int, default=None)
    fold.add_argument("--run-index", dest="run_index", type=int, default=None)
    fold.add_argument("--seed", type=int, default=None)
    fold.add_argument("--seed-mode", choices=["random", "derived"], default=None)
    fold.add_argument("--shot-seed", type=int, default=None)
    fold.add_argument("--max-iter", dest="max_iter", type=int, default=None)
    fold.add_argument("--scout-attempts", dest="scout_attempts", type=int, default=None)
    fold.add_argument("--top-k-snapshots", dest="top_k_snapshots", type=int, default=None)
    fold.add_argument("--snapshot-energy-gap", dest="snapshot_energy_gap", type=float, default=None)
    fold.add_argument("--snapshot-sort-by", choices=["energy", "rmsd"], default=None)
    fold.add_argument("--backend", default=None)
    fold.add_argument("--ibm-account", default=None)
    fold.add_argument("--ibm-token", default=None)
    fold.add_argument("--ibm-token-env", default=None)
    fold.add_argument("--ibm-token-file", default=None)
    fold.add_argument("--ibm-channel", choices=["ibm_quantum_platform", "ibm_cloud"], default=None)
    fold.add_argument("--ibm-url", default=None)
    fold.add_argument("--ibm-instance-crn", dest="ibm_instance_crn", default=None)
    fold.add_argument("--shots", type=int, default=None)
    fold.add_argument("--stop-on-error", dest="stop_on_error", action="store_true")
    fold.add_argument("--circuit-template", default=None)
    fold.add_argument("--circuit-template-source", choices=["qiskit-library", "qtf"], default=None)
    fold.add_argument("--circuit-template-option", action="append", default=[], metavar="key=value")
    fold.add_argument("--circuit-source", choices=["qpy", "qasm2", "qasm3"], default=None)
    fold.add_argument("--circuit-path", default=None)
    fold.add_argument("--circuit-index", type=int, default=None)

    fold.add_argument("--phase", action="append", default=[])
    fold.add_argument("--phase-label", action="append", default=[], metavar="NAME=LABEL")
    fold.add_argument("--phase-optimizer", action="append", default=[], metavar="NAME=OPTIMIZER")
    fold.add_argument("--phase-score", action="append", default=[], metavar="NAME=MODEL")
    fold.add_argument("--phase-backend", dest="phase_backend", action="append", default=[], metavar="NAME=BACKEND")
    fold.add_argument("--phase-readout-backend", action="append", default=[], metavar="NAME=BACKEND")
    fold.add_argument("--phase-shots", action="append", default=[], metavar="NAME=SHOTS")
    fold.add_argument("--phase-optimizer-shots", action="append", default=[], metavar="NAME=SHOTS")
    fold.add_argument("--phase-readout-shots", action="append", default=[], metavar="NAME=SHOTS")
    fold.add_argument("--phase-max-iter", dest="phase_maxiter", action="append", default=[], metavar="NAME=ITER")
    fold.add_argument(
        "--phase-optimizer-transpile-optimization-level",
        action="append",
        default=[],
        metavar="NAME=LEVEL",
    )
    fold.add_argument(
        "--phase-readout-transpile-optimization-level",
        action="append",
        default=[],
        metavar="NAME=LEVEL",
    )
    fold.add_argument("--phase-optimizer-transpile-seed", action="append", default=[], metavar="NAME=SEED")
    fold.add_argument("--phase-readout-transpile-seed", action="append", default=[], metavar="NAME=SEED")
    fold.add_argument("--phase-tol", action="append", default=[], metavar="NAME=TOL")
    fold.add_argument("--phase-option", action="append", default=[], metavar="NAME:key=value")
    fold.add_argument("--phase-score-option", action="append", default=[], metavar="NAME:key=value")
    fold.add_argument("--phase-geometry-option", action="append", default=[], metavar="NAME:key=value")

    fold.add_argument("--store-angles", default=None)
    fold.add_argument("--store-lengths", default=None)
    fold.add_argument("--angle-units", default=None)
    fold.add_argument("--max-chi", default=None)
    fold.add_argument("--selective-chi", action="append", default=[], metavar="RES=CHI1,CHI2")
    fold.add_argument("--include-terminal-oxt", action="store_true", default=False)
    fold.add_argument("--geometry-mode", default=None)
    fold.add_argument("--geometry-table", default=None)
    fold.add_argument("--geometry-profile", default=None)
    fold.add_argument("--bond-angle-encoding", default=None)
    fold.add_argument("--tau-center-deg", type=float, default=None)
    fold.add_argument("--tau-span-deg", type=float, default=None)
    fold.add_argument("--theta-center-deg", type=float, default=None)
    fold.add_argument("--theta-span-deg", type=float, default=None)
    fold.add_argument("--length-encoding-scope", choices=["shared-by-type", "per-residue"], default=None)
    fold.add_argument("--backbone-length-span", type=float, default=None)
    fold.add_argument("--sidechain-length-span", type=float, default=None)
    fold.add_argument("--basis-circuit-batching", default=None)
    fold.add_argument("--transpile-optimization-level", default=None)
    fold.add_argument("--transpile-seed", default=None)
    fold.add_argument("--estimate-gates", nargs="?", const="__selected_backend__", default=None)
    fold.add_argument("--gate-estimate-optimization-levels", default=None)
    fold.add_argument("--gate-estimate-transpile-seed", default=None)
    fold.add_argument(
        "--gate-estimate-backend-crn",
        action="append",
        default=[],
        metavar="BACKEND=CRN",
        help="Runtime instance CRN to use only for a named IBM gate-estimate backend; repeat for multiple backends.",
    )
    fold.add_argument("--optimizer-angle-mode", default=None)
    fold.add_argument("--scouting-score", default=None)
    fold.add_argument("--scouting-backend", default=None)
    fold.add_argument("--scouting-shots", type=int, default=None)
    fold.add_argument("--scouting-transpile-optimization-level", default=None)
    fold.add_argument("--scouting-transpile-seed", default=None)
    fold.add_argument("--scouting-score-option", action="append", default=[], metavar="key=value")
    fold.add_argument("--result-score", default=None)
    fold.add_argument("--result-score-option", action="append", default=[], metavar="key=value")
    fold.add_argument("--primary-result", default=None)
    fold.add_argument("--readout", action="append", default=[])
    fold.add_argument("--readout-backend", action="append", default=[])
    fold.add_argument("--readout-shots", action="append", default=[])
    fold.add_argument("--readout-score", action="append", default=[])
    fold.add_argument("--readout-transpile-optimization-level", action="append", default=[])
    fold.add_argument("--readout-transpile-seed", action="append", default=[])
    fold.add_argument("--report-command-line", default=None)

    for command, help_text in [
        ("bench", "Run the benchmark workflow."),
        ("eval", "Score or evaluate structures."),
        ("grid-search", "Run a parameter grid-search workflow."),
        ("relax", "Run GROMACS relaxation through QTF utilities."),
        ("vmd-trajectory", "Create a VMD-compatible multi-model PDB."),
    ]:
        passthrough = sub.add_parser(command, help=help_text)
        passthrough.add_argument("args", nargs=argparse.REMAINDER)

    return parser


def _run_passthrough(command: str, argv: Sequence[str]) -> int:
    module = importlib.import_module(_PASSTHROUGH_COMMANDS[command])
    result = module.main(list(argv))
    return int(result or 0)


def _fold(args) -> int:
    recipes = load_recipes(args.recipe_file)
    if args.list_recipes:
        for name in sorted(recipes):
            print(name)
        return 0
    if args.show_recipe:
        _print_recipe(args.show_recipe, args.recipe_file)
        return 0
    if not args.recipe_name:
        raise SystemExit("qtf fold requires a recipe name unless --list-recipes or --show-recipe is used.")

    recipe = resolve_recipe(args.recipe_name, args.recipe_file)
    _apply_common_recipe_overrides(recipe, args)
    n_runs = int(args.n_runs or 1)
    if n_runs < 1:
        raise SystemExit("--n-runs must be >= 1.")
    if args.run_index is not None or n_runs == 1:
        return _run_recipe(args, recipe)

    status = 0
    for run_index in range(n_runs):
        run_args = copy.copy(args)
        run_args.run_index = run_index
        status = _run_recipe(run_args, copy.deepcopy(recipe)) or status
    return status


def _run_recipe(args, recipe: dict) -> int:
    from qtf.engines import qtf

    return qtf.main(_qtf_argv(args, recipe))


def _print_recipe(name: str, recipe_file: Optional[str]) -> None:
    import yaml

    recipe = resolve_recipe(name, recipe_file)
    print(yaml.safe_dump({"recipes": {name: recipe}}, sort_keys=False))


def _apply_common_recipe_overrides(recipe: dict, args) -> None:
    fold = recipe.setdefault("fold", {})
    if args.max_iter is not None:
        fold["max_iter"] = args.max_iter
    if args.scout_attempts is not None:
        fold["scout_attempts"] = args.scout_attempts
    if args.phase:
        recipe["phases"] = [{"name": name} for name in args.phase]
    _apply_phase_assignments(recipe, args)


def _apply_phase_assignments(recipe: dict, args) -> None:
    phases = {str(phase.get("name")): phase for phase in recipe.get("phases") or [] if phase.get("name")}
    assignments = [
        (args.phase_label, "label", str),
        (args.phase_optimizer, "optimizer", str),
        (args.phase_maxiter, "maxiter", int),
        (args.phase_tol, "tol", float),
    ]
    for values, key, cast in assignments:
        for raw in values:
            name, value = _assignment(raw)
            if name not in phases:
                raise SystemExit(f"Unknown phase {name!r} in --phase override.")
            phases[name][key] = cast(value)
    for raw in args.phase_score:
        name, value = _assignment(raw)
        if name not in phases:
            raise SystemExit(f"Unknown phase {name!r} in --phase-score.")
        phase = phases[name]
        phase["score_model"] = value


def _assignment(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise SystemExit(f"Expected NAME=VALUE assignment, got {raw!r}.")
    name, value = raw.split("=", 1)
    if not name.strip() or not value.strip():
        raise SystemExit(f"Expected NAME=VALUE assignment, got {raw!r}.")
    return name.strip(), value.strip()


def _cli_scalar(value) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)


def _qtf_argv(args, recipe: dict) -> list[str]:
    fold = recipe.get("fold") or {}
    geometry = recipe.get("geometry") or {}
    metrics = recipe.get("metrics") or {}
    report = recipe.get("report") or {}
    if not args.sequence:
        raise SystemExit("qtf fold recipes require --sequence.")
    argv = [
        "--predict",
        args.sequence,
        "--replica-id",
        str(args.run_index if args.run_index is not None else 0),
        "--recipe",
        recipe["name"],
    ]
    if args.recipe_file:
        argv += ["--recipe-file", args.recipe_file]
    if args.reference_structure:
        argv += ["--reference-structure", args.reference_structure]
    argv += ["--forcefield", str(fold.get("force_field") or "protein-coarse-charge-v1")]
    argv += ["--maxiter", str(args.max_iter or fold.get("max_iter") or 2000)]
    if args.backend:
        argv += ["--backend", args.backend]
    for flag, value in [
        ("--ibm-account", args.ibm_account),
        ("--ibm-token", args.ibm_token),
        ("--ibm-token-env", args.ibm_token_env),
        ("--ibm-token-file", args.ibm_token_file),
        ("--ibm-channel", args.ibm_channel),
        ("--ibm-url", args.ibm_url),
        ("--ibm-instance-crn", args.ibm_instance_crn),
    ]:
        if value:
            argv += [flag, value]
    if args.shots is not None:
        argv += ["--shots", str(args.shots)]
    if args.outdir:
        argv += ["--outdir", args.outdir]
    if args.run_label:
        argv += ["--run-label", args.run_label]
    if args.stop_on_error:
        argv += ["--stop-on-phase-error"]
    for source, flag in [
        (args.phase_label, "--phase-label"),
        (args.phase_optimizer, "--phase-optimizer"),
        (args.phase_score, "--phase-score"),
        (args.phase_backend, "--phase-optimizer-backend"),
        (args.phase_readout_backend, "--phase-readout-backend"),
        (args.phase_shots, "--phase-shots"),
        (args.phase_optimizer_shots, "--phase-optimizer-shots"),
        (args.phase_readout_shots, "--phase-readout-shots"),
        (args.phase_maxiter, "--phase-maxiter"),
        (args.phase_optimizer_transpile_optimization_level, "--phase-optimizer-transpile-optimization-level"),
        (args.phase_readout_transpile_optimization_level, "--phase-readout-transpile-optimization-level"),
        (args.phase_optimizer_transpile_seed, "--phase-optimizer-transpile-seed"),
        (args.phase_readout_transpile_seed, "--phase-readout-transpile-seed"),
        (args.phase_tol, "--phase-tol"),
        (args.phase_option, "--phase-option"),
        (args.phase_score_option, "--phase-score-option"),
        (args.phase_geometry_option, "--phase-geometry-option"),
        (args.readout_backend, "--readout-backend"),
        (args.readout_shots, "--readout-shots"),
        (args.readout_score, "--readout-score"),
        (args.readout_transpile_optimization_level, "--readout-transpile-optimization-level"),
        (args.readout_transpile_seed, "--readout-transpile-seed"),
    ]:
        for value in source:
            argv += [flag, value]
    for value in args.phase:
        argv += ["--phase", value]
    for value in args.readout:
        argv += ["--readout", value]
    geometry_flags = [
        ("--store-angles", args.store_angles, "stored_angles"),
        ("--store-lengths", args.store_lengths, "stored_lengths"),
        ("--angle-units", args.angle_units, "angle_units"),
        ("--geometry-mode", args.geometry_mode, "geometry_mode"),
        ("--geometry-table", args.geometry_table, "geometry_table"),
        ("--geometry-profile", args.geometry_profile, "geometry_profile"),
        ("--max-chi", args.max_chi, "max_chi"),
        ("--bond-angle-encoding", args.bond_angle_encoding, "bond_angle_encoding"),
        ("--tau-center-deg", args.tau_center_deg, "tau_center_deg"),
        ("--tau-span-deg", args.tau_span_deg, "tau_span_deg"),
        ("--theta-center-deg", args.theta_center_deg, "theta_center_deg"),
        ("--theta-span-deg", args.theta_span_deg, "theta_span_deg"),
        ("--length-encoding-scope", args.length_encoding_scope, "length_encoding_scope"),
        ("--backbone-length-span", args.backbone_length_span, "backbone_length_span"),
        ("--sidechain-length-span", args.sidechain_length_span, "sidechain_length_span"),
    ]
    for flag, cli_value, *keys in geometry_flags:
        value = cli_value
        if value is None:
            for key in keys:
                if key in geometry:
                    value = geometry[key]
                    break
        if value is not None:
            argv += [flag, _cli_scalar(value)]
    selective_chi_values = list(args.selective_chi)
    if not selective_chi_values:
        selective_chi_values = _selective_chi_cli_values(geometry.get("selective_chi_map"))
    for value in selective_chi_values:
        argv += ["--selective-chi", value]

    scalar_flags = [
        ("--seed", args.seed),
        ("--seed-mode", args.seed_mode),
        ("--shot-seed", args.shot_seed),
        (
            "--basis-circuit-batching",
            args.basis_circuit_batching
            if args.basis_circuit_batching is not None
            else recipe.get("basis_circuit_batching"),
        ),
        (
            "--transpile-optimization-level",
            args.transpile_optimization_level
            if args.transpile_optimization_level is not None
            else (recipe.get("transpile") or {}).get("optimization_level"),
        ),
        (
            "--transpile-seed",
            args.transpile_seed if args.transpile_seed is not None else (recipe.get("transpile") or {}).get("seed"),
        ),
        (
            "--gate-estimate-optimization-levels",
            args.gate_estimate_optimization_levels
            if args.gate_estimate_optimization_levels is not None
            else (recipe.get("transpile") or {}).get("gate_estimate_optimization_levels"),
        ),
        (
            "--gate-estimate-transpile-seed",
            args.gate_estimate_transpile_seed
            if args.gate_estimate_transpile_seed is not None
            else (
                (recipe.get("transpile") or {}).get("gate_estimate_seed")
                if (recipe.get("transpile") or {}).get("gate_estimate_seed") is not None
                else (recipe.get("transpile") or {}).get("gate_estimate_transpile_seed")
            ),
        ),
        ("--metric-atom-sets", args.metric_atom_sets if args.metric_atom_sets is not None else metrics.get("atom_sets")),
        (
            "--rmsd-alignment-atom-set",
            args.rmsd_alignment_atom_set
            if args.rmsd_alignment_atom_set is not None
            else metrics.get("rmsd_alignment_atom_set"),
        ),
        (
            "--report-structure-domain",
            args.report_structure_domain
            if args.report_structure_domain is not None
            else report.get("structure_domain"),
        ),
        ("--optimizer-angle-mode", args.optimizer_angle_mode),
        ("--scouting-score", args.scouting_score),
        ("--scouting-backend", args.scouting_backend),
        ("--scouting-shots", args.scouting_shots),
        ("--scouting-attempts", args.scout_attempts),
        ("--scouting-transpile-optimization-level", args.scouting_transpile_optimization_level),
        ("--scouting-transpile-seed", args.scouting_transpile_seed),
        ("--result-score", args.result_score),
        ("--primary-result", args.primary_result),
        ("--report-command-line", args.report_command_line),
        ("--top-k-snapshots", args.top_k_snapshots),
        ("--snapshot-energy-gap", args.snapshot_energy_gap),
        ("--snapshot-sort-by", args.snapshot_sort_by),
        ("--circuit-template", args.circuit_template),
        ("--circuit-template-source", args.circuit_template_source),
        ("--circuit-source", args.circuit_source),
        ("--circuit-path", args.circuit_path),
        ("--circuit-index", args.circuit_index),
    ]
    for flag, value in scalar_flags:
        if value is not None:
            argv += [flag, _cli_scalar(value)]
    for value in args.circuit_template_option:
        argv += ["--circuit-template-option", value]
    for value in args.scouting_score_option:
        argv += ["--scouting-score-option", value]
    for value in args.result_score_option:
        argv += ["--result-score-option", value]
    if args.include_terminal_oxt or bool(geometry.get("include_terminal_oxt", False)):
        argv += ["--include-terminal-oxt"]
    if args.estimate_gates is not None:
        argv += ["--estimate-gates"]
        if args.estimate_gates != "__selected_backend__":
            argv += [args.estimate_gates]
    for value in args.gate_estimate_backend_crn:
        argv += ["--gate-estimate-backend-crn", value]
    return argv


def _selective_chi_cli_values(value) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, dict):
        raise SystemExit("geometry.selective_chi_map must be a mapping.")
    entries = []
    for residue, raw_items in value.items():
        residue_name = str(residue).strip()
        if not residue_name:
            raise SystemExit("geometry.selective_chi_map residue names must not be blank.")
        if raw_items is None:
            items = []
        elif isinstance(raw_items, str):
            items = [item.strip() for item in raw_items.split(",") if item.strip()]
        elif isinstance(raw_items, (list, tuple)):
            items = [str(item).strip() for item in raw_items if str(item).strip()]
        else:
            raise SystemExit(
                f"geometry.selective_chi_map[{residue_name!r}] must be a string or list."
            )
        entries.append(f"{residue_name}={','.join(items)}")
    return entries


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
