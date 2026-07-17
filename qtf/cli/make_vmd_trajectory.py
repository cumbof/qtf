"""Create a VMD-compatible multi-model PDB by keeping common atoms only."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


def _atom_key(line: str) -> tuple[str, str, str, str, str]:
    return (
        line[12:16].strip(),
        line[17:20].strip(),
        line[21:22].strip(),
        line[22:26].strip(),
        line[26:27].strip(),
    )


def _renumber_atom_line(line: str, serial: int) -> str:
    if len(line) < 11:
        return line
    return f"{line[:6]}{serial:5d}{line[11:]}"


def _read_models(path: Path) -> list[dict[str, object]]:
    models: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(True), start=1):
        if line.startswith("MODEL"):
            if current is not None:
                raise ValueError(f"Nested MODEL before ENDMDL near line {line_number}.")
            current = {"header": line, "other": [], "atoms": []}
            continue
        if line.startswith("ENDMDL"):
            if current is None:
                raise ValueError(f"ENDMDL without MODEL near line {line_number}.")
            models.append(current)
            current = None
            continue
        if current is None:
            continue
        if line.startswith(("ATOM", "HETATM")):
            current["atoms"].append(line)  # type: ignore[index]
        elif not line.startswith("END"):
            current["other"].append(line)  # type: ignore[index]
    if current is not None:
        raise ValueError("Input ended before final ENDMDL.")
    if not models:
        raise ValueError("No MODEL/ENDMDL blocks found.")
    return models


def _common_atom_order(models: list[dict[str, object]]) -> list[tuple[str, str, str, str, str]]:
    atom_key_sets = []
    for model in models:
        atoms = model["atoms"]  # type: ignore[index]
        atom_key_sets.append({_atom_key(line) for line in atoms})
    common = set.intersection(*atom_key_sets)
    order = []
    seen = set()
    for line in models[0]["atoms"]:  # type: ignore[index]
        key = _atom_key(line)
        if key in common and key not in seen:
            order.append(key)
            seen.add(key)
    if not order:
        raise ValueError("No common atoms found across MODEL blocks.")
    return order


def write_vmd_trajectory(input_path: Path, output_path: Path) -> dict[str, object]:
    models = _read_models(input_path)
    common_order = _common_atom_order(models)
    source_counts = Counter(len(model["atoms"]) for model in models)  # type: ignore[arg-type]

    with output_path.open("w", encoding="utf-8") as handle:
        for model_index, model in enumerate(models, start=1):
            handle.write(f"MODEL     {model_index:4d}\n")
            handle.write(
                "REMARK QTF_VMD_COMMON_TOPOLOGY "
                f"atoms_written={len(common_order)} "
                f"source_atoms={len(model['atoms'])} "
                "variable source atoms omitted for VMD-compatible frames\n"
            )
            for line in model["other"]:  # type: ignore[index]
                handle.write(line)
            atom_by_key = {
                _atom_key(line): line
                for line in model["atoms"]  # type: ignore[index]
            }
            for serial, key in enumerate(common_order, start=1):
                handle.write(_renumber_atom_line(atom_by_key[key], serial))
            handle.write("ENDMDL\n")

    return {
        "models": len(models),
        "atoms_per_model": len(common_order),
        "source_atom_count_groups": dict(sorted(source_counts.items())),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input multi-model PDB.")
    parser.add_argument("--output", required=True, help="Output VMD-compatible multi-model PDB.")
    args = parser.parse_args(argv)

    summary = write_vmd_trajectory(Path(args.input), Path(args.output))
    print(
        "Wrote "
        f"{args.output} with {summary['models']} models and "
        f"{summary['atoms_per_model']} common atoms per model "
        f"(source atom counts: {summary['source_atom_count_groups']})."
    )


if __name__ == "__main__":
    main()
