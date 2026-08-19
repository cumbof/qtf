"""GROMACS refinement and structural metrics for QTF hardware folds."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

from qtf.utils import gromacs as qtf_gromacs
from qtf.utils import workflow as qtf_workflow


def _as_json_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _write_aligned_pdb_like(source_pdb: Path, out_pdb: Path, aligned_coords: np.ndarray) -> None:
    """Write original PDB records with replacement coordinates."""

    coord_iter = iter(np.asarray(aligned_coords, dtype=float))
    lines = source_pdb.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_pdb.with_name(f"{out_pdb.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for line in lines:
            if line.startswith(("ATOM  ", "HETATM")):
                try:
                    x, y, z = next(coord_iter)
                except StopIteration as exc:
                    raise ValueError(f"More atom records than coordinates in {source_pdb}") from exc
                if len(line) < 54:
                    line = line.rstrip("\n").ljust(54) + "\n"
                handle.write(f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}")
            else:
                handle.write(line)
    temporary.replace(out_pdb)


def _ca_metrics(coords: np.ndarray, labels: list[tuple[int, str, str]]) -> dict[str, Optional[float]]:
    ca = qtf_gromacs.ca_coords(coords, labels)
    if len(ca) < 2:
        return {"e2e_A": None, "rg_A": None}
    metrics = qtf_workflow.calculate_metrics(ca)
    return {
        "e2e_A": _as_json_float(metrics["end_to_end"]),
        "rg_A": _as_json_float(metrics["radius_of_gyration"]),
    }


def refine_hardware_structure(
    input_pdb: str | Path,
    outdir: str | Path,
    *,
    reference_source: Optional[str] = None,
    rmsd_mode: str = "ca",
    rmsd_residue_scope: str = "core",
    forcefield: str = "amber99sb-ildn",
    water: str = "tip3p",
    nsteps: int = 5000,
    emtol: float = 100.0,
    maxwarn: int = 2,
) -> dict[str, Any]:
    """Minimize a hardware-built PDB and return legacy-compatible metadata."""

    input_pdb = Path(input_pdb).resolve()
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    raw_coords, raw_labels = qtf_gromacs.parse_pdb_atoms(str(input_pdb))
    raw_metrics = _ca_metrics(raw_coords, raw_labels)
    raw_rmsd = None
    raw_rmsd_meta = None
    ref_coords = ref_labels = None
    if reference_source:
        ref_coords, ref_labels, _ = qtf_workflow.load_reference_rmsd_coords(
            reference_source, rmsd_mode, average_backbone=False
        )
        aligned, value, raw_rmsd_meta, _ = qtf_workflow.align_structure_to_reference(
            raw_coords, raw_labels, ref_coords, ref_labels, rmsd_mode, rmsd_residue_scope
        )
        raw_rmsd = _as_json_float(value)
        raw_metrics = _ca_metrics(aligned, raw_labels)

    info = qtf_gromacs.refine_pdb_with_gromacs(
        str(input_pdb),
        outdir / "minimized.pdb",
        log_path=outdir / "gromacs_minimize.log",
        forcefield=forcefield,
        water=water,
        nsteps=int(nsteps),
        emtol=float(emtol),
        maxwarn=int(maxwarn),
    )

    refined_rmsd = None
    refined_rmsd_meta = None
    refined_metrics = {"e2e_A": None, "rg_A": None}
    minimized_pdb = str(info.get("gromacs_minimized_full_pdb_path") or "")
    if minimized_pdb and os.path.isfile(minimized_pdb):
        refined_coords, refined_labels = qtf_gromacs.parse_pdb_atoms(minimized_pdb)
        if ref_coords is not None and ref_labels is not None:
            aligned, value, refined_rmsd_meta, _ = qtf_workflow.align_structure_to_reference(
                refined_coords,
                refined_labels,
                ref_coords,
                ref_labels,
                rmsd_mode,
                rmsd_residue_scope,
            )
            refined_rmsd = _as_json_float(value)
            refined_metrics = _ca_metrics(aligned, refined_labels)
            _write_aligned_pdb_like(Path(minimized_pdb), Path(minimized_pdb), aligned)
        else:
            refined_metrics = _ca_metrics(refined_coords, refined_labels)

    effective_rmsd = refined_rmsd if refined_rmsd is not None else raw_rmsd
    effective_e2e = refined_metrics["e2e_A"] if refined_metrics["e2e_A"] is not None else raw_metrics["e2e_A"]
    effective_rg = refined_metrics["rg_A"] if refined_metrics["rg_A"] is not None else raw_metrics["rg_A"]
    return {
        "raw_hardware_rmsd_to_reference_A": raw_rmsd,
        "hardware_gromacs_enabled": True,
        "hardware_gromacs_forcefield": forcefield,
        "hardware_gromacs_water": water,
        "hardware_gromacs_nsteps": int(nsteps),
        "hardware_gromacs_emtol": float(emtol),
        "hardware_gromacs_maxwarn": int(maxwarn),
        "hardware_raw_e2e_A": raw_metrics["e2e_A"],
        "hardware_raw_rg_A": raw_metrics["rg_A"],
        "hardware_raw_rmsd_meta": raw_rmsd_meta,
        "hardware_gromacs_rmsd_to_reference_A": refined_rmsd,
        "hardware_gromacs_rmsd_meta": refined_rmsd_meta,
        "hardware_gromacs_e2e_A": refined_metrics["e2e_A"],
        "hardware_gromacs_rg_A": refined_metrics["rg_A"],
        "hardware_effective_rmsd_to_reference_A": effective_rmsd,
        "hardware_effective_e2e_A": effective_e2e,
        "hardware_effective_rg_A": effective_rg,
        "hardware_gromacs_aligned_pdb_path": minimized_pdb,
        **info,
    }
