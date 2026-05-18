#!/usr/bin/env python3
"""PHEAT-backed QTF single-replica runner.

This executable keeps the quantum circuit and optimization flow from
QTF.runner_hardware3, while using PHEAT residue geometry as the authoritative
structure encoding and the selected PHEAT score as the default objective.
"""

from __future__ import annotations

import argparse
import copy
import html
import importlib.metadata as importlib_metadata
import io
import json
import math
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import QTF.runner_hardware3 as runner

try:
    from pheat import (
        Atom,
        HeavyAtomStructure,
        ResidueGeometry,
        ResidueGeometryStructure,
        available_models,
        load_pdb,
        score_structure,
        structure_radius_of_gyration,
        structure_to_residue_geometry,
        write_pdb,
    )
    from pheat.metrics import radius_of_gyration_delta
    from pheat.pdbio import load_heavy_json, write_heavy_json
    from pheat.residue_geometry import (
        ANGLE_CA_C_N,
        ANGLE_N_CA_C,
        ANGLE_UNITS,
        load_residue_geometry,
        structure_from_residue_geometry,
        write_residue_geometry_json,
    )
    from pheat.residues import CANONICAL_RESIDUES, SIDECHAIN_STEPS, one_to_three, three_to_one
    from pheat.roundtrip import (
        align_reconstructed_to_original,
        normalize_max_chi,
        normalize_stored_angles,
    )
except ImportError as exc:
    raise SystemExit(
        "runner_hardware_pheat.py requires PHEAT to be importable in the active "
        "Python environment. Install PHEAT, for example with "
        "`python -m pip install -e ../pheat`, or run with a PHEAT-enabled "
        "environment."
    ) from exc


PHEAT_FAILURE_PENALTY = 1.0e12
OPTIONAL_PHEAT_ANGLES = ("omega", "tau", "theta")
DEFAULT_RESULT_SCORE_MODEL = "pheat-goap"
BASIS_CIRCUIT_BATCHING_MODES = ("auto", "on", "off")
GATE_ESTIMATE_SELECTED_BACKEND = "__selected_backend__"
GATE_ESTIMATE_OPTIMIZATION_LEVELS = (0, 3)
GATE_ESTIMATE_SEED_TRANSPILER = 42
NON_GATE_OPERATIONS = {"barrier", "measure", "delay", "reset", "snapshot"}
OPTIMIZER_ANGLE_MODES = ("auto", "statevector", "backend")
SUPPORTED_PHASE_OPTIMIZERS = {"COBYLA", "SLSQP", "Powell", "Nelder-Mead", "BFGS", "L-BFGS-B"}
SEED_MODES = ("random", "derived")
SEED_MODULUS = 2**32
RMSD_COMPARE_TOLERANCE = 1e-9
MOLSTAR_PROJECT_URL = "https://molstar.org/"
MOLSTAR_DOI_URL = "https://doi.org/10.1093/nar/gkab314"
PHEAT_REPOSITORY_URL = "https://github.com/BlankenbergLab/pheat"
VIEWER_REFERENCE_COLOR = "#0072B2"
VIEWER_FINAL_COLOR = "#D55E00"
VIEWER_SNAPSHOT_COLORS = (
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#882255",
    "#44AA99",
    "#AA4499",
    "#999933",
)
PHASE_PRESET_DIR = SCRIPT_DIR / "assets" / "pheat_phase_presets"
PHASE_PRESET_SCHEMA_PATH = PHASE_PRESET_DIR / "schema.json"
INHERIT_BACKEND_VALUES = {"inherit", "default", ""}
PRIMARY_LAST_PHASE = "last_phase_structure"
REPORT_ENVIRONMENT_KEYS = (
    "MPLCONFIGDIR",
    "CONDA_PREFIX",
    "VIRTUAL_ENV",
    "PYTHONPATH",
)
VERSION_PACKAGE_NAMES = (
    "numpy",
    "scipy",
    "matplotlib",
    "plotly",
    "PyYAML",
    "jsonschema",
    "qiskit",
    "qiskit-aer",
    "qiskit-ibm-runtime",
    "biopython",
    "mdtraj",
    "openmm",
    "pdbfixer",
)
TIMING_SECTION_ORDER = (
    "backend_access",
    "reference_load",
    "brickwork_encoding",
    "gate_estimates",
    "scouting_initialization",
    "__optimizer_phases__",
    "optional_readouts",
    "primary_result_selection",
    "artifact_writes",
    "landscape_plot",
    "interactive_landscape_plot",
    "diagnostics_rmsd",
    "report_generation",
    "total_run",
)
TIMING_SECTION_LABELS = {
    "backend_access": "Backend access",
    "reference_load": "Reference load",
    "brickwork_encoding": "Brickwork encoding",
    "gate_estimates": "Gate estimates",
    "scouting_initialization": "Scouting initialization",
    "optional_readouts": "Optional readouts",
    "primary_result_selection": "Primary result selection",
    "artifact_writes": "Artifact writes",
    "landscape_plot": "Landscape plot",
    "interactive_landscape_plot": "Interactive landscape plot",
    "diagnostics_rmsd": "Diagnostic RMSD",
    "report_generation": "Report generation",
    "total_run": "Total run",
}

WORKFLOW_STEP_LABELS = (
    "Backend access",
    "Reference loading",
    "Brickwork encoding",
    "Gate estimates",
    "Scouting initialization",
    "Optimization phases",
    "Optional readouts",
    "Primary result selection",
    "Artifact generation",
    "Evaluation and report",
)


def _console_prefix(replica_id: int) -> str:
    return f"[QTF PHEAT replica {replica_id}]"


def _derive_run_label(
    sequence: str,
    replica_id: int,
    reference_structure: Optional[str],
    explicit_label: Optional[str],
) -> str:
    if explicit_label is not None and explicit_label.strip():
        return explicit_label.strip()
    if reference_structure:
        reference_name = Path(reference_structure).stem or str(reference_structure)
        return f"{sequence} / {reference_name} / replica {replica_id}"
    return f"{sequence} / replica {replica_id}"


def _round_seconds(seconds: float) -> float:
    return round(float(seconds), 3)


def _format_elapsed(seconds) -> str:
    if seconds is None:
        return "n/a"
    seconds = float(seconds)
    if seconds >= 10.0:
        return f"{seconds:.1f}s"
    return f"{seconds:.3f}s"


class TimingRecorder:
    def __init__(self):
        self._origin = time.perf_counter()
        self._starts = {"total_run": self._origin}
        self.sections = {}

    def start(self, name: str) -> None:
        self._starts[name] = time.perf_counter()

    def stop(
        self,
        name: str,
        *,
        label: Optional[str] = None,
        status: str = "ok",
        metadata: Optional[dict] = None,
        print_elapsed: bool = True,
    ) -> dict:
        end = time.perf_counter()
        start = self._starts.pop(name, end)
        record = {
            "start_s": _round_seconds(start - self._origin),
            "end_s": _round_seconds(end - self._origin),
            "elapsed_s": _round_seconds(end - start),
            "status": status,
        }
        if metadata:
            record["metadata"] = dict(metadata)
        self.sections[name] = record
        if print_elapsed:
            suffix = "" if status == "ok" else f" ({status})"
            print(f"  {label or TIMING_SECTION_LABELS.get(name, name)} elapsed: {_format_elapsed(record['elapsed_s'])}{suffix}")
        return record

    def skip(self, name: str, *, label: Optional[str] = None, metadata: Optional[dict] = None) -> dict:
        record = {
            "start_s": None,
            "end_s": None,
            "elapsed_s": 0.0,
            "status": "skipped",
        }
        if metadata:
            record["metadata"] = dict(metadata)
        self.sections[name] = record
        print(f"  {label or TIMING_SECTION_LABELS.get(name, name)} elapsed: 0.000s (skipped)")
        return record

    @contextmanager
    def section(
        self,
        name: str,
        *,
        label: Optional[str] = None,
        metadata: Optional[dict] = None,
        print_elapsed: bool = True,
    ):
        self.start(name)
        failed = False
        try:
            yield
        except Exception:
            failed = True
            raise
        finally:
            self.stop(
                name,
                label=label,
                status="error" if failed else "ok",
                metadata=metadata,
                print_elapsed=print_elapsed,
            )

    def as_dict(self) -> dict:
        ordered = {}
        for name in TIMING_SECTION_ORDER:
            if name == "__optimizer_phases__":
                for phase_name, record in self.sections.items():
                    if phase_name.startswith("phase_") and phase_name not in ordered:
                        ordered[phase_name] = record
                continue
            if name in self.sections:
                ordered[name] = self.sections[name]
        for name, record in self.sections.items():
            if name not in ordered:
                ordered[name] = record
        return ordered


class WorkflowProgress:
    def __init__(self, replica_id: int, labels: Sequence[str] = WORKFLOW_STEP_LABELS):
        self.replica_id = replica_id
        self.labels = tuple(labels)
        self.total = len(self.labels)
        self.current = 0

    def start(self, label: str) -> None:
        self.current += 1
        print(f"\n{_console_prefix(self.replica_id)} Step {self.current}/{self.total}: {label}")


class PhaseLandscapeTracker:
    def __init__(self):
        self.history = []
        self.phase_markers = []
        self.current_iter = 0

    def log(self, energy):
        self.history.append(energy)
        self.current_iter += 1

    def mark_phase(self, name):
        self.phase_markers.append((self.current_iter, name))


class StatevectorShotsBackend:
    """Local shot-based backend: exact statevector evolution plus sampled counts."""

    name = "statevector-shots"

    def __init__(self, seed: Optional[int] = None):
        self.seed = None if seed is None else int(seed)

    def __str__(self) -> str:
        return self.name


class TeeStream:
    """Write-through stream used to capture console output without hiding it."""

    def __init__(self, primary, capture: io.StringIO):
        self.primary = primary
        self.capture = capture

    def write(self, text):
        self.primary.write(text)
        self.capture.write(text)
        return len(text)

    def flush(self):
        self.primary.flush()

    def isatty(self):
        return bool(getattr(self.primary, "isatty", lambda: False)())

    @property
    def encoding(self):
        return getattr(self.primary, "encoding", "utf-8")


@dataclass
class PheatReference:
    source_path: Path
    source_type: str
    residue_geometry: ResidueGeometryStructure
    structure: HeavyAtomStructure
    sequence: str


@dataclass
class ScoutingConfig:
    score_model: str
    backend: str
    shots: int
    attempts: int


@dataclass
class PhaseConfig:
    name: str
    label: str
    optimizer: str
    score_model: str
    optimizer_backend: str
    readout_backend: str
    shots: int
    optimizer_shots: int
    readout_shots: int
    maxiter: int
    tol: Optional[float]
    options: dict[str, Any]


@dataclass
class ResultConfig:
    primary: str
    score_model: str


@dataclass
class ReadoutConfig:
    name: str
    source: str
    backend: str
    shots: int
    score_model: str
    primary: bool


@dataclass
class PhaseSchedule:
    preset: str
    source: str
    config_path: Optional[str]
    basis_circuit_batching: str
    scouting: ScoutingConfig
    phases: list[PhaseConfig]
    result: ResultConfig
    readouts: list[ReadoutConfig]


class PhaseOptimizationError(RuntimeError):
    """Raised when --stop-on-phase-error encounters an optimizer phase error."""

    def __init__(self, phase_result: dict):
        self.phase_result = dict(phase_result)
        super().__init__(_phase_error_message(self.phase_result))


def _jsonify(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def _write_json(path: Path, payload: dict) -> None:
    text = json.dumps(_jsonify(payload), indent=4)
    path.write_text(text + "\n", encoding="utf-8")


def _run_git(path: Path, *args: str) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        return None
    return completed.stdout.strip()


def _git_provenance(path: Path) -> dict:
    git_path = path if path.is_dir() else path.parent
    root_text = _run_git(git_path, "rev-parse", "--show-toplevel")
    if not root_text:
        return {"available": False, "path": str(path)}
    root = Path(root_text)
    status = _run_git(root, "status", "--porcelain") or ""
    return {
        "available": True,
        "path": str(root),
        "branch": _run_git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        "commit": _run_git(root, "rev-parse", "HEAD"),
        "describe": _run_git(root, "describe", "--tags", "--always", "--dirty"),
        "dirty": bool(status),
    }


def _distribution_version(name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _installed_distributions() -> list[dict[str, str]]:
    rows = []
    for dist in importlib_metadata.distributions():
        metadata = dist.metadata
        name = metadata.get("Name") or dist.name
        version = metadata.get("Version") or ""
        if name:
            rows.append({"name": str(name), "version": str(version)})
    return sorted(rows, key=lambda item: item["name"].lower())


def _collect_software_versions() -> dict:
    try:
        import pheat

        pheat_path = Path(pheat.__file__).resolve()
        pheat_version = getattr(pheat, "__version__", None)
        pheat_git = _git_provenance(pheat_path)
    except Exception as exc:
        pheat_path = None
        pheat_version = None
        pheat_git = {"available": False, "error": str(exc)}

    packages = {name: _distribution_version(name) for name in VERSION_PACKAGE_NAMES}
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
        },
        "qtf": {
            "runner_path": str(Path(__file__).resolve()),
            "module_path": str(Path(runner.__file__).resolve()),
            "git": _git_provenance(SCRIPT_DIR),
        },
        "pheat": {
            "version": pheat_version,
            "module_path": None if pheat_path is None else str(pheat_path),
            "git": pheat_git,
        },
        "packages": packages,
        "installed_distributions": _installed_distributions(),
    }


def _software_summary_payload(software_versions: dict, sidecar_path: Path) -> dict:
    payload = {
        "python": software_versions.get("python") or {},
        "platform": software_versions.get("platform") or {},
        "qtf": software_versions.get("qtf") or {},
        "pheat": software_versions.get("pheat") or {},
        "packages": software_versions.get("packages") or {},
        "installed_distribution_count": len(software_versions.get("installed_distributions") or []),
        "sidecar_path": str(sidecar_path),
    }
    return payload


def _environment_snapshot() -> dict:
    return {key: os.environ.get(key) for key in REPORT_ENVIRONMENT_KEYS if os.environ.get(key)}


def _command_line(argv: Optional[Sequence[str]], override: Optional[str]) -> str:
    if override and override.strip():
        return override.strip()
    if argv is None:
        command = [sys.executable, *sys.argv]
    else:
        command = [sys.executable, str(Path(__file__).resolve()), *argv]
    return shlex.join(str(part) for part in command)


def _score_payload(structure: HeavyAtomStructure, model: str) -> dict:
    result = score_structure(structure, model=model).to_dict()
    result["status"] = "ok"
    return result


def _safe_score_payload(structure: HeavyAtomStructure, model: str) -> dict:
    try:
        return _score_payload(structure, model)
    except Exception as exc:
        return {
            "model": model,
            "status": "unavailable",
            "error": str(exc),
            "total": None,
            "units": None,
            "terms": {},
            "warnings": [],
            "citations": [],
            "metadata": {},
        }


def _ca_coords_from_structure(structure: HeavyAtomStructure) -> np.ndarray:
    coords = [
        [atom.x, atom.y, atom.z]
        for atom in structure.atoms
        if atom.name.strip().upper() == "CA"
    ]
    return np.asarray(coords, dtype=float)


def _physics_metrics(ca_coords: np.ndarray) -> tuple[float, float]:
    if len(ca_coords) == 0:
        return 0.0, 0.0
    end_to_end = float(np.linalg.norm(ca_coords[0] - ca_coords[-1]))
    centroid = np.mean(ca_coords, axis=0)
    rg = float(np.sqrt(np.mean(np.sum((ca_coords - centroid) ** 2, axis=1))))
    return end_to_end, rg


def _sequence_from_residue_geometry(residue_geometry: ResidueGeometryStructure) -> str:
    return "".join(three_to_one(residue.name) for residue in residue_geometry.residues)


def _chi_name_from_pheat_spec(spec: object) -> Optional[str]:
    if isinstance(spec, tuple):
        if not spec:
            return None
        spec = spec[0]
    if not isinstance(spec, str) or not spec.startswith("chi"):
        return None
    suffix = spec[3:]
    if not suffix.isdigit():
        return None
    return f"chi{int(suffix)}"


def _chi_sort_key(chi_name: str) -> int:
    if not chi_name.startswith("chi") or not chi_name[3:].isdigit():
        raise ValueError(f"Invalid chi name '{chi_name}'.")
    return int(chi_name[3:])


def _pheat_available_chis(residue_token: str) -> list[str]:
    resname = one_to_three(residue_token)
    chis = {
        chi_name
        for step in SIDECHAIN_STEPS.get(resname, [])
        for chi_name in [_chi_name_from_pheat_spec(step.dihedral)]
        if chi_name is not None
    }
    return sorted(chis, key=_chi_sort_key)


def _load_pheat_reference(
    reference: str,
    *,
    angle_units: str,
    stored_angles,
    max_chi: Optional[int],
    include_terminal_oxt: bool,
) -> PheatReference:
    path = Path(reference)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(
            "--reference_structure must be an existing PDB, PHEAT heavy JSON, "
            f"or PHEAT residue-geometry JSON file: {reference}"
        )

    source_type = "pdb"
    if path.suffix.lower() == ".json":
        json_errors = []
        try:
            residue_geometry = load_residue_geometry(path)
            source_type = "pheat_residue_geometry_json"
        except Exception as exc:
            json_errors.append(f"residue geometry: {exc}")
            try:
                source_structure = load_heavy_json(path)
                source_type = "pheat_heavy_json"
            except Exception as heavy_exc:
                json_errors.append(f"heavy atom: {heavy_exc}")
                raise ValueError(
                    "Reference JSON is neither PHEAT residue-geometry JSON nor "
                    "PHEAT heavy-atom JSON: "
                    + "; ".join(json_errors)
                ) from heavy_exc
            residue_geometry = structure_to_residue_geometry(
                _canonical_protein_structure(source_structure, source_path=path),
                name=f"pheat-reference:{path.name}",
                angle_units=angle_units,
                stored_angles=stored_angles,
                max_chi=max_chi,
            )
    else:
        source_structure = load_pdb(path)
        residue_geometry = structure_to_residue_geometry(
            _canonical_protein_structure(source_structure, source_path=path),
            name=f"pheat-reference:{path.name}",
            angle_units=angle_units,
            stored_angles=stored_angles,
            max_chi=max_chi,
        )

    reference_structure = structure_from_residue_geometry(
        residue_geometry,
        include_terminal_oxt=include_terminal_oxt,
    )
    return PheatReference(
        source_path=path,
        source_type=source_type,
        residue_geometry=residue_geometry,
        structure=reference_structure,
        sequence=_sequence_from_residue_geometry(residue_geometry),
    )


def _canonical_protein_structure(
    structure: HeavyAtomStructure,
    *,
    source_path: Path,
) -> HeavyAtomStructure:
    peptide_atoms = [
        atom
        for atom in structure.atoms
        if atom.record_name.upper().startswith("ATOM")
        and atom.resname.strip().upper() in CANONICAL_RESIDUES
    ]
    if not peptide_atoms:
        raise ValueError(f"Reference has no canonical protein ATOM records: {source_path}")
    return HeavyAtomStructure(
        atoms=peptide_atoms,
        name=structure.name or source_path.name,
        metadata={**structure.metadata, "filtered_for_pheat_rmsd": True},
        disulfide_bonds=structure.disulfide_bonds,
    )


def _pheat_alignment_details(
    reference_structure: HeavyAtomStructure,
    target_structure: HeavyAtomStructure,
) -> dict:
    alignment = align_reconstructed_to_original(reference_structure, target_structure)
    return {
        "all_heavy_rmsd": float(alignment["all_heavy_rmsd"]),
        "backbone_rmsd": float(alignment["backbone_rmsd"]),
        "matched_heavy_atoms": int(alignment["matched_heavy_atoms"]),
        "matched_backbone_atoms": int(alignment["matched_backbone_atoms"]),
        "unmatched_reference_atoms": int(alignment["unmatched_original_atoms"]),
        "unmatched_target_atoms": int(alignment["unmatched_reconstructed_atoms"]),
        "reference_atom_count": len(reference_structure.atoms),
        "target_atom_count": len(target_structure.atoms),
    }


def _pheat_heavy_atom_rmsd(
    reference_structure: HeavyAtomStructure,
    target_structure: HeavyAtomStructure,
) -> tuple[float, int, int, int]:
    details = _pheat_alignment_details(reference_structure, target_structure)
    return (
        details["all_heavy_rmsd"],
        details["matched_heavy_atoms"],
        details["reference_atom_count"],
        details["target_atom_count"],
    )


def _safe_pheat_radius_of_gyration(structure: Optional[HeavyAtomStructure]) -> dict:
    if structure is None:
        return {
            "status": "unavailable",
            "error": "structure is not available",
            "values": {},
            "units": "angstrom",
        }
    try:
        payload = structure_radius_of_gyration(structure, mode="both")
        payload["status"] = "ok"
        return payload
    except Exception as exc:
        return {
            "status": "unavailable",
            "error": str(exc),
            "atom_count": len(structure.atoms),
            "mode": "both",
            "values": {},
            "units": "angstrom",
        }


def _safe_pheat_radius_of_gyration_delta(before: dict, after: dict) -> dict:
    if before.get("status") != "ok" or after.get("status") != "ok":
        return {
            "status": "unavailable",
            "error": "radius of gyration is unavailable for one or both structures",
            "values": {},
            "units": after.get("units") or before.get("units") or "angstrom",
        }
    try:
        payload = radius_of_gyration_delta(before, after)
        payload["status"] = "ok"
        return payload
    except Exception as exc:
        return {
            "status": "unavailable",
            "error": str(exc),
            "values": {},
            "units": after.get("units") or before.get("units") or "angstrom",
        }


def _aligned_pheat_structures(
    reference_structure: HeavyAtomStructure,
    folded_structure: HeavyAtomStructure,
) -> tuple[HeavyAtomStructure, HeavyAtomStructure, int]:
    alignment = align_reconstructed_to_original(reference_structure, folded_structure)
    return (
        alignment["original_aligned"],
        alignment["reconstructed_aligned"],
        int(alignment["matched_heavy_atoms"]),
    )


def _pdb_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _json_script_payload(payload) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).replace("</", "<\\/")


def _format_optional_float(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _format_optional_angstrom(value, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f} A"


def _rg_value(payload: Optional[dict], key: str) -> Optional[float]:
    if not payload:
        return None
    values = payload.get("values") or {}
    value = values.get(key)
    return None if value is None else float(value)


def _pheat_rg_table_rows(result: dict) -> str:
    rg = result.get("pheat_radius_of_gyration") or {}
    rows = []
    for label, key in (
        ("Reference", "reference"),
        ("Primary result", "final"),
        ("Primary - reference", "delta_final_minus_reference"),
    ):
        payload = rg.get(key) or {}
        rows.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{html.escape(_format_optional_float(_rg_value(payload, 'unweighted'), 6))}</td>"
            f"<td>{html.escape(_format_optional_float(_rg_value(payload, 'mass_weighted'), 6))}</td>"
            f"<td>{html.escape(str(payload.get('units') or 'angstrom'))}</td>"
            f"<td>{html.escape(str(payload.get('status') or 'n/a'))}</td>"
            "</tr>"
        )
    return "\n".join(rows) or '<tr><td colspan="5">No PHEAT radius of gyration metrics.</td></tr>'


def _pheat_source_manifest_by_id() -> dict:
    try:
        import pheat

        manifest_path = Path(pheat.__file__).resolve().parent / "data" / "sources" / "manifest.json"
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    sources = data.get("sources") if isinstance(data, dict) else []
    return {
        str(source.get("id")): source
        for source in sources
        if isinstance(source, dict) and source.get("id")
    }


def _citation_item_html(citation_id: str, manifest: dict) -> str:
    source = manifest.get(citation_id)
    if source:
        citation = str(source.get("citation") or citation_id)
        urls = [str(url) for url in source.get("urls") or [] if url]
        if urls:
            return (
                f'<li><code>{html.escape(citation_id)}</code>: '
                f'<a href="{html.escape(urls[0])}" rel="noopener">{html.escape(citation)}</a></li>'
            )
        return f"<li><code>{html.escape(citation_id)}</code>: {html.escape(citation)}</li>"
    if citation_id.startswith("pheat-"):
        return (
            f"<li><code>{html.escape(citation_id)}</code>: "
            "PHEAT internal scorer/provenance identifier for the configured local model.</li>"
        )
    return f"<li><code>{html.escape(citation_id)}</code></li>"


def _pheat_citation_report_section(result: dict, final_score: dict) -> str:
    software = result.get("software_versions") or {}
    pheat_info = software.get("pheat") or {}
    pheat_version = str(pheat_info.get("version") or "n/a")
    pheat_git = _git_label(pheat_info.get("git") or {})
    model = str(final_score.get("model") or result.get("result_score_model") or "n/a")
    citations = [str(item) for item in final_score.get("citations") or []]
    manifest = _pheat_source_manifest_by_id()
    citation_rows = "\n".join(_citation_item_html(item, manifest) for item in citations)
    if not citation_rows:
        citation_rows = "<li>No score-specific citations were reported by the selected PHEAT scorer.</li>"
    warnings = [str(item) for item in final_score.get("warnings") or []]
    warning_rows = "".join(f"<li>{html.escape(item)}</li>" for item in warnings)
    warning_block = ""
    if warning_rows:
        warning_block = f"""
    <p class="citation-subhead">Score model notes</p>
    <ul>{warning_rows}</ul>
"""
    return f"""
  <section class="citation-section" aria-labelledby="pheat-citation-title">
    <h2 id="pheat-citation-title">PHEAT Citation</h2>
    <p>
      This run uses <strong>PHEAT</strong>, the Protein Heavy-atom Energy and
      Analysis Toolkit, for residue-geometry handling, heavy-atom reconstruction,
      angle template support, alignment/RMSD metrics, radius-of-gyration metrics,
      and the configured PHEAT score where selected.
    </p>
    <table>
      <tbody>
        <tr><th>PHEAT repository</th><td><a href="{PHEAT_REPOSITORY_URL}" rel="noopener">{PHEAT_REPOSITORY_URL}</a></td></tr>
        <tr><th>PHEAT version</th><td>{html.escape(pheat_version)}; {html.escape(pheat_git)}</td></tr>
        <tr><th>PHEAT module path</th><td><code>{html.escape(str(pheat_info.get("module_path") or "n/a"))}</code></td></tr>
        <tr><th>Selected score model</th><td><code>{html.escape(model)}</code></td></tr>
      </tbody>
    </table>
    <p class="citation-subhead">Score model citations and provenance</p>
    <ul>{citation_rows}</ul>
    {warning_block}
  </section>
"""


def _viewer_pdb_text(path_text: Optional[str]) -> Optional[str]:
    if not path_text:
        return None
    try:
        path = Path(path_text)
        if not path.exists():
            return None
        return _pdb_text(path)
    except Exception:
        return None


def _viewer_structure_entries(
    result: dict,
    *,
    reference_available: bool,
    reference_aligned_pdb_path: Optional[Path],
    folded_aligned_pdb_path: Path,
) -> list[dict]:
    entries = []
    if reference_available and reference_aligned_pdb_path is not None:
        reference_text = _viewer_pdb_text(str(reference_aligned_pdb_path))
        if reference_text:
            entries.append(
                {
                    "key": "reference",
                    "role": "reference",
                    "label": "PHEAT geometry-encoded reference",
                    "short_label": "Reference",
                    "pdb": reference_text,
                    "pdb_file": reference_aligned_pdb_path.name,
                    "color": VIEWER_REFERENCE_COLOR,
                    "visible_default": True,
                }
            )

    snapshot_index = 0
    for snapshot in result.get("structure_snapshots") or []:
        if snapshot.get("snapshot_status") != "ok":
            continue
        path_text = (
            snapshot.get("viewer_pdb_path")
            or snapshot.get("aligned_pdb_path")
            or snapshot.get("pdb_path")
        )
        pdb_text = _viewer_pdb_text(path_text)
        if not pdb_text:
            continue
        role = str(snapshot.get("role") or "structure")
        key = str(snapshot.get("key") or _slug(snapshot.get("label") or role))
        phase_index = snapshot.get("phase_index")
        short_label = (
            f"Phase {phase_index}"
            if role == "phase" and phase_index is not None
            else str(snapshot.get("short_label") or snapshot.get("label") or key)
        )
        entries.append(
            {
                "key": key,
                "role": role,
                "label": _snapshot_label(snapshot),
                "short_label": short_label,
                "phase_index": phase_index,
                "phase_status": snapshot.get("phase_status"),
                "phase_status_label": snapshot.get("phase_status_label"),
                "angle_mode": snapshot.get("angle_mode"),
                "backend": snapshot.get("backend"),
                "shots": snapshot.get("shots"),
                "score_model": snapshot.get("score_model"),
                "atom_count": snapshot.get("atom_count"),
                "is_primary_result": bool(snapshot.get("is_primary_result")),
                "pdb": pdb_text,
                "pdb_file": Path(path_text).name,
                "color": _viewer_color_for_snapshot(snapshot, snapshot_index),
                "visible_default": bool(snapshot.get("visible_default")),
            }
        )
        snapshot_index += 1

    if not any(entry.get("is_primary_result") for entry in entries):
        folded_text = _viewer_pdb_text(str(folded_aligned_pdb_path))
        if folded_text:
            entries.append(
                {
                    "key": "primary_result",
                    "role": "primary",
                    "label": "Primary folded result",
                    "short_label": "Primary",
                    "is_primary_result": True,
                    "pdb": folded_text,
                    "pdb_file": folded_aligned_pdb_path.name,
                    "color": VIEWER_FINAL_COLOR,
                    "visible_default": True,
                }
            )

    if entries and not any(entry.get("visible_default") for entry in entries):
        primary_entry = next((entry for entry in entries if entry.get("is_primary_result")), entries[-1])
        primary_entry["visible_default"] = True
    return entries


def _viewer_toggle_buttons(entries: Sequence[dict]) -> str:
    buttons = []
    for entry in entries:
        pressed = "true" if entry.get("visible_default") else "false"
        key = html.escape(str(entry.get("key") or "structure"))
        label = html.escape(str(entry.get("label") or key))
        color = html.escape(str(entry.get("color") or "#888888"))
        status = str(entry.get("phase_status") or "")
        status_badge = ""
        status_class = ""
        if status in {"warning", "error"}:
            status_class = f" snapshot-status-{html.escape(status)}"
            status_badge = (
                f'<span class="legend-status-badge phase-status-{html.escape(status)}">'
                f'{html.escape(status)}</span>'
            )
        buttons.append(
            f"""
        <button type="button" class="legend-toggle{status_class}" data-structure-toggle="{key}" aria-pressed="{pressed}">
          <span class="legend-swatch" style="background: {color};" aria-hidden="true"></span>
          <span class="structure-toggle-label">{label}</span>
          {status_badge}
        </button>
"""
        )
    return "".join(buttons)


def _viewer_pdb_link_rows(entries: Sequence[dict]) -> str:
    links = []
    for entry in entries:
        pdb_file = entry.get("pdb_file")
        if not pdb_file:
            continue
        links.append(
            f'<a href="{html.escape(str(pdb_file))}">{html.escape(str(entry.get("short_label") or entry.get("label") or pdb_file))}</a>'
        )
    if not links:
        return """
          <dt>Structure PDBs</dt>
          <dd>n/a</dd>
"""
    return f"""
          <dt>Structure PDBs</dt>
          <dd>{' '.join(links)}</dd>
"""


def _format_seed(value) -> str:
    return "unseeded" if value is None else str(value)


def _rmsd_progress_label(current: Optional[float], previous: Optional[float]) -> str:
    if current is None or previous is None:
        return "n/a"
    delta = float(current) - float(previous)
    if delta < -RMSD_COMPARE_TOLERANCE:
        return "↓ better"
    if delta > RMSD_COMPARE_TOLERANCE:
        return "↑ worse"
    return "≈ same"


def _phase_status_category(success, status, message: str) -> str:
    if bool(success):
        return "ok"
    status_text = "" if status is None else str(status).strip()
    message_text = str(message or "").strip().lower()
    if status_text == "9" or "iteration limit" in message_text:
        return "warning"
    if "maximum number of function evaluations" in message_text:
        return "warning"
    if "maxiter" in message_text or "maxfun" in message_text:
        return "warning"
    return "error"


def _phase_status_label_from_values(success, status, message: str) -> str:
    category = _phase_status_category(success, status, message)
    message_text = str(message or "").strip()
    if category == "ok":
        return "ok" if not message_text else f"ok: {message_text}"
    if category == "warning":
        if message_text:
            return f"warning: {message_text}"
        return f"warning: status {status}" if status is not None else "warning"
    if message_text:
        return f"error: {message_text}"
    return f"error: status {status}" if status is not None else "error"


def _phase_error_message(phase_result: dict) -> str:
    label = phase_result.get("label") or phase_result.get("name") or "phase"
    status = phase_result.get("status")
    message = str(phase_result.get("message") or "").strip()
    detail = f"status {status}" if status is not None else "optimizer returned an error"
    if message:
        detail += f": {message}"
    return f"Stopping after phase '{label}' because it was classified as an error ({detail})."


def _slug(text: str) -> str:
    chars = []
    for char in str(text).lower():
        if char.isalnum():
            chars.append(char)
        elif chars and chars[-1] != "_":
            chars.append("_")
    slug = "".join(chars).strip("_")
    return slug or "phase"


def _phase_timing_key(index: int, name: str) -> str:
    return f"phase_{index:02d}_{_slug(name)}"


def _snapshot_payload(snapshot: dict) -> dict:
    return {key: value for key, value in snapshot.items() if key != "structure"}


def _snapshot_file_stem(prefix: str, snapshot: dict) -> str:
    key = _slug(snapshot.get("key") or snapshot.get("label") or snapshot.get("role") or "structure")
    return f"{prefix}_{key}"


def _snapshot_label(snapshot: dict) -> str:
    label = snapshot.get("label") or snapshot.get("key") or "Structure"
    return str(label)


def _viewer_color_for_snapshot(snapshot: dict, index: int) -> str:
    role = str(snapshot.get("role") or "").lower()
    if role in {"final", "primary"} or snapshot.get("is_primary_result"):
        return VIEWER_FINAL_COLOR
    color_index = max(0, index) % len(VIEWER_SNAPSHOT_COLORS)
    return VIEWER_SNAPSHOT_COLORS[color_index]


def _phase_schedule_payload(schedule: PhaseSchedule) -> dict:
    return {
        "preset": schedule.preset,
        "source": schedule.source,
        "config_path": schedule.config_path,
        "basis_circuit_batching": schedule.basis_circuit_batching,
        "scouting": asdict(schedule.scouting),
        "phases": [asdict(phase) for phase in schedule.phases],
        "result": asdict(schedule.result),
        "readouts": [asdict(readout) for readout in schedule.readouts],
    }


def _load_yaml_mapping(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "YAML phase presets require PyYAML. Install it in the active environment "
            "with `python -m pip install PyYAML`."
        ) from exc

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Could not parse YAML phase config {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML phase config {path} must contain a mapping at the top level.")
    return data


def _validate_phase_preset_yaml(path: Path, data: dict) -> None:
    try:
        import jsonschema
    except ImportError as exc:
        raise RuntimeError(
            "PHEAT phase workflow YAML validation requires jsonschema. Install it "
            "in the active environment with `python -m pip install jsonschema`."
        ) from exc
    if not PHASE_PRESET_SCHEMA_PATH.exists():
        raise FileNotFoundError(f"PHEAT phase workflow schema is missing: {PHASE_PRESET_SCHEMA_PATH}")
    try:
        schema = json.loads(PHASE_PRESET_SCHEMA_PATH.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.validate(instance=data, schema=schema)
    except jsonschema.ValidationError as exc:
        location = ".".join(str(part) for part in exc.absolute_path) or "<root>"
        raise ValueError(f"YAML phase config {path} failed schema validation at {location}: {exc.message}") from exc
    except Exception as exc:
        raise ValueError(f"Could not validate YAML phase config {path}: {exc}") from exc


def _load_phase_presets_from_path(path: Path) -> dict[str, dict]:
    data = _load_yaml_mapping(path)
    _validate_phase_preset_yaml(path, data)
    presets = data.get("presets")
    if not isinstance(presets, dict):
        raise ValueError(f"YAML phase config {path} must define a top-level `presets` mapping.")
    return {str(name): copy.deepcopy(config) for name, config in presets.items()}


def _load_builtin_phase_presets() -> dict[str, dict]:
    if not PHASE_PRESET_DIR.exists():
        raise FileNotFoundError(f"Built-in PHEAT phase preset directory is missing: {PHASE_PRESET_DIR}")
    paths = sorted(PHASE_PRESET_DIR.glob("*.yaml")) + sorted(PHASE_PRESET_DIR.glob("*.yml"))
    if not paths:
        raise FileNotFoundError(f"No built-in PHEAT phase preset YAML files found in {PHASE_PRESET_DIR}")

    presets: dict[str, dict] = {}
    for path in paths:
        presets.update(_load_phase_presets_from_path(path))
    return presets


def _parse_cli_scalar(value: str):
    text = str(value).strip()
    lowered = text.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _parse_assignment(parser: argparse.ArgumentParser, raw: str, flag: str) -> tuple[str, str]:
    if "=" not in raw:
        parser.error(f"{flag} entries must use NAME=value syntax: {raw!r}")
    name, value = raw.split("=", 1)
    name = name.strip()
    if not name:
        parser.error(f"{flag} phase name must not be blank.")
    if not value.strip():
        parser.error(f"{flag} value must not be blank for phase {name!r}.")
    return name, value.strip()


def _parse_phase_option(parser: argparse.ArgumentParser, raw: str) -> tuple[str, str, Any]:
    if ":" not in raw:
        parser.error(f"--phase-option entries must use NAME:key=value syntax: {raw!r}")
    name, remainder = raw.split(":", 1)
    name = name.strip()
    if not name:
        parser.error("--phase-option phase name must not be blank.")
    key, value = _parse_assignment(parser, remainder, "--phase-option")
    return name, key, _parse_cli_scalar(value)


def _set_phase_override(
    parser: argparse.ArgumentParser,
    phases_by_name: dict[str, dict],
    raw: str,
    flag: str,
    key: str,
    *,
    cast=None,
    item_label: str = "phase",
    define_hint: Optional[str] = None,
) -> None:
    name, value = _parse_assignment(parser, raw, flag)
    if name not in phases_by_name:
        hint = define_hint or "Define it with --phase or in the selected preset."
        parser.error(f"{flag} references unknown {item_label} {name!r}. {hint}")
    try:
        phases_by_name[name][key] = cast(value) if cast is not None else value
    except Exception as exc:
        parser.error(f"{flag} has an invalid value for {item_label} {name!r}: {exc}")


def _positive_int(parser: argparse.ArgumentParser, value, context: str) -> int:
    try:
        resolved = int(value)
    except Exception as exc:
        parser.error(f"{context} must be an integer.")
        raise AssertionError from exc
    if resolved <= 0:
        parser.error(f"{context} must be > 0.")
    return resolved


def _optional_positive_int(parser: argparse.ArgumentParser, value, context: str) -> Optional[int]:
    if value is None:
        return None
    return _positive_int(parser, value, context)


def _optional_float(parser: argparse.ArgumentParser, value, context: str) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception as exc:
        parser.error(f"{context} must be a float.")
        raise AssertionError from exc


def _phase_options(parser: argparse.ArgumentParser, value, context: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        parser.error(f"{context} options must be a mapping.")
    if "maxiter" in value:
        parser.error(f"{context} options must not contain maxiter; use top-level maxiter.")
    return dict(value)


def _normalize_phase_optimizer(parser: argparse.ArgumentParser, value, context: str) -> str:
    if value is None or not str(value).strip():
        parser.error(f"{context} optimizer is required.")
    requested = str(value).strip()
    by_lower = {name.lower(): name for name in SUPPORTED_PHASE_OPTIMIZERS}
    normalized = by_lower.get(requested.lower())
    if normalized is None:
        parser.error(
            f"{context} optimizer '{requested}' is not supported. "
            f"Supported optimizers: {', '.join(sorted(SUPPORTED_PHASE_OPTIMIZERS))}"
        )
    return normalized


def _normalize_score_model(parser: argparse.ArgumentParser, value, context: str) -> str:
    if value is None or not str(value).strip():
        parser.error(f"{context} score_model is required.")
    model = str(value).strip().lower()
    if model not in available_models():
        parser.error(
            f"{context} score_model '{model}' is not available. "
            f"Available models: {', '.join(available_models())}"
        )
    return model


def _normalize_basis_circuit_batching(parser: argparse.ArgumentParser, value, context: str) -> str:
    mode = "auto" if value is None else str(value).strip().lower()
    if mode not in BASIS_CIRCUIT_BATCHING_MODES:
        parser.error(
            f"{context} must be one of {', '.join(BASIS_CIRCUIT_BATCHING_MODES)}."
        )
    return mode


def _normalize_backend_spec(value, context: str = "backend") -> str:
    if value is None:
        return "inherit"
    spec = str(value).strip()
    if not spec:
        raise ValueError(f"{context} must not be blank.")
    return spec


def _backend_spec_from_cli(parser: argparse.ArgumentParser, value, context: str) -> str:
    try:
        return _normalize_backend_spec(value, context)
    except ValueError as exc:
        parser.error(str(exc))
        raise AssertionError from exc


def _effective_backend_spec(spec: Optional[str], inherited_spec: str) -> str:
    normalized = "inherit" if spec is None else str(spec).strip()
    if normalized.lower() in INHERIT_BACKEND_VALUES:
        return str(inherited_spec).strip()
    return normalized


def _resolve_phase_schedule(
    args,
    parser: argparse.ArgumentParser,
    *,
    global_shots: int,
    global_maxiter: int,
) -> PhaseSchedule:
    try:
        presets = _load_builtin_phase_presets()
        config_path = None
        if args.phase_config:
            user_path = Path(args.phase_config)
            config_path = str(user_path)
            presets.update(_load_phase_presets_from_path(user_path))
    except Exception as exc:
        parser.error(str(exc))

    if args.phase_preset not in presets and not args.phase:
        parser.error(
            f"--phase-preset {args.phase_preset!r} was not found. "
            f"Available presets: {', '.join(sorted(presets)) or 'none'}"
        )

    raw = copy.deepcopy(presets.get(args.phase_preset, {}))
    scouting_raw = copy.deepcopy(raw.get("scouting") or {})
    result_raw = copy.deepcopy(raw.get("result") or {})
    phases_raw = copy.deepcopy(raw.get("phases") or [])
    readouts_raw = copy.deepcopy(raw.get("readouts") or [])
    basis_circuit_batching_raw = raw.get("basis_circuit_batching", "auto")

    if args.phase:
        phases_raw = [{"name": name} for name in args.phase]

    if not isinstance(phases_raw, list):
        parser.error("Selected phase preset must define phases as a list.")
    phase_names = []
    phases_by_name = {}
    for index, phase in enumerate(phases_raw, start=1):
        if not isinstance(phase, dict):
            parser.error(f"Phase entry #{index} must be a mapping.")
        name = str(phase.get("name") or "").strip()
        if not name:
            parser.error(f"Phase entry #{index} must define a non-blank name.")
        if name in phases_by_name:
            parser.error(f"Duplicate phase name in selected schedule: {name!r}")
        phase_names.append(name)
        phases_by_name[name] = phase

    for raw_value in args.phase_label or []:
        _set_phase_override(parser, phases_by_name, raw_value, "--phase-label", "label")
    for raw_value in args.phase_optimizer or []:
        _set_phase_override(parser, phases_by_name, raw_value, "--phase-optimizer", "optimizer")
    for raw_value in args.phase_score or []:
        _set_phase_override(parser, phases_by_name, raw_value, "--phase-score", "score_model")
    for raw_value in args.phase_optimizer_backend or []:
        _set_phase_override(
            parser,
            phases_by_name,
            raw_value,
            "--phase-optimizer-backend",
            "optimizer_backend",
        )
    for raw_value in args.phase_readout_backend or []:
        _set_phase_override(
            parser,
            phases_by_name,
            raw_value,
            "--phase-readout-backend",
            "readout_backend",
        )
    for raw_value in args.phase_shots or []:
        _set_phase_override(parser, phases_by_name, raw_value, "--phase-shots", "shots", cast=int)
    for raw_value in args.phase_optimizer_shots or []:
        _set_phase_override(
            parser,
            phases_by_name,
            raw_value,
            "--phase-optimizer-shots",
            "optimizer_shots",
            cast=int,
        )
    for raw_value in args.phase_readout_shots or []:
        _set_phase_override(parser, phases_by_name, raw_value, "--phase-readout-shots", "readout_shots", cast=int)
    for raw_value in args.phase_maxiter or []:
        _set_phase_override(parser, phases_by_name, raw_value, "--phase-maxiter", "maxiter", cast=int)
    for raw_value in args.phase_tol or []:
        _set_phase_override(parser, phases_by_name, raw_value, "--phase-tol", "tol", cast=float)
    for raw_value in args.phase_option or []:
        name, key, value = _parse_phase_option(parser, raw_value)
        if name not in phases_by_name:
            parser.error(
                f"--phase-option references unknown phase {name!r}. "
                "Define it with --phase or in the selected preset."
            )
        phases_by_name[name].setdefault("options", {})[key] = value

    if args.scouting_score is not None:
        scouting_raw["score_model"] = args.scouting_score
    if args.scouting_backend is not None:
        scouting_raw["backend"] = args.scouting_backend
    if args.scouting_shots is not None:
        scouting_raw["shots"] = args.scouting_shots
    if args.scouting_attempts is not None:
        scouting_raw["attempts"] = args.scouting_attempts
    if args.result_score is not None:
        result_raw["score_model"] = args.result_score
    elif result_raw.get("score_model") is None:
        result_raw["score_model"] = DEFAULT_RESULT_SCORE_MODEL
    if args.primary_result is not None:
        result_raw["primary"] = args.primary_result
    if args.basis_circuit_batching is not None:
        basis_circuit_batching_raw = args.basis_circuit_batching

    if not isinstance(readouts_raw, list):
        parser.error("Selected phase preset must define readouts as a list when present.")
    readouts_by_name = {}
    readout_names = []
    for index, readout in enumerate(readouts_raw, start=1):
        if not isinstance(readout, dict):
            parser.error(f"Readout entry #{index} must be a mapping.")
        name = str(readout.get("name") or "").strip()
        if not name:
            parser.error(f"Readout entry #{index} must define a non-blank name.")
        if name == PRIMARY_LAST_PHASE:
            parser.error(f"Readout name {PRIMARY_LAST_PHASE!r} is reserved.")
        if name in readouts_by_name:
            parser.error(f"Duplicate readout name in selected schedule: {name!r}")
        readouts_by_name[name] = readout
        readout_names.append(name)
    for name in args.readout or []:
        readout_name = str(name).strip()
        if not readout_name:
            parser.error("--readout name must not be blank.")
        if readout_name == PRIMARY_LAST_PHASE:
            parser.error(f"--readout name {PRIMARY_LAST_PHASE!r} is reserved.")
        if readout_name not in readouts_by_name:
            readouts_by_name[readout_name] = {"name": readout_name}
            readout_names.append(readout_name)
    for raw_value in args.readout_backend or []:
        _set_phase_override(
            parser,
            readouts_by_name,
            raw_value,
            "--readout-backend",
            "backend",
            item_label="readout",
            define_hint="Define it with --readout or in the selected preset.",
        )
    for raw_value in args.readout_shots or []:
        _set_phase_override(
            parser,
            readouts_by_name,
            raw_value,
            "--readout-shots",
            "shots",
            cast=int,
            item_label="readout",
            define_hint="Define it with --readout or in the selected preset.",
        )
    for raw_value in args.readout_score or []:
        _set_phase_override(
            parser,
            readouts_by_name,
            raw_value,
            "--readout-score",
            "score_model",
            item_label="readout",
            define_hint="Define it with --readout or in the selected preset.",
        )

    if not phase_names:
        parser.error("The resolved phase schedule must contain at least one phase.")

    scouting_score = _normalize_score_model(
        parser,
        scouting_raw.get("score_model", "generic"),
        "scouting",
    )
    scouting_backend = _backend_spec_from_cli(parser, scouting_raw.get("backend", "inherit"), "scouting backend")
    scouting_shots = _positive_int(parser, scouting_raw.get("shots") or global_shots, "scouting shots")
    scouting_attempts = _positive_int(
        parser,
        scouting_raw.get("attempts", 50),
        "scouting attempts",
    )
    scouting = ScoutingConfig(
        score_model=scouting_score,
        backend=scouting_backend,
        shots=scouting_shots,
        attempts=scouting_attempts,
    )

    result_score = _normalize_score_model(
        parser,
        result_raw.get("score_model") or DEFAULT_RESULT_SCORE_MODEL,
        "result",
    )
    primary_result = str(result_raw.get("primary") or PRIMARY_LAST_PHASE).strip()
    if not primary_result:
        parser.error("result primary must not be blank.")

    phases = []
    for index, name in enumerate(phase_names, start=1):
        phase_raw = phases_by_name[name]
        phase_context = f"phase {name!r}"
        score_model = _normalize_score_model(parser, phase_raw.get("score_model"), phase_context)
        optimizer = _normalize_phase_optimizer(parser, phase_raw.get("optimizer"), phase_context)
        optimizer_backend = _backend_spec_from_cli(
            parser,
            phase_raw.get("optimizer_backend", "inherit"),
            f"{phase_context} optimizer_backend",
        )
        readout_backend = _backend_spec_from_cli(
            parser,
            phase_raw.get("readout_backend", "inherit"),
            f"{phase_context} readout_backend",
        )
        phase_shots = _positive_int(
            parser,
            phase_raw.get("shots") or global_shots,
            f"{phase_context} shots",
        )
        optimizer_shots = _positive_int(
            parser,
            phase_raw.get("optimizer_shots") or phase_shots,
            f"{phase_context} optimizer_shots",
        )
        readout_shots = _positive_int(
            parser,
            phase_raw.get("readout_shots") or phase_shots,
            f"{phase_context} readout_shots",
        )
        maxiter = _positive_int(
            parser,
            phase_raw.get("maxiter") or global_maxiter,
            f"{phase_context} maxiter",
        )
        phases.append(
            PhaseConfig(
                name=name,
                label=str(phase_raw.get("label") or name),
                optimizer=optimizer,
                score_model=score_model,
                optimizer_backend=optimizer_backend,
                readout_backend=readout_backend,
                shots=phase_shots,
                optimizer_shots=optimizer_shots,
                readout_shots=readout_shots,
                maxiter=maxiter,
                tol=_optional_float(parser, phase_raw.get("tol"), f"{phase_context} tol"),
                options=_phase_options(parser, phase_raw.get("options"), phase_context),
            )
        )

    readouts = []
    primary_flags = []
    for name in readout_names:
        readout_raw = readouts_by_name[name]
        readout_context = f"readout {name!r}"
        source = str(readout_raw.get("source") or "optimized_params").strip()
        if source != "optimized_params":
            parser.error(f"{readout_context} source must be 'optimized_params'.")
        readout_backend = _backend_spec_from_cli(
            parser,
            readout_raw.get("backend", "inherit"),
            f"{readout_context} backend",
        )
        readout_shots = _positive_int(
            parser,
            readout_raw.get("shots") or global_shots,
            f"{readout_context} shots",
        )
        readout_score = _normalize_score_model(
            parser,
            readout_raw.get("score_model") or result_score,
            readout_context,
        )
        primary = bool(readout_raw.get("primary", False))
        if primary:
            primary_flags.append(name)
        readouts.append(
            ReadoutConfig(
                name=name,
                source=source,
                backend=readout_backend,
                shots=readout_shots,
                score_model=readout_score,
                primary=primary,
            )
        )
    if primary_result == PRIMARY_LAST_PHASE and len(primary_flags) == 1 and args.primary_result is None:
        primary_result = primary_flags[0]
    if primary_result != PRIMARY_LAST_PHASE and primary_result not in {readout.name for readout in readouts}:
        parser.error(
            "--primary-result must be 'last_phase_structure' or the name of a configured readout. "
            f"Configured readouts: {', '.join(readout.name for readout in readouts) or 'none'}"
        )
    return PhaseSchedule(
        preset=args.phase_preset,
        source="cli" if args.phase else "yaml",
        config_path=config_path,
        basis_circuit_batching=_normalize_basis_circuit_batching(
            parser,
            basis_circuit_batching_raw,
            "basis_circuit_batching",
        ),
        scouting=scouting,
        phases=phases,
        result=ResultConfig(primary=primary_result, score_model=result_score),
        readouts=readouts,
    )


def _format_timing_metadata_value(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value) or "none"
    if isinstance(value, dict):
        return ", ".join(f"{key}={item}" for key, item in value.items()) or "none"
    return str(value)


def _timing_table_rows(timings: dict) -> str:
    if not timings:
        return '<tr><td colspan="4">No timings recorded.</td></tr>'
    rows = []
    ordered_keys = []
    for key in TIMING_SECTION_ORDER:
        if key == "__optimizer_phases__":
            ordered_keys.extend(
                phase_key
                for phase_key in timings
                if phase_key.startswith("phase_") and phase_key not in ordered_keys
            )
            continue
        if key in timings:
            ordered_keys.append(key)
    ordered_keys.extend(key for key in timings if key not in set(ordered_keys))
    for key in ordered_keys:
        record = timings.get(key) or {}
        metadata = record.get("metadata") or {}
        label = TIMING_SECTION_LABELS.get(key)
        if label is None and key.startswith("phase_"):
            phase_index = metadata.get("phase_index")
            phase_label = metadata.get("phase_label") or metadata.get("phase") or key
            label = f"Phase {phase_index}: {phase_label}" if phase_index else f"Phase: {phase_label}"
        if label is None:
            label = key
        rows.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{html.escape(_format_elapsed(record.get('elapsed_s')))}</td>"
            f"<td>{html.escape(str(record.get('status') or 'n/a'))}</td>"
            f"<td>{html.escape(_format_timing_metadata_value(metadata))}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _validate_shots(parser: argparse.ArgumentParser, value: Optional[int], flag: str) -> Optional[int]:
    if value is None:
        return None
    shots = int(value)
    if shots <= 0:
        parser.error(f"{flag} must be > 0.")
    return shots


def _resolve_shot_settings(args, parser: argparse.ArgumentParser) -> int:
    shots = _validate_shots(parser, args.shots, "--shots")
    if shots is None:
        parser.error("--shots must be provided.")
    return shots


def _copy_molstar_assets(outdir: Path) -> tuple[Optional[Path], Optional[str]]:
    try:
        import pheat

        pheat_root = Path(pheat.__file__).resolve().parents[2]
        source_dir = pheat_root / "examples" / "vendor" / "molstar"
        missing = [
            filename
            for filename in ("molstar.js", "molstar.css", "LICENSE")
            if not (source_dir / filename).exists()
        ]
        if missing:
            return None, f"Missing Mol* assets in {source_dir}: {', '.join(missing)}"

        target_dir = outdir / "vendor" / "molstar"
        target_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("molstar.js", "molstar.css", "LICENSE"):
            shutil.copy2(source_dir / filename, target_dir / filename)
        return target_dir, None
    except Exception as exc:
        return None, str(exc)


def _format_optional_int(value) -> str:
    if value is None:
        return "n/a"
    return str(int(value))


def _backend_display_name(backend) -> str:
    name = getattr(backend, "name", None)
    if callable(name):
        name = name()
    return str(name if name is not None else backend)


def _backend_attr(backend, name: str):
    try:
        value = getattr(backend, name, None)
        return value() if callable(value) else value
    except Exception:
        return None


def _backend_configuration(backend):
    try:
        return backend.configuration()
    except Exception:
        return None


def _backend_processor_metadata(backend) -> dict:
    config = _backend_configuration(backend)
    processor = _backend_attr(backend, "processor_type")
    if processor is None and config is not None:
        processor = getattr(config, "processor_type", None)

    processor_type = None
    processor_revision = None
    processor_segment = None
    if isinstance(processor, dict):
        processor_type = processor.get("family") or processor.get("type") or processor.get("name")
        processor_revision = processor.get("revision")
        processor_segment = processor.get("segment")
    elif processor is not None:
        processor_type = str(processor)

    backend_version = _backend_attr(backend, "backend_version")
    if backend_version is None and config is not None:
        backend_version = getattr(config, "backend_version", None)
    if backend_version is None:
        backend_version = _backend_attr(backend, "version")

    is_simulator = bool(_backend_attr(backend, "simulator"))
    if not is_simulator and config is not None:
        is_simulator = bool(getattr(config, "simulator", False))
    if processor_type is None and is_simulator:
        processor_type = "simulator"
    if processor_revision is None and is_simulator and backend_version is not None:
        processor_revision = backend_version

    num_qubits = _backend_attr(backend, "num_qubits")
    if num_qubits is None and config is not None:
        num_qubits = getattr(config, "n_qubits", None)

    return {
        "processor_type": None if processor_type is None else str(processor_type),
        "processor_revision": None if processor_revision is None else str(processor_revision),
        "processor_segment": None if processor_segment is None else str(processor_segment),
        "backend_version": None if backend_version is None else str(backend_version),
        "backend_num_qubits": None if num_qubits is None else int(num_qubits),
    }


def _backend_lookup_error(context: str, exc: Exception) -> ValueError:
    error = ValueError(f"Backend lookup failed for {context}: {exc}")
    error.__cause__ = exc
    return error


def _least_busy_context(min_num_qubits: int) -> str:
    return (
        "'least_busy' with criteria "
        f"simulator=False, operational=True, min_num_qubits={int(min_num_qubits)}"
    )


def _ibm_runtime_service(context: str, ibm_instance_crn: Optional[str] = None):
    from qiskit_ibm_runtime import QiskitRuntimeService

    try:
        if ibm_instance_crn:
            return QiskitRuntimeService(channel="ibm_cloud", instance=ibm_instance_crn)
        return QiskitRuntimeService()
    except Exception as exc:
        raise _backend_lookup_error(context, exc)


def _least_busy_backend(service, min_num_qubits: int):
    min_qubits = max(1, int(min_num_qubits))
    criteria = _least_busy_context(min_qubits)
    try:
        candidates = service.backends(
            simulator=False,
            operational=True,
            min_num_qubits=min_qubits,
        )
    except Exception as exc:
        raise _backend_lookup_error(criteria, exc)
    if not candidates:
        raise ValueError(f"Backend lookup failed for {criteria}: no matching backends returned.")
    return sorted(candidates, key=lambda b: b.status().pending_jobs)[0]


def _parse_gate_estimate_backend_spec(estimate_arg, selected_backend: str) -> tuple[Optional[str], str]:
    if estimate_arg is None:
        return None, "not_requested"
    selected_key = str(selected_backend).strip().lower()
    if estimate_arg == GATE_ESTIMATE_SELECTED_BACKEND:
        if selected_key == "none":
            raise ValueError("--estimate-gates cannot infer a backend when --hw_backend none.")
        if selected_key == StatevectorShotsBackend.name:
            return "aer", "implicit_aer_for_statevector_shots"
        return str(selected_backend), "selected_backend"

    entries = [entry.strip() for entry in str(estimate_arg).split(",")]
    if not entries or any(not entry for entry in entries):
        raise ValueError("--estimate-gates requires one or more non-empty comma-separated backend names.")
    invalid = [
        entry
        for entry in entries
        if entry.lower() in {"none", StatevectorShotsBackend.name}
    ]
    if invalid:
        raise ValueError(
            "--estimate-gates targets must be transpilation backends; invalid entries: "
            + ", ".join(invalid)
        )
    return ",".join(entries), "explicit"


def _verify_gate_estimate_backend_access(
    backend_spec: Optional[str],
    ibm_instance_crn: Optional[str] = None,
) -> None:
    if backend_spec is None:
        return
    ibm_service = None
    for raw_name in backend_spec.split(","):
        name = raw_name.strip()
        key = name.lower()
        if key in {"aer", "aer_simulator"}:
            continue
        if key == "least_busy":
            criteria = _least_busy_context(1)
            if ibm_service is None:
                ibm_service = _ibm_runtime_service(criteria, ibm_instance_crn)
            _least_busy_backend(ibm_service, 1)
        else:
            context = f"'{name}'"
            if ibm_service is None:
                ibm_service = _ibm_runtime_service(context, ibm_instance_crn)
            try:
                ibm_service.backend(name)
            except Exception as exc:
                raise _backend_lookup_error(context, exc)


def _resolve_gate_estimate_backends(
    backend_spec: Optional[str],
    min_num_qubits: int,
    ibm_instance_crn: Optional[str] = None,
):
    if backend_spec is None:
        return []
    backends = []
    ibm_service = None
    for raw_name in backend_spec.split(","):
        name = raw_name.strip()
        key = name.lower()
        if key in {"aer", "aer_simulator"}:
            from qiskit_aer import AerSimulator

            backends.append({"requested": name, "backend": AerSimulator(), "source": "aer"})
            continue
        if key == "least_busy":
            min_qubits = max(1, int(min_num_qubits))
            criteria = _least_busy_context(min_qubits)
            if ibm_service is None:
                ibm_service = _ibm_runtime_service(criteria, ibm_instance_crn)
            backend = _least_busy_backend(ibm_service, min_qubits)
            backends.append({"requested": name, "backend": backend, "source": "least_busy"})
        else:
            context = f"'{name}'"
            if ibm_service is None:
                ibm_service = _ibm_runtime_service(context, ibm_instance_crn)
            try:
                backend = ibm_service.backend(name)
            except Exception as exc:
                raise _backend_lookup_error(context, exc)
            backends.append({"requested": name, "backend": backend, "source": "ibm"})
    return backends


def _is_quantum_gate(operation) -> bool:
    return str(getattr(operation, "name", "")).lower() not in NON_GATE_OPERATIONS


def _instruction_parts(instruction):
    operation = getattr(instruction, "operation", None)
    qubits = getattr(instruction, "qubits", None)
    if operation is None:
        operation = instruction[0]
    if qubits is None:
        qubits = instruction[1]
    return operation, qubits


def _filtered_quantum_gate_circuit(qc, predicate):
    filtered = runner.QuantumCircuit(qc.num_qubits)
    for instruction in qc.data:
        operation, qubits = _instruction_parts(instruction)
        if not _is_quantum_gate(operation) or not predicate(operation):
            continue
        qargs = [qc.find_bit(qubit).index for qubit in qubits]
        filtered.append(operation, qargs)
    return filtered


def _gate_op_breakdown(qc) -> dict[str, int]:
    counts: dict[str, int] = {}
    for instruction in qc.data:
        operation, _qubits = _instruction_parts(instruction)
        if not _is_quantum_gate(operation):
            continue
        name = str(getattr(operation, "name", operation))
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def _excluded_op_breakdown(qc) -> dict[str, int]:
    counts: dict[str, int] = {}
    for instruction in qc.data:
        operation, _qubits = _instruction_parts(instruction)
        if _is_quantum_gate(operation):
            continue
        name = str(getattr(operation, "name", operation))
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def _multi_qubit_gate_count(qc) -> int:
    count = 0
    for instruction in qc.data:
        operation, _qubits = _instruction_parts(instruction)
        if _is_quantum_gate(operation) and int(getattr(operation, "num_qubits", 0)) >= 2:
            count += 1
    return count


def _build_gate_estimate_circuits(ansatz) -> dict[str, object]:
    base = runner.QuantumCircuit(ansatz.num_qubits)
    base.append(ansatz, range(ansatz.num_qubits))
    base = base.decompose()

    circuits = {"brickwork": base.copy()}
    z_circuit = base.copy()
    z_circuit.measure_all()
    circuits["measurement_z"] = z_circuit

    x_circuit = base.copy()
    for q in range(x_circuit.num_qubits):
        x_circuit.h(q)
    x_circuit.measure_all()
    circuits["measurement_x"] = x_circuit

    y_circuit = base.copy()
    for q in range(y_circuit.num_qubits):
        y_circuit.sdg(q)
        y_circuit.h(q)
    y_circuit.measure_all()
    circuits["measurement_y"] = y_circuit
    return circuits


def _circuit_gate_metrics(qc) -> dict:
    gate_only = _filtered_quantum_gate_circuit(qc, lambda operation: True)
    multi_qubit_only = _filtered_quantum_gate_circuit(
        qc,
        lambda operation: int(getattr(operation, "num_qubits", 0)) >= 2,
    )
    op_breakdown = _gate_op_breakdown(qc)
    return {
        "all_gate_count": int(sum(op_breakdown.values())),
        "all_gate_depth": int(gate_only.depth()),
        "multi_qubit_gate_count": int(_multi_qubit_gate_count(qc)),
        "multi_qubit_gate_depth": int(multi_qubit_only.depth()),
        "op_breakdown": op_breakdown,
        "excluded_op_breakdown": _excluded_op_breakdown(qc),
    }


def _estimate_gate_costs(folder: "PheatQuantumBiophysicsFolder", backend_refs) -> list[dict]:
    if not runner.QISKIT_AVAILABLE or folder.ansatz is None:
        raise RuntimeError("Qiskit is required for --estimate-gates.")

    estimates = []
    circuits = _build_gate_estimate_circuits(folder.ansatz)
    for backend_ref in backend_refs:
        backend = backend_ref["backend"]
        backend_name = _backend_display_name(backend)
        backend_metadata = _backend_processor_metadata(backend)
        for circuit_label, circuit in circuits.items():
            for optimization_level in GATE_ESTIMATE_OPTIMIZATION_LEVELS:
                kwargs = {"backend": backend, "optimization_level": optimization_level}
                seed_transpiler = None
                if optimization_level > 0:
                    kwargs["seed_transpiler"] = GATE_ESTIMATE_SEED_TRANSPILER
                    seed_transpiler = GATE_ESTIMATE_SEED_TRANSPILER
                t0 = time.time()
                transpiled = runner.transpile(circuit, **kwargs)
                transpile_time_s = time.time() - t0
                estimates.append(
                    {
                        "backend": backend_name,
                        "requested_backend": backend_ref["requested"],
                        "backend_source": backend_ref["source"],
                        **backend_metadata,
                        "circuit": circuit_label,
                        "optimization_level": optimization_level,
                        "seed_transpiler": seed_transpiler,
                        "transpile_time_s": round(transpile_time_s, 3),
                        **_circuit_gate_metrics(transpiled),
                    }
                )
    return estimates


def _energy_summary(values: Sequence[float]) -> dict:
    numeric = [float(value) for value in values]
    if not numeric:
        return {
            "count": 0,
            "first": None,
            "last": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(numeric),
        "first": numeric[0],
        "last": numeric[-1],
        "min": min(numeric),
        "max": max(numeric),
    }


def _build_energy_trace(tracker, phase_results: Sequence[dict]) -> dict:
    history = [float(value) for value in getattr(tracker, "history", [])]
    phase_traces = []
    for phase in phase_results:
        start = int(phase.get("energy_start_index") or 0)
        end = int(phase.get("energy_end_index") or start)
        start = max(0, min(start, len(history)))
        end = max(start, min(end, len(history)))
        values = history[start:end]
        rows = [
            {
                "phase_evaluation": index + 1,
                "global_evaluation": start + index + 1,
                "energy": float(value),
            }
            for index, value in enumerate(values)
        ]
        phase_traces.append(
            {
                "index": phase.get("index"),
                "name": phase.get("name"),
                "label": phase.get("label") or phase.get("name"),
                "score_model": phase.get("score_model"),
                "optimizer": phase.get("optimizer"),
                "energy_start_index": start,
                "energy_end_index": end,
                "summary": _energy_summary(values),
                "evaluations": rows,
            }
        )
    return {
        "history": history,
        "current_iter": int(getattr(tracker, "current_iter", len(history))),
        "phase_markers": [
            {"global_evaluation_start": int(index) + 1, "label": str(label)}
            for index, label in getattr(tracker, "phase_markers", [])
        ],
        "phase_traces": phase_traces,
    }


def _html_artifact_link(path_text: Optional[str], label: str) -> str:
    if not path_text:
        return "n/a"
    return f'<a href="{html.escape(Path(path_text).name)}">{html.escape(label)}</a>'


def _energy_trace_report_sections(result: dict) -> str:
    energy_trace = result.get("energy_trace") or {}
    phase_traces = energy_trace.get("phase_traces") or []
    if not phase_traces:
        return """
  <h2>Energy Evaluation Details</h2>
  <p class="muted">No optimizer energy evaluations were recorded.</p>
"""

    detail_blocks = []
    for trace in phase_traces:
        summary = trace.get("summary") or {}
        label = trace.get("label") or trace.get("name") or "phase"
        evaluation_rows = "\n".join(
            "<tr>"
            f"<td>{html.escape(str(row.get('phase_evaluation')))}</td>"
            f"<td>{html.escape(str(row.get('global_evaluation')))}</td>"
            f"<td>{html.escape(_format_optional_float(row.get('energy'), 8))}</td>"
            "</tr>"
            for row in trace.get("evaluations") or []
        ) or '<tr><td colspan="3">No energy evaluations recorded.</td></tr>'
        detail_blocks.append(
            f"""
  <details class="report-detail">
    <summary>{html.escape(str(label))} energy evaluations ({html.escape(str(summary.get('count') or 0))})</summary>
    <div class="report-table-wrap">
      <table class="sortable-table">
        <thead><tr><th>Phase evaluation</th><th>Global evaluation</th><th>Objective energy</th></tr></thead>
        <tbody>{evaluation_rows}</tbody>
      </table>
    </div>
  </details>
"""
        )
    return f"""
  <h2>Energy Evaluation Details</h2>
  <p class="muted">
    Energy rows are objective-function evaluations recorded by the QTF landscape tracker;
    these are not necessarily the same as SciPy optimizer iterations.
  </p>
  {''.join(detail_blocks)}
"""


def _phase_status_label(phase: dict) -> str:
    return str(
        phase.get("phase_status_label")
        or _phase_status_label_from_values(
            phase.get("success"),
            phase.get("status"),
            str(phase.get("message") or ""),
        )
    )


def _phase_status_badge(phase: dict) -> str:
    category = str(
        phase.get("phase_status")
        or _phase_status_category(
            phase.get("success"),
            phase.get("status"),
            str(phase.get("message") or ""),
        )
    )
    if category not in {"ok", "warning", "error"}:
        category = "error"
    label = _phase_status_label(phase)
    raw_status = phase.get("status")
    raw_message = str(phase.get("message") or "").strip()
    raw_parts = []
    if raw_status is not None:
        raw_parts.append(f"SciPy status {raw_status}")
    if raw_message:
        raw_parts.append(raw_message)
    raw_detail = " - ".join(raw_parts) if raw_parts else label
    return (
        f'<span class="phase-status-badge phase-status-{html.escape(category)}">'
        f'{html.escape(category)}</span>'
        f'<span class="phase-status-detail">{html.escape(raw_detail)}</span>'
    )


def _phase_status_anchor_markup(phase: dict, warning_anchor_used: bool, error_anchor_used: bool) -> tuple[str, str, bool, bool]:
    index = phase.get("index")
    row_id_attr = ""
    if index not in (None, ""):
        row_id_attr = f' id="phase-result-{html.escape(str(index))}"'
    category = str(
        phase.get("phase_status")
        or _phase_status_category(
            phase.get("success"),
            phase.get("status"),
            str(phase.get("message") or ""),
        )
    )
    inline_anchors = []
    if category == "warning" and not warning_anchor_used:
        inline_anchors.append('<span id="first-phase-warning" class="phase-row-anchor"></span>')
        warning_anchor_used = True
    if category == "error" and not error_anchor_used:
        inline_anchors.append('<span id="first-phase-error" class="phase-row-anchor"></span>')
        error_anchor_used = True
    return row_id_attr, "".join(inline_anchors), warning_anchor_used, error_anchor_used


def _phase_status_alerts(result: dict) -> str:
    phase_results = result.get("phase_results") or []
    warnings = [phase for phase in phase_results if str(phase.get("phase_status")) == "warning"]
    errors = [phase for phase in phase_results if str(phase.get("phase_status")) == "error"]

    def _alert(kind: str, phases: list[dict], href: str) -> str:
        if not phases:
            return ""
        first = phases[0]
        label = first.get("label") or first.get("name") or f"phase {first.get('index') or ''}"
        count = len(phases)
        noun = "phase" if count == 1 else "phases"
        detail = _phase_status_label(first)
        title = "Phase errors detected" if kind == "error" else "Phase warnings detected"
        return f"""
  <aside class="phase-alert phase-alert-{html.escape(kind)}" role="status">
    <strong>{html.escape(title)}</strong>
    <span>{count} {noun}; first: {html.escape(str(label))} - {html.escape(detail)}</span>
    <a href="{html.escape(href)}">Jump to first {html.escape(kind)}</a>
  </aside>
"""

    return (
        _alert("error", errors, "#first-phase-error")
        + _alert("warning", warnings, "#first-phase-warning")
    )


def _phase_results_report_section(result: dict) -> str:
    phase_results = result.get("phase_results") or []
    reference_available = bool(result.get("reference_available"))
    reference_note = ""
    if not reference_available:
        reference_note = (
            " Reference-dependent RMSD columns are shown as n/a because no "
            "ground-truth reference structure was provided."
        )
    if not phase_results:
        rows = '<tr><td colspan="21">No optimizer phases recorded.</td></tr>'
    else:
        rows_parts = []
        warning_anchor_used = False
        error_anchor_used = False
        for item in phase_results:
            id_attr, anchor_markup, warning_anchor_used, error_anchor_used = _phase_status_anchor_markup(
                item,
                warning_anchor_used,
                error_anchor_used,
            )
            rows_parts.append(
                f"<tr{id_attr}>"
                f"<td>{anchor_markup}{html.escape(str(item.get('index') or ''))}</td>"
                f"<td>{html.escape(str(item.get('label') or item.get('name') or ''))}</td>"
                f"<td><code>{html.escape(str(item.get('optimizer') or ''))}</code></td>"
                f"<td><code>{html.escape(str(item.get('score_model') or ''))}</code></td>"
                f"<td><code>{html.escape(str(item.get('optimizer_backend') or ''))}</code></td>"
                f"<td><code>{html.escape(str(item.get('readout_backend') or ''))}</code></td>"
                f"<td>{html.escape(str(item.get('maxiter') or ''))}</td>"
                f"<td>{html.escape(_format_optional_float(item.get('tol'), 6))}</td>"
                f"<td>{html.escape(str(item.get('optimizer_shots') or ''))}</td>"
                f"<td>{html.escape(str(item.get('readout_shots') or ''))}</td>"
                f"<td>{html.escape(_format_optional_float(item.get('objective'), 6))}</td>"
                f"<td>{html.escape(_format_optional_float(item.get('energy_first'), 6))}</td>"
                f"<td>{html.escape(_format_optional_float(item.get('energy_last'), 6))}</td>"
                f"<td>{html.escape(_format_optional_float(item.get('energy_min'), 6))}</td>"
                f"<td>{html.escape(_format_optional_float(item.get('energy_max'), 6))}</td>"
                f"<td>{html.escape(str(item.get('energy_evaluations') or 0))}</td>"
                f"<td>{html.escape(_format_optional_angstrom(item.get('rmsd'), 6))}</td>"
                f"<td>{html.escape('n/a' if item.get('rmsd') is None else str(item.get('rmsd_progress') or 'n/a'))}</td>"
                f"<td>{html.escape(str(item.get('rmsd_atom_count') or 'n/a'))}</td>"
                f"<td>{html.escape(_format_elapsed(item.get('elapsed_s')))}</td>"
                f"<td>{_phase_status_badge(item)}</td>"
                "</tr>"
            )
        rows = "\n".join(rows_parts)
    return f"""
  <h2>Phase Results</h2>
  <p class="muted">
    One row is shown for each optimizer phase. Energy values are objective
    evaluations recorded for that phase, and RMSD values use
    <code>{html.escape(str(result.get("rmsd_angle_mode") or "n/a"))}</code>
    angle decoding.{html.escape(reference_note)}
  </p>
  <div class="report-table-wrap">
    <table class="sortable-table">
      <thead>
        <tr>
          <th>#</th>
          <th>Phase</th>
          <th>Optimizer</th>
          <th>Score</th>
          <th>Optimizer backend</th>
          <th>Readout backend</th>
          <th>Max iter</th>
          <th>Tol</th>
          <th>Optimizer shots</th>
          <th>Readout shots</th>
          <th>Objective</th>
          <th>Energy first</th>
          <th>Energy last</th>
          <th>Energy min</th>
          <th>Energy max</th>
          <th>Energy evals</th>
          <th>RMSD</th>
          <th>RMSD change</th>
          <th>Matched atoms</th>
          <th>Elapsed</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
"""


def _structure_snapshots_report_section(result: dict) -> str:
    snapshots = result.get("structure_snapshots") or []
    if not snapshots:
        return """
  <h2>Structure Snapshots</h2>
  <p class="muted">No structure snapshots were recorded.</p>
"""
    rows = []
    for snapshot in snapshots:
        status = str(snapshot.get("snapshot_status") or "n/a")
        status_markup = (
            f'<span class="phase-status-badge phase-status-{html.escape(status if status in {"ok", "warning", "error"} else "warning")}">'
            f'{html.escape(status)}</span>'
        )
        if snapshot.get("error"):
            status_markup += f'<span class="phase-status-detail">{html.escape(str(snapshot.get("error")))}</span>'
        phase_status_markup = _phase_status_badge(snapshot) if snapshot.get("phase_status") else "n/a"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(snapshot.get('label') or snapshot.get('key') or ''))}</td>"
            f"<td>{html.escape(str(snapshot.get('role') or ''))}</td>"
            f"<td>{html.escape(str(snapshot.get('phase_index') or ''))}</td>"
            f"<td><code>{html.escape(str(snapshot.get('score_model') or 'n/a'))}</code></td>"
            f"<td>{html.escape(str(snapshot.get('angle_mode') or 'n/a'))}</td>"
            f"<td>{html.escape(str(snapshot.get('backend') or 'statevector'))}</td>"
            f"<td>{html.escape(str(snapshot.get('shots') or 'n/a'))}</td>"
            f"<td>{html.escape(str(snapshot.get('atom_count') or 'n/a'))}</td>"
            f"<td>{'yes' if snapshot.get('is_primary_result') else 'no'}</td>"
            f"<td>{'yes' if snapshot.get('visible_default') else 'no'}</td>"
            f"<td>{_html_artifact_link(snapshot.get('viewer_pdb_path') or snapshot.get('pdb_path'), 'PDB')}</td>"
            f"<td>{phase_status_markup}</td>"
            f"<td>{status_markup}</td>"
            "</tr>"
        )
    return f"""
  <h2>Structure Snapshots</h2>
  <p class="muted">
    These PHEAT heavy-atom structures are captured after brickwork initialization,
    after each optimizer phase, and after any optional readouts. The Mol* viewer
    below can toggle each available snapshot; the configured primary result is
    shown by default.
  </p>
  <div class="report-table-wrap">
    <table class="sortable-table">
      <thead>
        <tr>
          <th>Structure</th>
          <th>Role</th>
          <th>Phase</th>
          <th>Score</th>
          <th>Angle mode</th>
          <th>Backend</th>
          <th>Shots</th>
          <th>Atoms</th>
          <th>Primary result</th>
          <th>Default visible</th>
          <th>Viewer PDB</th>
          <th>Phase status</th>
          <th>Snapshot status</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
"""


def _landscape_report_section(result: dict) -> str:
    landscape_path = result.get("landscape_path")
    interactive_path = result.get("interactive_landscape_path")
    if not landscape_path:
        return """
  <h2>Energy Landscape</h2>
  <p class="muted">No energy landscape artifact was recorded.</p>
"""
    name = html.escape(Path(landscape_path).name)
    interactive_block = ""
    if interactive_path:
        interactive_name = html.escape(Path(interactive_path).name)
        interactive_block = f"""
  <p>{_html_artifact_link(interactive_path, "Open interactive landscape HTML")}</p>
  <iframe
    class="interactive-landscape-frame"
    src="{interactive_name}"
    title="Interactive energy landscape"
    loading="lazy">
  </iframe>
"""
    fallback_open = "" if interactive_path else " open"
    return f"""
  <h2>Energy Landscape</h2>
  {interactive_block}
  <details class="report-detail"{fallback_open}>
    <summary>Static PNG fallback</summary>
    <p>{_html_artifact_link(landscape_path, "Open landscape PNG")}</p>
    <img class="landscape-img" src="{name}" alt="Energy landscape for this PHEAT run">
  </details>
"""


def _git_label(git_info: Optional[dict]) -> str:
    if not git_info or not git_info.get("available"):
        return "unavailable"
    describe = git_info.get("describe") or git_info.get("commit") or "unknown"
    branch = git_info.get("branch") or "unknown"
    dirty = " dirty" if git_info.get("dirty") else ""
    return f"{describe} ({branch}{dirty})"


def _software_report_section(result: dict) -> str:
    software = result.get("software_versions") or {}
    packages = software.get("packages") or {}
    package_rows = "\n".join(
        "<tr>"
        f"<td><code>{html.escape(str(name))}</code></td>"
        f"<td>{html.escape(str(version or 'not installed'))}</td>"
        "</tr>"
        for name, version in packages.items()
    ) or '<tr><td colspan="2">No package versions recorded.</td></tr>'
    python_info = software.get("python") or {}
    platform_info = software.get("platform") or {}
    qtf_info = software.get("qtf") or {}
    pheat_info = software.get("pheat") or {}
    sidecar_link = _html_artifact_link(software.get("sidecar_path"), "Open full software version JSON")
    return f"""
  <h2>Software Versions</h2>
  <table>
    <tbody>
      <tr><th>Python</th><td>{html.escape(str(python_info.get("version") or "n/a"))} ({html.escape(str(python_info.get("implementation") or "n/a"))})</td></tr>
      <tr><th>Python executable</th><td><code>{html.escape(str(python_info.get("executable") or "n/a"))}</code></td></tr>
      <tr><th>Platform</th><td>{html.escape(str(platform_info.get("platform") or "n/a"))}</td></tr>
      <tr><th>QTF</th><td>{html.escape(_git_label((qtf_info.get("git") or {})))}<br><code>{html.escape(str(qtf_info.get("runner_path") or "n/a"))}</code></td></tr>
      <tr><th>PHEAT</th><td>version {html.escape(str(pheat_info.get("version") or "n/a"))}; {html.escape(_git_label((pheat_info.get("git") or {})))}<br><code>{html.escape(str(pheat_info.get("module_path") or "n/a"))}</code></td></tr>
      <tr><th>Installed distributions</th><td>{html.escape(str(software.get("installed_distribution_count") or 0))}; {sidecar_link}</td></tr>
    </tbody>
  </table>
  <div class="report-table-wrap">
    <table class="sortable-table">
      <thead><tr><th>Package</th><th>Version</th></tr></thead>
      <tbody>{package_rows}</tbody>
    </table>
  </div>
"""


def _execution_report_section(result: dict) -> str:
    execution = result.get("execution") or {}
    environment = execution.get("environment") or {}
    env_rows = "\n".join(
        "<tr>"
        f"<td><code>{html.escape(str(key))}</code></td>"
        f"<td><code>{html.escape(str(value))}</code></td>"
        "</tr>"
        for key, value in environment.items()
    ) or '<tr><td colspan="2">No selected environment variables were set.</td></tr>'
    command_line = html.escape(str(execution.get("command_line") or "n/a"))
    console_output = html.escape(str(execution.get("console_output") or ""))
    console_link = _html_artifact_link(execution.get("console_output_path"), "Open console log")
    return f"""
  <h2>Execution Log</h2>
  <table>
    <tbody>
      <tr><th>Working directory</th><td><code>{html.escape(str(execution.get("working_directory") or "n/a"))}</code></td></tr>
      <tr><th>Console log</th><td>{console_link}</td></tr>
    </tbody>
  </table>
  <details class="report-detail">
    <summary>Command line</summary>
    <pre class="log-pre">{command_line}</pre>
  </details>
  <details class="report-detail">
    <summary>Selected environment</summary>
    <table><tbody>{env_rows}</tbody></table>
  </details>
  <details class="report-detail">
    <summary>Console output</summary>
    <pre class="log-pre">{console_output}</pre>
  </details>
"""


def _write_pheat_html_report(
    report_path: Path,
    *,
    sequence: str,
    result: dict,
    final_score: dict,
    pheat_reference: Optional[PheatReference],
    reference_aligned_pdb_path: Optional[Path],
    folded_aligned_pdb_path: Path,
    molstar_dir: Optional[Path],
    molstar_error: Optional[str],
) -> None:
    rmsd_details = result.get("rmsd_details") or {}
    pheat_rg = result.get("pheat_radius_of_gyration") or {}
    final_rg = pheat_rg.get("final") or {}
    rg_delta = pheat_rg.get("delta_final_minus_reference") or {}
    reference_available = (
        pheat_reference is not None
        and reference_aligned_pdb_path is not None
        and reference_aligned_pdb_path.exists()
    )
    viewer_structures = _viewer_structure_entries(
        result,
        reference_available=reference_available,
        reference_aligned_pdb_path=reference_aligned_pdb_path,
        folded_aligned_pdb_path=folded_aligned_pdb_path,
    )
    run_label = str(result.get("run_label") or f"replica_{result['replica_id']}")
    case_id = f"replica_{result['replica_id']}"
    viewer_case = {
        "case_id": case_id,
        "case_label": run_label,
        "has_reference": reference_available,
        "structures": viewer_structures,
        "all_heavy_rmsd": result.get("rmsd_to_reference"),
        "backbone_rmsd": rmsd_details.get("backbone_rmsd"),
        "matched_heavy_atoms": result.get("rmsd_atom_count"),
        "matched_backbone_atoms": rmsd_details.get("matched_backbone_atoms"),
        "unmatched_reference_atoms": rmsd_details.get("unmatched_reference_atoms"),
        "unmatched_target_atoms": rmsd_details.get("unmatched_target_atoms"),
        "rg_unweighted_final": _rg_value(final_rg, "unweighted"),
        "rg_mass_weighted_final": _rg_value(final_rg, "mass_weighted"),
        "rg_unweighted_delta": _rg_value(rg_delta, "unweighted"),
        "rg_mass_weighted_delta": _rg_value(rg_delta, "mass_weighted"),
    }
    viewer_data = _json_script_payload(
        [viewer_case]
    )
    case_options = (
        f'<option value="{html.escape(case_id)}">'
        f'{html.escape(run_label)}</option>'
    )
    structure_toggle_buttons = _viewer_toggle_buttons(viewer_structures)
    viewer_pdb_link_rows = _viewer_pdb_link_rows(viewer_structures)
    css_link = ""
    js_script = ""
    if molstar_dir is not None:
        css_link = '<link rel="stylesheet" href="vendor/molstar/molstar.css">'
        js_script = '<script src="vendor/molstar/molstar.js"></script>'

    phase_results = result.get("phase_results") or []
    phase_schedule = result.get("phase_schedule") or {}
    scouting_config = result.get("scouting_config") or phase_schedule.get("scouting") or {}
    result_config = result.get("result_config") or phase_schedule.get("result") or {}
    readout_results = result.get("readout_results") or []
    basis_batching_stats = result.get("basis_circuit_batching_stats") or {}
    phase_alerts = _phase_status_alerts(result)
    phase_results_section = _phase_results_report_section(result)
    structure_snapshots_section = _structure_snapshots_report_section(result)
    score_total = _format_optional_float(final_score.get("total"), 6)
    rg_rows = _pheat_rg_table_rows(result)
    timings = result.get("timings") or {}
    timing_rows = _timing_table_rows(timings)
    brickwork_timing = timings.get("brickwork_encoding") or {}
    brickwork_metadata = brickwork_timing.get("metadata") or {}
    terms = final_score.get("terms") or {}
    score_terms = "\n".join(
        "<tr>"
        f"<td><code>{html.escape(str(name))}</code></td>"
        f"<td>{html.escape(_format_optional_float(value, 6))}</td>"
        "</tr>"
        for name, value in terms.items()
    ) or '<tr><td colspan="2">No terms reported.</td></tr>'
    gate_estimates = result.get("gate_estimates") or []
    gate_estimate_section = ""
    if gate_estimates:
        gate_rows = "\n".join(
            "<tr>"
            f"<td><code>{html.escape(str(item.get('backend')))}</code></td>"
            f"<td>{html.escape(str(item.get('processor_type') or 'n/a'))}</td>"
            f"<td>{html.escape(str(item.get('processor_revision') or 'n/a'))}</td>"
            f"<td>{html.escape(str(item.get('circuit')))}</td>"
            f"<td>{html.escape(str(item.get('optimization_level')))}</td>"
            f"<td>{html.escape(_format_optional_int(item.get('all_gate_depth')))}</td>"
            f"<td>{html.escape(_format_optional_int(item.get('multi_qubit_gate_depth')))}</td>"
            f"<td>{html.escape(_format_optional_int(item.get('all_gate_count')))}</td>"
            f"<td>{html.escape(_format_optional_int(item.get('multi_qubit_gate_count')))}</td>"
            f"<td>{html.escape(_format_optional_float(item.get('transpile_time_s'), 3))}</td>"
            "</tr>"
            for item in gate_estimates
        )
        gate_estimate_section = f"""
  <h2>Circuit Cost Estimates</h2>
  <p class="muted">
    Estimates are transpiled backend resources for the brickwork ansatz and the
    three measurement-basis circuits. Multi-qubit metrics count quantum
    gates with arity greater than or equal to two; measurement and barrier
    operations are excluded. Processor values use backend metadata when exposed
    by Qiskit; simulator rows use the simulator backend version as the revision.
  </p>
  <div class="report-table-wrap">
    <table class="sortable-table">
      <thead>
        <tr>
          <th>Backend</th>
          <th>Processor</th>
          <th>Revision</th>
          <th>Circuit</th>
          <th>Opt level</th>
          <th>Depth</th>
          <th>Multi-Q depth</th>
          <th>Gates</th>
          <th>Multi-Q gates</th>
          <th>Transpile (s)</th>
        </tr>
      </thead>
      <tbody>{gate_rows}</tbody>
    </table>
  </div>
"""
    molstar_status = (
        "Mol* viewer assets loaded from self-hosted PHEAT vendor files."
        if molstar_dir is not None
        else f"Mol* viewer assets unavailable: {molstar_error or 'unknown error'}"
    )
    landscape_section = _landscape_report_section(result)
    energy_trace_section = _energy_trace_report_sections(result)
    software_section = _software_report_section(result)
    execution_section = _execution_report_section(result)
    pheat_citation_section = _pheat_citation_report_section(result, final_score)
    reference_source = str(pheat_reference.source_path) if pheat_reference is not None else "none provided"
    reference_source_type = str(pheat_reference.source_type) if pheat_reference is not None else "sequence-only"
    report_summary = (
        "This report compares the PHEAT geometry-encoded reference to the folded result."
        if reference_available
        else "This sequence-only report shows the folded result and optimizer metrics; no ground-truth reference was provided."
    )
    reference_metric_cards = (
        f"""
    <dl class="metric"><dt>All-heavy RMSD</dt><dd>{html.escape(_format_optional_angstrom(result.get("rmsd_to_reference"), 6))}</dd></dl>
    <dl class="metric"><dt>Backbone RMSD</dt><dd>{html.escape(_format_optional_angstrom(rmsd_details.get("backbone_rmsd"), 6))}</dd></dl>
    <dl class="metric"><dt>Matched atoms</dt><dd>{html.escape(str(result.get("rmsd_atom_count") or "n/a"))}</dd></dl>
    <dl class="metric"><dt>Unmatched reference / folded</dt><dd>{html.escape(str(rmsd_details.get("unmatched_reference_atoms", "n/a")))} / {html.escape(str(rmsd_details.get("unmatched_target_atoms", "n/a")))}</dd></dl>
"""
        if reference_available
        else """
    <dl class="metric"><dt>Reference RMSD</dt><dd>n/a</dd></dl>
    <dl class="metric"><dt>Reference status</dt><dd>not provided</dd></dl>
"""
    )
    alignment_metrics_section = ""
    if reference_available:
        alignment_metrics_section = f"""
  <h2>PHEAT Alignment Metrics</h2>
  <table>
    <tbody>
      <tr><th>All-heavy RMSD</th><td>{html.escape(_format_optional_angstrom(rmsd_details.get("all_heavy_rmsd"), 6))}</td></tr>
      <tr><th>Backbone RMSD</th><td>{html.escape(_format_optional_angstrom(rmsd_details.get("backbone_rmsd"), 6))}</td></tr>
      <tr><th>Matched heavy atoms</th><td>{html.escape(str(rmsd_details.get("matched_heavy_atoms") or "n/a"))}</td></tr>
      <tr><th>Matched backbone atoms</th><td>{html.escape(str(rmsd_details.get("matched_backbone_atoms") or "n/a"))}</td></tr>
      <tr><th>Unmatched reference atoms</th><td>{html.escape(str(rmsd_details.get("unmatched_reference_atoms", "n/a")))}</td></tr>
      <tr><th>Unmatched folded atoms</th><td>{html.escape(str(rmsd_details.get("unmatched_target_atoms", "n/a")))}</td></tr>
    </tbody>
  </table>
"""
    primary_vs_last_phase_rmsd = result.get("primary_vs_last_phase_rmsd")
    readout_drift_values = [
        item.get("primary_drift_rmsd")
        for item in readout_results
        if not item.get("primary") and item.get("primary_drift_rmsd") is not None
    ]
    readout_drift_card = (
        f"""
    <dl class="metric"><dt>Readout drift</dt><dd>{html.escape(_format_optional_angstrom(readout_drift_values[0], 6))}</dd></dl>
"""
        if readout_drift_values
        else ""
    )
    primary_vs_last_phase_card = f"""
    <dl class="metric"><dt>Primary vs last phase</dt><dd>{html.escape(_format_optional_angstrom(primary_vs_last_phase_rmsd, 6))}</dd></dl>
"""
    diagnostic_rmsd_section = ""
    if readout_results:
        readout_rows = "\n".join(
            "<tr>"
            f"<td>{html.escape(str(item.get('name') or ''))}</td>"
            f"<td><code>{html.escape(str(item.get('backend') or ''))}</code></td>"
            f"<td>{html.escape(str(item.get('shots') or 'n/a'))}</td>"
            f"<td><code>{html.escape(str(item.get('score_model') or ''))}</code></td>"
            f"<td>{'yes' if item.get('primary') else 'no'}</td>"
            f"<td>{html.escape(_format_optional_angstrom(item.get('rmsd'), 6))}</td>"
            f"<td>{html.escape(_format_optional_angstrom(item.get('primary_drift_rmsd'), 6))}</td>"
            "</tr>"
            for item in readout_results
        )
        diagnostic_rmsd_section = f"""
  <h2>Optional Readout Diagnostics</h2>
  <div class="report-table-wrap">
    <table class="sortable-table">
      <thead>
        <tr>
          <th>Readout</th>
          <th>Backend</th>
          <th>Shots</th>
          <th>Score</th>
          <th>Primary</th>
          <th>RMSD to reference</th>
          <th>Drift from primary</th>
        </tr>
      </thead>
      <tbody>{readout_rows}</tbody>
    </table>
  </div>
"""
    viewer_title = (
        "Interactive Mol* Structure Snapshot Viewer"
        if reference_available
        else "Interactive Mol* Structure Snapshot Viewer"
    )
    reference_metric_rows = ""
    if reference_available:
        reference_metric_rows = f"""
          <dt>All-heavy RMSD</dt>
          <dd id="viewer-all-heavy-rmsd">{html.escape(_format_optional_angstrom(result.get("rmsd_to_reference"), 6))}</dd>
          <dt>Backbone RMSD</dt>
          <dd id="viewer-backbone-rmsd">{html.escape(_format_optional_angstrom(rmsd_details.get("backbone_rmsd"), 6))}</dd>
          <dt>Matched atoms</dt>
          <dd id="viewer-matched-atoms">{html.escape(str(result.get("rmsd_atom_count") or "n/a"))}</dd>
          <dt>Unmatched reference / folded</dt>
          <dd id="viewer-unmatched">{html.escape(str(rmsd_details.get("unmatched_reference_atoms", "n/a")))} / {html.escape(str(rmsd_details.get("unmatched_target_atoms", "n/a")))}</dd>
"""
    else:
        reference_metric_rows = """
          <dt>Reference metrics</dt>
          <dd>No reference structure was provided.</dd>
"""
    viewer_section = f"""
  <section class="viewer-section" aria-labelledby="viewer-title">
    <h2 id="viewer-title">{html.escape(viewer_title)}</h2>
    <p class="viewer-citations">
      Citations:
      <a href="{MOLSTAR_PROJECT_URL}" rel="noopener">Mol*</a>;
      Sehnal et al.,
      <a href="{MOLSTAR_DOI_URL}" rel="noopener">Mol* Viewer: modern web app for 3D visualization and analysis of large biomolecular structures</a>,
      <em>Nucleic Acids Research</em>, 2021.
    </p>
    <div class="viewer-controls">
      <label for="case-select">Case</label>
      <select id="case-select">{case_options}</select>
      <div class="legend" aria-label="Viewer toggles">
        {structure_toggle_buttons}
        <span class="display-mode-group" role="group" aria-label="Display mode">
          <button type="button" class="legend-toggle display-mode-toggle" data-display-mode="ribbon" aria-pressed="true">Ribbon</button>
          <button type="button" class="legend-toggle display-mode-toggle" data-display-mode="all-atom" aria-pressed="false">All atom</button>
        </span>
        <button type="button" class="legend-toggle" data-recolor>Recolor</button>
      </div>
    </div>
    <div class="viewer-layout">
      <div id="molstar-viewer"></div>
      <aside class="viewer-details" aria-live="polite">
        <h3 id="viewer-case-id">{html.escape(run_label)}</h3>
        <dl>
          {reference_metric_rows}
          <dt>Rg unweighted</dt>
          <dd id="viewer-rg-unweighted">{html.escape(_format_optional_angstrom(_rg_value(final_rg, "unweighted"), 6))}</dd>
          <dt>Rg mass-weighted</dt>
          <dd id="viewer-rg-mass-weighted">{html.escape(_format_optional_angstrom(_rg_value(final_rg, "mass_weighted"), 6))}</dd>
          <dt>Visible structures</dt>
          <dd id="viewer-visible-structures">n/a</dd>
          {viewer_pdb_link_rows}
        </dl>
        <p class="viewer-status" id="viewer-status">Initializing Mol*...</p>
      </aside>
    </div>
  </section>
"""

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(run_label)} | QTF PHEAT Geometry Report</title>
  {css_link}
  <style>
    :root {{
      color-scheme: light;
      --border: #d8d8d8;
      --soft: #f6f8fb;
      --soft-border: #e7e7e7;
      --table-head: #f4f4f4;
      --text: #1f252d;
      --muted: #5c6673;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
      margin: 32px;
    }}
    h1 {{ margin: 0 0 12px; }}
    h2 {{ margin: 28px 0 12px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 13px; }}
    th, td {{ border: 1px solid var(--border); padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: var(--table-head); }}
    .sortable-table thead th {{
      background: var(--table-head);
      box-shadow: 0 1px 0 var(--border), 0 2px 6px rgba(31, 37, 45, 0.08);
      cursor: pointer;
      position: sticky;
      top: 0;
      user-select: none;
      z-index: 4;
    }}
    .sortable-table thead th:focus {{
      outline: 2px solid #205fa6;
      outline-offset: -2px;
    }}
    .sort-indicator {{
      color: var(--muted);
      display: inline-block;
      font-size: 11px;
      margin-left: 4px;
      min-width: 10px;
    }}
    .phase-status-badge {{
      border: 1px solid transparent;
      border-radius: 4px;
      display: inline-block;
      font-size: 11px;
      font-weight: 700;
      line-height: 1.2;
      margin: 0 0 4px;
      padding: 2px 6px;
      text-transform: uppercase;
      white-space: nowrap;
    }}
    .phase-status-ok {{ background: #e8f5e9; border-color: #9ccc9c; color: #1b5e20; }}
    .phase-status-warning {{ background: #fff7e0; border-color: #f1c56a; color: #7a4a00; }}
    .phase-status-error {{ background: #fde8e8; border-color: #f4a6a6; color: #9b1c1c; }}
    .phase-status-detail {{
      color: var(--muted);
      display: block;
      font-size: 12px;
      max-width: 360px;
    }}
    .phase-alert {{
      align-items: flex-start;
      border: 1px solid transparent;
      display: grid;
      gap: 4px;
      margin: 16px 0;
      padding: 12px 14px;
    }}
    .phase-alert strong {{
      display: block;
      font-size: 14px;
    }}
    .phase-alert a {{
      font-weight: 700;
    }}
    .phase-alert-error {{
      background: #fff1f1;
      border-color: #f4a6a6;
      color: #7f1d1d;
    }}
    .phase-alert-warning {{
      background: #fff8e6;
      border-color: #f1c56a;
      color: #6f4300;
    }}
    .phase-row-anchor,
    .sortable-table tbody tr {{
      scroll-margin-top: 72px;
    }}
    code {{ background: #f4f4f4; padding: 1px 3px; }}
    button, select {{
      border: 1px solid var(--border);
      border-radius: 4px;
      background: #fff;
      color: var(--text);
      font: inherit;
    }}
    button {{ cursor: pointer; padding: 3px 7px; }}
    button:hover, select:hover {{ border-color: #aab3bd; }}
    .muted {{ color: var(--muted); }}
    .metric-grid {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); }}
    .metric {{ border: 1px solid var(--soft-border); padding: 12px; }}
    .metric dt {{ color: var(--muted); font-size: 12px; font-weight: 600; }}
    .metric dd {{ font-size: 20px; margin: 2px 0 0; }}
    .landscape-img {{
      border: 1px solid var(--soft-border);
      display: block;
      height: auto;
      max-width: 100%;
    }}
    .interactive-landscape-frame {{
      border: 1px solid var(--soft-border);
      display: block;
      height: min(980px, 92vh);
      margin: 10px 0 16px;
      width: 100%;
    }}
    .report-detail {{
      border: 1px solid var(--soft-border);
      margin: 12px 0;
      padding: 8px 12px;
    }}
    .report-detail > summary {{
      cursor: pointer;
      font-weight: 600;
    }}
    .log-pre {{
      background: #111827;
      color: #e5e7eb;
      margin: 10px 0 4px;
      max-height: 460px;
      overflow: auto;
      padding: 12px;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .report-table-wrap {{
      border: 1px solid var(--soft-border);
      margin: 16px 0;
      max-height: 520px;
      overflow: auto;
      position: relative;
    }}
    .report-table-wrap table {{
      border-collapse: separate;
      border-spacing: 0;
      margin: 0;
      min-width: 100%;
    }}
    .report-table-wrap th,
    .report-table-wrap td {{
      border-left: 0;
      border-top: 0;
    }}
    .report-table-wrap th:last-child,
    .report-table-wrap td:last-child {{
      border-right: 0;
    }}
    .report-table-wrap tbody tr:last-child td {{
      border-bottom: 0;
    }}
    .citation-section {{
      background: #fbfcfe;
      border: 1px solid var(--soft-border);
      margin: 18px 0 20px;
      padding: 14px 16px;
    }}
    .citation-section h2 {{
      margin-top: 0;
    }}
    .citation-section table {{
      margin: 10px 0;
    }}
    .citation-section ul {{
      margin: 8px 0 0 18px;
      padding: 0;
    }}
    .citation-subhead {{
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
      margin: 12px 0 0;
    }}
    .viewer-citations {{ color: var(--muted); font-size: 13px; margin: -4px 0 12px; }}
    .viewer-section {{
      clear: both;
      isolation: isolate;
      margin: 32px 0 0;
      position: relative;
      z-index: 0;
    }}
    .viewer-controls {{
      align-items: center;
      display: flex;
      flex-wrap: wrap;
      gap: 12px 18px;
      margin: 12px 0;
    }}
    .viewer-controls label {{ color: var(--muted); font-size: 13px; font-weight: 600; }}
    .viewer-controls select {{ min-width: 220px; padding: 4px 28px 4px 8px; }}
    .legend {{ align-items: center; display: inline-flex; flex-wrap: wrap; gap: 8px; font-size: 13px; }}
    .legend-toggle {{
      align-items: center;
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: 4px;
      display: inline-flex;
      gap: 6px;
      padding: 3px 7px;
    }}
    .legend-toggle[data-structure-toggle][aria-pressed="false"] {{ color: var(--muted); opacity: 0.55; }}
    .legend-toggle.snapshot-status-warning {{ border-color: #f1c56a; }}
    .legend-toggle.snapshot-status-error {{ border-color: #f4a6a6; }}
    .structure-toggle-label {{ white-space: nowrap; }}
    .legend-status-badge {{
      border-radius: 4px;
      font-size: 10px;
      font-weight: 700;
      line-height: 1.1;
      margin-left: 2px;
      padding: 2px 4px;
      text-transform: uppercase;
    }}
    .display-mode-group {{
      align-items: center;
      display: inline-flex;
      gap: 6px;
    }}
    .display-mode-toggle {{
      justify-content: center;
      min-width: 70px;
    }}
    .display-mode-toggle[aria-pressed="true"] {{
      background: #eef2f7;
      border-color: #1f2937;
      box-shadow: inset 0 0 0 1px #1f2937;
      color: #111827;
    }}
    .legend-swatch {{
      border: 1px solid rgba(0, 0, 0, 0.18);
      border-radius: 50%;
      display: inline-block;
      flex: 0 0 auto;
      height: 10px;
      width: 10px;
    }}
    .legend-swatch-reference {{ background: #0072B2; }}
    .legend-swatch-folded {{ background: #D55E00; }}
    .viewer-layout {{
      border: 1px solid var(--soft-border);
      display: grid;
      grid-template-columns: minmax(0, 1fr) 300px;
      height: 620px;
      isolation: isolate;
      min-height: 620px;
      overflow: hidden;
      position: relative;
    }}
    #molstar-viewer {{
      contain: layout paint size;
      height: 620px;
      isolation: isolate;
      min-height: 0;
      overflow: hidden;
      position: relative;
    }}
    #molstar-viewer .msp-plugin,
    #molstar-viewer .msp-plugin-content,
    #molstar-viewer .msp-layout {{
      height: 100%;
      inset: 0;
      max-height: 100%;
      overflow: hidden;
      position: absolute;
      width: 100%;
    }}
    #molstar-viewer .msp-layout-expanded,
    #molstar-viewer .msp-viewport-expanded {{
      height: 100% !important;
      inset: 0 !important;
      max-height: 100% !important;
      position: absolute !important;
      width: 100% !important;
      z-index: 5 !important;
    }}
    .viewer-details {{ border-left: 1px solid var(--soft-border); background: #fafafa; padding: 16px; }}
    .viewer-details h3 {{ font-size: 16px; margin: 0 0 12px; }}
    .viewer-details dl {{ display: grid; gap: 8px; grid-template-columns: 1fr; margin: 0; }}
    .viewer-details dt {{ color: var(--muted); font-size: 12px; font-weight: 600; }}
    .viewer-details dd {{ margin: -4px 0 4px; }}
    .viewer-details a {{ margin-right: 10px; }}
    .viewer-status {{ color: var(--muted); font-size: 13px; margin: 8px 0 0; }}
    @media (max-width: 900px) {{
      body {{ margin: 18px; }}
      .viewer-layout {{
        grid-template-columns: 1fr;
        height: auto;
        min-height: 0;
      }}
      #molstar-viewer {{ height: 560px; }}
      .viewer-details {{ border-left: 0; border-top: 1px solid var(--soft-border); }}
    }}
  </style>
</head>
<body>
  <h1>{html.escape(run_label)}</h1>
  <p class="muted">
    QTF PHEAT geometry report. Sequence <code>{html.escape(sequence)}</code>,
    replica {result["replica_id"]}. {html.escape(report_summary)}
  </p>

  {pheat_citation_section}

  <section class="metric-grid" aria-label="Key metrics">
    {reference_metric_cards}
    <dl class="metric"><dt>Primary Rg</dt><dd>{html.escape(_format_optional_angstrom(_rg_value(final_rg, "unweighted"), 6))}</dd></dl>
    {readout_drift_card}
    {primary_vs_last_phase_card}
    <dl class="metric"><dt>PHEAT score</dt><dd>{html.escape(score_total)}</dd></dl>
    <dl class="metric"><dt>Default backend</dt><dd>{html.escape(str(result.get("hw_backend")))}</dd></dl>
  </section>

  {phase_alerts}

  <h2>Run Configuration</h2>
  <table>
    <tbody>
      <tr><th>Run label</th><td>{html.escape(run_label)}</td></tr>
      <tr><th>Replica ID</th><td>{html.escape(str(result.get("replica_id")))}</td></tr>
      <tr><th>Reference source</th><td><code>{html.escape(reference_source)}</code></td></tr>
      <tr><th>Reference source type</th><td>{html.escape(reference_source_type)}</td></tr>
      <tr><th>Phase preset / source</th><td><code>{html.escape(str(result.get("phase_preset") or phase_schedule.get("preset") or "n/a"))}</code> / {html.escape(str(result.get("phase_source") or phase_schedule.get("source") or "n/a"))}</td></tr>
      <tr><th>Phase config path</th><td>{html.escape(str(result.get("phase_config_path") or "built-in assets"))}</td></tr>
      <tr><th>Basis circuit batching</th><td>{html.escape(str(result.get("basis_circuit_batching_requested") or phase_schedule.get("basis_circuit_batching") or "auto"))} / effective={html.escape(str(result.get("basis_circuit_batching_effective") or "n/a"))}, calls={html.escape(str(basis_batching_stats.get("calls", 0)))}, jobs={html.escape(str(basis_batching_stats.get("backend_jobs", 0)))}</td></tr>
      <tr><th>Basis batching fallback</th><td>{html.escape("; ".join(str(item) for item in basis_batching_stats.get("fallback_reasons") or []) or "none")}</td></tr>
      <tr><th>Scouting</th><td><code>{html.escape(str(scouting_config.get("score_model") or "n/a"))}</code>, shots={html.escape(str(scouting_config.get("shots") or "n/a"))}, attempts={html.escape(str(scouting_config.get("attempts") or "n/a"))}</td></tr>
      <tr><th>Result score model</th><td><code>{html.escape(str(result.get("result_score_model") or result.get("pheat_score_model")))}</code></td></tr>
      <tr><th>Primary result</th><td><code>{html.escape(str(result_config.get("primary") or (result.get("primary_result") or {}).get("source") or "n/a"))}</code></td></tr>
      <tr><th>Optional readouts</th><td>{html.escape(str(len(readout_results)))}</td></tr>
      <tr><th>Global shots</th><td>{html.escape(str(result.get("shots")))}</td></tr>
      <tr><th>Seed mode</th><td>{html.escape(str(result.get("seed_mode")))}</td></tr>
      <tr><th>Run seed</th><td>{html.escape(_format_seed(result.get("run_seed")))} ({html.escape(str(result.get("seed_source")))})</td></tr>
      <tr><th>Shot seed</th><td>{html.escape(_format_seed(result.get("hw_shot_seed")))} ({html.escape(str(result.get("shot_seed_source")))})</td></tr>
      <tr><th>IBM instance CRN</th><td>{'provided' if result.get("ibm_instance_crn_provided") else 'not provided'}</td></tr>
      <tr><th>Optimizer angle mode</th><td>{html.escape(str(result.get("optimizer_angle_mode")))} ({html.escape(str(result.get("optimizer_angle_mode_requested")))})</td></tr>
      <tr><th>Readout / primary angle mode</th><td>{html.escape(str(result.get("rmsd_angle_mode")))} / {html.escape(str(result.get("primary_angle_mode")))}</td></tr>
      <tr><th>Stop on phase error</th><td>{'yes' if result.get("stop_on_phase_error") else 'no'}</td></tr>
      <tr><th>Angles / qubits / reps / params</th><td>{result.get("total_angles")} / {result.get("n_qubits")} / {result.get("reps")} / {result.get("n_params")}</td></tr>
      <tr><th>Stored angles</th><td>{html.escape(", ".join(result.get("stored_angles") or []) or "none")}</td></tr>
      <tr><th>Chi source / mode</th><td>{html.escape(str(result.get("chi_source")))} / {html.escape(str(result.get("chi_mode")))}</td></tr>
      <tr><th>Max chi</th><td>{html.escape("all" if result.get("max_chi") is None else str(result.get("max_chi")))}</td></tr>
      <tr><th>Viewer status</th><td>{html.escape(molstar_status)}</td></tr>
    </tbody>
  </table>

  <h2>Brickwork Encoding</h2>
  <table>
    <tbody>
      <tr><th>Elapsed</th><td>{html.escape(_format_elapsed(brickwork_timing.get("elapsed_s")))}</td></tr>
      <tr><th>Angles / qubits / reps / params</th><td>{html.escape(str(brickwork_metadata.get("total_angles", result.get("total_angles"))))} / {html.escape(str(brickwork_metadata.get("n_qubits", result.get("n_qubits"))))} / {html.escape(str(brickwork_metadata.get("reps", result.get("reps"))))} / {html.escape(str(brickwork_metadata.get("n_params", result.get("n_params"))))}</td></tr>
      <tr><th>Stored angles</th><td>{html.escape(_format_timing_metadata_value(brickwork_metadata.get("stored_angles", result.get("stored_angles"))))}</td></tr>
      <tr><th>Chi mode / max chi</th><td>{html.escape(str(brickwork_metadata.get("chi_mode", result.get("chi_mode"))))} / {html.escape(str(brickwork_metadata.get("max_chi", result.get("max_chi"))))}</td></tr>
      <tr><th>Optimizer</th><td>{html.escape(str(brickwork_metadata.get("optimizer_angle_mode", result.get("optimizer_angle_mode"))))} / {html.escape(str(brickwork_metadata.get("optimizer_backend_mode", result.get("optimizer_backend_mode"))))}</td></tr>
      <tr><th>Phase preset / count</th><td>{html.escape(str(brickwork_metadata.get("phase_preset", result.get("phase_preset"))))} / {html.escape(str(brickwork_metadata.get("phase_count", len(phase_results))))}</td></tr>
      <tr><th>Basis circuit batching</th><td>{html.escape(str(brickwork_metadata.get("basis_circuit_batching", result.get("basis_circuit_batching_requested"))))}</td></tr>
      <tr><th>Result score / primary</th><td>{html.escape(str(result_config.get("score_model") or result.get("result_score_model")))} / {html.escape(str(result_config.get("primary") or "n/a"))}</td></tr>
    </tbody>
  </table>

  {phase_results_section}

  {structure_snapshots_section}

  {viewer_section}

  <h2>Timing</h2>
  <div class="report-table-wrap">
    <table class="sortable-table">
      <thead><tr><th>Section</th><th>Elapsed</th><th>Status</th><th>Details</th></tr></thead>
      <tbody>{timing_rows}</tbody>
    </table>
  </div>

  {landscape_section}

  {energy_trace_section}

  {alignment_metrics_section}

  <h2>PHEAT Heavy-atom Radius Of Gyration</h2>
  <div class="report-table-wrap">
    <table class="sortable-table">
      <thead><tr><th>Structure</th><th>Unweighted</th><th>Mass-weighted</th><th>Units</th><th>Status</th></tr></thead>
      <tbody>{rg_rows}</tbody>
    </table>
  </div>

  {diagnostic_rmsd_section}

  <h2>Selected PHEAT Score Terms</h2>
  <div class="report-table-wrap">
    <table class="sortable-table">
      <thead><tr><th>Term</th><th>Value</th></tr></thead>
      <tbody>{score_terms}</tbody>
    </table>
  </div>

  {gate_estimate_section}

  {software_section}

  {execution_section}

  <script type="application/json" id="roundtrip-viewer-data">{viewer_data}</script>
  {js_script}
  <script>
    const roundtripCases = JSON.parse(document.getElementById('roundtrip-viewer-data').textContent);
    const caseSelect = document.getElementById('case-select');
    const statusNode = document.getElementById('viewer-status');
    function structuresForCase(caseData) {{
      return Array.isArray(caseData?.structures) ? caseData.structures : [];
    }}
    function colorValueFromHex(hex) {{
      const cleaned = `${{hex || ''}}`.replace('#', '').trim();
      const parsed = Number.parseInt(cleaned, 16);
      return Number.isFinite(parsed) ? parsed : 0x888888;
    }}
    function colorMapForCase(caseData) {{
      const map = {{}};
      for (const structure of structuresForCase(caseData)) {{
        map[structure.key] = {{
          hex: structure.color || '#888888',
          value: colorValueFromHex(structure.color),
        }};
      }}
      return map;
    }}
    function initialVisibleState(caseData) {{
      const visible = {{}};
      for (const structure of structuresForCase(caseData)) {{
        visible[structure.key] = Boolean(structure.visible_default);
      }}
      return visible;
    }}
    const REPRESENTATION_PRESETS = {{
      ribbon: 'preset-structure-representation-polymer-cartoon',
      'all-atom': 'preset-structure-representation-atomic-detail',
    }};
    const viewerState = {{
      viewer: null,
      objectUrls: [],
      visible: initialVisibleState(roundtripCases[0]),
      displayMode: 'ribbon',
      representationKeys: new Map(),
      colors: colorMapForCase(roundtripCases[0]),
      currentCaseData: roundtripCases[0] || null,
    }};
    const detailNodes = {{
      caseId: document.getElementById('viewer-case-id'),
      allHeavyRmsd: document.getElementById('viewer-all-heavy-rmsd'),
      backboneRmsd: document.getElementById('viewer-backbone-rmsd'),
      matchedAtoms: document.getElementById('viewer-matched-atoms'),
      rgUnweighted: document.getElementById('viewer-rg-unweighted'),
      rgMassWeighted: document.getElementById('viewer-rg-mass-weighted'),
      unmatched: document.getElementById('viewer-unmatched'),
      visibleStructures: document.getElementById('viewer-visible-structures'),
    }};
    function setDetailText(node, text) {{
      if (node) node.textContent = text;
    }}

    function sortableCellValue(cell) {{
      const text = (cell?.textContent || '').trim().replace(/\\s+/g, ' ');
      if (!text || text.toLowerCase() === 'n/a') {{
        return {{ kind: 'empty', value: Number.POSITIVE_INFINITY, text: '' }};
      }}
      const numericMatch = text.match(/^[-+]?(?:\\d+\\.?\\d*|\\.\\d+)(?:[eE][-+]?\\d+)?/);
      if (numericMatch) {{
        return {{ kind: 'number', value: Number.parseFloat(numericMatch[0]), text }};
      }}
      return {{ kind: 'text', value: text.toLocaleLowerCase(), text }};
    }}
    function compareSortableRows(left, right, columnIndex, direction) {{
      const leftValue = sortableCellValue(left.row.cells[columnIndex]);
      const rightValue = sortableCellValue(right.row.cells[columnIndex]);
      let result = 0;
      if (leftValue.kind === 'empty' && rightValue.kind !== 'empty') result = 1;
      else if (leftValue.kind !== 'empty' && rightValue.kind === 'empty') result = -1;
      else if (leftValue.kind === 'number' && rightValue.kind === 'number') {{
        result = leftValue.value - rightValue.value;
      }} else {{
        result = `${{leftValue.value}}`.localeCompare(`${{rightValue.value}}`, undefined, {{
          numeric: true,
          sensitivity: 'base',
        }});
      }}
      if (result === 0) result = left.index - right.index;
      return direction === 'ascending' ? result : -result;
    }}
    function sortTableByColumn(table, header, columnIndex) {{
      const tbody = table.tBodies[0];
      if (!tbody) return;
      const currentDirection = header.getAttribute('aria-sort');
      const nextDirection = currentDirection === 'ascending' ? 'descending' : 'ascending';
      for (const sortableHeader of table.querySelectorAll('thead th')) {{
        sortableHeader.setAttribute('aria-sort', 'none');
        const indicator = sortableHeader.querySelector('.sort-indicator');
        if (indicator) indicator.textContent = '';
      }}
      header.setAttribute('aria-sort', nextDirection);
      const activeIndicator = header.querySelector('.sort-indicator');
      if (activeIndicator) activeIndicator.textContent = nextDirection === 'ascending' ? '^' : 'v';
      const rows = Array.from(tbody.rows).map((row, index) => ({{ row, index }}));
      rows.sort((left, right) => compareSortableRows(left, right, columnIndex, nextDirection));
      for (const entry of rows) tbody.appendChild(entry.row);
    }}
    function initializeSortableTables() {{
      for (const table of document.querySelectorAll('table.sortable-table')) {{
        for (const [columnIndex, header] of Array.from(table.querySelectorAll('thead th')).entries()) {{
          header.tabIndex = 0;
          header.setAttribute('aria-sort', 'none');
          header.title = 'Sort table by this column';
          const indicator = document.createElement('span');
          indicator.className = 'sort-indicator';
          indicator.setAttribute('aria-hidden', 'true');
          header.appendChild(indicator);
          header.addEventListener('click', () => sortTableByColumn(table, header, columnIndex));
          header.addEventListener('keydown', (event) => {{
            if (event.key === 'Enter' || event.key === ' ') {{
              event.preventDefault();
              sortTableByColumn(table, header, columnIndex);
            }}
          }});
        }}
      }}
    }}

    function releaseObjectUrls() {{
      for (const url of viewerState.objectUrls) URL.revokeObjectURL(url);
      viewerState.objectUrls = [];
    }}
    function pdbBlobUrl(pdbText) {{
      const blob = new Blob([pdbText], {{ type: 'chemical/x-pdb' }});
      const url = URL.createObjectURL(blob);
      viewerState.objectUrls.push(url);
      return url;
    }}
    function representationParams(key) {{
      const color = viewerState.colors[key]?.value ?? 0x888888;
      return {{ theme: {{ globalName: 'uniform', globalColorParams: {{ value: color }} }} }};
    }}
    function hasCoordinateRecords(pdbText) {{
      return /^(ATOM  |HETATM)/m.test(pdbText);
    }}
    async function loadPdbStructure(pdbText, key, label) {{
      if (!pdbText || !hasCoordinateRecords(pdbText)) return;
      const plugin = viewerState.viewer.plugin;
      const beforeRefs = representationRefs();
      const data = await plugin.builders.data.download(
        {{ url: pdbBlobUrl(pdbText), isBinary: false, label }},
        {{ state: {{ isGhost: true }} }},
      );
      const trajectory = await plugin.builders.structure.parseTrajectory(data, 'pdb');
      const preset = REPRESENTATION_PRESETS[viewerState.displayMode] || REPRESENTATION_PRESETS.ribbon;
      await plugin.builders.structure.hierarchy.applyPreset(trajectory, 'default', {{
        representationPreset: preset,
        representationPresetParams: representationParams(key),
      }});
      for (const ref of representationRefs()) {{
        if (!beforeRefs.has(ref)) viewerState.representationKeys.set(ref, key);
      }}
    }}
    function formatAngstrom(value) {{
      return typeof value === 'number' ? `${{value.toFixed(6)}} A` : 'n/a';
    }}
    function formatRg(value, delta) {{
      if (typeof value !== 'number') return 'n/a';
      const deltaText = typeof delta === 'number'
        ? ` (delta ${{delta >= 0 ? '+' : ''}}${{delta.toFixed(6)}} A)`
        : '';
      return `${{value.toFixed(6)}} A${{deltaText}}`;
    }}
    function setDetails(caseData) {{
      setDetailText(detailNodes.caseId, caseData.case_label || caseData.case_id || 'n/a');
      setDetailText(detailNodes.allHeavyRmsd, formatAngstrom(caseData.all_heavy_rmsd));
      setDetailText(detailNodes.backboneRmsd, formatAngstrom(caseData.backbone_rmsd));
      setDetailText(
        detailNodes.matchedAtoms,
        `${{caseData.matched_heavy_atoms ?? 'n/a'}} heavy, ${{caseData.matched_backbone_atoms ?? 'n/a'}} backbone`,
      );
      setDetailText(detailNodes.rgUnweighted, formatRg(caseData.rg_unweighted_final, caseData.rg_unweighted_delta));
      setDetailText(detailNodes.rgMassWeighted, formatRg(caseData.rg_mass_weighted_final, caseData.rg_mass_weighted_delta));
      setDetailText(
        detailNodes.unmatched,
        `${{caseData.unmatched_reference_atoms ?? 'n/a'}} reference, ${{caseData.unmatched_target_atoms ?? 'n/a'}} folded`,
      );
      setDetailText(
        detailNodes.visibleStructures,
        `${{visibleStructureCount(caseData)}} of ${{structuresForCase(caseData).length}}`,
      );
    }}
    function stateCells() {{
      const cells = viewerState.viewer?.plugin?.state?.data?.cells;
      if (!cells) return [];
      if (typeof cells.values === 'function') return Array.from(cells.values());
      return Object.values(cells);
    }}
    function isRepresentationCell(cell) {{
      const transformer = `${{cell?.transform?.transformer?.id || cell?.transform?.transformer || ''}}`.toLowerCase();
      const type = `${{cell?.obj?.type?.name || cell?.obj?.type || ''}}`.toLowerCase();
      return transformer.includes('representation') || type.includes('representation');
    }}
    function representationRefs() {{
      return new Set(
        stateCells()
          .filter((cell) => cell?.transform?.ref && isRepresentationCell(cell))
          .map((cell) => cell.transform.ref),
      );
    }}
    function setToggleStates() {{
      for (const button of document.querySelectorAll('[data-structure-toggle]')) {{
        const key = button.dataset.structureToggle;
        button.setAttribute('aria-pressed', viewerState.visible[key] ? 'true' : 'false');
      }}
      for (const button of document.querySelectorAll('[data-display-mode]')) {{
        button.setAttribute('aria-pressed', button.dataset.displayMode === viewerState.displayMode ? 'true' : 'false');
      }}
    }}
    function visibleStructureCount(caseData = viewerState.currentCaseData) {{
      return structuresForCase(caseData).filter((structure) => viewerState.visible[structure.key]).length;
    }}
    async function loadCase(caseId, options = {{}}) {{
      const selectedCaseId = typeof caseId === 'string' ? caseId : (caseSelect?.value || roundtripCases[0]?.case_id);
      const loadOptions = typeof caseId === 'object' ? caseId : options;
      const caseData = roundtripCases.find((entry) => entry.case_id === selectedCaseId) || roundtripCases[0];
      if (!caseData || !viewerState.viewer) return;
      viewerState.currentCaseData = caseData;
      viewerState.colors = colorMapForCase(caseData);
      for (const structure of structuresForCase(caseData)) {{
        if (!(structure.key in viewerState.visible)) viewerState.visible[structure.key] = Boolean(structure.visible_default);
      }}
      setDetails(caseData);
      statusNode.textContent = 'Loading selected structures...';
      releaseObjectUrls();
      await viewerState.viewer.plugin.clear(false);
      viewerState.representationKeys.clear();
      const visibleStructures = structuresForCase(caseData).filter((structure) => viewerState.visible[structure.key]);
      if (visibleStructures.length === 0) {{
        statusNode.textContent = 'All structures hidden.';
        setToggleStates();
        setDetails(caseData);
        return;
      }}
      for (const structure of visibleStructures) {{
        await loadPdbStructure(structure.pdb, structure.key, structure.label);
      }}
      await recolorCurrentRepresentations({{ silent: true }});
      if (loadOptions.resetCamera !== false) {{
        try {{ await viewerState.viewer.plugin.managers.camera.reset(); }} catch (error) {{}}
      }}
      setToggleStates();
      setDetails(caseData);
      statusNode.textContent = `Loaded ${{visibleStructures.length}} selected structure${{visibleStructures.length === 1 ? '' : 's'}} from embedded PDB data; no network access is required.`;
    }}
    function textForCell(cell, cellByRef, seen = new Set()) {{
      if (!cell || seen.has(cell)) return '';
      seen.add(cell);
      const parts = [cell.obj?.label, cell.obj?.data?.label, cell.params?.values?.label, cell.transform?.params?.label];
      const parentRef = cell.transform?.parent || cell.parent?.ref || cell.sourceRef;
      if (parentRef && cellByRef.has(parentRef)) parts.push(textForCell(cellByRef.get(parentRef), cellByRef, seen));
      return parts.filter(Boolean).join(' ');
    }}
    function colorKeyForCell(cell, cellByRef) {{
      const storedKey = viewerState.representationKeys.get(cell?.transform?.ref);
      if (storedKey) return storedKey;
      const text = textForCell(cell, cellByRef).toLowerCase();
      for (const structure of structuresForCase(viewerState.currentCaseData)) {{
        const key = `${{structure.key || ''}}`.toLowerCase();
        const label = `${{structure.label || ''}}`.toLowerCase();
        const shortLabel = `${{structure.short_label || ''}}`.toLowerCase();
        if ((key && text.includes(key)) || (label && text.includes(label)) || (shortLabel && text.includes(shortLabel))) {{
          return structure.key;
        }}
      }}
      return null;
    }}
    function recoloredParams(params, key) {{
      const next = {{ ...(params || {{}}) }};
      const color = viewerState.colors[key]?.value ?? 0x888888;
      next.theme = {{ ...(next.theme || {{}}), globalName: 'uniform', globalColorParams: {{ ...(next.theme?.globalColorParams || {{}}), value: color }} }};
      next.colorTheme = {{ ...(next.colorTheme || {{}}), name: 'uniform', params: {{ ...(next.colorTheme?.params || {{}}), value: color }} }};
      if (next.color !== undefined) next.color = 'uniform';
      if (next.colorParams !== undefined) next.colorParams = {{ ...(next.colorParams || {{}}), value: color }};
      return next;
    }}
    async function recolorCurrentRepresentations(options = {{}}) {{
      const plugin = viewerState.viewer?.plugin;
      const builder = plugin?.build ? plugin.build() : null;
      if (!plugin || !builder) {{
        if (!options.silent) statusNode.textContent = 'Mol* state is not ready for recoloring.';
        return 0;
      }}
      const cells = stateCells();
      const cellByRef = new Map(cells.map((cell) => [cell?.transform?.ref, cell]).filter(([ref]) => ref));
      let changed = 0;
      for (const cell of cells) {{
        if (!cell?.transform?.ref || !isRepresentationCell(cell)) continue;
        const key = colorKeyForCell(cell, cellByRef);
        if (!key) continue;
        builder.to(cell.transform.ref).update((old) => recoloredParams(old, key));
        changed += 1;
      }}
      if (changed) await builder.commit();
      if (!options.silent) statusNode.textContent = changed ? `Recolored ${{changed}} Mol* representations.` : 'No recolorable Mol* representations were found.';
      return changed;
    }}
    for (const button of document.querySelectorAll('[data-structure-toggle]')) {{
      button.addEventListener('click', () => {{
        const key = button.dataset.structureToggle;
        viewerState.visible[key] = !viewerState.visible[key];
        setToggleStates();
        loadCase(caseSelect.value, {{ resetCamera: false }});
      }});
    }}
    for (const button of document.querySelectorAll('[data-display-mode]')) {{
      button.addEventListener('click', () => {{
        viewerState.displayMode = button.dataset.displayMode;
        setToggleStates();
        loadCase(caseSelect.value, {{ resetCamera: false }});
      }});
    }}
    caseSelect.addEventListener('change', () => loadCase(caseSelect.value, {{ resetCamera: true }}));
    const recolorButton = document.querySelector('[data-recolor]');
    if (recolorButton) recolorButton.addEventListener('click', () => recolorCurrentRepresentations());
    async function initViewer() {{
      setToggleStates();
      if (!window.molstar || !roundtripCases.length) {{
        statusNode.textContent = {json.dumps(molstar_status)};
        return;
      }}
      try {{
        viewerState.viewer = await molstar.Viewer.create('molstar-viewer', {{
          layoutIsExpanded: false,
          layoutShowControls: true,
          layoutShowRemoteState: false,
          layoutShowSequence: true,
          layoutShowLog: true,
          layoutShowLeftPanel: true,
          viewportShowReset: true,
          viewportShowScreenshotControls: true,
          viewportShowControls: true,
          viewportShowExpand: true,
          viewportShowToggleFullscreen: true,
          viewportShowSettings: true,
          viewportShowSelectionMode: true,
          viewportShowAnimation: true,
          viewportShowTrajectoryControls: true,
          viewportBackgroundColor: '#ffffff',
          viewportFocusBehavior: 'disabled',
          volumeStreamingDisabled: true,
        }});
        await loadCase(caseSelect.value || roundtripCases[0].case_id, {{ resetCamera: true }});
      }} catch (error) {{
        statusNode.textContent = `Mol* failed to load: ${{error.message || error}}`;
      }}
    }}
    initializeSortableTables();
    initViewer();
    window.addEventListener('unload', releaseObjectUrls);
  </script>
</body>
</html>
"""
    report_path.write_text(document, encoding="utf-8")


def get_hw_backend(
    hw_backend_name,
    shot_seed: Optional[int] = None,
    ibm_instance_crn: Optional[str] = None,
):
    if hw_backend_name is None:
        return None
    backend_key = str(hw_backend_name).lower()
    if backend_key == "none":
        return None
    if backend_key == StatevectorShotsBackend.name:
        return StatevectorShotsBackend(seed=shot_seed)
    if backend_key == "aer":
        from qiskit_aer import AerSimulator

        if shot_seed is not None:
            return AerSimulator(seed_simulator=int(shot_seed))
        return AerSimulator()
    if backend_key == "least_busy":
        criteria = _least_busy_context(4)
        service = _ibm_runtime_service(criteria, ibm_instance_crn)
        return _least_busy_backend(service, 4)
    service = _ibm_runtime_service(f"'{hw_backend_name}'", ibm_instance_crn)
    try:
        return service.backend(hw_backend_name)
    except Exception as exc:
        raise _backend_lookup_error(f"'{hw_backend_name}'", exc)


def _workflow_backend_specs(schedule: PhaseSchedule, inherited_spec: str) -> list[str]:
    specs = [
        _effective_backend_spec(schedule.scouting.backend, inherited_spec),
    ]
    for phase in schedule.phases:
        specs.append(_effective_backend_spec(phase.optimizer_backend, inherited_spec))
        specs.append(_effective_backend_spec(phase.readout_backend, inherited_spec))
    for readout in schedule.readouts:
        specs.append(_effective_backend_spec(readout.backend, inherited_spec))
    deduped = []
    seen = set()
    for spec in specs:
        key = spec.lower()
        if key not in seen:
            deduped.append(spec)
            seen.add(key)
    return deduped


def _resolve_workflow_backends(
    schedule: PhaseSchedule,
    *,
    inherited_spec: str,
    inherited_backend,
    shot_seed: Optional[int],
    ibm_instance_crn: Optional[str],
) -> dict[str, object]:
    registry = {}
    inherited_key = str(inherited_spec).strip().lower()
    for spec in _workflow_backend_specs(schedule, inherited_spec):
        key = spec.lower()
        if key == inherited_key:
            registry[key] = inherited_backend
        else:
            registry[key] = get_hw_backend(
                spec,
                shot_seed=shot_seed,
                ibm_instance_crn=ibm_instance_crn,
            )
    return registry


def _backend_from_registry(registry: dict[str, object], spec: Optional[str], inherited_spec: str):
    effective = _effective_backend_spec(spec, inherited_spec)
    return registry[effective.lower()]


def _backend_label(backend) -> str:
    return "statevector" if backend is None else f"sampler:{_backend_display_name(backend)}"


class PheatQuantumBiophysicsFolder(runner.QuantumBiophysicsFolder):
    """runner_hardware3 folder with PHEAT-backed structure and objective."""

    def __init__(
        self,
        sequence,
        force_field="charmm",
        chi_mode="all",
        selective_chi_map=None,
        *,
        angle_units="radians",
        stored_angles="all",
        max_chi=None,
        include_terminal_oxt=False,
        pheat_score_model="pheat-goap",
        qtf_energy_weight=0.0,
        bond_angle_encoding="centered",
        tau_center_deg=ANGLE_N_CA_C,
        tau_span_deg=25.0,
        theta_center_deg=ANGLE_CA_C_N,
        theta_span_deg=25.0,
        optimizer_angle_mode="statevector",
        optimizer_backend=None,
        optimizer_shots=4096,
        basis_circuit_batching="auto",
        reference_residue_geometry: Optional[ResidueGeometryStructure] = None,
    ):
        self.angle_units = str(angle_units).lower()
        self.stored_angles = normalize_stored_angles(stored_angles)
        self.max_chi = normalize_max_chi(max_chi)
        self.include_terminal_oxt = bool(include_terminal_oxt)
        self.reference_residue_geometry = reference_residue_geometry
        self.pheat_score_model = str(pheat_score_model).lower()
        self.qtf_energy_weight = float(qtf_energy_weight)
        self.bond_angle_encoding = str(bond_angle_encoding).lower()
        self.tau_center_deg = float(tau_center_deg)
        self.tau_span_deg = float(tau_span_deg)
        self.theta_center_deg = float(theta_center_deg)
        self.theta_span_deg = float(theta_span_deg)
        self.optimizer_angle_mode = str(optimizer_angle_mode).lower()
        self.optimizer_backend = optimizer_backend
        self.optimizer_shots = int(optimizer_shots)
        self.basis_circuit_batching = str(basis_circuit_batching).strip().lower()
        self.basis_circuit_batching_stats = {
            "requested": self.basis_circuit_batching,
            "calls": 0,
            "batched_calls": 0,
            "serial_calls": 0,
            "local_statevector_calls": 0,
            "fallback_calls": 0,
            "basis_circuits": 0,
            "backend_jobs": 0,
            "last_effective": None,
            "fallback_reasons": [],
        }
        self.active_pheat_score_model = self.pheat_score_model
        self.last_pheat_score = None
        self.last_pheat_error = None
        self.last_pheat_structure = None
        self.last_pheat_residue_geometry = None
        self.pheat_chi_dofs_by_residue = {}

        if self.angle_units not in ANGLE_UNITS:
            raise ValueError(f"angle_units must be one of {', '.join(ANGLE_UNITS)}")
        if self.pheat_score_model not in available_models():
            raise ValueError(
                f"Unknown PHEAT score model '{self.pheat_score_model}'. "
                f"Available models: {', '.join(available_models())}"
            )
        if self.bond_angle_encoding not in {"centered", "raw"}:
            raise ValueError("bond_angle_encoding must be 'centered' or 'raw'.")
        if self.optimizer_angle_mode not in {"statevector", "sampler"}:
            raise ValueError("optimizer_angle_mode must be 'statevector' or 'sampler'.")
        if self.optimizer_angle_mode == "sampler" and self.optimizer_backend is None:
            raise ValueError("optimizer_backend is required when optimizer_angle_mode is 'sampler'.")
        if self.optimizer_shots <= 0:
            raise ValueError("optimizer_shots must be positive.")
        if self.basis_circuit_batching not in BASIS_CIRCUIT_BATCHING_MODES:
            raise ValueError(
                "basis_circuit_batching must be one of "
                f"{', '.join(BASIS_CIRCUIT_BATCHING_MODES)}."
            )

        super().__init__(
            sequence,
            force_field=force_field,
            chi_mode=chi_mode,
            selective_chi_map=selective_chi_map,
        )
        self._apply_pheat_angle_configuration()

    def _apply_pheat_angle_configuration(self) -> None:
        self.dof_map = []
        self.pheat_chi_dofs_by_residue = {}

        for res_idx, aa in enumerate(self.sequence):
            self.dof_map.append({"res": res_idx, "type": "phi"})
            self.dof_map.append({"res": res_idx, "type": "psi"})

            available_chis = _pheat_available_chis(aa)
            allowed_chis = self._allowed_chis_for_residue(res_idx, aa, available_chis)
            if self.max_chi is not None:
                allowed_chis = [
                    chi_name
                    for chi_name in allowed_chis
                    if _chi_sort_key(chi_name) <= self.max_chi
                ]
            allowed_chis = sorted(set(allowed_chis), key=_chi_sort_key)
            self.pheat_chi_dofs_by_residue[res_idx] = allowed_chis
            for chi_name in allowed_chis:
                self.dof_map.append({"res": res_idx, "type": chi_name})

        present = {(dof["res"], dof["type"]) for dof in self.dof_map}
        for angle_name in OPTIONAL_PHEAT_ANGLES:
            if angle_name not in self.stored_angles:
                continue
            residue_count = self.n_residues if angle_name == "tau" else self.n_residues - 1
            for res_idx in range(max(0, residue_count)):
                key = (res_idx, angle_name)
                if key not in present:
                    self.dof_map.append({"res": res_idx, "type": angle_name})
                    present.add(key)

        self._rebuild_quantum_register()

    def _rebuild_quantum_register(self) -> None:
        self.total_angles = len(self.dof_map)
        self.n_qubits = max(2, int(np.ceil(np.log2(max(self.total_angles, 1)))))
        min_reps = int(np.ceil(self.total_angles / (2 * self.n_qubits)))
        self.reps = max(3, min(min_reps, 6))

        if runner.QISKIT_AVAILABLE:
            self.ansatz = self._build_brickwork_ansatz(self.n_qubits, self.reps)
            self.n_params = self.ansatz.num_parameters
        else:
            self.ansatz = None
            self.n_params = 1

        self._cache_initialized = False
        self._initialize_topology_cache()

    @staticmethod
    def _build_brickwork_ansatz(n_qubits, reps):
        n_params_total = 2 * n_qubits * (reps + 1)
        params = runner.ParameterVector("theta", n_params_total)
        qc = runner.QuantumCircuit(n_qubits)
        p_idx = 0

        for _rep in range(reps):
            for q in range(n_qubits):
                qc.ry(params[p_idx], q)
                p_idx += 1
                qc.rz(params[p_idx], q)
                p_idx += 1
            for q in range(0, n_qubits - 1, 2):
                qc.cx(q, q + 1)
            for q in range(1, n_qubits - 1, 2):
                qc.cx(q, q + 1)

        for q in range(n_qubits):
            qc.ry(params[p_idx], q)
            p_idx += 1
            qc.rz(params[p_idx], q)
            p_idx += 1
        return qc

    def _record_basis_circuit_batching(
        self,
        *,
        effective: str,
        backend,
        reason: Optional[str] = None,
    ) -> None:
        stats = getattr(self, "basis_circuit_batching_stats", None)
        if not isinstance(stats, dict):
            return
        stats["calls"] = int(stats.get("calls") or 0) + 1
        stats["basis_circuits"] = int(stats.get("basis_circuits") or 0) + 3
        stats["last_effective"] = effective
        stats["last_backend"] = _backend_display_name(backend) if backend is not None else "statevector"
        if effective == "batched":
            stats["batched_calls"] = int(stats.get("batched_calls") or 0) + 1
            stats["backend_jobs"] = int(stats.get("backend_jobs") or 0) + 1
        elif effective.startswith("local_statevector"):
            stats["local_statevector_calls"] = int(stats.get("local_statevector_calls") or 0) + 1
        else:
            stats["serial_calls"] = int(stats.get("serial_calls") or 0) + 1
            stats["backend_jobs"] = int(stats.get("backend_jobs") or 0) + 3
            if effective == "fallback_serial":
                stats["fallback_calls"] = int(stats.get("fallback_calls") or 0) + 1
        if reason:
            reasons = list(stats.get("fallback_reasons") or [])
            if reason not in reasons:
                reasons.append(reason)
            stats["fallback_reasons"] = reasons[-10:]

    def _angles_from_probability_vectors(self, pZ, pX, pY):
        n = self.n_qubits
        n_states = 2 ** n
        state_angles = 2.0 * np.pi * np.arange(n_states) / n_states

        def _marginal_angles(pvec):
            angles = []
            for q in range(n):
                mask = np.array([(k >> q) & 1 for k in range(n_states)], dtype=float)
                p1 = np.dot(pvec, mask)
                angles.append(2.0 * np.pi * p1 - np.pi)
            return np.array(angles)

        def _circular_mean(pvec):
            s = np.sum(pvec * np.sin(state_angles))
            c = np.sum(pvec * np.cos(state_angles))
            return np.arctan2(s, c)

        def _cdf_angles(pvec):
            cdf = np.cumsum(pvec)
            return 2.0 * np.pi * cdf - np.pi

        mZ = _marginal_angles(pZ)
        mX = _marginal_angles(pX)
        mY = _marginal_angles(pY)
        cmZ = _circular_mean(pZ)
        cmX = _circular_mean(pX)
        cmY = _circular_mean(pY)
        cdf_angles = _cdf_angles(pZ)

        eps = 1e-12
        kl_ZX = float(np.sum(pZ * np.log((pZ + eps) / (pX + eps))))
        kl_ZY = float(np.sum(pZ * np.log((pZ + eps) / (pY + eps))))
        cross_1 = np.arctan(kl_ZX) * 2.0 - np.pi / 2.0
        cross_2 = np.arctan(kl_ZY) * 2.0 - np.pi / 2.0

        base = np.concatenate([
            mZ,
            mX,
            mY,
            [cmZ, cmX, cmY],
            cdf_angles,
            [cross_1, cross_2],
        ])
        base = np.clip(base, -np.pi, np.pi)

        if len(base) == 0:
            base = np.zeros(1, dtype=float)
        if len(base) >= self.total_angles:
            return base[:self.total_angles]

        out = np.zeros(self.total_angles, dtype=float)
        out[:len(base)] = base
        length = len(base)
        for k in range(length, self.total_angles):
            i = k % length
            j = (k * 3 + 1) % length
            m = (k * 7 + 2) % length
            value = 0.60 * base[i] + 0.30 * np.sin(base[j]) + 0.10 * np.cos(base[m])
            out[k] = (value + np.pi) % (2 * np.pi) - np.pi
        return out

    def _get_angles(self, params, mode: str = "statevector", backend=None, shots: int = 4096):
        if mode != "sampler":
            return super()._get_angles(params, mode=mode, backend=backend, shots=shots)
        if self.ansatz is None:
            raise RuntimeError("Qiskit not available.")
        if shots <= 0:
            raise ValueError("shots must be positive for sampler angle extraction.")
        if backend is None:
            if runner.AerSimulator is None:
                raise RuntimeError("Sampler angle extraction requires a backend or qiskit-aer.")
            backend = runner.AerSimulator()

        param_dict = dict(zip(self.ansatz.parameters, params))
        bound_circuit = self.ansatz.assign_parameters(param_dict)
        n = self.n_qubits
        n_states = 2 ** n

        def _basis_circuit(qc, basis: str):
            c = qc.copy()
            if basis == "X":
                for q in range(n):
                    c.h(q)
            elif basis == "Y":
                for q in range(n):
                    c.sdg(q)
                    c.h(q)
            if not isinstance(backend, StatevectorShotsBackend):
                c.measure_all()
            return c

        circuits = [
            _basis_circuit(bound_circuit, "Z"),
            _basis_circuit(bound_circuit, "X"),
            _basis_circuit(bound_circuit, "Y"),
        ]

        def _statevector_counts(qc, basis_offset: int):
            if runner.Statevector is None:
                raise RuntimeError("Statevector not available. Install qiskit.")
            sv = runner.Statevector(qc)
            if backend.seed is not None:
                sv.seed(backend.seed + basis_offset)
            return sv.sample_counts(shots)

        def _counts_to_pvec(counts):
            pvec = np.zeros(n_states, dtype=float)
            total = sum(counts.values())
            if total <= 0:
                raise ValueError("statevector-shots produced no measurement counts.")
            for bitstring, count in counts.items():
                bs = str(bitstring).replace(" ", "")[::-1]
                idx = int(bs, 2)
                pvec[idx] += count / total
            return pvec

        def _serial_counts():
            counts = []
            for qc in circuits:
                tqc = runner.transpile(qc, backend)
                counts.append(backend.run(tqc, shots=shots).result().get_counts())
            return counts

        def _batched_counts():
            tqcs = runner.transpile(circuits, backend)
            job = backend.run(tqcs, shots=shots)
            result = job.result()
            return [result.get_counts(index) for index in range(len(circuits))]

        requested = getattr(self, "basis_circuit_batching", "auto")
        if isinstance(backend, StatevectorShotsBackend):
            counts = [_statevector_counts(qc, offset) for offset, qc in enumerate(circuits)]
            effective = "local_statevector_serial" if requested == "off" else "local_statevector"
            self._record_basis_circuit_batching(effective=effective, backend=backend)
            return self._angles_from_probability_vectors(*[_counts_to_pvec(counts_i) for counts_i in counts])

        if requested == "off":
            counts = _serial_counts()
            self._record_basis_circuit_batching(effective="serial", backend=backend)
            return self._angles_from_probability_vectors(*[_counts_to_pvec(counts_i) for counts_i in counts])

        try:
            counts = _batched_counts()
            self._record_basis_circuit_batching(effective="batched", backend=backend)
        except Exception as exc:
            if requested == "on":
                raise RuntimeError(
                    "Basis-circuit batching was requested with --basis-circuit-batching on, "
                    f"but backend {_backend_display_name(backend)} did not accept the batched "
                    f"Z/X/Y circuit job: {exc}"
                ) from exc
            counts = _serial_counts()
            self._record_basis_circuit_batching(
                effective="fallback_serial",
                backend=backend,
                reason=f"{_backend_display_name(backend)}: {exc}",
            )
        return self._angles_from_probability_vectors(*[_counts_to_pvec(counts_i) for counts_i in counts])

    def _angle_dict(self, angle_vector) -> dict[str, float]:
        return {
            f"{dof['res']}_{dof['type']}": float(value)
            for dof, value in zip(self.dof_map, angle_vector)
        }

    def _to_configured_units(self, angle_radians: Optional[float]) -> Optional[float]:
        if angle_radians is None:
            return None
        if self.angle_units == "degrees":
            return float(math.degrees(angle_radians))
        return float(angle_radians)

    def _bond_angle_value(
        self,
        raw_radians: Optional[float],
        *,
        center_deg: float,
        span_deg: float,
    ) -> Optional[float]:
        if raw_radians is None:
            return None
        if self.bond_angle_encoding == "raw":
            return self._to_configured_units(float(raw_radians))
        angle_deg = center_deg + span_deg * math.sin(float(raw_radians))
        return self._to_configured_units(math.radians(angle_deg))

    def _template_residue(self, res_idx: int) -> Optional[ResidueGeometry]:
        template = self.reference_residue_geometry
        if template is None or not (0 <= res_idx < len(template.residues)):
            return None
        return template.residues[res_idx]

    def _template_disulfide_bonds(self):
        template = self.reference_residue_geometry
        if template is None:
            return []
        return list(template.disulfide_bonds)

    def angle_vector_to_residue_geometry(self, angle_vector) -> ResidueGeometryStructure:
        angle_dict = self._angle_dict(angle_vector)
        residues = []

        for res_idx, aa in enumerate(self.sequence):
            template_residue = self._template_residue(res_idx)
            chain_id = template_residue.chain_id if template_residue is not None else "A"
            resseq = template_residue.resseq if template_residue is not None else res_idx + 1
            icode = template_residue.icode if template_residue is not None else ""
            chi_values = []
            for chi_name in self.pheat_chi_dofs_by_residue.get(res_idx, []):
                key = f"{res_idx}_{chi_name}"
                if key in angle_dict:
                    chi_values.append(self._to_configured_units(angle_dict[key]))

            omega = None
            theta = None
            if res_idx < self.n_residues - 1:
                if "omega" in self.stored_angles:
                    omega = self._to_configured_units(angle_dict.get(f"{res_idx}_omega"))
                if "theta" in self.stored_angles:
                    theta = self._bond_angle_value(
                        angle_dict.get(f"{res_idx}_theta"),
                        center_deg=self.theta_center_deg,
                        span_deg=self.theta_span_deg,
                    )

            tau = None
            if "tau" in self.stored_angles:
                tau = self._bond_angle_value(
                    angle_dict.get(f"{res_idx}_tau"),
                    center_deg=self.tau_center_deg,
                    span_deg=self.tau_span_deg,
                )

            residues.append(
                ResidueGeometry(
                    name=one_to_three(aa),
                    phi=self._to_configured_units(angle_dict.get(f"{res_idx}_phi")),
                    psi=self._to_configured_units(angle_dict.get(f"{res_idx}_psi")),
                    omega=omega,
                    tau=tau,
                    theta=theta,
                    chi=chi_values,
                    chain_id=chain_id,
                    resseq=resseq,
                    icode=icode,
                )
            )

        residue_geometry = ResidueGeometryStructure(
            residues=residues,
            name=f"qtf-pheat:{self.sequence}",
            angle_units=self.angle_units,
            metadata={
                "source": "qtf_runner_hardware_pheat",
                "sequence": self.sequence,
                "chi_source": "pheat.sidechain_steps",
                "chi_mode": self.chi_mode,
                "stored_angles": list(self.stored_angles),
                "max_chi": self.max_chi,
                "bond_angle_encoding": self.bond_angle_encoding,
                "tau_center_deg": self.tau_center_deg,
                "tau_span_deg": self.tau_span_deg,
                "theta_center_deg": self.theta_center_deg,
                "theta_span_deg": self.theta_span_deg,
            },
            stored_angles=self.stored_angles,
            disulfide_bonds=self._template_disulfide_bonds(),
        )
        self.last_pheat_residue_geometry = residue_geometry
        return residue_geometry

    def structure_from_angle_vector(self, angle_vector) -> HeavyAtomStructure:
        residue_geometry = self.angle_vector_to_residue_geometry(angle_vector)
        structure = structure_from_residue_geometry(
            residue_geometry,
            include_terminal_oxt=self.include_terminal_oxt,
        )
        self.last_pheat_structure = structure
        return structure

    def structure_from_coords_labels(self, coords, labels) -> HeavyAtomStructure:
        atoms = []
        for serial, (pos, label) in enumerate(zip(coords, labels), start=1):
            if len(label) >= 8:
                rid, atom_name, element, chain_id, resseq, icode, resname, record_name = label[:8]
            else:
                rid, atom_name, element = label[:3]
                template_residue = self._template_residue(int(rid))
                chain_id = template_residue.chain_id if template_residue is not None else "A"
                resseq = template_residue.resseq if template_residue is not None else int(rid) + 1
                icode = template_residue.icode if template_residue is not None else ""
                resname = one_to_three(self.sequence[int(rid)])
                record_name = "ATOM"
            rid = int(rid)
            atom_name = str(atom_name).strip()
            atoms.append(
                Atom(
                    name=atom_name,
                    element=str(element).strip().upper() or atom_name[0].upper(),
                    x=float(pos[0]),
                    y=float(pos[1]),
                    z=float(pos[2]),
                    resname=str(resname).strip().upper() or one_to_three(self.sequence[rid]),
                    chain_id=str(chain_id or ""),
                    resseq=int(resseq),
                    icode=str(icode or ""),
                    record_name=str(record_name or "ATOM"),
                    serial=serial,
                    occupancy=1.0,
                    bfactor=0.0,
                )
            )
        return HeavyAtomStructure(
            atoms=atoms,
            name=f"qtf-pheat:{self.sequence}",
            metadata={"source": "qtf_runner_hardware_pheat"},
            disulfide_bonds=self._template_disulfide_bonds(),
        )

    def _structure_to_arrays(self, structure: HeavyAtomStructure):
        coords = []
        labels = []
        residue_index = {key: index for index, key in enumerate(structure.residue_keys())}
        for atom in structure.atoms:
            rid = residue_index.get(atom.residue_key, int(atom.resseq) - 1)
            coords.append([atom.x, atom.y, atom.z])
            labels.append((rid, atom.name.strip().upper(), atom.element.strip().upper()))
        return np.asarray(coords, dtype=float), labels, []

    def build_full_structure(self, angle_vector):
        structure = self.structure_from_angle_vector(angle_vector)
        return self._structure_to_arrays(structure)

    def _angle_vector_from_params(
        self,
        params,
        *,
        angle_mode: str,
        backend=None,
        shots: int = 4096,
    ):
        if angle_mode == "statevector":
            return self._get_angles(params, mode="statevector")
        if angle_mode == "sampler":
            if backend is None:
                raise ValueError("sampler angle mode requires a backend.")
            return self._get_angles(params, mode="sampler", backend=backend, shots=shots)
        raise ValueError(f"Unknown angle mode: {angle_mode}")

    @staticmethod
    def _angle_mode_for_backend(backend) -> str:
        return "statevector" if backend is None else "sampler"

    def energy_function(self, params, return_terms: bool = False):
        angle_vec = self._angle_vector_from_params(
            params,
            angle_mode=self.optimizer_angle_mode,
            backend=self.optimizer_backend,
            shots=self.optimizer_shots,
        )
        structure = self.structure_from_angle_vector(angle_vec)

        qtf_total = 0.0
        qtf_terms = {}
        if self.qtf_energy_weight:
            old_tracker = self.tracker
            self.tracker = None
            try:
                qtf_total = runner.QuantumBiophysicsFolder.energy_function(
                    self, params, return_terms=True
                )
                qtf_terms = dict(getattr(self, "last_energy_terms", {}) or {})
            finally:
                self.tracker = old_tracker

        active_model = getattr(self, "active_pheat_score_model", self.pheat_score_model)
        try:
            score = score_structure(structure, model=active_model)
            score_payload = score.to_dict()
            score_payload["status"] = "ok"
            pheat_total = float(score.total)
            self.last_pheat_error = None
        except Exception as exc:
            score_payload = {
                "model": active_model,
                "status": "unavailable",
                "error": str(exc),
                "total": None,
                "units": None,
                "terms": {},
                "warnings": [],
                "citations": [],
                "metadata": {},
            }
            pheat_total = PHEAT_FAILURE_PENALTY
            self.last_pheat_error = str(exc)

        objective = pheat_total + self.qtf_energy_weight * float(qtf_total)
        self.last_pheat_score = score_payload
        self.last_energy_terms = {
            "pheat_model": active_model,
            "pheat_total": pheat_total,
            "qtf_weight": self.qtf_energy_weight,
            "qtf_total": float(qtf_total),
            "total": float(objective),
            **{f"pheat_{k}": float(v) for k, v in score_payload.get("terms", {}).items()},
            **{f"qtf_{k}": float(v) for k, v in qtf_terms.items() if isinstance(v, (int, float))},
        }

        if self.tracker:
            self.tracker.log(objective)
        return objective

    def get_smart_initialization(self, n_attempts=20, seed=None):
        rng = np.random.default_rng(seed)

        print(f"--- SCOUTING: Checking {n_attempts} starting points ---")
        print(f" > RNG Seed: {_format_seed(seed)}")
        print(f" > PHEAT score: {getattr(self, 'active_pheat_score_model', self.pheat_score_model)}")

        best_params = None
        best_energy = float("inf")
        for _i in range(n_attempts):
            trial_params = rng.uniform(-np.pi, np.pi, self.n_params)
            energy = self.energy_function(trial_params)
            if energy < best_energy:
                best_energy = energy
                best_params = trial_params
        print(f" > Best Start Found: Energy {best_energy:.2f}")
        return best_params

    def fold(
        self,
        max_iter=2000,
        initial_params=None,
        hw_backend=None,
        shots=4096,
        phase_schedule: Optional[Sequence[PhaseConfig]] = None,
        result_config: Optional[ResultConfig] = None,
        readout_schedule: Optional[Sequence[ReadoutConfig]] = None,
        backend_registry: Optional[dict[str, object]] = None,
        inherited_backend_spec: str = "none",
        timings: Optional[TimingRecorder] = None,
        workflow_progress: Optional[WorkflowProgress] = None,
        true_ca=None,
        pheat_reference_structure: Optional[HeavyAtomStructure] = None,
        stop_on_phase_error: bool = False,
        initial_snapshot_score_model: Optional[str] = None,
        initial_snapshot_shots: Optional[int] = None,
        initial_snapshot_backend_spec: Optional[str] = None,
    ):
        if workflow_progress is not None:
            workflow_progress.start("Optimization phases")
        print(f"--- STARTING OPTIMIZATION PHASES (HYBRID PIPELINE) ---")
        timings = timings or TimingRecorder()
        self.timing_recorder = timings
        self.tracker = PhaseLandscapeTracker()
        base_shots = int(shots)
        if base_shots <= 0:
            raise ValueError("shots must be positive.")
        phases = list(phase_schedule or [])
        if not phases:
            raise ValueError("phase_schedule must contain at least one phase.")
        if result_config is None:
            raise ValueError("result_config is required.")
        readouts = list(readout_schedule or [])
        backend_registry = backend_registry or {
            str(inherited_backend_spec).strip().lower(): hw_backend,
        }
        self.phase_schedule = [asdict(phase) for phase in phases]
        self.result_config = asdict(result_config)
        self.readout_schedule = [asdict(readout) for readout in readouts]
        self.readout_results = []
        self.structure_snapshots = []

        def _backend_for_spec(spec: Optional[str]):
            return _backend_from_registry(backend_registry, spec, inherited_backend_spec)

        self.rmsd_angle_mode = "per_phase_readout"
        self.primary_angle_mode = "primary_result"
        print("  Optimizer angles : phase-specific")
        print("  Readout angles   : phase-specific")
        print("  Phase schedule   :")
        for index, phase in enumerate(phases, start=1):
            phase_optimizer_backend = _backend_for_spec(phase.optimizer_backend)
            phase_readout_backend = _backend_for_spec(phase.readout_backend)
            print(
                "    "
                f"Phase {index}/{len(phases)}: {phase.name} "
                f"({phase.optimizer}, score={phase.score_model}, "
                f"optimizer={_backend_label(phase_optimizer_backend)}, "
                f"optimizer_shots={phase.optimizer_shots}, "
                f"readout={_backend_label(phase_readout_backend)}, "
                f"readout_shots={phase.readout_shots}, "
                f"maxiter={phase.maxiter})"
            )
        print(f"  Primary result   : {result_config.primary}")
        print(f"  Result score     : {result_config.score_model}")
        print(f"  Optional readouts: {len(readouts)}")

        if initial_params is None:
            init_params = self.get_smart_initialization()
        else:
            init_params = initial_params

        def _structure_from_params(
            params,
            *,
            angle_mode: str,
            backend,
            shots: int,
        ) -> HeavyAtomStructure:
            angles = self._angle_vector_from_params(
                params,
                angle_mode=angle_mode,
                backend=backend,
                shots=shots,
            )
            return self.structure_from_angle_vector(angles)

        def _record_structure_snapshot(
            *,
            key: str,
            role: str,
            label: str,
            params=None,
            structure: Optional[HeavyAtomStructure] = None,
            angle_mode: str,
            backend,
            shots: int,
            score_model: Optional[str],
            phase_index: Optional[int] = None,
            phase_name: Optional[str] = None,
            phase_label: Optional[str] = None,
            phase_status: Optional[str] = None,
            phase_status_label: Optional[str] = None,
            visible_default: bool = False,
        ) -> Optional[HeavyAtomStructure]:
            snapshot = {
                "key": key,
                "role": role,
                "label": label,
                "phase_index": phase_index,
                "phase_name": phase_name,
                "phase_label": phase_label,
                "phase_status": phase_status,
                "phase_status_label": phase_status_label,
                "angle_mode": angle_mode,
                "backend": None if backend is None else _backend_display_name(backend),
                "shots": int(shots),
                "score_model": score_model,
                "visible_default": bool(visible_default),
            }
            try:
                if structure is None:
                    if params is None:
                        raise ValueError("snapshot requires params or a structure")
                    structure = _structure_from_params(
                        params,
                        angle_mode=angle_mode,
                        backend=backend,
                        shots=int(shots),
                    )
                snapshot["structure"] = structure
                snapshot["atom_count"] = len(structure.atoms)
                snapshot["snapshot_status"] = "ok"
            except Exception as exc:
                snapshot["structure"] = None
                snapshot["atom_count"] = None
                snapshot["snapshot_status"] = "error"
                snapshot["error"] = str(exc)
                print(f" > Structure snapshot warning: {label} failed: {exc}")
            self.structure_snapshots.append(snapshot)
            return snapshot.get("structure")

        initial_backend = _backend_for_spec(initial_snapshot_backend_spec)
        initial_angle_mode = self._angle_mode_for_backend(initial_backend)
        _record_structure_snapshot(
            key="brickwork_initial",
            role="brickwork",
            label="Brickwork initial structure",
            params=init_params,
            angle_mode=initial_angle_mode,
            backend=initial_backend,
            shots=int(initial_snapshot_shots or base_shots),
            score_model=initial_snapshot_score_model or getattr(
                self,
                "active_pheat_score_model",
                self.pheat_score_model,
            ),
            visible_default=False,
        )

        def _get_rmsd(params, phase: PhaseConfig, *, readout_backend, readout_angle_mode: str):
            if pheat_reference_structure is None:
                return None
            angles = self._angle_vector_from_params(
                params,
                angle_mode=readout_angle_mode,
                backend=readout_backend,
                shots=phase.readout_shots,
            )
            structure = self.structure_from_angle_vector(angles)
            return _pheat_alignment_details(pheat_reference_structure, structure)

        current_params = init_params
        final_result = None
        phase_results = []
        previous_rmsd = None

        for phase_index, phase in enumerate(phases, start=1):
            timing_key = _phase_timing_key(phase_index, phase.name)
            phase_optimizer_backend = _backend_for_spec(phase.optimizer_backend)
            phase_readout_backend = _backend_for_spec(phase.readout_backend)
            phase_optimizer_angle_mode = self._angle_mode_for_backend(phase_optimizer_backend)
            phase_readout_angle_mode = self._angle_mode_for_backend(phase_readout_backend)
            phase_optimizer_label = _backend_label(phase_optimizer_backend)
            phase_readout_label = _backend_label(phase_readout_backend)
            self.active_pheat_score_model = phase.score_model
            self.optimizer_angle_mode = phase_optimizer_angle_mode
            self.optimizer_backend = phase_optimizer_backend
            self.optimizer_shots = phase.optimizer_shots
            self.current_stage = phase_index
            phase_label = phase.label or phase.name
            minimize_options = dict(phase.options)
            minimize_options["maxiter"] = phase.maxiter
            minimize_kwargs = {
                "method": phase.optimizer,
                "options": minimize_options,
            }
            if phase.tol is not None:
                minimize_kwargs["tol"] = phase.tol

            phase_start_iter = int(self.tracker.current_iter)
            phase_end_iter = phase_start_iter
            with timings.section(
                timing_key,
                label=phase_label,
                metadata={
                    "phase": phase.name,
                    "phase_index": phase_index,
                    "phase_label": phase_label,
                    "phase_count": len(phases),
                    "method": phase.optimizer,
                    "score_model": phase.score_model,
                    "optimizer": phase_optimizer_label,
                    "optimizer_backend": _effective_backend_spec(
                        phase.optimizer_backend,
                        inherited_backend_spec,
                    ),
                    "readout_backend": _effective_backend_spec(
                        phase.readout_backend,
                        inherited_backend_spec,
                    ),
                    "optimizer_shots": phase.optimizer_shots,
                    "readout": phase_readout_label,
                    "readout_shots": phase.readout_shots,
                    "maxiter": phase.maxiter,
                    "tol": phase.tol,
                    "options": dict(phase.options),
                },
            ):
                print(
                    f"Phase {phase_index}/{len(phases)}: {phase_label} "
                    f"({phase.optimizer}, score={phase.score_model}, "
                    f"optimizer={phase_optimizer_label}, optimizer_shots={self.optimizer_shots}, "
                    f"readout={phase_readout_label}, readout_shots={phase.readout_shots})..."
                )
                self.tracker.mark_phase(phase_label)
                final_result = runner.minimize(
                    self.energy_function,
                    current_params,
                    **minimize_kwargs,
                )
                details = _get_rmsd(
                    final_result.x,
                    phase,
                    readout_backend=phase_readout_backend,
                    readout_angle_mode=phase_readout_angle_mode,
                )
                rmsd_value = details["all_heavy_rmsd"] if details is not None else None
                rmsd_atoms = details["matched_heavy_atoms"] if details is not None else None
                print(f" > Phase Energy : {final_result.fun:.2f}")
                if rmsd_value is not None:
                    print(
                        f" > Phase RMSD   : {rmsd_value:.4f} Å "
                        f"({rmsd_atoms} PHEAT heavy atoms, readout_shots={phase.readout_shots})"
                    )
                phase_end_iter = int(self.tracker.current_iter)

            timing_record = timings.sections.get(timing_key) or {}
            phase_energy_values = self.tracker.history[phase_start_iter:phase_end_iter]
            phase_energy_summary = _energy_summary(phase_energy_values)
            optimizer_success = bool(getattr(final_result, "success", False))
            optimizer_status = getattr(final_result, "status", None)
            optimizer_message = str(getattr(final_result, "message", ""))
            phase_status = _phase_status_category(
                optimizer_success,
                optimizer_status,
                optimizer_message,
            )
            phase_status_label = _phase_status_label_from_values(
                optimizer_success,
                optimizer_status,
                optimizer_message,
            )
            if phase_status != "ok":
                timing_record["status"] = phase_status
            snapshot_key = f"phase_{phase_index:02d}_{_slug(phase.name)}"
            phase_result = {
                "index": phase_index,
                "name": phase.name,
                "label": phase_label,
                "timing_key": timing_key,
                "structure_snapshot_key": snapshot_key,
                "optimizer": phase.optimizer,
                "score_model": phase.score_model,
                "optimizer_backend": _effective_backend_spec(phase.optimizer_backend, inherited_backend_spec),
                "readout_backend": _effective_backend_spec(phase.readout_backend, inherited_backend_spec),
                "shots": phase.shots,
                "optimizer_shots": phase.optimizer_shots,
                "readout_shots": phase.readout_shots,
                "maxiter": phase.maxiter,
                "tol": phase.tol,
                "options": dict(phase.options),
                "objective": float(final_result.fun),
                "success": optimizer_success,
                "status": optimizer_status,
                "message": optimizer_message,
                "phase_status": phase_status,
                "phase_status_label": phase_status_label,
                "rmsd": rmsd_value,
                "rmsd_atom_count": rmsd_atoms,
                "rmsd_details": details,
                "elapsed_s": timing_record.get("elapsed_s"),
                "energy_start_index": phase_start_iter,
                "energy_end_index": phase_end_iter,
                "energy_evaluations": phase_energy_summary["count"],
                "energy_first": phase_energy_summary["first"],
                "energy_last": phase_energy_summary["last"],
                "energy_min": phase_energy_summary["min"],
                "energy_max": phase_energy_summary["max"],
                "rmsd_progress": _rmsd_progress_label(rmsd_value, previous_rmsd)
                if previous_rmsd is not None
                else "baseline",
            }
            phase_results.append(phase_result)
            _record_structure_snapshot(
                key=snapshot_key,
                role="phase",
                label=f"Phase {phase_index}: {phase_label}",
                params=final_result.x,
                angle_mode=phase_readout_angle_mode,
                backend=phase_readout_backend,
                shots=phase.readout_shots,
                score_model=phase.score_model,
                phase_index=phase_index,
                phase_name=phase.name,
                phase_label=phase_label,
                phase_status=phase_status,
                phase_status_label=phase_status_label,
                visible_default=False,
            )
            if phase_status != "ok":
                print(f" > Phase Status : {phase_status_label}")
            if rmsd_value is not None:
                previous_rmsd = rmsd_value
            current_params = final_result.x
            if stop_on_phase_error and phase_status == "error":
                self.phase_results = phase_results
                self.phase_rmsds = {result["name"]: result["rmsd"] for result in phase_results}
                self.phase_rmsd_details = {
                    result["name"]: result["rmsd_details"] for result in phase_results
                }
                raise PhaseOptimizationError(phase_result)

        if final_result is None:
            raise RuntimeError("No optimizer phases ran.")

        if pheat_reference_structure is not None:
            print(f"\n── RMSD Progress by Phase (PHEAT heavy atoms) ──────")
            for phase_result in phase_results:
                if phase_result["rmsd"] is None:
                    print(f"  {phase_result['label']} : N/A")
                    continue
                suffix = (
                    ""
                    if phase_result["rmsd_progress"] == "baseline"
                    else f"  {phase_result['rmsd_progress']}"
                )
                print(
                    f"  {phase_result['label']} : "
                    f"{phase_result['rmsd']:.4f} Å{suffix}"
                )
            print("──────────────────────────────────────────")

        self.phase_results = phase_results
        self.phase_rmsds = {result["name"]: result["rmsd"] for result in phase_results}
        self.phase_rmsd_details = {
            result["name"]: result["rmsd_details"] for result in phase_results
        }

        optimal_params = final_result.x

        if workflow_progress is not None:
            workflow_progress.start("Optional readouts")
        if readouts:
            with timings.section(
                "optional_readouts",
                label="Optional readouts",
                metadata={"count": len(readouts)},
            ):
                print("\n[READOUTS] Optional post-optimization readouts")
                for readout in readouts:
                    readout_backend = _backend_for_spec(readout.backend)
                    readout_angle_mode = self._angle_mode_for_backend(readout_backend)
                    readout_key = f"readout_{_slug(readout.name)}"
                    print(
                        "  "
                        f"{readout.name}: source={readout.source}, "
                        f"backend={_backend_label(readout_backend)}, shots={readout.shots}"
                    )
                    structure = _record_structure_snapshot(
                        key=readout_key,
                        role="readout",
                        label=f"Readout: {readout.name}",
                        params=optimal_params,
                        angle_mode=readout_angle_mode,
                        backend=readout_backend,
                        shots=readout.shots,
                        score_model=readout.score_model,
                        visible_default=(
                            readout.primary
                            or result_config.primary == readout.name
                        ),
                    )
                    details = None
                    if pheat_reference_structure is not None and structure is not None:
                        details = _pheat_alignment_details(pheat_reference_structure, structure)
                    self.readout_results.append(
                        {
                            "name": readout.name,
                            "key": readout_key,
                            "source": readout.source,
                            "backend": _effective_backend_spec(readout.backend, inherited_backend_spec),
                            "angle_mode": readout_angle_mode,
                            "shots": readout.shots,
                            "score_model": readout.score_model,
                            "primary": bool(
                                readout.primary
                                or result_config.primary == readout.name
                            ),
                            "snapshot_key": readout_key,
                            "atom_count": None if structure is None else len(structure.atoms),
                            "rmsd": None if details is None else details["all_heavy_rmsd"],
                            "rmsd_details": details,
                        }
                    )
        else:
            timings.skip(
                "optional_readouts",
                label="Optional readouts",
                metadata={"count": 0},
            )

        if workflow_progress is not None:
            workflow_progress.start("Primary result selection")
        with timings.section(
            "primary_result_selection",
            label="Primary result selection",
            metadata={
                "primary": result_config.primary,
                "score_model": result_config.score_model,
            },
        ):
            last_phase_snapshot = next(
                (
                    snapshot
                    for snapshot in reversed(self.structure_snapshots)
                    if snapshot.get("role") == "phase"
                    and snapshot.get("snapshot_status") == "ok"
                    and snapshot.get("structure") is not None
                ),
                None,
            )
            if result_config.primary == PRIMARY_LAST_PHASE:
                primary_snapshot = last_phase_snapshot
            else:
                primary_snapshot = next(
                    (
                        snapshot
                        for snapshot in self.structure_snapshots
                        if snapshot.get("role") == "readout"
                        and snapshot.get("key") == f"readout_{_slug(result_config.primary)}"
                        and snapshot.get("snapshot_status") == "ok"
                        and snapshot.get("structure") is not None
                    ),
                    None,
                )
            if primary_snapshot is None:
                raise RuntimeError(f"Primary result source is unavailable: {result_config.primary}")
            primary_snapshot["is_primary_result"] = True
            primary_snapshot["visible_default"] = True
            primary_structure = primary_snapshot["structure"]
            self.primary_angle_mode = primary_snapshot.get("angle_mode")
            self.primary_backend_mode = primary_snapshot.get("backend")
            self.primary_shots = primary_snapshot.get("shots")
            self.primary_result = {
                "source": result_config.primary,
                "snapshot_key": primary_snapshot.get("key"),
                "label": primary_snapshot.get("label"),
                "score_model": result_config.score_model,
                "angle_mode": self.primary_angle_mode,
                "backend": self.primary_backend_mode,
                "shots": self.primary_shots,
                "atom_count": len(primary_structure.atoms),
            }
            coords, labels, bonds = self._structure_to_arrays(primary_structure)
            print(
                "\n[PRIMARY RESULT] "
                f"{self.primary_result['label']} "
                f"({self.primary_result['source']}, {self.primary_result['atom_count']} atoms)"
            )
        print(f"\n[RESULT] Final objective: {final_result.fun:.4f}")
        return coords, labels, bonds, self.tracker, optimal_params, final_result.fun

    def save_pdb(self, coords, labels, filename="structure.pdb", energy=0.0):
        write_pdb(self.structure_from_coords_labels(coords, labels), filename)

    def save_reduced_pdb(self, ca_coords, filename="reduced.pdb", sidechain_centroids=None, energy=0.0):
        atoms = []
        for serial, pos in enumerate(ca_coords, start=1):
            rid = serial - 1
            template_residue = self._template_residue(rid)
            chain_id = template_residue.chain_id if template_residue is not None else "A"
            resseq = template_residue.resseq if template_residue is not None else rid + 1
            icode = template_residue.icode if template_residue is not None else ""
            atoms.append(
                Atom(
                    name="CA",
                    element="C",
                    x=float(pos[0]),
                    y=float(pos[1]),
                    z=float(pos[2]),
                    resname=one_to_three(self.sequence[rid]),
                    chain_id=chain_id,
                    resseq=resseq,
                    icode=icode,
                    record_name="ATOM",
                    serial=serial,
                    occupancy=1.0,
                    bfactor=0.0,
                )
            )
        write_pdb(
            HeavyAtomStructure(
                atoms=atoms,
                name=f"qtf-pheat:{self.sequence}:ca",
                metadata={"source": "qtf_runner_hardware_pheat", "representation": "ca"},
                disulfide_bonds=self._template_disulfide_bonds(),
            ),
            filename,
        )


def _selective_chi_map() -> dict[str, list[str]]:
    return {
        "Y": ["chi1", "chi2"],
        "W": ["chi1", "chi2"],
        "F": ["chi1", "chi2"],
        "H": ["chi1", "chi2"],
        "D": ["chi1"],
        "E": ["chi1"],
        "N": ["chi1"],
        "Q": ["chi1"],
        "T": ["chi1"],
        "S": ["chi1"],
        "V": ["chi1"],
        "I": ["chi1"],
        "L": ["chi1"],
        "M": ["chi1"],
        "K": ["chi1"],
        "R": ["chi1"],
        "C": ["chi1"],
        "P": ["chi1"],
        "A": [],
        "G": [],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predict", required=True)
    parser.add_argument(
        "--reference_structure",
        default=None,
        help="Reference PDB, PHEAT heavy-atom JSON, or PHEAT residue-geometry JSON for RMSD/reporting.",
    )
    parser.add_argument("--forcefield", default="amber")
    parser.add_argument(
        "--chi_mode",
        default="all",
        choices=["chi1_only", "selective", "all"],
        help="Filter PHEAT-derived chi angles; default keeps all PHEAT chis per residue.",
    )
    parser.add_argument("--replica_id", type=int, required=True)
    parser.add_argument(
        "--run-label",
        default=None,
        help="Optional human-readable label for console/report display; does not affect seeds or filenames.",
    )
    parser.add_argument(
        "--stop-on-phase-error",
        action="store_true",
        default=False,
        help="Abort after a phase classified as an optimizer error; warnings still continue.",
    )
    parser.add_argument("--maxiter", type=int, default=2000)
    parser.add_argument(
        "--hw_backend",
        default="aer",
        help="Default backend inherited by scouting, optimizer phases, readouts, and gate estimates unless overridden.",
    )
    parser.add_argument(
        "--ibm-instance-crn",
        "--ibm-crn",
        dest="ibm_instance_crn",
        default=None,
        help="Optional IBM Cloud quantum instance CRN for QiskitRuntimeService IBM backend lookup.",
    )
    parser.add_argument("--shots", type=int, default=4096, help="Default shot count for sampled angle decoding.")
    parser.add_argument("--phase-preset", default="pheat-phased", help="PHEAT phase preset name from YAML assets or --phase-config.")
    parser.add_argument("--phase-config", default=None, help="Optional YAML file containing additional phase presets.")
    parser.add_argument(
        "--basis-circuit-batching",
        choices=BASIS_CIRCUIT_BATCHING_MODES,
        default=None,
        help=(
            "Submit Z/X/Y measurement-basis circuits as one sampler job: "
            "auto batches when supported, on requires it, off submits serially."
        ),
    )
    parser.add_argument("--phase", action="append", default=[], help="Define a CLI-only phase name; repeat in execution order.")
    parser.add_argument("--phase-label", action="append", default=[], metavar="NAME=LABEL")
    parser.add_argument("--phase-optimizer", action="append", default=[], metavar="NAME=OPTIMIZER")
    parser.add_argument("--phase-score", action="append", default=[], metavar="NAME=MODEL")
    parser.add_argument("--phase-optimizer-backend", action="append", default=[], metavar="NAME=BACKEND")
    parser.add_argument("--phase-readout-backend", action="append", default=[], metavar="NAME=BACKEND")
    parser.add_argument("--phase-shots", action="append", default=[], metavar="NAME=SHOTS")
    parser.add_argument("--phase-optimizer-shots", action="append", default=[], metavar="NAME=SHOTS")
    parser.add_argument("--phase-readout-shots", action="append", default=[], metavar="NAME=SHOTS")
    parser.add_argument("--phase-maxiter", action="append", default=[], metavar="NAME=ITERATIONS")
    parser.add_argument("--phase-tol", action="append", default=[], metavar="NAME=TOL")
    parser.add_argument("--phase-option", action="append", default=[], metavar="NAME:key=value")
    parser.add_argument("--scouting-score", choices=available_models(), default=None)
    parser.add_argument("--scouting-backend", default=None)
    parser.add_argument("--scouting-shots", type=int, default=None)
    parser.add_argument("--scouting-attempts", type=int, default=None)
    parser.add_argument("--result-score", choices=available_models(), default=None)
    parser.add_argument(
        "--primary-result",
        default=None,
        help="Primary result source: last_phase_structure or a configured readout name.",
    )
    parser.add_argument("--readout", action="append", default=[], help="Define an optional post-optimization readout name.")
    parser.add_argument("--readout-backend", action="append", default=[], metavar="NAME=BACKEND")
    parser.add_argument("--readout-shots", action="append", default=[], metavar="NAME=SHOTS")
    parser.add_argument("--readout-score", action="append", default=[], metavar="NAME=MODEL")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Explicit run-level seed for optimizer initialization; overrides --seed-mode.",
    )
    parser.add_argument(
        "--seed-mode",
        choices=SEED_MODES,
        default="random",
        help="Seed policy when --seed is omitted: random/unseeded or sequence+replica derived.",
    )
    parser.add_argument(
        "--shot-seed",
        type=int,
        default=None,
        help="Seed for shot-based sampling; defaults to the run seed.",
    )
    parser.add_argument(
        "--optimizer-angle-mode",
        choices=OPTIMIZER_ANGLE_MODES,
        default="auto",
        help=(
            "Initial/fallback optimizer angle decoding mode. Phase optimizer backends "
            "override this during the configured workflow."
        ),
    )
    parser.add_argument(
        "--estimate-gates",
        nargs="?",
        const=GATE_ESTIMATE_SELECTED_BACKEND,
        default=None,
        metavar="BACKENDS",
        help=(
            "Estimate transpiled gate counts/depths for BACKENDS, a comma-separated "
            "list such as aer,ibm_brisbane. If BACKENDS is omitted, use --hw_backend; "
            "statevector-shots implicitly estimates against aer."
        ),
    )
    parser.add_argument("--outdir", default="outputs/replica")
    parser.add_argument(
        "--report-command-line",
        default=None,
        help="Exact shell command to show in the report; defaults to a reconstructed Python command.",
    )
    parser.add_argument("--average_reference_backbone", default=False, type=bool)

    parser.add_argument("--angle-units", choices=ANGLE_UNITS, default="radians")
    parser.add_argument(
        "--store-angles",
        default="all",
        help="Comma-separated PHEAT optional angles to optimize/store: omega,tau,theta,all, or blank.",
    )
    parser.add_argument("--max-chi", default=None, help="Maximum chi angles per residue; blank/all/none keeps all.")
    parser.add_argument("--include-terminal-oxt", action="store_true", default=False)
    parser.add_argument(
        "--bond-angle-encoding",
        choices=["centered", "raw"],
        default="centered",
        help="How quantum values encode PHEAT tau/theta bond angles.",
    )
    parser.add_argument("--tau-center-deg", type=float, default=ANGLE_N_CA_C)
    parser.add_argument("--tau-span-deg", type=float, default=25.0)
    parser.add_argument("--theta-center-deg", type=float, default=ANGLE_CA_C_N)
    parser.add_argument("--theta-span-deg", type=float, default=25.0)

    parser.add_argument("--qtf-energy-weight", type=float, default=0.0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    prefix = f"replica_{args.replica_id}"
    if args.run_label is not None and not args.run_label.strip():
        parser.error("--run-label must not be blank when provided.")
    run_label = _derive_run_label(
        args.predict,
        args.replica_id,
        args.reference_structure,
        args.run_label,
    )
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    console_capture = io.StringIO()
    sys.stdout = TeeStream(sys.stdout, console_capture)
    sys.stderr = TeeStream(sys.stderr, console_capture)
    console_log_path = outdir / f"{prefix}_console.log"
    software_versions_path = outdir / f"{prefix}_software_versions.json"
    command_line = _command_line(argv, args.report_command_line)
    execution_context = {
        "command_line": command_line,
        "working_directory": str(Path.cwd()),
        "environment": _environment_snapshot(),
        "console_output_path": str(console_log_path),
        "software_versions_path": str(software_versions_path),
    }
    stored_angles = normalize_stored_angles(args.store_angles)
    max_chi = normalize_max_chi(args.max_chi)
    strat = "random"
    base_seed = int(__import__("hashlib").sha256(args.predict.encode()).hexdigest(), 16) % (2**32)
    derived_seed = (base_seed + args.replica_id) % SEED_MODULUS
    if args.maxiter <= 0:
        parser.error("--maxiter must be > 0.")
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be a non-negative integer.")
    if args.shot_seed is not None and args.shot_seed < 0:
        parser.error("--shot-seed must be a non-negative integer.")
    shots = _resolve_shot_settings(args, parser)
    phase_schedule = _resolve_phase_schedule(
        args,
        parser,
        global_shots=shots,
        global_maxiter=args.maxiter,
    )
    timings = TimingRecorder()
    workflow_progress = WorkflowProgress(args.replica_id)
    if args.seed is not None:
        run_seed = int(args.seed)
        seed_source = "cli"
    elif args.seed_mode == "derived":
        run_seed = derived_seed
        seed_source = "derived"
    else:
        run_seed = None
        seed_source = "random"

    if args.shot_seed is not None:
        resolved_shot_seed = int(args.shot_seed)
        shot_seed_source = "cli"
    elif run_seed is not None:
        resolved_shot_seed = run_seed
        shot_seed_source = "run_seed"
    else:
        resolved_shot_seed = None
        shot_seed_source = "unseeded"
    backend_key = str(args.hw_backend).lower()
    ibm_instance_crn = args.ibm_instance_crn.strip() if args.ibm_instance_crn else None
    gate_estimate_requested = args.estimate_gates is not None
    gate_estimate_backend_spec = None
    gate_estimate_source = "not_requested"
    gate_estimate_backends = []
    gate_estimates = []
    workflow_backend_registry = {}

    if args.ibm_instance_crn is not None and not ibm_instance_crn:
        parser.error("--ibm-instance-crn must not be blank when provided.")

    try:
        gate_estimate_backend_spec, gate_estimate_source = _parse_gate_estimate_backend_spec(
            args.estimate_gates,
            args.hw_backend,
        )
    except ValueError as exc:
        parser.error(str(exc))

    try:
        print(f"{_console_prefix(args.replica_id)} Starting...")
        workflow_progress.start("Backend access")
        with timings.section(
            "backend_access",
            label="Backend access",
            metadata={
                "hw_backend": args.hw_backend,
                "gate_estimate_requested": gate_estimate_requested,
                "ibm_instance_crn_provided": bool(ibm_instance_crn),
            },
        ):
            hw_backend = get_hw_backend(
                args.hw_backend,
                shot_seed=resolved_shot_seed,
                ibm_instance_crn=ibm_instance_crn,
            )
            workflow_backend_registry = _resolve_workflow_backends(
                phase_schedule,
                inherited_spec=args.hw_backend,
                inherited_backend=hw_backend,
                shot_seed=resolved_shot_seed,
                ibm_instance_crn=ibm_instance_crn,
            )
            if gate_estimate_requested:
                _verify_gate_estimate_backend_access(
                    gate_estimate_backend_spec,
                    ibm_instance_crn,
                )
    except Exception as exc:
        parser.error(str(exc))

    if args.optimizer_angle_mode == "auto":
        optimizer_angle_mode = "statevector" if hw_backend is None else "sampler"
    elif args.optimizer_angle_mode == "statevector":
        optimizer_angle_mode = "statevector"
    else:
        optimizer_angle_mode = "sampler" if hw_backend is not None else "statevector"

    print(f"  Sequence       : {args.predict}")
    print(f"  Run label      : {run_label}")
    print(f"  Replica ID     : {args.replica_id}")
    print(f"  Forcefield     : {args.forcefield}")
    print(f"  Backend        : {args.hw_backend}")
    if ibm_instance_crn:
        print("  IBM instance   : CRN provided")
    print(f"  Derived seed   : {derived_seed}")
    print(f"  Seed mode      : {args.seed_mode}")
    print(f"  Run seed       : {_format_seed(run_seed)} ({seed_source})")
    print(f"  Shot seed      : {_format_seed(resolved_shot_seed)} ({shot_seed_source})")
    print(f"  Optimizer mode : {args.optimizer_angle_mode} -> {optimizer_angle_mode}")
    print(f"  Stop on error  : {'yes' if args.stop_on_phase_error else 'no'}")
    print(f"  Shots          : {shots}")
    print(f"  Max iter       : {args.maxiter}")
    print(f"  Phase preset   : {phase_schedule.preset}")
    print(f"  Phase source   : {phase_schedule.source}")
    print(f"  Basis batching : {phase_schedule.basis_circuit_batching}")
    print(
        f"  Scouting       : score={phase_schedule.scouting.score_model}, "
        f"backend={phase_schedule.scouting.backend}, shots={phase_schedule.scouting.shots}, "
        f"attempts={phase_schedule.scouting.attempts}"
    )
    print("  Phases         :")
    for phase in phase_schedule.phases:
        print(
            "    "
            f"{phase.name}: optimizer={phase.optimizer}, score={phase.score_model}, "
            f"optimizer_backend={phase.optimizer_backend}, readout_backend={phase.readout_backend}, "
            f"optimizer_shots={phase.optimizer_shots}, readout_shots={phase.readout_shots}, "
            f"maxiter={phase.maxiter}"
        )
    print(f"  Primary result : {phase_schedule.result.primary}")
    print(f"  Result score   : {phase_schedule.result.score_model}")
    print(f"  Readouts       : {len(phase_schedule.readouts)}")
    print(f"  Stored angles  : {','.join(stored_angles) if stored_angles else 'none'}")
    print(f"  Angle units    : {args.angle_units}")
    print(f"  Chi mode       : {args.chi_mode} (PHEAT side-chain templates)")
    print(f"  Max chi        : {'all' if max_chi is None else max_chi}")
    print(f"  QTF weight     : {args.qtf_energy_weight}")
    if gate_estimate_requested:
        print(f"  Gate estimates : {gate_estimate_backend_spec} ({gate_estimate_source})")

    pheat_reference = None
    workflow_progress.start("Reference loading")
    if args.reference_structure:
        try:
            with timings.section(
                "reference_load",
                label="Reference load",
                metadata={"source": args.reference_structure},
            ):
                pheat_reference = _load_pheat_reference(
                    args.reference_structure,
                    angle_units=args.angle_units,
                    stored_angles=stored_angles,
                    max_chi=max_chi,
                    include_terminal_oxt=args.include_terminal_oxt,
                )
                if pheat_reference.sequence != args.predict:
                    parser.error(
                        "--reference_structure sequence does not match --predict: "
                        f"{pheat_reference.sequence} != {args.predict}"
                    )
                print(
                    f"{_console_prefix(args.replica_id)} PHEAT reference loaded: "
                    f"{len(pheat_reference.residue_geometry.residues)} residues, "
                    f"{len(pheat_reference.structure.atoms)} reconstructed heavy atoms "
                    f"({pheat_reference.source_type})"
                )
        except Exception as exc:
            parser.error(str(exc))
    else:
        timings.skip("reference_load", label="Reference load")

    workflow_progress.start("Brickwork encoding")
    timings.start("brickwork_encoding")
    try:
        folder = PheatQuantumBiophysicsFolder(
            args.predict,
            force_field=args.forcefield,
            chi_mode=args.chi_mode,
            selective_chi_map=_selective_chi_map(),
            angle_units=args.angle_units,
            stored_angles=stored_angles,
            max_chi=max_chi,
            include_terminal_oxt=args.include_terminal_oxt,
            pheat_score_model=phase_schedule.result.score_model,
            qtf_energy_weight=args.qtf_energy_weight,
            bond_angle_encoding=args.bond_angle_encoding,
            tau_center_deg=args.tau_center_deg,
            tau_span_deg=args.tau_span_deg,
            theta_center_deg=args.theta_center_deg,
            theta_span_deg=args.theta_span_deg,
            optimizer_angle_mode=optimizer_angle_mode,
            optimizer_backend=hw_backend,
            optimizer_shots=shots,
            basis_circuit_batching=phase_schedule.basis_circuit_batching,
            reference_residue_geometry=(
                pheat_reference.residue_geometry if pheat_reference is not None else None
            ),
        )
    except Exception:
        timings.stop("brickwork_encoding", label="Brickwork encoding", status="error")
        raise
    brickwork_metadata = {
        "total_angles": folder.total_angles,
        "n_qubits": folder.n_qubits,
        "reps": folder.reps,
        "n_params": folder.n_params,
        "stored_angles": list(stored_angles),
        "chi_mode": args.chi_mode,
        "max_chi": "all" if max_chi is None else max_chi,
        "optimizer_angle_mode": optimizer_angle_mode,
        "optimizer_backend_mode": None if hw_backend is None else _backend_display_name(hw_backend),
        "phase_preset": phase_schedule.preset,
        "phase_source": phase_schedule.source,
        "phase_count": len(phase_schedule.phases),
        "basis_circuit_batching": phase_schedule.basis_circuit_batching,
        "scouting_score_model": phase_schedule.scouting.score_model,
        "result_score_model": phase_schedule.result.score_model,
        "primary_result": phase_schedule.result.primary,
        "readout_count": len(phase_schedule.readouts),
    }
    brickwork_record = timings.stop(
        "brickwork_encoding",
        label="Brickwork encoding",
        metadata=brickwork_metadata,
        print_elapsed=False,
    )
    folder.current_stage = 3
    print(f"{_console_prefix(args.replica_id)} Brickwork encoding complete")
    print(
        f"  Optimized DOFs : {folder.total_angles} angles, {folder.n_qubits} qubits, "
        f"{folder.reps} reps, {folder.n_params} params"
    )
    print(f"  Brickwork elapsed: {_format_elapsed(brickwork_record['elapsed_s'])}")

    workflow_progress.start("Gate estimates")
    if gate_estimate_requested:
        try:
            with timings.section(
                "gate_estimates",
                label="Gate estimates",
                metadata={"backend_spec": gate_estimate_backend_spec},
            ):
                backend_refs = _resolve_gate_estimate_backends(
                    gate_estimate_backend_spec,
                    folder.n_qubits,
                    ibm_instance_crn,
                )
                gate_estimates = _estimate_gate_costs(folder, backend_refs)
                gate_estimate_backends = sorted(
                    {_backend_display_name(ref["backend"]) for ref in backend_refs}
                )
                print(
                    f"{_console_prefix(args.replica_id)} Gate estimates computed for "
                    f"{', '.join(gate_estimate_backends)}"
                )
                for estimate in gate_estimates:
                    if estimate["circuit"] != "measurement_z" or estimate["optimization_level"] != 3:
                        continue
                    print(
                        "  "
                        f"{estimate['backend']} opt3 measurement_z: "
                        f"depth={estimate['all_gate_depth']}, "
                        f"multiQ_depth={estimate['multi_qubit_gate_depth']}, "
                        f"gates={estimate['all_gate_count']}, "
                        f"multiQ_gates={estimate['multi_qubit_gate_count']}"
                    )
        except Exception as exc:
            parser.error(f"--estimate-gates failed: {exc}")
    else:
        timings.skip("gate_estimates", label="Gate estimates")

    if optimizer_angle_mode == "sampler":
        if isinstance(hw_backend, StatevectorShotsBackend):
            seed_label = "seeded" if resolved_shot_seed is not None else "unseeded"
            print(
                f"{_console_prefix(args.replica_id)} Optimizer uses {seed_label} statevector-shots sampling."
            )
        elif backend_key in {"aer", "aer_simulator"}:
            print(f"{_console_prefix(args.replica_id)} Optimizer uses Aer shot-based sampling; this is slower than statevector.")
        else:
            print(
                f"{_console_prefix(args.replica_id)} WARNING: optimizer objective uses the selected backend; "
                "this may submit many sampled circuit jobs."
            )

    workflow_progress.start("Scouting initialization")
    with timings.section(
        "scouting_initialization",
        label="Scouting initialization",
        metadata={
            "attempts": phase_schedule.scouting.attempts,
            "seed": _format_seed(run_seed),
            "score_model": phase_schedule.scouting.score_model,
            "backend": phase_schedule.scouting.backend,
            "shots": phase_schedule.scouting.shots,
        },
    ):
        scouting_backend = _backend_from_registry(
            workflow_backend_registry,
            phase_schedule.scouting.backend,
            args.hw_backend,
        )
        folder.active_pheat_score_model = phase_schedule.scouting.score_model
        folder.optimizer_backend = scouting_backend
        folder.optimizer_angle_mode = folder._angle_mode_for_backend(scouting_backend)
        folder.optimizer_shots = phase_schedule.scouting.shots
        start_params = folder.get_smart_initialization(
            n_attempts=phase_schedule.scouting.attempts,
            seed=run_seed,
        )
    print(f"{_console_prefix(args.replica_id)} Init strategy: {strat}, seed: {_format_seed(run_seed)}")

    try:
        coords, labels, bonds, tracker, opt_params, final_energy = folder.fold(
            max_iter=args.maxiter,
            initial_params=start_params,
            hw_backend=hw_backend,
            shots=shots,
            phase_schedule=phase_schedule.phases,
            result_config=phase_schedule.result,
            readout_schedule=phase_schedule.readouts,
            backend_registry=workflow_backend_registry,
            inherited_backend_spec=args.hw_backend,
            timings=timings,
            workflow_progress=workflow_progress,
            pheat_reference_structure=pheat_reference.structure if pheat_reference is not None else None,
            stop_on_phase_error=args.stop_on_phase_error,
            initial_snapshot_score_model=phase_schedule.scouting.score_model,
            initial_snapshot_shots=phase_schedule.scouting.shots,
            initial_snapshot_backend_spec=phase_schedule.scouting.backend,
        )
    except PhaseOptimizationError as exc:
        print(f"{_console_prefix(args.replica_id)} {exc}")
        execution_context["console_output"] = console_capture.getvalue()
        console_log_path.write_text(execution_context["console_output"], encoding="utf-8")
        return 2

    final_structure = folder.structure_from_coords_labels(coords, labels)
    final_score = _safe_score_payload(final_structure, phase_schedule.result.score_model)
    final_residue_geometry = structure_to_residue_geometry(
        final_structure,
        angle_units=args.angle_units,
        stored_angles=stored_angles,
        max_chi=max_chi,
    )

    pdb_path = outdir / f"{prefix}.pdb"
    ca_pdb_path = outdir / f"{prefix}_ca.pdb"
    heavy_json_path = outdir / f"{prefix}_heavy.json"
    residue_geometry_path = outdir / f"{prefix}_residue_geometry.json"
    score_path = outdir / f"{prefix}_pheat_score.json"
    tracker_path = outdir / f"{prefix}_tracker.json"
    result_path = outdir / f"{prefix}_result.json"
    reference_pdb_path = None
    reference_residue_geometry_path = None
    structure_snapshots = list(getattr(folder, "structure_snapshots", []) or [])
    structure_snapshot_payloads = []
    snapshot_structures = {}

    workflow_progress.start("Artifact generation")
    with timings.section(
        "artifact_writes",
        label="Artifact writes",
        metadata={"outdir": str(outdir)},
    ):
        write_pdb(final_structure, pdb_path)
        write_heavy_json(final_structure, heavy_json_path)
        write_residue_geometry_json(
            final_residue_geometry,
            residue_geometry_path,
            stored_angles=stored_angles,
            max_chi=max_chi,
        )
        _write_json(score_path, final_score)

        if pheat_reference is not None:
            reference_pdb_path = outdir / f"{prefix}_reference_pheat.pdb"
            reference_residue_geometry_path = outdir / f"{prefix}_reference_residue_geometry.json"
            write_pdb(pheat_reference.structure, reference_pdb_path)
            write_residue_geometry_json(
                pheat_reference.residue_geometry,
                reference_residue_geometry_path,
                stored_angles=stored_angles,
                max_chi=max_chi,
            )

        pred_ca = _ca_coords_from_structure(final_structure)
        folder.save_reduced_pdb(pred_ca, filename=ca_pdb_path, energy=final_energy)

        for snapshot in structure_snapshots:
            payload = _snapshot_payload(snapshot)
            key = str(payload.get("key") or _slug(payload.get("label") or payload.get("role") or "structure"))
            payload["key"] = key
            structure = snapshot.get("structure")
            raw_pdb_path = None
            if payload.get("snapshot_status") == "ok" and structure is not None:
                raw_pdb_path = outdir / f"{_snapshot_file_stem(prefix, payload)}.pdb"
                write_pdb(structure, raw_pdb_path)
            if raw_pdb_path is not None:
                payload["pdb_path"] = str(raw_pdb_path)
                payload["viewer_pdb_path"] = str(raw_pdb_path)
            if structure is not None:
                snapshot_structures[key] = structure
                payload["atom_count"] = len(structure.atoms)
            structure_snapshot_payloads.append(payload)

    phase_results = getattr(folder, "phase_results", [])
    readout_results = list(getattr(folder, "readout_results", []) or [])
    primary_vs_last_phase_rmsd = None
    primary_vs_last_phase_atom_count = None
    primary_vs_last_phase_details = None
    last_phase_snapshot = next(
        (
            snapshot
            for snapshot in reversed(structure_snapshot_payloads)
            if snapshot.get("role") == "phase"
            and snapshot.get("snapshot_status") == "ok"
            and snapshot_structures.get(snapshot.get("key")) is not None
        ),
        None,
    )
    if last_phase_snapshot is not None:
        try:
            primary_vs_last_phase_details = _pheat_alignment_details(
                snapshot_structures[last_phase_snapshot["key"]],
                final_structure,
            )
            primary_vs_last_phase_rmsd = primary_vs_last_phase_details["all_heavy_rmsd"]
            primary_vs_last_phase_atom_count = primary_vs_last_phase_details["matched_heavy_atoms"]
            print(
                f"{_console_prefix(args.replica_id)} Primary vs last phase RMSD: "
                f"{primary_vs_last_phase_rmsd:.3f} A "
                f"({primary_vs_last_phase_atom_count} PHEAT heavy atoms)"
            )
        except Exception as exc:
            primary_vs_last_phase_details = {"status": "unavailable", "error": str(exc)}
    for readout_result in readout_results:
        readout_structure = snapshot_structures.get(readout_result.get("snapshot_key"))
        if readout_structure is None:
            continue
        try:
            drift_details = _pheat_alignment_details(final_structure, readout_structure)
            readout_result["primary_drift_rmsd"] = drift_details["all_heavy_rmsd"]
            readout_result["primary_drift_atom_count"] = drift_details["matched_heavy_atoms"]
            readout_result["primary_drift_details"] = drift_details
        except Exception as exc:
            readout_result["primary_drift_details"] = {"status": "unavailable", "error": str(exc)}
    landscape_path = outdir / f"{prefix}_landscape.png"
    interactive_landscape_path = outdir / f"{prefix}_landscape_interactive.html"
    timings.start("landscape_plot")
    try:
        from qtf_landscape_viz import plot_energy_landscape

        plot_energy_landscape(
            tracker,
            sequence=args.predict,
            forcefield=args.forcefield,
            save_path=landscape_path,
            show=False,
            title_extra=f"{run_label} | PHEAT {phase_schedule.result.score_model}",
        )
        timings.stop("landscape_plot", label="Landscape plot")
    except Exception as exc:
        timings.stop(
            "landscape_plot",
            label="Landscape plot",
            status="error",
            metadata={"error": str(exc)},
        )
        print(f"{_console_prefix(args.replica_id)} Landscape plot failed: {exc}")

    timings.start("interactive_landscape_plot")
    try:
        from qtf_landscape_viz import plot_energy_landscape_interactive

        plot_energy_landscape_interactive(
            tracker,
            sequence=args.predict,
            forcefield=args.forcefield,
            save_path=interactive_landscape_path,
            title_extra=f"{run_label} | PHEAT {phase_schedule.result.score_model}",
            phase_results=phase_results,
            include_plotlyjs="directory",
            full_html=True,
        )
        timings.stop(
            "interactive_landscape_plot",
            label="Interactive landscape plot",
            metadata={"interactive_landscape_path": str(interactive_landscape_path)},
        )
    except Exception as exc:
        interactive_landscape_path = None
        timings.stop(
            "interactive_landscape_plot",
            label="Interactive landscape plot",
            status="error",
            metadata={"error": str(exc)},
        )
        print(f"{_console_prefix(args.replica_id)} Interactive landscape plot failed: {exc}")

    rmsd = None
    rmsd_atom_count = None
    rmsd_reference_atom_count = None
    rmsd_predicted_atom_count = None
    rmsd_details = None
    pheat_rg_payload = None
    t_e2e = None
    t_rg = None
    report_path = None
    reference_aligned_pdb_path = None
    folded_aligned_pdb_path = None
    report_error = None
    molstar_vendor_path = None
    molstar_error = None
    workflow_progress.start("Evaluation and report")
    if pheat_reference is not None:
        with timings.section(
            "diagnostics_rmsd",
            label="Diagnostic RMSD",
            metadata={"reference": str(pheat_reference.source_path)},
        ):
            try:
                rmsd_details = _pheat_alignment_details(pheat_reference.structure, final_structure)
                rmsd = rmsd_details["all_heavy_rmsd"]
                rmsd_atom_count = rmsd_details["matched_heavy_atoms"]
                rmsd_reference_atom_count = rmsd_details["reference_atom_count"]
                rmsd_predicted_atom_count = rmsd_details["target_atom_count"]
                reference_ca = _ca_coords_from_structure(pheat_reference.structure)
                t_e2e, t_rg = _physics_metrics(reference_ca)
                print(
                    f"{_console_prefix(args.replica_id)} Primary RMSD: {rmsd:.3f} A "
                    f"({rmsd_atom_count} PHEAT heavy atoms)"
                )
            except Exception as exc:
                print(f"{_console_prefix(args.replica_id)} Primary RMSD failed: {exc}")
            reference_rg = _safe_pheat_radius_of_gyration(pheat_reference.structure)
            final_rg = _safe_pheat_radius_of_gyration(final_structure)
            pheat_rg_payload = {
                "reference": reference_rg,
                "final": final_rg,
                "delta_final_minus_reference": _safe_pheat_radius_of_gyration_delta(
                    reference_rg,
                    final_rg,
                ),
            }
    else:
        timings.skip("diagnostics_rmsd", label="Diagnostic RMSD")
        final_rg = _safe_pheat_radius_of_gyration(final_structure)
        unavailable_reference_rg = _safe_pheat_radius_of_gyration(None)
        pheat_rg_payload = {
            "reference": unavailable_reference_rg,
            "final": final_rg,
            "delta_final_minus_reference": _safe_pheat_radius_of_gyration_delta(
                unavailable_reference_rg,
                final_rg,
            ),
        }

    p_e2e, p_rg = _physics_metrics(pred_ca)
    phase_rmsds = getattr(folder, "phase_rmsds", {})
    phase_rmsd_details = getattr(folder, "phase_rmsd_details", {})
    energy_trace = _build_energy_trace(tracker, phase_results)
    software_versions_full = _collect_software_versions()
    software_versions_summary = _software_summary_payload(
        software_versions_full,
        software_versions_path,
    )

    report_path = outdir / f"{prefix}_report.html"
    folded_aligned_pdb_path = pdb_path
    timings.start("report_generation")
    try:
        try:
            if pheat_reference is not None:
                reference_aligned_pdb_path = outdir / f"{prefix}_reference_pheat_aligned.pdb"
                folded_aligned_pdb_path = outdir / f"{prefix}_folded_aligned.pdb"
                aligned_reference, aligned_folded, _aligned_atom_count = _aligned_pheat_structures(
                    pheat_reference.structure,
                    final_structure,
                )
                write_pdb(aligned_reference, reference_aligned_pdb_path)
                write_pdb(aligned_folded, folded_aligned_pdb_path)
                for snapshot in structure_snapshot_payloads:
                    if snapshot.get("snapshot_status") != "ok":
                        continue
                    key = snapshot.get("key")
                    structure = snapshot_structures.get(key)
                    if structure is None:
                        continue
                    aligned_snapshot_path = outdir / f"{_snapshot_file_stem(prefix, snapshot)}_aligned.pdb"
                    _snapshot_reference, aligned_snapshot, aligned_atom_count = _aligned_pheat_structures(
                        pheat_reference.structure,
                        structure,
                    )
                    write_pdb(aligned_snapshot, aligned_snapshot_path)
                    snapshot["aligned_pdb_path"] = str(aligned_snapshot_path)
                    snapshot["viewer_pdb_path"] = str(aligned_snapshot_path)
                    snapshot["aligned_matched_heavy_atoms"] = aligned_atom_count
            molstar_dir, molstar_error = _copy_molstar_assets(outdir)
            molstar_vendor_path = molstar_dir
        except Exception as exc:
            report_error = str(exc)
            print(f"{_console_prefix(args.replica_id)} Report preparation failed: {exc}")
    finally:
        timings.stop(
            "report_generation",
            label="Report generation",
            status="error" if report_error else "ok",
            metadata={"report_path": str(report_path), "error": report_error} if report_error else {"report_path": str(report_path)},
        )

    backend_display = None if hw_backend is None else _backend_display_name(hw_backend)
    rmsd_angle_mode = getattr(folder, "rmsd_angle_mode", "statevector" if hw_backend is None else "sampler")
    primary_angle_mode = getattr(folder, "primary_angle_mode", rmsd_angle_mode)
    primary_backend_mode = getattr(folder, "primary_backend_mode", None)
    basis_circuit_batching_stats = dict(getattr(folder, "basis_circuit_batching_stats", {}) or {})
    total_record = timings.stop("total_run", label="Total run", print_elapsed=False)
    runtime_s = total_record["elapsed_s"]
    timing_payload = timings.as_dict()

    result = {
        "replica_id": args.replica_id,
        "run_label": run_label,
        "seed": run_seed,
        "derived_seed": derived_seed,
        "run_seed": run_seed,
        "seed_source": seed_source,
        "seed_mode": args.seed_mode,
        "stop_on_phase_error": bool(args.stop_on_phase_error),
        "init_type": strat,
        "sequence": args.predict,
        "forcefield": args.forcefield,
        "energy": float(final_energy),
        "objective_total": float(final_energy),
        "phase_preset": phase_schedule.preset,
        "phase_source": phase_schedule.source,
        "phase_config_path": phase_schedule.config_path,
        "phase_schedule": _phase_schedule_payload(phase_schedule),
        "basis_circuit_batching_requested": phase_schedule.basis_circuit_batching,
        "basis_circuit_batching_effective": basis_circuit_batching_stats.get("last_effective"),
        "basis_circuit_batching_stats": basis_circuit_batching_stats,
        "phase_results": phase_results,
        "phase_rmsds": phase_rmsds,
        "phase_rmsd_details": phase_rmsd_details,
        "result_config": asdict(phase_schedule.result),
        "primary_result": getattr(folder, "primary_result", None),
        "readout_schedule": [asdict(readout) for readout in phase_schedule.readouts],
        "readout_results": readout_results,
        "structure_snapshots": structure_snapshot_payloads,
        "primary_vs_last_phase_rmsd": primary_vs_last_phase_rmsd,
        "primary_vs_last_phase_atom_count": primary_vs_last_phase_atom_count,
        "primary_vs_last_phase_details": primary_vs_last_phase_details,
        "energy_trace": energy_trace,
        "scouting_config": asdict(phase_schedule.scouting),
        "pheat_score_model": phase_schedule.result.score_model,
        "result_score_model": phase_schedule.result.score_model,
        "pheat_score_status": final_score.get("status"),
        "pheat_score_total": final_score.get("total"),
        "pheat_score_units": final_score.get("units"),
        "pheat_score_terms": final_score.get("terms"),
        "qtf_energy_weight": args.qtf_energy_weight,
        "qtf_supplemental_total": getattr(folder, "last_energy_terms", {}).get("qtf_total"),
        "angle_units": args.angle_units,
        "stored_angles": list(stored_angles),
        "chi_source": "pheat.sidechain_steps",
        "chi_mode": args.chi_mode,
        "max_chi": max_chi,
        "include_terminal_oxt": args.include_terminal_oxt,
        "bond_angle_encoding": args.bond_angle_encoding,
        "tau_center_deg": args.tau_center_deg,
        "tau_span_deg": args.tau_span_deg,
        "theta_center_deg": args.theta_center_deg,
        "theta_span_deg": args.theta_span_deg,
        "total_angles": folder.total_angles,
        "n_qubits": folder.n_qubits,
        "reps": folder.reps,
        "n_params": folder.n_params,
        "rmsd_to_reference": rmsd,
        "reference_available": pheat_reference is not None and reference_aligned_pdb_path is not None,
        "rmsd_mode": "pheat_heavy_atom" if pheat_reference is not None else None,
        "rmsd_reference_source": str(pheat_reference.source_path) if pheat_reference is not None else None,
        "rmsd_reference_source_type": pheat_reference.source_type if pheat_reference is not None else None,
        "rmsd_reference_geometry_path": (
            str(reference_residue_geometry_path) if reference_residue_geometry_path is not None else None
        ),
        "rmsd_reference_pdb_path": str(reference_pdb_path) if reference_pdb_path is not None else None,
        "rmsd_atom_count": rmsd_atom_count,
        "rmsd_reference_atom_count": rmsd_reference_atom_count,
        "rmsd_predicted_atom_count": rmsd_predicted_atom_count,
        "rmsd_details": rmsd_details,
        "pheat_radius_of_gyration": pheat_rg_payload,
        "pred_e2e_A": float(p_e2e),
        "pred_rg_A": float(p_rg),
        "ref_e2e_A": float(t_e2e) if t_e2e is not None else None,
        "ref_rg_A": float(t_rg) if t_rg is not None else None,
        "runtime_s": round(runtime_s, 2),
        "timings": timing_payload,
        "execution": dict(execution_context),
        "command_line": command_line,
        "console_output_path": str(console_log_path),
        "software_versions": software_versions_summary,
        "software_versions_path": str(software_versions_path),
        "hw_backend": args.hw_backend,
        "hw_backend_resolved": backend_display,
        "shots": shots,
        "hw_shot_seed": resolved_shot_seed,
        "shot_seed_source": shot_seed_source,
        "optimizer_angle_mode_requested": args.optimizer_angle_mode,
        "optimizer_angle_mode": optimizer_angle_mode,
        "optimization_angle_mode": optimizer_angle_mode,
        "optimizer_backend_mode": backend_display if optimizer_angle_mode == "sampler" else None,
        "rmsd_angle_mode": rmsd_angle_mode,
        "primary_angle_mode": primary_angle_mode,
        "primary_backend_mode": primary_backend_mode,
        "rmsd_backend_mode": backend_display if rmsd_angle_mode == "sampler" else None,
        "ibm_instance_crn_provided": bool(ibm_instance_crn),
        "gate_estimate_requested": gate_estimate_requested,
        "gate_estimate_backend_spec": gate_estimate_backend_spec,
        "gate_estimate_source": gate_estimate_source,
        "gate_estimate_backends": gate_estimate_backends,
        "gate_estimates": gate_estimates,
        "pdb_path": str(pdb_path),
        "ca_pdb_path": str(ca_pdb_path),
        "heavy_json_path": str(heavy_json_path),
        "residue_geometry_path": str(residue_geometry_path),
        "score_path": str(score_path),
        "reference_pheat_pdb_path": str(reference_pdb_path) if reference_pdb_path is not None else None,
        "reference_residue_geometry_path": (
            str(reference_residue_geometry_path) if reference_residue_geometry_path is not None else None
        ),
        "reference_aligned_pdb_path": (
            str(reference_aligned_pdb_path) if reference_aligned_pdb_path is not None else None
        ),
        "folded_aligned_pdb_path": str(folded_aligned_pdb_path) if folded_aligned_pdb_path is not None else None,
        "folded_viewer_pdb_path": str(folded_aligned_pdb_path) if folded_aligned_pdb_path is not None else None,
        "report_path": str(report_path) if report_path is not None else None,
        "report_error": report_error,
        "molstar_vendor_path": str(molstar_vendor_path) if molstar_vendor_path is not None else None,
        "molstar_error": molstar_error,
        "tracker_path": str(tracker_path),
        "landscape_path": str(landscape_path),
        "interactive_landscape_path": (
            str(interactive_landscape_path) if interactive_landscape_path is not None else None
        ),
    }

    print(f"\n{_console_prefix(args.replica_id)} Done")
    print(f"  Objective  : {final_energy:.4f}")
    print(f"  PHEAT score: {final_score.get('total')} {final_score.get('units')}")
    print(
        f"  RMSD primary: {rmsd:.3f} A ({rmsd_atom_count} PHEAT heavy atoms)"
        if rmsd is not None
        else "  RMSD primary: N/A"
    )
    print(f"  Runtime    : {runtime_s:.1f}s")
    if report_path is not None and report_error is None:
        print(f"  Report     : {report_path}")
    print(f"  Saved to   : {result_path}")

    console_output = console_capture.getvalue()
    execution_context["console_output"] = console_output
    result["execution"] = dict(execution_context)
    _write_json(software_versions_path, software_versions_full)
    console_log_path.write_text(console_output, encoding="utf-8")

    if (
        report_path is not None
        and folded_aligned_pdb_path is not None
        and report_error is None
    ):
        try:
            _write_pheat_html_report(
                report_path,
                sequence=args.predict,
                result=result,
                final_score=final_score,
                pheat_reference=pheat_reference,
                reference_aligned_pdb_path=reference_aligned_pdb_path,
                folded_aligned_pdb_path=folded_aligned_pdb_path,
                molstar_dir=molstar_vendor_path,
                molstar_error=molstar_error if molstar_vendor_path is None else None,
            )
        except Exception as exc:
            report_error = str(exc)
            result["report_error"] = report_error
            print(f"{_console_prefix(args.replica_id)} HTML report failed: {exc}")
            execution_context["console_output"] = console_capture.getvalue()
            result["execution"] = dict(execution_context)
            console_log_path.write_text(execution_context["console_output"], encoding="utf-8")

    _write_json(
        tracker_path,
        {
            "history": tracker.history,
            "phase_markers": [[m[0], m[1]] for m in tracker.phase_markers],
            "current_iter": tracker.current_iter,
            "energy_trace": energy_trace,
            "timings": timing_payload,
        },
    )
    _write_json(result_path, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
