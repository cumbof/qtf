#!/usr/bin/env python3
"""QTF single-replica runner using PHEAT geometry.

This executable keeps QTF's hybrid quantum/classical optimization flow while
using PHEAT residue geometry as the authoritative structure encoding. Score
models may be QTF-native or provided by PHEAT.
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
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from scipy.optimize import minimize

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _ensure_python_environment_bin_on_path() -> None:
    bin_dir = str(Path(sys.executable).resolve().parent)
    existing = os.environ.get("PATH", "")
    entries = [entry for entry in existing.split(os.pathsep) if entry]
    if bin_dir not in entries:
        os.environ["PATH"] = os.pathsep.join([bin_dir, *entries])


_ensure_python_environment_bin_on_path()

from qtf.core.circuits import DEFAULT_CIRCUIT_TEMPLATE, build_circuit
from qtf.core.folder import (
    TRANSPILE_OPTIMIZATION_LEVELS,
    QuantumBiophysicsFolder,
)
from qtf.metrics import (
    DEFAULT_RMSD_ALIGNMENT_ATOM_SET,
    METRIC_ATOM_SETS,
    PRIMARY_RMSD_ATOM_SET,
    normalize_metric_atom_sets,
    normalize_rmsd_alignment_atom_set,
    radius_of_gyration_delta_summary,
    radius_of_gyration_summary,
    structure_metric_summary,
)
from qtf.scoring import (
    canonical_score_model,
    is_qtf_score_model,
    pheat_score_model_capabilities,
    score_pheat_structure,
)

try:
    from pheat import (
        Atom,
        collect_software_provenance,
        SCORING_DOMAINS,
        HeavyAtomStructure,
        ResidueGeometry,
        ResidueGeometryStructure,
        filter_structure_for_domain,
        load_structure_json,
        load_pdb,
        normalize_domain,
        structure_to_residue_geometry,
        write_pdb,
        write_structure_json,
    )
    from pheat.metrics import align_structure_to_reference
    from pheat.geometry import radius_of_gyration as pheat_radius_of_gyration
    from pheat.residue_geometry import (
        ANGLE_CA_C_N,
        ANGLE_N_CA_C,
        ANGLE_UNITS,
        load_residue_geometry,
        structure_from_residue_geometry,
        write_residue_geometry_json,
    )
    from pheat.residues import CANONICAL_RESIDUES, three_to_one
    from pheat.roundtrip import normalize_max_chi, normalize_stored_angles
except ImportError as exc:
    raise SystemExit(
        "qtf fold requires PHEAT to be importable in the active Python "
        "environment. Install PHEAT into this environment before running."
    ) from exc


PHEAT_FAILURE_PENALTY = 1.0e12
OPTIONAL_PHEAT_ANGLES = ("omega", "tau", "theta")
DEFAULT_RESULT_SCORE_MODEL = "pheat-generic"
DEFAULT_REPORT_STRUCTURE_DOMAIN = "protein-heavy"
BASIS_CIRCUIT_BATCHING_MODES = ("auto", "on", "off")
GATE_ESTIMATE_SELECTED_BACKEND = "__selected_backend__"
DEFAULT_GATE_ESTIMATE_OPTIMIZATION_LEVELS = (0, 3)
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
PHASE_PRESET_DIR = PACKAGE_ROOT / "assets" / "recipes"
PHASE_PRESET_SCHEMA_PATH = PHASE_PRESET_DIR / "schema.json"
INHERIT_BACKEND_VALUES = {"inherit", "default", ""}
PRIMARY_LAST_PHASE = "last_phase_structure"
REPORT_ENVIRONMENT_KEYS = (
    "MPLCONFIGDIR",
    "CONDA_PREFIX",
    "VIRTUAL_ENV",
    "PYTHONPATH",
)
IBM_RUNTIME_CHANNELS = ("ibm_quantum_platform", "ibm_cloud")
IBM_RUNTIME_DEFAULT_TOKEN_CHANNEL = "ibm_quantum_platform"
SENSITIVE_COMMAND_OPTIONS = frozenset({"--ibm-token", "--ibm-instance-crn"})
BASE_SOFTWARE_PACKAGES = (
    {"name": "numpy", "role": "core numeric arrays", "required": True},
    {"name": "scipy", "role": "classical optimization", "required": True},
    {"name": "PyYAML", "role": "recipe loading", "required": True},
    {"name": "jsonschema", "role": "recipe validation", "required": True},
    {"name": "plotly", "role": "interactive report plots", "required": True},
    {"name": "qiskit", "role": "quantum circuit construction", "required": True},
)
OPTIONAL_QUANTUM_PACKAGES = {
    "qiskit-aer": "Aer simulator and Aer gate estimates",
    "qiskit-ibm-runtime": "IBM Quantum backend access and estimates",
}
TIMING_SECTION_ORDER = (
    "backend_access",
    "reference_load",
    "circuit_construction",
    "gate_estimates",
    "scouting_initialization",
    "__optimizer_phases__",
    "optional_readouts",
    "primary_result_selection",
    "artifact_writes",
    "phase_gate_estimates",
    "external_validation",
    "landscape_plot",
    "interactive_landscape_plot",
    "diagnostics_rmsd",
    "report_generation",
    "total_run",
)
TIMING_SECTION_LABELS = {
    "backend_access": "Backend access",
    "reference_load": "Reference load",
    "circuit_construction": "Circuit construction",
    "gate_estimates": "Gate estimates",
    "scouting_initialization": "Scouting initialization",
    "optional_readouts": "Optional readouts",
    "primary_result_selection": "Primary result selection",
    "artifact_writes": "Artifact writes",
    "phase_gate_estimates": "Phase gate estimates",
    "external_validation": "External validation",
    "landscape_plot": "Landscape plot",
    "interactive_landscape_plot": "Interactive landscape plot",
    "diagnostics_rmsd": "Diagnostic RMSD",
    "report_generation": "Report generation",
    "total_run": "Total run",
}

WORKFLOW_STEP_LABELS = (
    "Backend access",
    "Reference loading",
    "Circuit construction",
    "Gate estimates",
    "Scouting initialization",
    "Optimization phases",
    "Optional readouts",
    "Primary result selection",
    "Artifact generation",
    "Phase gate estimates",
    "External validation",
    "Evaluation and report",
)
STATUS_UPDATE_INTERVAL_S = 30.0
STATUS_EVAL_INTERVAL = 25
STATUS_CONSOLE_HEARTBEAT_INTERVAL_S = 60.0


def _console_prefix(replica_id: int) -> str:
    return f"[QTF run {replica_id}]"


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
    def __init__(
        self,
        replica_id: int,
        labels: Sequence[str] = WORKFLOW_STEP_LABELS,
        status_writer=None,
    ):
        self.replica_id = replica_id
        self.labels = tuple(labels)
        self.total = len(self.labels)
        self.current = 0
        self.status_writer = status_writer

    def start(self, label: str) -> None:
        self.current += 1
        print(f"\n{_console_prefix(self.replica_id)} Step {self.current}/{self.total}: {label}")
        if self.status_writer is not None:
            self.status_writer.update(
                status="running",
                step={
                    "index": self.current,
                    "total": self.total,
                    "label": label,
                },
                force=True,
                flush_console=True,
            )


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


class RunStatusWriter:
    """Small live status artifact for long-running single-replica jobs."""

    def __init__(
        self,
        path: Path,
        *,
        replica_id: int,
        run_label: str,
        command_line: str,
        console_output_path: Path,
        flush_console=None,
    ):
        self.path = Path(path)
        self.started_at_epoch_s = time.time()
        self.flush_console = flush_console
        self.payload: dict[str, Any] = {
            "status": "starting",
            "replica_id": int(replica_id),
            "run_label": run_label,
            "command_line": command_line,
            "console_output_path": str(console_output_path),
            "started_at_epoch_s": self.started_at_epoch_s,
            "started_at": self._timestamp(self.started_at_epoch_s),
            "updated_at_epoch_s": self.started_at_epoch_s,
            "updated_at": self._timestamp(self.started_at_epoch_s),
            "elapsed_s": 0.0,
        }

    @staticmethod
    def _timestamp(epoch_s: float) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch_s))

    @staticmethod
    def _json_default(value):
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        return str(value)

    def update(
        self,
        *,
        status: Optional[str] = None,
        force: bool = False,
        flush_console: bool = False,
        **fields,
    ) -> None:
        now = time.time()
        if status is not None:
            self.payload["status"] = str(status)
        self.payload.update(fields)
        self.payload["updated_at_epoch_s"] = now
        self.payload["updated_at"] = self._timestamp(now)
        self.payload["elapsed_s"] = _round_seconds(now - self.started_at_epoch_s)
        if force:
            self.write(flush_console=flush_console)

    def write(self, *, flush_console: bool = False) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_name(f"{self.path.name}.tmp")
        text = json.dumps(self.payload, indent=4, default=self._json_default)
        tmp_path.write_text(text + "\n", encoding="utf-8")
        tmp_path.replace(self.path)
        if flush_console and self.flush_console is not None:
            self.flush_console()


@dataclass
class PheatReference:
    source_path: Path
    source_type: str
    residue_geometry: ResidueGeometryStructure
    structure: HeavyAtomStructure
    metric_residue_geometry: ResidueGeometryStructure
    metric_structure: HeavyAtomStructure
    sequence: str
    source_domain_coverage: Optional[dict[str, Any]] = None


@dataclass
class TranspileConfig:
    optimization_level: Optional[int]
    seed: Optional[int]
    description: Optional[str] = None


@dataclass(frozen=True)
class IBMRuntimeAuthConfig:
    account_name: Optional[str] = None
    token: Optional[str] = None
    token_source: str = "none"
    channel: Optional[str] = None
    instance_crn: Optional[str] = None
    url: Optional[str] = None


@dataclass
class ScoutingConfig:
    score_model: str
    backend: str
    shots: int
    attempts: int
    score_options: dict[str, Any]
    transpile: TranspileConfig
    description: Optional[str] = None


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
    score_options: dict[str, Any]
    geometry: dict[str, Any]
    handoff_guard: dict[str, Any]
    optimizer_transpile: TranspileConfig
    readout_transpile: TranspileConfig
    description: Optional[str] = None


@dataclass
class ResultConfig:
    primary: str
    score_model: str
    score_options: dict[str, Any]
    description: Optional[str] = None


@dataclass
class ReadoutConfig:
    name: str
    source: str
    backend: str
    shots: int
    score_model: str
    primary: bool
    transpile: TranspileConfig
    description: Optional[str] = None


@dataclass
class MetricsConfig:
    atom_sets: list[str]
    rmsd_alignment_atom_set: str
    description: Optional[str] = None


@dataclass
class ReportConfig:
    structure_domain: str
    description: Optional[str] = None


@dataclass
class EvaluatorConfig:
    name: str
    score_model: str
    required: bool
    options: dict[str, Any]
    description: Optional[str] = None


@dataclass
class PhaseComparisonConfig:
    enabled: bool
    evaluators: list[str]
    compare: str
    affect_selection: bool
    description: Optional[str] = None


@dataclass
class RerankingConfig:
    enabled: bool
    evaluator: Optional[str]
    triggers: list[dict[str, Any]]
    candidate_pool: dict[str, Any]
    apply: str
    description: Optional[str] = None


@dataclass
class PhaseReadinessConfig:
    enabled: bool
    evaluator: Optional[str]
    phases: list[str]
    on_fail: str
    max_clash_count: Optional[int]
    max_short_contact_count: Optional[int]
    min_nonlocal_distance_a: Optional[float]
    description: Optional[str] = None


@dataclass
class HandoffGuardConfig:
    enabled: bool
    evaluator: Optional[str]
    phases: list[str]
    fallback: str
    abort_on_reject: bool
    allow_improving_unsafe: bool
    max_clash_count: Optional[int]
    max_short_contact_count: Optional[int]
    min_nonlocal_distance_a: Optional[float]
    unsafe_transition_max_short_contact_count: Optional[int]
    unsafe_transition_min_nonlocal_distance_a: Optional[float]
    unsafe_transition_require_clash_count_decrease: bool
    reject_on_score_worse: bool
    reject_on_clash_count_increase: bool
    reject_on_short_contact_count_increase: bool
    reject_on_min_nonlocal_distance_decrease: bool
    reject_on_nonfinite: bool
    description: Optional[str] = None


@dataclass
class ValidationConfig:
    enabled: bool
    candidates: list[str]
    evaluators: list[str]
    description: Optional[str] = None


@dataclass
class PhaseSchedule:
    preset: str
    source: str
    config_path: Optional[str]
    fold: dict[str, Any]
    basis_circuit_batching: str
    circuit_template: Optional[dict[str, Any]]
    circuit: Optional[dict[str, Any]]
    scouting: ScoutingConfig
    phases: list[PhaseConfig]
    result: ResultConfig
    readouts: list[ReadoutConfig]
    metrics: MetricsConfig
    report: ReportConfig
    evaluators: dict[str, EvaluatorConfig]
    phase_comparisons: PhaseComparisonConfig
    reranking: RerankingConfig
    phase_readiness: PhaseReadinessConfig
    handoff_guard: HandoffGuardConfig
    validation: ValidationConfig
    default_transpile: TranspileConfig
    gate_estimate_optimization_levels: list[int]
    gate_estimate_transpile_seed: Optional[int]
    description: Optional[str] = None


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


def _report_structure_domain(value: Optional[str]) -> str:
    return normalize_domain(value or DEFAULT_REPORT_STRUCTURE_DOMAIN)


def _filter_structure_for_report_domain(
    structure: HeavyAtomStructure,
    domain: str,
) -> tuple[HeavyAtomStructure, dict[str, Any]]:
    filtered, coverage = filter_structure_for_domain(structure, domain=domain)
    return filtered, dict(coverage)


def _write_report_pdb(
    structure: HeavyAtomStructure,
    path: Path,
    *,
    domain: str,
) -> tuple[HeavyAtomStructure, dict[str, Any]]:
    filtered, coverage = _filter_structure_for_report_domain(structure, domain)
    write_pdb(structure, path, domain=domain)
    return filtered, coverage


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


def _package_component(name: str, role: str, *, required: bool, selected: bool = True) -> dict[str, Any]:
    version = _distribution_version(name)
    if not selected:
        status = "not_selected"
    elif version:
        status = "available"
    else:
        status = "missing"
    return {
        "name": name,
        "version": version,
        "role": role,
        "required": bool(required),
        "selected": bool(selected),
        "status": status,
    }


def _merge_package_component(rows: dict[str, dict[str, Any]], component: dict[str, Any]) -> None:
    name = str(component.get("name") or "")
    if not name:
        return
    existing = rows.get(name)
    if existing is None:
        rows[name] = component
        return
    existing["required"] = bool(existing.get("required")) or bool(component.get("required"))
    existing["selected"] = bool(existing.get("selected")) or bool(component.get("selected"))
    if existing.get("status") == "not_selected" and component.get("status") != "not_selected":
        existing["status"] = component.get("status")
    elif component.get("status") == "available":
        existing["status"] = "available"
    roles = [str(part) for part in str(existing.get("role") or "").split("; ") if part]
    new_role = str(component.get("role") or "")
    if new_role and new_role not in roles:
        roles.append(new_role)
    existing["role"] = "; ".join(roles)


def _pheat_raw_score_models_for_public(public_models: Sequence[str]) -> list[str]:
    capabilities = _pheat_capabilities_by_public_name()
    raw_models = []
    for model in public_models:
        public_model = str(model or "").strip()
        if not public_model or public_model not in capabilities:
            continue
        raw_model = _pheat_raw_model_name(public_model)
        if raw_model:
            raw_models.append(raw_model)
    return list(dict.fromkeys(raw_models))


def _merge_external_tool_component(rows: dict[str, dict[str, Any]], component: Mapping[str, Any]) -> None:
    name = str(component.get("name") or "")
    if not name:
        return
    existing = rows.get(name)
    if existing is None:
        rows[name] = dict(component)
        return
    existing["required"] = bool(existing.get("required")) or bool(component.get("required"))
    if component.get("status") == "available":
        existing["status"] = "available"
        existing["path"] = component.get("path")
        existing["version"] = component.get("version")
        existing["details"] = component.get("details")
    elif not existing.get("details") and component.get("details"):
        existing["details"] = component.get("details")
    roles = [str(part) for part in str(existing.get("role") or "").split("; ") if part]
    new_role = str(component.get("role") or "")
    if new_role and new_role not in roles:
        roles.append(new_role)
    existing["role"] = "; ".join(roles)


def _selected_score_models_for_schedule(schedule: PhaseSchedule) -> list[str]:
    selected: list[str] = []

    def add(model: Optional[str]) -> None:
        if model is None:
            return
        value = str(model).strip()
        if value and value not in selected:
            selected.append(value)

    add(schedule.scouting.score_model)
    add(schedule.result.score_model)
    for phase in schedule.phases:
        add(phase.score_model)
    for readout in schedule.readouts:
        add(readout.score_model)
    for evaluator in schedule.evaluators.values():
        add(evaluator.score_model)
    return selected


def _selected_quantum_package_names(
    args: Optional[argparse.Namespace],
    gate_estimate_backend_spec: Optional[str] = None,
) -> set[str]:
    selected = {"qiskit"}
    backend_tokens = []
    for value in (getattr(args, "hw_backend", None), gate_estimate_backend_spec, getattr(args, "estimate_gates", None)):
        if value is None:
            continue
        backend_tokens.extend(token.strip().lower() for token in str(value).split(",") if token.strip())
    if any(token in {"aer", "aer_simulator"} or token.startswith("aer_") for token in backend_tokens):
        selected.add("qiskit-aer")
    if any(token.startswith("ibm_") for token in backend_tokens):
        selected.add("qiskit-ibm-runtime")
    return selected


def _base_pheat_package_names() -> set[str]:
    try:
        payload = collect_software_provenance(selected_score_models=[])
    except Exception:
        return set()
    return {str(item.get("name")) for item in payload.get("package_components") or [] if item.get("name")}


def _collect_pheat_dependency_provenance(
    *,
    selected_score_models: Sequence[str],
    evaluator_statuses: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    phase_raw_models = _pheat_raw_score_models_for_public(selected_score_models)
    try:
        payload = collect_software_provenance(selected_score_models=phase_raw_models)
    except Exception as exc:
        return {
            "format": "pheat.software-provenance",
            "version": 1,
            "selected_score_models": phase_raw_models,
            "selected_features": [],
            "package_components": [],
            "external_tools": [],
            "warnings": [f"PHEAT software provenance collection failed: {exc}"],
        }

    package_rows_by_name = {
        str(item.get("name")): dict(item)
        for item in payload.get("package_components") or []
        if item.get("name")
    }
    tool_rows_by_name = {
        str(item.get("name")): dict(item)
        for item in payload.get("external_tools") or []
        if item.get("name")
    }
    base_package_names = _base_pheat_package_names()
    selected_raw_models = list(phase_raw_models)

    for status in evaluator_statuses:
        score_model = str(status.get("score_model") or "")
        raw_models = _pheat_raw_score_models_for_public([score_model])
        if not raw_models:
            continue
        for raw_model in raw_models:
            if raw_model not in selected_raw_models:
                selected_raw_models.append(raw_model)
        evaluator_name = str(status.get("name") or score_model or "evaluator")
        evaluator_required = bool(status.get("required"))
        try:
            evaluator_payload = collect_software_provenance(selected_score_models=raw_models)
        except Exception as exc:
            payload.setdefault("warnings", []).append(
                f"PHEAT software provenance collection failed for evaluator {evaluator_name}: {exc}"
            )
            continue
        for component in evaluator_payload.get("package_components") or []:
            name = str(component.get("name") or "")
            if not name or name in base_package_names:
                continue
            adjusted = dict(component)
            adjusted["required"] = evaluator_required
            adjusted["role"] = f"selected evaluator: {evaluator_name}"
            _merge_package_component(package_rows_by_name, adjusted)
        for tool in evaluator_payload.get("external_tools") or []:
            name = str(tool.get("name") or "")
            if not name:
                continue
            adjusted = dict(tool)
            adjusted["required"] = evaluator_required
            adjusted["role"] = f"selected evaluator: {evaluator_name}"
            _merge_external_tool_component(tool_rows_by_name, adjusted)

    payload["selected_score_models"] = selected_raw_models
    payload["package_components"] = sorted(
        package_rows_by_name.values(), key=lambda item: str(item.get("name") or "").lower()
    )
    payload["external_tools"] = sorted(
        tool_rows_by_name.values(), key=lambda item: str(item.get("name") or "").lower()
    )
    return payload


def _collect_software_versions(
    *,
    selected_score_models: Optional[Sequence[str]] = None,
    evaluator_statuses: Optional[Sequence[Mapping[str, Any]]] = None,
    selected_quantum_packages: Optional[set[str]] = None,
) -> dict:
    try:
        import pheat

        pheat_path = Path(pheat.__file__).resolve()
        pheat_version = getattr(pheat, "__version__", None)
        pheat_git = _git_provenance(pheat_path)
    except Exception as exc:
        pheat_path = None
        pheat_version = None
        pheat_git = {"available": False, "error": str(exc)}

    selected_score_models = list(selected_score_models or [])
    evaluator_statuses = list(evaluator_statuses or [])
    selected_quantum_packages = set(selected_quantum_packages or {"qiskit"})
    pheat_provenance = _collect_pheat_dependency_provenance(
        selected_score_models=selected_score_models,
        evaluator_statuses=evaluator_statuses,
    )

    package_rows_by_name: dict[str, dict[str, Any]] = {}
    for component in pheat_provenance.get("package_components") or []:
        _merge_package_component(package_rows_by_name, dict(component))
    for spec in BASE_SOFTWARE_PACKAGES:
        _merge_package_component(
            package_rows_by_name,
            _package_component(
                str(spec["name"]),
                str(spec["role"]),
                required=bool(spec["required"]),
                selected=True,
            ),
        )
    for name, role in OPTIONAL_QUANTUM_PACKAGES.items():
        if name not in selected_quantum_packages:
            continue
        _merge_package_component(
            package_rows_by_name,
            _package_component(
                name,
                role,
                required=False,
                selected=True,
            ),
        )

    external_tool_rows_by_name: dict[str, dict[str, Any]] = {}
    for tool in pheat_provenance.get("external_tools") or []:
        _merge_external_tool_component(external_tool_rows_by_name, dict(tool))

    package_components = sorted(package_rows_by_name.values(), key=lambda item: str(item.get("name") or "").lower())
    external_tools = sorted(external_tool_rows_by_name.values(), key=lambda item: str(item.get("name") or "").lower())
    packages = {str(item["name"]): item.get("version") for item in package_components}
    return {
        "format": "qtf.software-provenance",
        "version": 1,
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
            "module_path": str((PACKAGE_ROOT / "__init__.py").resolve()),
            "git": _git_provenance(SCRIPT_DIR),
        },
        "pheat": {
            "version": pheat_version,
            "module_path": None if pheat_path is None else str(pheat_path),
            "git": pheat_git,
        },
        "packages": packages,
        "package_components": package_components,
        "external_tools": external_tools,
        "selected_score_models": selected_score_models,
        "pheat_software_provenance": pheat_provenance,
        "installed_distributions": _installed_distributions(),
    }

def _software_summary_payload(software_versions: dict, sidecar_path: Path) -> dict:
    payload = {
        "format": software_versions.get("format") or "qtf.software-provenance",
        "version": software_versions.get("version") or 1,
        "python": software_versions.get("python") or {},
        "platform": software_versions.get("platform") or {},
        "qtf": software_versions.get("qtf") or {},
        "pheat": software_versions.get("pheat") or {},
        "packages": software_versions.get("packages") or {},
        "package_components": software_versions.get("package_components") or [],
        "external_tools": software_versions.get("external_tools") or [],
        "selected_score_models": software_versions.get("selected_score_models") or [],
        "pheat_software_provenance": software_versions.get("pheat_software_provenance") or {},
        "installed_distribution_count": len(software_versions.get("installed_distributions") or []),
        "sidecar_path": str(sidecar_path),
    }
    return payload

def _environment_snapshot() -> dict:
    return {key: os.environ.get(key) for key in REPORT_ENVIRONMENT_KEYS if os.environ.get(key)}


def _redact_command_args(parts: Sequence[object]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for raw_part in parts:
        part = str(raw_part)
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        matched_inline = False
        for option in SENSITIVE_COMMAND_OPTIONS:
            if part == option:
                redacted.append(part)
                redact_next = True
                matched_inline = True
                break
            if part.startswith(f"{option}="):
                redacted.append(f"{option}=<redacted>")
                matched_inline = True
                break
        if not matched_inline:
            redacted.append(part)
    return redacted


def _redact_command_line_text(command_line: str) -> str:
    if not any(option in command_line for option in SENSITIVE_COMMAND_OPTIONS):
        return command_line
    try:
        return shlex.join(_redact_command_args(shlex.split(command_line)))
    except ValueError:
        return "<redacted command line containing IBM credentials>"


def _command_line(argv: Optional[Sequence[str]], override: Optional[str]) -> str:
    if override and override.strip():
        return _redact_command_line_text(override.strip())
    if argv is None:
        command = [sys.executable, *sys.argv]
    else:
        command = [sys.executable, str(Path(__file__).resolve()), *argv]
    return shlex.join(_redact_command_args(command))


def _score_payload(structure: HeavyAtomStructure, model: str, options: Optional[dict[str, Any]] = None) -> dict:
    if is_qtf_score_model(model):
        raise ValueError("QTF-native score models require folder/parameter context.")
    result = score_pheat_structure(structure, model=model, **dict(options or {})).to_dict()
    metadata_status = str((result.get("metadata") or {}).get("status") or "").strip().lower()
    result["status"] = metadata_status if metadata_status else "ok"
    if metadata_status == "unavailable" and not result.get("error"):
        result["error"] = (result.get("metadata") or {}).get("reason")
    return result


def _score_options_for_folder_params(
    folder,
    params,
    model: str,
    options: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    score_options = dict(options or {})
    if str(model).strip().lower().replace("_", "-") != "pheat-coarse-protein-folding-v1":
        return score_options
    if "decoded_torsions" in score_options or params is None:
        return score_options
    angle_mode = getattr(folder, "primary_angle_mode", None) or getattr(folder, "optimizer_angle_mode", "statevector")
    backend = getattr(folder, "optimizer_backend", None) if angle_mode == "sampler" else None
    shots = int(getattr(folder, "primary_shots", None) or getattr(folder, "optimizer_shots", 4096))
    angle_vec = folder._angle_vector_from_params(
        params,
        angle_mode=angle_mode,
        backend=backend,
        shots=shots,
    )
    score_options["decoded_torsions"] = folder._angle_dict(angle_vec)
    return score_options


def _safe_score_payload(structure: HeavyAtomStructure, model: str, options: Optional[dict[str, Any]] = None) -> dict:
    try:
        return _score_payload(structure, model, options=options)
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


def _safe_score_payload_for_folder(
    folder,
    params,
    model: str,
    *,
    fallback_structure: HeavyAtomStructure,
    options: Optional[dict[str, Any]] = None,
) -> dict:
    try:
        if is_qtf_score_model(model):
            angle_mode = getattr(folder, "primary_angle_mode", None) or getattr(
                folder, "optimizer_angle_mode", "statevector"
            )
            backend = getattr(folder, "optimizer_backend", None) if angle_mode == "sampler" else None
            shots = int(getattr(folder, "primary_shots", None) or getattr(folder, "optimizer_shots", 4096))
            payload, _total = folder.score_model_for_params(
                params,
                model,
                angle_mode=angle_mode,
                backend=backend,
                shots=shots,
                options=options,
            )
            return payload
        score_options = _score_options_for_folder_params(folder, params, model, options=options)
        return _score_payload(fallback_structure, model, options=score_options)
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


def _evaluator_options_for_run(
    evaluator: EvaluatorConfig,
    *,
    outdir: Optional[Path] = None,
    prefix: Optional[str] = None,
    candidate_key: Optional[str] = None,
    include_prepared_output: bool = False,
) -> dict[str, Any]:
    options = dict(evaluator.options or {})
    cache_mode = str(options.get("prep_cache_mode") or "off").strip().lower()
    if outdir is not None and cache_mode != "off" and not options.get("prep_cache_dir"):
        options["prep_cache_dir"] = str(outdir / "pheat_prep_cache")
    if include_prepared_output and outdir is not None and prefix is not None and candidate_key is not None:
        raw_model = _pheat_raw_model_name(evaluator.score_model)
        if raw_model in {"ambertools-sander", "gromacs-mdrun"} and not options.get("prepared_output"):
            extension = "gro" if raw_model == "gromacs-mdrun" else "pdb"
            prepared_dir = outdir / "external_structures"
            options["prepared_output"] = str(
                prepared_dir / f"{prefix}_{_slug(candidate_key)}_{_slug(evaluator.name)}_prepared.{extension}"
            )
    return options


def _validate_external_evaluator_options(
    evaluator: EvaluatorConfig,
    *,
    outdir: Optional[Path] = None,
) -> dict[str, Any]:
    raw_model = _pheat_raw_model_name(evaluator.score_model)
    options = _evaluator_options_for_run(evaluator, outdir=outdir)
    capability = dict(_pheat_capabilities_by_public_name().get(evaluator.score_model) or {})
    implementation = capability.get("implementation") or {}
    external = bool(implementation.get("external"))
    payload = {
        "ok": True,
        "model": raw_model,
        "external": external,
        "errors": [],
        "warnings": [],
    }
    try:
        from pheat.scoring import validate_scoring_options
    except Exception as exc:
        payload["warnings"] = [f"PHEAT score option validation is unavailable: {exc}"]
        return payload
    try:
        validation_payload = validate_scoring_options(
            raw_model,
            options,
            require_executables=False,
        )
    except Exception as exc:
        payload.update({"ok": False, "errors": [str(exc)]})
        return payload
    validation_payload = dict(validation_payload or {})
    validation_payload.setdefault("model", raw_model)
    validation_payload.setdefault("errors", [])
    validation_payload.setdefault("warnings", [])
    validation_payload["external"] = external
    return validation_payload


def _validate_recipe_evaluators_for_run(
    parser: argparse.ArgumentParser,
    schedule: PhaseSchedule,
    *,
    outdir: Path,
) -> list[dict[str, Any]]:
    active_names = set()
    if schedule.phase_comparisons.enabled:
        active_names.update(schedule.phase_comparisons.evaluators)
    if schedule.reranking.enabled and schedule.reranking.evaluator:
        active_names.add(schedule.reranking.evaluator)
    if schedule.handoff_guard.enabled and schedule.handoff_guard.evaluator:
        active_names.add(schedule.handoff_guard.evaluator)
    if schedule.validation.enabled:
        active_names.update(schedule.validation.evaluators)
    if not schedule.evaluators or not active_names:
        return []
    capabilities = _pheat_capabilities_by_public_name()
    statuses = []
    for evaluator in schedule.evaluators.values():
        if evaluator.name not in active_names:
            continue
        capability = dict(capabilities.get(evaluator.score_model) or {})
        validation_payload = _validate_external_evaluator_options(evaluator, outdir=outdir)
        available = bool(capability.get("available"))
        validation_ok = bool(validation_payload.get("ok", True))
        status = "ok" if available and validation_ok else "skipped"
        errors = []
        if not available:
            errors.append(str(capability.get("reason") or "score model is unavailable"))
        validation_errors = [str(item) for item in validation_payload.get("errors") or []]
        if not available:
            validation_errors = [
                error
                for error in validation_errors
                if "executable was not found" not in error
            ]
        errors.extend(validation_errors)
        errors = list(dict.fromkeys(error for error in errors if error))
        if evaluator.required and errors:
            parser.error(f"required evaluator {evaluator.name!r} is unavailable: {'; '.join(errors)}")
        statuses.append(
            {
                "name": evaluator.name,
                "score_model": evaluator.score_model,
                "pheat_model": _pheat_raw_model_name(evaluator.score_model),
                "required": bool(evaluator.required),
                "status": status,
                "available": available,
                "capability": capability,
                "validation": validation_payload,
                "errors": errors,
                "warnings": list(validation_payload.get("warnings") or []),
            }
        )
    return statuses


def _evaluator_status_map(statuses: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item.get("name")): dict(item) for item in statuses}


def _score_structure_with_evaluator(
    structure: HeavyAtomStructure,
    evaluator: EvaluatorConfig,
    *,
    outdir: Optional[Path] = None,
    prefix: Optional[str] = None,
    candidate_key: Optional[str] = None,
    include_prepared_output: bool = False,
) -> dict[str, Any]:
    options = _evaluator_options_for_run(
        evaluator,
        outdir=outdir,
        prefix=prefix,
        candidate_key=candidate_key,
        include_prepared_output=include_prepared_output,
    )
    score = _safe_score_payload(structure, evaluator.score_model, options=options)
    score["evaluator"] = evaluator.name
    score["score_model"] = evaluator.score_model
    score["options"] = _jsonify(options)
    if options.get("prepared_output"):
        score["prepared_output"] = str(options["prepared_output"])
    return score


def _validation_candidate_entries(
    validation_config: ValidationConfig,
    *,
    primary_structure: HeavyAtomStructure,
    primary_snapshot_key: Optional[str],
    structure_snapshot_payloads: Sequence[dict[str, Any]],
    snapshot_structures: dict[str, HeavyAtomStructure],
    reranking_results: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if "primary" in validation_config.candidates:
        entries.append(
            {
                "candidate_set": "primary",
                "candidate_key": str(primary_snapshot_key or "primary"),
                "label": "Primary result",
                "snapshot_key": primary_snapshot_key,
                "structure": primary_structure,
            }
        )
    if "phase_ends" in validation_config.candidates:
        for snapshot in structure_snapshot_payloads:
            key = snapshot.get("key")
            if snapshot.get("role") != "phase" or key not in snapshot_structures:
                continue
            entries.append(
                {
                    "candidate_set": "phase_ends",
                    "candidate_key": str(key),
                    "label": str(snapshot.get("label") or key),
                    "snapshot_key": key,
                    "structure": snapshot_structures[key],
                }
            )
    if "reranked_top" in validation_config.candidates:
        for result in reranking_results:
            if str(result.get("trigger") or "phase_end") != "phase_end":
                continue
            selected = result.get("selected") or {}
            key = selected.get("snapshot_key")
            if key not in snapshot_structures:
                continue
            entries.append(
                {
                    "candidate_set": "reranked_top",
                    "candidate_key": str(key),
                    "label": f"Reranked top: {result.get('phase_label') or result.get('phase_name') or key}",
                    "snapshot_key": key,
                    "structure": snapshot_structures[key],
                }
            )
    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in entries:
        deduped[(entry["candidate_set"], entry["candidate_key"])] = entry
    return list(deduped.values())


def _validation_warnings_for_score(score: Mapping[str, Any]) -> list[str]:
    warnings = [str(item) for item in score.get("warnings") or []]
    units = str(score.get("units") or "").lower()
    try:
        total = float(score.get("total"))
    except Exception:
        total = math.nan
    if math.isfinite(total) and "kj/mol" in units and abs(total) >= 1.0e9:
        warnings.append(
            "force-field validation energy is extremely large; treat this candidate as physically suspect"
        )
    return warnings


def _run_validation_evaluators(
    validation_config: ValidationConfig,
    evaluator_configs: dict[str, EvaluatorConfig],
    evaluator_status_by_name: dict[str, dict[str, Any]],
    *,
    primary_structure: HeavyAtomStructure,
    primary_snapshot_key: Optional[str],
    structure_snapshot_payloads: Sequence[dict[str, Any]],
    snapshot_structures: dict[str, HeavyAtomStructure],
    reranking_results: Sequence[dict[str, Any]],
    outdir: Path,
    prefix: str,
) -> list[dict[str, Any]]:
    if not validation_config.enabled:
        return []
    candidates = _validation_candidate_entries(
        validation_config,
        primary_structure=primary_structure,
        primary_snapshot_key=primary_snapshot_key,
        structure_snapshot_payloads=structure_snapshot_payloads,
        snapshot_structures=snapshot_structures,
        reranking_results=reranking_results,
    )
    results = []
    for candidate in candidates:
        for evaluator_name in validation_config.evaluators:
            evaluator = evaluator_configs.get(evaluator_name)
            if evaluator is None:
                continue
            evaluator_status = evaluator_status_by_name.get(evaluator_name) or {}
            if evaluator_status.get("status") not in {None, "ok"}:
                results.append(
                    {
                        "candidate_set": candidate["candidate_set"],
                        "candidate_key": candidate["candidate_key"],
                        "label": candidate["label"],
                        "snapshot_key": candidate.get("snapshot_key"),
                        "evaluator": evaluator_name,
                        "score_model": evaluator.score_model,
                        "status": "skipped",
                        "error": "; ".join(str(item) for item in evaluator_status.get("errors") or [])
                        or "evaluator is unavailable",
                    }
                )
                continue
            score = _score_structure_with_evaluator(
                candidate["structure"],
                evaluator,
                outdir=outdir,
                prefix=prefix,
                candidate_key=f"{candidate['candidate_key']}_{candidate['candidate_set']}",
                include_prepared_output=True,
            )
            validation_warnings = _validation_warnings_for_score(score)
            results.append(
                {
                    "candidate_set": candidate["candidate_set"],
                    "candidate_key": candidate["candidate_key"],
                    "label": candidate["label"],
                    "snapshot_key": candidate.get("snapshot_key"),
                    "evaluator": evaluator_name,
                    "score_model": evaluator.score_model,
                    "status": score.get("status"),
                    "score": score,
                    "score_total": score.get("total"),
                    "score_units": score.get("units"),
                    "prepared_output": score.get("prepared_output"),
                    "warnings": validation_warnings,
                    "error": score.get("error"),
                }
            )
    return results


def _physical_count_from_score(score: Mapping[str, Any], key: str) -> int:
    metadata = score.get("metadata") or {}
    counts = metadata.get("checked_counts") or {}
    value = counts.get(key)
    if value is None:
        value = metadata.get(key)
    try:
        return int(value or 0)
    except Exception:
        return 0


def _physical_min_distance_from_score(score: Mapping[str, Any]) -> Optional[float]:
    metadata = score.get("metadata") or {}
    value = metadata.get("min_nonlocal_distance")
    if value is None:
        component = ((metadata.get("component_models") or {}).get("physical_integrity") or {})
        value = (component.get("metadata") or {}).get("min_nonlocal_distance")
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _physical_counts_payload(score: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "clash_count": _physical_count_from_score(score, "clash_count"),
        "short_contact_count": _physical_count_from_score(score, "short_contact_count"),
        "nonfinite_atom_count": _physical_count_from_score(score, "nonfinite_atom_count"),
        "min_nonlocal_distance": _physical_min_distance_from_score(score),
    }


def _physical_absolute_threshold_violations(
    score: Mapping[str, Any],
    *,
    max_clash_count: Optional[int],
    max_short_contact_count: Optional[int],
    min_nonlocal_distance_a: Optional[float],
) -> list[str]:
    violations: list[str] = []
    clash_count = _physical_count_from_score(score, "clash_count")
    short_count = _physical_count_from_score(score, "short_contact_count")
    min_distance = _physical_min_distance_from_score(score)
    if max_clash_count is not None and clash_count > max_clash_count:
        violations.append(f"clash count {clash_count} exceeds limit {max_clash_count}")
    if max_short_contact_count is not None and short_count > max_short_contact_count:
        violations.append(f"short-contact count {short_count} exceeds limit {max_short_contact_count}")
    if (
        min_nonlocal_distance_a is not None
        and min_distance is not None
        and min_distance < min_nonlocal_distance_a
    ):
        violations.append(
            f"min nonlocal distance {min_distance:.6g} A is below "
            f"{min_nonlocal_distance_a:.6g} A"
        )
    return violations


def _handoff_guard_decision_payload(
    start_score: Mapping[str, Any],
    handoff_score: Mapping[str, Any],
    config: HandoffGuardConfig,
) -> dict[str, Any]:
    hard_reasons: list[str] = []
    threshold_violations: list[str] = []
    start_threshold_violations = _physical_absolute_threshold_violations(
        start_score,
        max_clash_count=config.max_clash_count,
        max_short_contact_count=config.max_short_contact_count,
        min_nonlocal_distance_a=config.min_nonlocal_distance_a,
    )

    if handoff_score.get("status") != "ok" or handoff_score.get("total") is None:
        hard_reasons.append(f"handoff score unavailable: {handoff_score.get('error') or handoff_score.get('status')}")

    delta = None
    if start_score.get("status") == "ok" and start_score.get("total") is not None and handoff_score.get("total") is not None:
        start_total = float(start_score["total"])
        handoff_total = float(handoff_score["total"])
        delta = handoff_total - start_total
        if config.reject_on_score_worse and handoff_total > start_total:
            hard_reasons.append(f"physical score worsened ({handoff_total:.6g} > {start_total:.6g})")

    start_clashes = _physical_count_from_score(start_score, "clash_count")
    handoff_clashes = _physical_count_from_score(handoff_score, "clash_count")
    start_short = _physical_count_from_score(start_score, "short_contact_count")
    handoff_short = _physical_count_from_score(handoff_score, "short_contact_count")
    start_nonfinite = _physical_count_from_score(start_score, "nonfinite_atom_count")
    handoff_nonfinite = _physical_count_from_score(handoff_score, "nonfinite_atom_count")
    start_min_distance = _physical_min_distance_from_score(start_score)
    handoff_min_distance = _physical_min_distance_from_score(handoff_score)
    unsafe_transition_configured = (
        config.unsafe_transition_max_short_contact_count is not None
        or config.unsafe_transition_min_nonlocal_distance_a is not None
        or config.unsafe_transition_require_clash_count_decrease
    )
    unsafe_transition_active = (
        config.allow_improving_unsafe
        and unsafe_transition_configured
        and config.max_clash_count is not None
        and start_clashes > config.max_clash_count
    )
    unsafe_transition_failures: list[str] = []
    if unsafe_transition_active:
        if delta is None or delta >= 0:
            unsafe_transition_failures.append("unsafe transition requires physical score improvement")
        if config.unsafe_transition_require_clash_count_decrease and handoff_clashes >= start_clashes:
            unsafe_transition_failures.append(
                f"unsafe transition requires clash count decrease ({handoff_clashes} >= {start_clashes})"
            )
        if (
            config.unsafe_transition_max_short_contact_count is not None
            and handoff_short > config.unsafe_transition_max_short_contact_count
        ):
            unsafe_transition_failures.append(
                "unsafe transition short-contact count "
                f"{handoff_short} exceeds limit {config.unsafe_transition_max_short_contact_count}"
            )
        if config.unsafe_transition_min_nonlocal_distance_a is not None:
            if handoff_min_distance is None:
                unsafe_transition_failures.append("unsafe transition min nonlocal distance is unavailable")
            elif handoff_min_distance < config.unsafe_transition_min_nonlocal_distance_a:
                unsafe_transition_failures.append(
                    f"unsafe transition min nonlocal distance {handoff_min_distance:.6g} A is below "
                    f"{config.unsafe_transition_min_nonlocal_distance_a:.6g} A"
                )
    unsafe_transition_candidate_ok = unsafe_transition_active and not unsafe_transition_failures

    if config.reject_on_clash_count_increase and handoff_clashes > start_clashes:
        hard_reasons.append(f"clash count increased ({handoff_clashes} > {start_clashes})")
    if (
        config.reject_on_short_contact_count_increase
        and handoff_short > start_short
        and not unsafe_transition_candidate_ok
    ):
        hard_reasons.append(f"short-contact count increased ({handoff_short} > {start_short})")
    if (
        config.reject_on_min_nonlocal_distance_decrease
        and start_min_distance is not None
        and handoff_min_distance is not None
        and handoff_min_distance < start_min_distance
        and not unsafe_transition_candidate_ok
    ):
        hard_reasons.append(
            f"min nonlocal distance decreased ({handoff_min_distance:.6g} A < {start_min_distance:.6g} A)"
        )
    if config.reject_on_nonfinite and handoff_nonfinite > 0:
        hard_reasons.append(f"non-finite atom count is {handoff_nonfinite}")

    threshold_violations = _physical_absolute_threshold_violations(
        handoff_score,
        max_clash_count=config.max_clash_count,
        max_short_contact_count=config.max_short_contact_count,
        min_nonlocal_distance_a=config.min_nonlocal_distance_a,
    )

    if unsafe_transition_active:
        accepted_with_violations = bool(threshold_violations) and not hard_reasons and unsafe_transition_candidate_ok
    else:
        accepted_with_violations = (
            bool(threshold_violations)
            and not hard_reasons
            and config.allow_improving_unsafe
            and bool(start_threshold_violations or start_nonfinite > 0)
        )
    if hard_reasons:
        status = "rejected"
        decision = "fallback"
        reasons = [*hard_reasons, *threshold_violations]
    elif threshold_violations and not accepted_with_violations:
        status = "rejected"
        decision = "fallback"
        reasons = list(threshold_violations)
    elif accepted_with_violations:
        status = "accepted_with_violations"
        decision = "accept"
        reasons = list(threshold_violations)
    else:
        status = "accepted"
        decision = "accept"
        reasons = []

    return {
        "status": status,
        "decision": decision,
        "reasons": reasons,
        "hard_reasons": hard_reasons,
        "absolute_threshold_violations": threshold_violations,
        "phase_start_absolute_threshold_violations": start_threshold_violations,
        "unsafe_transition": {
            "configured": unsafe_transition_configured,
            "allow_improving_unsafe": bool(config.allow_improving_unsafe),
            "active": unsafe_transition_active,
            "accepted": accepted_with_violations and unsafe_transition_candidate_ok,
            "failures": unsafe_transition_failures,
            "max_short_contact_count": config.unsafe_transition_max_short_contact_count,
            "min_nonlocal_distance_a": config.unsafe_transition_min_nonlocal_distance_a,
            "require_clash_count_decrease": config.unsafe_transition_require_clash_count_decrease,
        },
        "delta_current_minus_start": delta,
        "phase_start_counts": {
            "clash_count": start_clashes,
            "short_contact_count": start_short,
            "nonfinite_atom_count": start_nonfinite,
            "min_nonlocal_distance": start_min_distance,
        },
        "handoff_counts": {
            "clash_count": handoff_clashes,
            "short_contact_count": handoff_short,
            "nonfinite_atom_count": handoff_nonfinite,
            "min_nonlocal_distance": handoff_min_distance,
        },
    }


_PHASE_HANDOFF_GUARD_OVERRIDE_KEYS = {
    "fallback",
    "abort_on_reject",
    "allow_improving_unsafe",
    "max_clash_count",
    "max_short_contact_count",
    "min_nonlocal_distance_a",
    "unsafe_transition_max_short_contact_count",
    "unsafe_transition_min_nonlocal_distance_a",
    "unsafe_transition_require_clash_count_decrease",
    "reject_on_score_worse",
    "reject_on_clash_count_increase",
    "reject_on_short_contact_count_increase",
    "reject_on_min_nonlocal_distance_decrease",
    "reject_on_nonfinite",
}


def _phase_handoff_guard_enabled(config: HandoffGuardConfig, phase: PhaseConfig) -> bool:
    phase_guard = phase.handoff_guard or {}
    if "enabled" in phase_guard:
        return bool(phase_guard["enabled"])
    if config.phases:
        return phase.name in set(config.phases)
    return False


def _effective_handoff_guard_config(config: HandoffGuardConfig, phase: PhaseConfig) -> HandoffGuardConfig:
    phase_guard = phase.handoff_guard or {}
    updates = {
        key: phase_guard[key]
        for key in _PHASE_HANDOFF_GUARD_OVERRIDE_KEYS
        if key in phase_guard
    }
    if phase_guard and "allow_improving_unsafe" not in phase_guard:
        updates["allow_improving_unsafe"] = False
    return replace(config, **updates)


def _physical_readiness_payload(
    final_score: Mapping[str, Any],
    validation_results: Sequence[Mapping[str, Any]],
    *,
    max_clash_count: int = 0,
    max_short_contact_count: int = 0,
    min_nonlocal_distance_a: float = 0.7,
) -> dict[str, Any]:
    source = "final_score"
    score = final_score
    for item in validation_results:
        if item.get("candidate_set") == "primary" and str(item.get("score_model") or "").endswith("physical-integrity"):
            candidate_score = item.get("score") or {}
            if candidate_score.get("status") == "ok":
                source = f"validation:{item.get('evaluator')}"
                score = candidate_score
                break
    counts = _physical_counts_payload(score)
    reasons: list[str] = []
    if score.get("status") != "ok" or score.get("total") is None:
        reasons.append(f"physical score unavailable: {score.get('error') or score.get('status')}")
    if counts["nonfinite_atom_count"] > 0:
        reasons.append(f"non-finite atom count is {counts['nonfinite_atom_count']}")
    if counts["clash_count"] > max_clash_count:
        reasons.append(f"clash count {counts['clash_count']} exceeds limit {max_clash_count}")
    if counts["short_contact_count"] > max_short_contact_count:
        reasons.append(f"short-contact count {counts['short_contact_count']} exceeds limit {max_short_contact_count}")
    min_distance = counts["min_nonlocal_distance"]
    if min_distance is None:
        reasons.append("min nonlocal distance is unavailable")
    elif min_distance < min_nonlocal_distance_a:
        reasons.append(
            f"min nonlocal distance {min_distance:.6g} A is below {min_nonlocal_distance_a:.6g} A"
        )
    return {
        "status": "ready" if not reasons else "not_ready",
        "ready_for_length_tuning": not reasons,
        "source": source,
        "score_model": score.get("score_model") or score.get("model"),
        "score_total": score.get("total"),
        "score_units": score.get("units"),
        "thresholds": {
            "max_clash_count": max_clash_count,
            "max_short_contact_count": max_short_contact_count,
            "min_nonlocal_distance_a": min_nonlocal_distance_a,
        },
        "counts": counts,
        "reasons": reasons,
    }


def _phase_readiness_decision_payload(
    score: Mapping[str, Any],
    config: PhaseReadinessConfig,
) -> dict[str, Any]:
    counts = _physical_counts_payload(score)
    thresholds = {
        "max_clash_count": config.max_clash_count,
        "max_short_contact_count": config.max_short_contact_count,
        "min_nonlocal_distance_a": config.min_nonlocal_distance_a,
    }
    reasons: list[str] = []
    if score.get("status") != "ok" or score.get("total") is None:
        reasons.append(f"physical score unavailable: {score.get('error') or score.get('status')}")
    if counts["nonfinite_atom_count"] > 0:
        reasons.append(f"non-finite atom count is {counts['nonfinite_atom_count']}")
    if config.max_clash_count is not None and counts["clash_count"] > config.max_clash_count:
        reasons.append(f"clash count {counts['clash_count']} exceeds limit {config.max_clash_count}")
    if (
        config.max_short_contact_count is not None
        and counts["short_contact_count"] > config.max_short_contact_count
    ):
        reasons.append(
            f"short-contact count {counts['short_contact_count']} exceeds limit "
            f"{config.max_short_contact_count}"
        )
    min_distance = counts["min_nonlocal_distance"]
    if config.min_nonlocal_distance_a is not None:
        if min_distance is None:
            reasons.append("min nonlocal distance is unavailable")
        elif min_distance < config.min_nonlocal_distance_a:
            reasons.append(
                f"min nonlocal distance {min_distance:.6g} A is below "
                f"{config.min_nonlocal_distance_a:.6g} A"
            )
    return {
        "status": "ready" if not reasons else "not_ready",
        "ready": not reasons,
        "thresholds": thresholds,
        "counts": counts,
        "reasons": reasons,
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
    rg = float(pheat_radius_of_gyration(ca_coords.tolist()))
    return end_to_end, rg


def _sequence_from_residue_geometry(residue_geometry: ResidueGeometryStructure) -> str:
    return "".join(three_to_one(residue.name) for residue in residue_geometry.residues)


def _sequence_indexed_residue_geometry(
    residue_geometry: ResidueGeometryStructure,
) -> ResidueGeometryStructure:
    old_to_new_refs: dict[tuple[str, int, str], tuple[str, int, str]] = {}
    residues = []
    for index, residue in enumerate(residue_geometry.residues, start=1):
        old_ref = (
            residue.chain_id or "",
            int(residue.resseq if residue.resseq is not None else index),
            residue.icode or "",
        )
        new_ref = ("A", index, "")
        old_to_new_refs[old_ref] = new_ref
        residues.append(replace(residue, chain_id=new_ref[0], resseq=new_ref[1], icode=new_ref[2]))

    disulfide_bonds = []
    for bond in residue_geometry.disulfide_bonds:
        ref_1 = old_to_new_refs.get((bond.chain_id_1 or "", int(bond.resseq_1), bond.icode_1 or ""))
        ref_2 = old_to_new_refs.get((bond.chain_id_2 or "", int(bond.resseq_2), bond.icode_2 or ""))
        if ref_1 is None or ref_2 is None:
            continue
        disulfide_bonds.append(
            replace(
                bond,
                chain_id_1=ref_1[0],
                resseq_1=ref_1[1],
                icode_1=ref_1[2],
                chain_id_2=ref_2[0],
                resseq_2=ref_2[1],
                icode_2=ref_2[2],
            )
        )

    return replace(
        residue_geometry,
        residues=residues,
        disulfide_bonds=disulfide_bonds,
        metadata={
            **dict(residue_geometry.metadata or {}),
            "qtf_metric_numbering": "sequence-indexed",
            "qtf_metric_chain_id": "A",
        },
    )


def _load_pheat_reference(
    reference: str,
    *,
    angle_units: str,
    stored_angles,
    stored_lengths,
    max_chi: Optional[int],
    include_terminal_oxt: bool,
    report_structure_domain: str,
    geometry_mode: Optional[str],
    geometry_table: Optional[str],
    geometry_profile: Optional[str],
) -> PheatReference:
    path = Path(reference)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(
            "--reference-structure must be an existing PDB, PHEAT heavy JSON, "
            f"or PHEAT residue-geometry JSON file: {reference}"
        )

    source_type = "pdb"
    source_domain_coverage = None
    if path.suffix.lower() == ".json":
        json_errors = []
        try:
            residue_geometry = load_residue_geometry(path)
            source_type = "pheat_residue_geometry_json"
        except Exception as exc:
            json_errors.append(f"residue geometry: {exc}")
            try:
                source_structure = load_structure_json(path)
                source_type = "pheat_structure_json"
            except Exception as heavy_exc:
                json_errors.append(f"atom structure: {heavy_exc}")
                raise ValueError(
                    "Reference JSON is neither PHEAT residue-geometry JSON nor "
                    "PHEAT atom-structure JSON: "
                    + "; ".join(json_errors)
                ) from heavy_exc
            _source_filtered, source_domain_coverage = _filter_structure_for_report_domain(
                source_structure,
                report_structure_domain,
            )
            residue_geometry = structure_to_residue_geometry(
                _canonical_protein_structure(source_structure, source_path=path),
                name=f"pheat-reference:{path.name}",
                angle_units=angle_units,
                stored_angles=stored_angles,
                stored_lengths=stored_lengths,
                max_chi=max_chi,
            )
    else:
        source_structure = load_pdb(path)
        _source_filtered, source_domain_coverage = _filter_structure_for_report_domain(
            source_structure,
            report_structure_domain,
        )
        residue_geometry = structure_to_residue_geometry(
            _canonical_protein_structure(source_structure, source_path=path),
            name=f"pheat-reference:{path.name}",
            angle_units=angle_units,
            stored_angles=stored_angles,
            stored_lengths=stored_lengths,
            max_chi=max_chi,
        )

    reference_structure = structure_from_residue_geometry(
        residue_geometry,
        include_terminal_oxt=include_terminal_oxt,
        geometry_mode=geometry_mode,
        geometry_table=geometry_table,
        geometry_profile=geometry_profile,
    )
    metric_residue_geometry = _sequence_indexed_residue_geometry(residue_geometry)
    metric_reference_structure = structure_from_residue_geometry(
        metric_residue_geometry,
        include_terminal_oxt=include_terminal_oxt,
        geometry_mode=geometry_mode,
        geometry_table=geometry_table,
        geometry_profile=geometry_profile,
    )
    return PheatReference(
        source_path=path,
        source_type=source_type,
        residue_geometry=residue_geometry,
        structure=reference_structure,
        metric_residue_geometry=metric_residue_geometry,
        metric_structure=metric_reference_structure,
        sequence=_sequence_from_residue_geometry(residue_geometry),
        source_domain_coverage=source_domain_coverage,
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
        metadata={
            "source": "qtf_reference_filter",
            "source_metadata": dict(structure.metadata or {}),
            "source_name": structure.name or source_path.name,
            "filtered_for_pheat_rmsd": True,
            "dropped_bonds_reason": "atom filtering can invalidate index-based bonds",
        },
        disulfide_bonds=structure.disulfide_bonds,
        atom_scope="heavy",
    )


def _pheat_alignment_details(
    reference_structure: HeavyAtomStructure,
    target_structure: HeavyAtomStructure,
    *,
    atom_sets: Sequence[str] = METRIC_ATOM_SETS,
    alignment_atom_set: str = DEFAULT_RMSD_ALIGNMENT_ATOM_SET,
) -> dict:
    normalized_atom_sets = normalize_metric_atom_sets(atom_sets)
    normalized_alignment_atom_set = normalize_rmsd_alignment_atom_set(alignment_atom_set)
    metrics = structure_metric_summary(
        reference_structure,
        target_structure,
        atom_sets=normalized_atom_sets,
        alignment_atom_set=normalized_alignment_atom_set,
    )
    all_heavy = metrics.get("all-heavy") or {}
    backbone = metrics.get("backbone") or {}
    ca = metrics.get("ca") or {}
    ok_metrics = [payload for payload in metrics.values() if payload.get("status") == "ok"]
    return {
        "status": "ok" if ok_metrics else "unavailable",
        "atom_sets": list(normalized_atom_sets),
        "alignment_atom_set": normalized_alignment_atom_set,
        "metrics": metrics,
        "all_heavy_rmsd": _metric_payload_value(all_heavy),
        "backbone_rmsd": _metric_payload_value(backbone),
        "ca_rmsd": _metric_payload_value(ca),
        "matched_heavy_atoms": _metric_payload_int(all_heavy, "matched_atoms"),
        "matched_backbone_atoms": _metric_payload_int(backbone, "matched_atoms"),
        "matched_ca_atoms": _metric_payload_int(ca, "matched_atoms"),
        "unmatched_reference_atoms": _metric_payload_int(all_heavy, "unmatched_reference_atoms"),
        "unmatched_target_atoms": _metric_payload_int(all_heavy, "unmatched_target_atoms"),
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


def _safe_pheat_radius_of_gyration(
    structure: Optional[HeavyAtomStructure],
    *,
    atom_sets: Sequence[str] = METRIC_ATOM_SETS,
) -> dict:
    atom_sets = normalize_metric_atom_sets(atom_sets)
    if structure is None:
        return _unavailable_rg_summary(atom_sets, "structure is not available")
    try:
        return radius_of_gyration_summary(structure, atom_sets=atom_sets)
    except Exception as exc:
        return _unavailable_rg_summary(atom_sets, str(exc))


def _safe_pheat_radius_of_gyration_delta(before: dict, after: dict) -> dict:
    return radius_of_gyration_delta_summary(before, after, atom_sets=before.keys() or after.keys() or METRIC_ATOM_SETS)


def _aligned_pheat_structures(
    reference_structure: HeavyAtomStructure,
    folded_structure: HeavyAtomStructure,
    *,
    atom_set: str = PRIMARY_RMSD_ATOM_SET,
) -> tuple[HeavyAtomStructure, HeavyAtomStructure, int]:
    alignment = align_structure_to_reference(reference_structure, folded_structure, atom_set=atom_set)
    return (
        reference_structure,
        alignment["aligned_target"],
        int(alignment["matched_atoms"]),
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


def _metric_payload_value(payload: Optional[dict]) -> Optional[float]:
    if not payload or payload.get("status") != "ok" or payload.get("value") is None:
        return None
    return float(payload["value"])


def _metric_payload_int(payload: Optional[dict], key: str) -> Optional[int]:
    if not payload or payload.get("status") != "ok" or payload.get(key) is None:
        return None
    return int(payload[key])


def _metric_payload_for_atom_set(result_or_details: dict, atom_set: str) -> dict:
    payload = result_or_details.get("rmsd_details") if "rmsd_details" in result_or_details else result_or_details
    metrics = (payload or {}).get("metrics") or {}
    return metrics.get(atom_set) or {}


def _rg_value(payload: Optional[dict], key: str) -> Optional[float]:
    if not payload:
        return None
    values = payload.get("values") or {}
    value = values.get(key)
    return None if value is None else float(value)


def _rg_payload_for_atom_set(summary: Optional[dict], atom_set: str) -> dict:
    if not summary:
        return {}
    if "values" in summary:
        return summary
    return summary.get(atom_set) or {}


def _unavailable_rg_summary(atom_sets: Iterable[str], error: str) -> dict:
    return {
        atom_set: {
            "status": "unavailable",
            "error": error,
            "atom_set": atom_set,
            "mode": "both",
            "values": {},
            "units": "angstrom",
        }
        for atom_set in normalize_metric_atom_sets(atom_sets)
    }


def _pheat_rg_table_rows(result: dict) -> str:
    rg = result.get("pheat_radius_of_gyration") or {}
    atom_sets = normalize_metric_atom_sets(result.get("metric_atom_sets") or METRIC_ATOM_SETS)
    rows = []
    for label, key in (
        ("Reference", "reference"),
        ("Primary result", "final"),
        ("Primary - reference", "delta_final_minus_reference"),
    ):
        summary = rg.get(key) or {}
        for atom_set in atom_sets:
            payload = _rg_payload_for_atom_set(summary, atom_set)
            rows.append(
                "<tr>"
                f"<td>{html.escape(label)}</td>"
                f"<td><code>{html.escape(atom_set)}</code></td>"
                f"<td>{html.escape(_format_optional_float(_rg_value(payload, 'unweighted'), 6))}</td>"
                f"<td>{html.escape(_format_optional_float(_rg_value(payload, 'mass_weighted'), 6))}</td>"
                f"<td>{html.escape(str(payload.get('atom_count') or 'n/a'))}</td>"
                f"<td>{html.escape(str(payload.get('units') or 'angstrom'))}</td>"
                f"<td>{html.escape(str(payload.get('status') or 'n/a'))}</td>"
                "</tr>"
            )
    return "\n".join(rows) or '<tr><td colspan="7">No PHEAT radius of gyration metrics.</td></tr>'


def _structure_metrics_table_rows(details: Optional[dict], atom_sets: Sequence[str]) -> str:
    rows = []
    metrics = (details or {}).get("metrics") or {}
    for atom_set in normalize_metric_atom_sets(atom_sets):
        payload = metrics.get(atom_set) or {}
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(atom_set)}</code></td>"
            f"<td>{html.escape(_format_optional_angstrom(_metric_payload_value(payload), 6))}</td>"
            f"<td><code>{html.escape(str(payload.get('alignment_atom_set') or (details or {}).get('alignment_atom_set') or 'n/a'))}</code></td>"
            f"<td>{html.escape(str(payload.get('matched_atoms') or 'n/a'))}</td>"
            f"<td>{html.escape(str(payload.get('unmatched_reference_atoms', 'n/a')))}</td>"
            f"<td>{html.escape(str(payload.get('unmatched_target_atoms', 'n/a')))}</td>"
            f"<td>{html.escape(str(payload.get('units') or 'angstrom'))}</td>"
            f"<td>{html.escape(str(payload.get('status') or 'unavailable'))}</td>"
            "</tr>"
        )
    return "\n".join(rows) or '<tr><td colspan="8">No structure metrics.</td></tr>'


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
        citation_rows = "<li>No score-specific citations were reported by the selected score model.</li>"
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
      and scoring where a PHEAT score model is selected.
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
        "recipe": schedule.preset,
        "recipe_source": schedule.source,
        "preset": schedule.preset,
        "source": schedule.source,
        "config_path": schedule.config_path,
        "description": schedule.description,
        "fold": dict(schedule.fold),
        "basis_circuit_batching": schedule.basis_circuit_batching,
        "default_transpile": asdict(schedule.default_transpile),
        "gate_estimate_optimization_levels": list(schedule.gate_estimate_optimization_levels),
        "gate_estimate_transpile_seed": schedule.gate_estimate_transpile_seed,
        "circuit_template": schedule.circuit_template,
        "circuit": schedule.circuit,
        "scouting": asdict(schedule.scouting),
        "phases": [asdict(phase) for phase in schedule.phases],
        "result": asdict(schedule.result),
        "readouts": [asdict(readout) for readout in schedule.readouts],
        "metrics": asdict(schedule.metrics),
        "report": asdict(schedule.report),
        "evaluators": {name: asdict(config) for name, config in schedule.evaluators.items()},
        "phase_comparisons": asdict(schedule.phase_comparisons),
        "reranking": asdict(schedule.reranking),
        "phase_readiness": asdict(schedule.phase_readiness),
        "handoff_guard": asdict(schedule.handoff_guard),
        "validation": asdict(schedule.validation),
    }


def _load_yaml_mapping(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "YAML fold recipes require PyYAML. Install it in the active environment "
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
            "QTF fold recipe YAML validation requires jsonschema. Install it "
            "in the active environment with `python -m pip install jsonschema`."
        ) from exc
    if not PHASE_PRESET_SCHEMA_PATH.exists():
        raise FileNotFoundError(f"QTF fold recipe schema is missing: {PHASE_PRESET_SCHEMA_PATH}")
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
    recipes = data.get("recipes")
    if not isinstance(recipes, dict):
        raise ValueError(f"YAML recipe config {path} must define a top-level `recipes` mapping.")
    qtf_recipes = {}
    for name, config in recipes.items():
        if not isinstance(config, dict):
            continue
        recipe = copy.deepcopy(config)
        recipe.pop("engine", None)
        qtf_recipes[str(name)] = recipe
    return qtf_recipes


def _load_builtin_phase_presets() -> dict[str, dict]:
    if not PHASE_PRESET_DIR.exists():
        raise FileNotFoundError(f"Built-in QTF recipe directory is missing: {PHASE_PRESET_DIR}")
    paths = sorted(PHASE_PRESET_DIR.glob("*.yaml")) + sorted(PHASE_PRESET_DIR.glob("*.yml"))
    if not paths:
        raise FileNotFoundError(f"No built-in QTF recipe YAML files found in {PHASE_PRESET_DIR}")

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


def _normalize_stored_lengths_for_cli(value) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return ()
    return ResidueGeometryStructure(residues=[], stored_lengths=value).stored_lengths


def _normalize_selective_chi_mapping(value) -> dict[str, list[str]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("selective_chi_map must be a mapping.")
    normalized: dict[str, list[str]] = {}
    for residue, raw_items in value.items():
        residue_name = str(residue).strip()
        if not residue_name:
            raise ValueError("selective_chi_map residue names must not be blank.")
        if raw_items is None:
            items = []
        elif isinstance(raw_items, str):
            items = [item.strip() for item in raw_items.split(",") if item.strip()]
        else:
            items = [str(item).strip() for item in raw_items if str(item).strip()]
        normalized[residue_name] = items
    return normalized


PHASE_GEOMETRY_OPTION_KEYS = {
    "stored_angles",
    "stored_lengths",
    "max_chi",
    "selective_chi_map",
    "length_encoding_scope",
    "backbone_length_span",
    "sidechain_length_span",
}
PHASE_GEOMETRY_METADATA_KEYS = {"description"}
PHASE_GEOMETRY_ALLOWED_KEYS = PHASE_GEOMETRY_OPTION_KEYS | PHASE_GEOMETRY_METADATA_KEYS
PHASE_GEOMETRY_CLI_OPTION_KEYS = PHASE_GEOMETRY_OPTION_KEYS - {"selective_chi_map"}


def _phase_geometry_update_kwargs(geometry: Mapping[str, Any]) -> dict[str, Any]:
    if not geometry:
        return {}
    updates: dict[str, Any] = {}
    for key in PHASE_GEOMETRY_OPTION_KEYS:
        if key not in geometry:
            continue
        value = geometry[key]
        if key == "selective_chi_map":
            value = _normalize_selective_chi_mapping(value)
        updates[key] = value
    return updates


def _effective_initial_geometry(
    *,
    stored_angles,
    stored_lengths,
    max_chi,
    selective_chi_map,
    length_encoding_scope: str,
    backbone_length_span: float,
    sidechain_length_span: float,
    phase: Optional[PhaseConfig],
) -> dict[str, Any]:
    config = {
        "stored_angles": stored_angles,
        "stored_lengths": stored_lengths,
        "max_chi": max_chi,
        "selective_chi_map": selective_chi_map,
        "length_encoding_scope": length_encoding_scope,
        "backbone_length_span": backbone_length_span,
        "sidechain_length_span": sidechain_length_span,
    }
    if phase is not None:
        config.update(_phase_geometry_update_kwargs(phase.geometry))
    config["stored_angles"] = normalize_stored_angles(config["stored_angles"] or ())
    config["stored_lengths"] = _normalize_stored_lengths_for_cli(config["stored_lengths"])
    config["max_chi"] = normalize_max_chi(config["max_chi"])
    config["selective_chi_map"] = _normalize_selective_chi_mapping(config["selective_chi_map"])
    config["length_encoding_scope"] = str(config["length_encoding_scope"])
    config["backbone_length_span"] = float(config["backbone_length_span"])
    config["sidechain_length_span"] = float(config["sidechain_length_span"])
    return config


def _parse_selective_chi_map(
    parser: argparse.ArgumentParser,
    values: Sequence[str],
) -> dict[str, list[str]]:
    selective: dict[str, list[str]] = {}
    for raw in values or []:
        if "=" not in raw:
            parser.error(f"--selective-chi entries must use RES=chi1,chi2 syntax: {raw!r}")
        residue, raw_chis = raw.split("=", 1)
        residue = residue.strip()
        if not residue:
            parser.error("--selective-chi residue name must not be blank.")
        selective[residue] = [
            item.strip()
            for item in raw_chis.split(",")
            if item.strip()
        ]
    return selective


def _format_selective_chi_map(selective_chi_map: Mapping[str, Sequence[str]]) -> str:
    if not selective_chi_map:
        return "none"
    entries = []
    for residue in sorted(selective_chi_map):
        chis = [str(item) for item in selective_chi_map[residue]]
        entries.append(f"{residue}={','.join(chis) if chis else 'none'}")
    return "; ".join(entries)


def _chi_selection_summary(max_chi: Optional[int], selective_chi_map: Mapping[str, Sequence[str]]) -> str:
    cap = "all" if max_chi is None else f"chi<=chi{max_chi}"
    if selective_chi_map:
        return f"{cap}; selective map"
    return cap


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


def _parse_phase_option(parser: argparse.ArgumentParser, raw: str, flag: str = "--phase-option") -> tuple[str, str, Any]:
    if ":" not in raw:
        parser.error(f"{flag} entries must use NAME:key=value syntax: {raw!r}")
    name, remainder = raw.split(":", 1)
    name = name.strip()
    if not name:
        parser.error(f"{flag} phase name must not be blank.")
    key, value = _parse_assignment(parser, remainder, flag)
    return name, key, _parse_cli_scalar(value)


def _parse_key_value_option(parser: argparse.ArgumentParser, raw: str, flag: str) -> tuple[str, Any]:
    if "=" not in raw:
        parser.error(f"{flag} entries must use key=value syntax: {raw!r}")
    key, value = raw.split("=", 1)
    key = key.strip()
    if not key:
        parser.error(f"{flag} key must not be blank.")
    if not value.strip():
        parser.error(f"{flag} value must not be blank for key {key!r}.")
    return key, _parse_cli_scalar(value)


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
        hint = define_hint or "Define it with --phase or in the selected recipe."
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


def _nonnegative_int(parser: argparse.ArgumentParser, value, context: str) -> int:
    try:
        resolved = int(value)
    except Exception as exc:
        parser.error(f"{context} must be an integer.")
        raise AssertionError from exc
    if resolved < 0:
        parser.error(f"{context} must be >= 0.")
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


def _optional_description(parser: argparse.ArgumentParser, value, context: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        parser.error(f"{context} description must be a string.")
    text = value.strip()
    return text or None


def _positive_float(parser: argparse.ArgumentParser, value, context: str) -> float:
    resolved = _optional_float(parser, value, context)
    if resolved is None:
        parser.error(f"{context} must be provided.")
    if resolved <= 0.0:
        parser.error(f"{context} must be > 0.")
    return resolved


def _phase_options(parser: argparse.ArgumentParser, value, context: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        parser.error(f"{context} options must be a mapping.")
    if "maxiter" in value:
        parser.error(f"{context} options must not contain maxiter; use top-level maxiter.")
    return dict(value)


def _phase_minimize_options(phase: PhaseConfig) -> dict[str, Any]:
    options = dict(phase.options)
    options["maxiter"] = phase.maxiter
    if phase.optimizer in {"Powell", "Nelder-Mead"}:
        maxfev = options.get("maxfev")
        if maxfev is None or int(maxfev) > int(phase.maxiter):
            options["maxfev"] = int(phase.maxiter)
    return options


def _score_options(parser: argparse.ArgumentParser, value, context: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        parser.error(f"{context} score_options must be a mapping.")
    return dict(value)


def _phase_geometry_options(parser: argparse.ArgumentParser, value, context: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        parser.error(f"{context} geometry must be a mapping.")
    options = dict(value)
    if "description" in options:
        options["description"] = _optional_description(parser, options.get("description"), f"{context} geometry")
    unknown = sorted(str(key) for key in options if str(key) not in PHASE_GEOMETRY_ALLOWED_KEYS)
    if unknown:
        parser.error(
            f"{context} geometry contains unsupported keys: {', '.join(unknown)}. "
            f"Supported keys: {', '.join(sorted(PHASE_GEOMETRY_ALLOWED_KEYS))}."
        )
    return options


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
    model = str(value).strip().lower().replace("_", "-")
    try:
        return canonical_score_model(model)
    except ValueError as exc:
        parser.error(f"{context} score_model {exc}")
    raise AssertionError("unreachable")


def _normalize_supported_pheat_score_model(parser: argparse.ArgumentParser, value, context: str) -> str:
    if value is None or not str(value).strip():
        parser.error(f"{context} score_model is required.")
    model = str(value).strip().lower().replace("_", "-")
    capabilities = _pheat_capabilities_by_public_name()
    if model not in capabilities:
        supported = ", ".join(sorted(capabilities)) or "none"
        parser.error(f"{context} score_model must be one of supported PHEAT models: {supported}.")
    return model


def _pheat_capabilities_by_public_name() -> dict[str, dict[str, Any]]:
    return {
        str(item.get("public_model") or item.get("model")): dict(item)
        for item in pheat_score_model_capabilities()
    }


def _pheat_raw_model_name(public_model: str) -> str:
    capabilities = _pheat_capabilities_by_public_name()
    capability = capabilities.get(str(public_model))
    if capability is None:
        return str(public_model).removeprefix("pheat-")
    return str(capability.get("pheat_model") or capability.get("raw_model") or public_model)


def _normalize_basis_circuit_batching(parser: argparse.ArgumentParser, value, context: str) -> str:
    mode = "auto" if value is None else str(value).strip().lower()
    if mode not in BASIS_CIRCUIT_BATCHING_MODES:
        parser.error(
            f"{context} must be one of {', '.join(BASIS_CIRCUIT_BATCHING_MODES)}."
        )
    return mode


def _normalize_transpile_optimization_level(parser: argparse.ArgumentParser, value, context: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "default", "auto"}:
        return None
    try:
        level = int(value)
    except (TypeError, ValueError):
        parser.error(f"{context} must be one of none, 0, 1, 2, or 3.")
    if level not in TRANSPILE_OPTIMIZATION_LEVELS:
        parser.error(f"{context} must be one of none, 0, 1, 2, or 3.")
    return level


def _normalize_transpile_seed(parser: argparse.ArgumentParser, value, context: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "default", "auto"}:
        return None
    try:
        seed = int(value)
    except (TypeError, ValueError):
        parser.error(f"{context} must be a non-negative integer or none.")
    if seed < 0:
        parser.error(f"{context} must be a non-negative integer or none.")
    return seed


def _raw_config_value(mapping: Mapping[str, Any], key: str, default=None):
    return mapping[key] if key in mapping else default


def _transpile_config(
    parser: argparse.ArgumentParser,
    *,
    optimization_level,
    seed,
    context: str,
    description=None,
) -> TranspileConfig:
    return TranspileConfig(
        optimization_level=_normalize_transpile_optimization_level(
            parser,
            optimization_level,
            f"{context} transpile optimization_level",
        ),
        seed=_normalize_transpile_seed(parser, seed, f"{context} transpile seed"),
        description=_optional_description(parser, description, context),
    )


def _transpile_config_dict(config: Optional[TranspileConfig]) -> dict[str, Any]:
    if config is None:
        return {"optimization_level": None, "seed": None}
    payload = {
        "optimization_level": config.optimization_level,
        "seed": config.seed,
    }
    if config.description is not None:
        payload["description"] = config.description
    return payload


def _transpile_kwargs(config: Optional[TranspileConfig]) -> dict[str, int]:
    kwargs: dict[str, int] = {}
    if config is None:
        return kwargs
    if config.optimization_level is not None:
        kwargs["optimization_level"] = int(config.optimization_level)
    if config.seed is not None:
        kwargs["seed_transpiler"] = int(config.seed)
    return kwargs


def _normalize_gate_estimate_optimization_levels(
    parser: argparse.ArgumentParser,
    value,
    context: str,
) -> list[int]:
    if value is None:
        requested = list(DEFAULT_GATE_ESTIMATE_OPTIMIZATION_LEVELS)
    elif isinstance(value, str):
        requested = [item.strip() for item in value.split(",")]
    elif isinstance(value, (list, tuple)):
        requested = list(value)
    else:
        parser.error(f"{context} must be a comma-separated list or sequence of levels.")

    levels: list[int] = []
    for item in requested:
        if item is None or (isinstance(item, str) and not item.strip()):
            continue
        level = _normalize_transpile_optimization_level(parser, item, context)
        if level is None:
            parser.error(f"{context} cannot include none; use levels 0, 1, 2, or 3.")
        if level not in levels:
            levels.append(level)
    if 0 not in levels:
        levels.insert(0, 0)
    if not any(level > 0 for level in levels):
        levels.append(3)
    return levels


def _resolve_circuit_config(args, parser: argparse.ArgumentParser, raw_recipe: dict) -> tuple[Optional[dict], Optional[dict]]:
    circuit_template = copy.deepcopy(raw_recipe.get("circuit_template"))
    circuit = copy.deepcopy(raw_recipe.get("circuit"))

    cli_template_requested = any(
        item is not None
        for item in (
            args.circuit_template,
            args.circuit_template_source,
        )
    ) or bool(args.circuit_template_option)
    cli_circuit_requested = any(
        item is not None
        for item in (
            args.circuit_source,
            args.circuit_path,
            args.circuit_index,
        )
    )

    if cli_template_requested:
        circuit_template = dict(circuit_template or {})
        if args.circuit_template is not None:
            circuit_template["name"] = args.circuit_template
        if args.circuit_template_source is not None:
            circuit_template["source"] = args.circuit_template_source
        options = dict(circuit_template.get("options") or {})
        for raw_value in args.circuit_template_option or []:
            key, value = _parse_key_value_option(parser, raw_value, "--circuit-template-option")
            options[key] = value
        circuit_template["options"] = options
        if cli_circuit_requested:
            parser.error("--circuit-* and --circuit-template-* options are mutually exclusive.")
        circuit = None

    if cli_circuit_requested:
        circuit = dict(circuit or {})
        if args.circuit_source is not None:
            circuit["source"] = args.circuit_source
        if args.circuit_path is not None:
            circuit["path"] = args.circuit_path
        if args.circuit_index is not None:
            circuit["index"] = args.circuit_index
        circuit_template = None

    if circuit_template and circuit:
        parser.error("Recipe must define either circuit_template or circuit, not both.")

    if circuit:
        if not isinstance(circuit, dict):
            parser.error("circuit must be a mapping.")
        source = str(circuit.get("source") or "").strip().lower()
        if source not in {"qpy", "qasm2", "qasm3"}:
            parser.error("circuit source must be one of qpy, qasm2, or qasm3.")
        if not circuit.get("path"):
            parser.error("circuit path is required.")
        resolved = dict(circuit)
        if "index" in resolved:
            resolved["index"] = int(resolved["index"])
        if "description" in resolved:
            resolved["description"] = _optional_description(parser, resolved.get("description"), "circuit")
        return None, resolved

    if circuit_template is None:
        circuit_template = copy.deepcopy(DEFAULT_CIRCUIT_TEMPLATE)
    if not isinstance(circuit_template, dict):
        parser.error("circuit_template must be a mapping.")
    source = str(circuit_template.get("source") or "qiskit-library").strip().lower()
    if source not in {"qiskit-library", "qtf"}:
        parser.error("circuit_template source must be one of qiskit-library or qtf.")
    name = str(circuit_template.get("name") or "").strip()
    if not name:
        parser.error("circuit_template name must not be blank.")
    resolved_template = {
        "source": source,
        "name": name,
        "options": dict(circuit_template.get("options") or {}),
    }
    description = _optional_description(parser, circuit_template.get("description"), "circuit_template")
    if description is not None:
        resolved_template["description"] = description
    return resolved_template, None


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


def _effective_backend_spec(spec: Optional[str], inherited_spec: Optional[str]) -> str:
    normalized = "inherit" if spec is None else str(spec).strip()
    if normalized.lower() in INHERIT_BACKEND_VALUES:
        if inherited_spec is None:
            return "none"
        inherited = str(inherited_spec).strip()
        return inherited if inherited else "none"
    return normalized


def _backend_spec_display(spec: Optional[str]) -> str:
    if spec is None:
        return "exact-statevector"
    normalized = str(spec).strip()
    return normalized if normalized else "exact-statevector"


def _config_bool(parser: argparse.ArgumentParser, value, context: str, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    parser.error(f"{context} must be a boolean.")
    raise AssertionError("unreachable")


def _config_string_list(parser: argparse.ArgumentParser, value, context: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    elif isinstance(value, (list, tuple)):
        items = [str(item).strip() for item in value]
    else:
        parser.error(f"{context} must be a string or list of strings.")
    return [item for item in items if item]


def _normalize_evaluator_configs(
    parser: argparse.ArgumentParser,
    raw_evaluators,
) -> dict[str, EvaluatorConfig]:
    if raw_evaluators is None:
        return {}
    if not isinstance(raw_evaluators, dict):
        parser.error("evaluators must be a mapping.")
    evaluators: dict[str, EvaluatorConfig] = {}
    for name, raw in raw_evaluators.items():
        evaluator_name = str(name).strip()
        if not evaluator_name:
            parser.error("evaluator names must not be blank.")
        if not isinstance(raw, dict):
            parser.error(f"evaluator {evaluator_name!r} must be a mapping.")
        score_model = _normalize_supported_pheat_score_model(
            parser,
            raw.get("score_model"),
            f"evaluator {evaluator_name!r}",
        )
        options = raw.get("options") or {}
        if not isinstance(options, dict):
            parser.error(f"evaluator {evaluator_name!r} options must be a mapping.")
        evaluators[evaluator_name] = EvaluatorConfig(
            name=evaluator_name,
            score_model=score_model,
            required=_config_bool(parser, raw.get("required"), f"evaluator {evaluator_name!r} required", False),
            options=dict(options),
            description=_optional_description(parser, raw.get("description"), f"evaluator {evaluator_name!r}"),
        )
    return evaluators


def _validate_evaluator_references(
    parser: argparse.ArgumentParser,
    evaluator_names: Sequence[str],
    evaluators: dict[str, EvaluatorConfig],
    context: str,
) -> list[str]:
    names = [str(name).strip() for name in evaluator_names if str(name).strip()]
    for name in names:
        if name not in evaluators:
            parser.error(f"{context} references unknown evaluator {name!r}.")
    return names


def _normalize_phase_comparison_config(
    parser: argparse.ArgumentParser,
    raw_config,
    evaluators: dict[str, EvaluatorConfig],
) -> PhaseComparisonConfig:
    if raw_config is not None and not isinstance(raw_config, dict):
        parser.error("phase_comparisons must be a mapping.")
    raw = dict(raw_config or {})
    evaluator_names = _validate_evaluator_references(
        parser,
        _config_string_list(parser, raw.get("evaluators"), "phase_comparisons evaluators"),
        evaluators,
        "phase_comparisons",
    )
    compare = str(raw.get("compare") or "consecutive_phase_ends").strip()
    if compare != "consecutive_phase_ends":
        parser.error("phase_comparisons compare must be 'consecutive_phase_ends'.")
    return PhaseComparisonConfig(
        enabled=_config_bool(parser, raw.get("enabled"), "phase_comparisons enabled", False),
        evaluators=evaluator_names,
        compare=compare,
        affect_selection=_config_bool(parser, raw.get("affect_selection"), "phase_comparisons affect_selection", False),
        description=_optional_description(parser, raw.get("description"), "phase_comparisons"),
    )


def _normalize_reranking_config(
    parser: argparse.ArgumentParser,
    raw_config,
    evaluators: dict[str, EvaluatorConfig],
) -> RerankingConfig:
    if raw_config is not None and not isinstance(raw_config, dict):
        parser.error("reranking must be a mapping.")
    raw = dict(raw_config or {})
    evaluator = raw.get("evaluator")
    evaluator_name = str(evaluator).strip() if evaluator is not None else None
    if evaluator_name:
        _validate_evaluator_references(parser, [evaluator_name], evaluators, "reranking")
    triggers = raw.get("triggers") or []
    if triggers and not isinstance(triggers, list):
        parser.error("reranking triggers must be a list.")
    normalized_triggers = []
    for index, trigger in enumerate(triggers, start=1):
        if not isinstance(trigger, dict):
            parser.error(f"reranking trigger #{index} must be a mapping.")
        when = str(trigger.get("when") or "").strip()
        if when not in {"phase_end", "every_evaluations"}:
            parser.error("reranking trigger when must be 'phase_end' or 'every_evaluations'.")
        item = dict(trigger)
        item["when"] = when
        if when == "every_evaluations":
            item["interval"] = _positive_int(parser, item.get("interval"), "reranking every_evaluations interval")
        normalized_triggers.append(item)
    candidate_pool = raw.get("candidate_pool") or {}
    if not isinstance(candidate_pool, dict):
        parser.error("reranking candidate_pool must be a mapping.")
    apply_mode = str(raw.get("apply") or "next_phase_start").strip()
    if apply_mode not in {"next_phase_start", "report_only"}:
        parser.error("reranking apply must be 'next_phase_start' or 'report_only'.")
    return RerankingConfig(
        enabled=_config_bool(parser, raw.get("enabled"), "reranking enabled", False),
        evaluator=evaluator_name,
        triggers=normalized_triggers,
        candidate_pool=dict(candidate_pool),
        apply=apply_mode,
        description=_optional_description(parser, raw.get("description"), "reranking"),
    )


def _normalize_phase_readiness_config(
    parser: argparse.ArgumentParser,
    raw_config,
    evaluators: dict[str, EvaluatorConfig],
) -> PhaseReadinessConfig:
    if raw_config is not None and not isinstance(raw_config, dict):
        parser.error("phase_readiness must be a mapping.")
    raw = dict(raw_config or {})
    evaluator = raw.get("evaluator")
    evaluator_name = str(evaluator).strip() if evaluator is not None else None
    if evaluator_name:
        _validate_evaluator_references(parser, [evaluator_name], evaluators, "phase_readiness")
    phases = _config_string_list(parser, raw.get("phases"), "phase_readiness phases")
    on_fail = str(raw.get("on_fail") or "continue").strip()
    if on_fail not in {"continue", "skip_phase"}:
        parser.error("phase_readiness on_fail must be 'continue' or 'skip_phase'.")
    max_clash = raw.get("max_clash_count")
    max_short = raw.get("max_short_contact_count")
    min_distance = raw.get("min_nonlocal_distance_a")
    return PhaseReadinessConfig(
        enabled=_config_bool(parser, raw.get("enabled"), "phase_readiness enabled", False),
        evaluator=evaluator_name,
        phases=phases,
        on_fail=on_fail,
        max_clash_count=None if max_clash is None else _nonnegative_int(parser, max_clash, "phase_readiness max_clash_count"),
        max_short_contact_count=None if max_short is None else _nonnegative_int(parser, max_short, "phase_readiness max_short_contact_count"),
        min_nonlocal_distance_a=None if min_distance is None else _positive_float(parser, min_distance, "phase_readiness min_nonlocal_distance_a"),
        description=_optional_description(parser, raw.get("description"), "phase_readiness"),
    )


def _normalize_handoff_guard_config(
    parser: argparse.ArgumentParser,
    raw_config,
    evaluators: dict[str, EvaluatorConfig],
) -> HandoffGuardConfig:
    if raw_config is not None and not isinstance(raw_config, dict):
        parser.error("handoff_guard must be a mapping.")
    raw = dict(raw_config or {})
    evaluator = raw.get("evaluator")
    evaluator_name = str(evaluator).strip() if evaluator is not None else None
    if evaluator_name:
        _validate_evaluator_references(parser, [evaluator_name], evaluators, "handoff_guard")
    phases = _config_string_list(parser, raw.get("phases"), "handoff_guard phases")
    fallback = str(raw.get("fallback") or "phase_start").strip()
    if fallback != "phase_start":
        parser.error("handoff_guard fallback must be 'phase_start'.")
    max_clash = raw.get("max_clash_count")
    max_clash_count = None if max_clash is None else _nonnegative_int(
        parser,
        max_clash,
        "handoff_guard max_clash_count",
    )
    max_short = raw.get("max_short_contact_count")
    max_short_contact_count = None if max_short is None else _nonnegative_int(
        parser,
        max_short,
        "handoff_guard max_short_contact_count",
    )
    min_distance = raw.get("min_nonlocal_distance_a")
    min_nonlocal_distance_a = None if min_distance is None else _positive_float(
        parser,
        min_distance,
        "handoff_guard min_nonlocal_distance_a",
    )
    unsafe_max_short = raw.get("unsafe_transition_max_short_contact_count")
    unsafe_transition_max_short_contact_count = None if unsafe_max_short is None else _nonnegative_int(
        parser,
        unsafe_max_short,
        "handoff_guard unsafe_transition_max_short_contact_count",
    )
    unsafe_min_distance = raw.get("unsafe_transition_min_nonlocal_distance_a")
    unsafe_transition_min_nonlocal_distance_a = None if unsafe_min_distance is None else _positive_float(
        parser,
        unsafe_min_distance,
        "handoff_guard unsafe_transition_min_nonlocal_distance_a",
    )
    return HandoffGuardConfig(
        enabled=_config_bool(parser, raw.get("enabled"), "handoff_guard enabled", False),
        evaluator=evaluator_name,
        phases=phases,
        fallback=fallback,
        abort_on_reject=_config_bool(parser, raw.get("abort_on_reject"), "handoff_guard abort_on_reject", False),
        allow_improving_unsafe=_config_bool(
            parser,
            raw.get("allow_improving_unsafe"),
            "handoff_guard allow_improving_unsafe",
            True,
        ),
        max_clash_count=max_clash_count,
        max_short_contact_count=max_short_contact_count,
        min_nonlocal_distance_a=min_nonlocal_distance_a,
        unsafe_transition_max_short_contact_count=unsafe_transition_max_short_contact_count,
        unsafe_transition_min_nonlocal_distance_a=unsafe_transition_min_nonlocal_distance_a,
        unsafe_transition_require_clash_count_decrease=_config_bool(
            parser,
            raw.get("unsafe_transition_require_clash_count_decrease"),
            "handoff_guard unsafe_transition_require_clash_count_decrease",
            False,
        ),
        reject_on_score_worse=_config_bool(parser, raw.get("reject_on_score_worse"), "handoff_guard reject_on_score_worse", True),
        reject_on_clash_count_increase=_config_bool(parser, raw.get("reject_on_clash_count_increase"), "handoff_guard reject_on_clash_count_increase", True),
        reject_on_short_contact_count_increase=_config_bool(parser, raw.get("reject_on_short_contact_count_increase"), "handoff_guard reject_on_short_contact_count_increase", True),
        reject_on_min_nonlocal_distance_decrease=_config_bool(
            parser,
            raw.get("reject_on_min_nonlocal_distance_decrease"),
            "handoff_guard reject_on_min_nonlocal_distance_decrease",
            True,
        ),
        reject_on_nonfinite=_config_bool(parser, raw.get("reject_on_nonfinite"), "handoff_guard reject_on_nonfinite", True),
        description=_optional_description(parser, raw.get("description"), "handoff_guard"),
    )


def _normalize_phase_handoff_guard_options(
    parser: argparse.ArgumentParser,
    raw_config,
    context: str,
) -> dict[str, Any]:
    if raw_config is None:
        return {}
    if not isinstance(raw_config, dict):
        parser.error(f"{context} handoff_guard must be a mapping.")
    raw = dict(raw_config)
    options: dict[str, Any] = {}
    for key, value in raw.items():
        field_context = f"{context} handoff_guard {key}"
        if key == "enabled":
            options[key] = _config_bool(parser, value, field_context, False)
        elif key == "description":
            options[key] = _optional_description(parser, value, f"{context} handoff_guard")
        elif key == "fallback":
            fallback = str(value or "phase_start").strip()
            if fallback != "phase_start":
                parser.error(f"{field_context} must be 'phase_start'.")
            options[key] = fallback
        elif key in {
            "abort_on_reject",
            "allow_improving_unsafe",
            "unsafe_transition_require_clash_count_decrease",
            "reject_on_score_worse",
            "reject_on_clash_count_increase",
            "reject_on_short_contact_count_increase",
            "reject_on_min_nonlocal_distance_decrease",
            "reject_on_nonfinite",
        }:
            options[key] = _config_bool(parser, value, field_context, False)
        elif key in {"max_clash_count", "max_short_contact_count", "unsafe_transition_max_short_contact_count"}:
            options[key] = None if value is None else _nonnegative_int(parser, value, field_context)
        elif key in {"min_nonlocal_distance_a", "unsafe_transition_min_nonlocal_distance_a"}:
            options[key] = None if value is None else _positive_float(parser, value, field_context)
        else:
            allowed = (
                "enabled, fallback, abort_on_reject, allow_improving_unsafe, max_clash_count, "
                "max_short_contact_count, min_nonlocal_distance_a, unsafe_transition_max_short_contact_count, "
                "unsafe_transition_min_nonlocal_distance_a, unsafe_transition_require_clash_count_decrease, "
                "reject_on_score_worse, reject_on_clash_count_increase, reject_on_short_contact_count_increase, "
                "reject_on_min_nonlocal_distance_decrease, reject_on_nonfinite, description"
            )
            parser.error(f"{field_context} is not supported. Supported keys: {allowed}.")
    return options


def _normalize_validation_config(
    parser: argparse.ArgumentParser,
    raw_config,
    evaluators: dict[str, EvaluatorConfig],
) -> ValidationConfig:
    if raw_config is not None and not isinstance(raw_config, dict):
        parser.error("validation must be a mapping.")
    raw = dict(raw_config or {})
    candidate_sets = _config_string_list(parser, raw.get("candidates"), "validation candidates")
    for candidate in candidate_sets:
        if candidate not in {"primary", "phase_ends", "reranked_top"}:
            parser.error("validation candidates must be one of primary, phase_ends, reranked_top.")
    evaluator_names = _validate_evaluator_references(
        parser,
        _config_string_list(parser, raw.get("evaluators"), "validation evaluators"),
        evaluators,
        "validation",
    )
    return ValidationConfig(
        enabled=_config_bool(parser, raw.get("enabled"), "validation enabled", False),
        candidates=candidate_sets,
        evaluators=evaluator_names,
        description=_optional_description(parser, raw.get("description"), "validation"),
    )


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
            f"--recipe {args.phase_preset!r} was not found. "
            f"Available recipes: {', '.join(sorted(presets)) or 'none'}"
        )

    raw = copy.deepcopy(presets.get(args.phase_preset, {}))
    recipe_description = _optional_description(parser, raw.get("description"), "recipe")
    fold_raw = copy.deepcopy(raw.get("fold") or {})
    if not isinstance(fold_raw, dict):
        parser.error("Selected recipe fold block must be a mapping when present.")
    if "description" in fold_raw:
        fold_raw["description"] = _optional_description(parser, fold_raw.get("description"), "fold")
    scouting_raw = copy.deepcopy(raw.get("scouting") or {})
    result_raw = copy.deepcopy(raw.get("result") or {})
    phases_raw = copy.deepcopy(raw.get("phases") or [])
    readouts_raw = copy.deepcopy(raw.get("readouts") or [])
    metrics_raw = copy.deepcopy(raw.get("metrics") or {})
    report_raw = copy.deepcopy(raw.get("report") or {})
    evaluators_raw = copy.deepcopy(raw.get("evaluators") or {})
    phase_comparisons_raw = copy.deepcopy(raw.get("phase_comparisons") or {})
    reranking_raw = copy.deepcopy(raw.get("reranking") or {})
    phase_readiness_raw = copy.deepcopy(raw.get("phase_readiness") or {})
    handoff_guard_raw = copy.deepcopy(raw.get("handoff_guard") or {})
    validation_raw = copy.deepcopy(raw.get("validation") or {})
    transpile_raw = copy.deepcopy(raw.get("transpile") or {})
    basis_circuit_batching_raw = raw.get("basis_circuit_batching", "auto")
    circuit_template, circuit = _resolve_circuit_config(args, parser, raw)

    if args.phase:
        phases_raw = [{"name": name} for name in args.phase]

    if not isinstance(phases_raw, list):
        parser.error("Selected recipe must define phases as a list.")
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
    for raw_value in args.phase_optimizer_transpile_optimization_level or []:
        _set_phase_override(
            parser,
            phases_by_name,
            raw_value,
            "--phase-optimizer-transpile-optimization-level",
            "optimizer_transpile_optimization_level",
        )
    for raw_value in args.phase_readout_transpile_optimization_level or []:
        _set_phase_override(
            parser,
            phases_by_name,
            raw_value,
            "--phase-readout-transpile-optimization-level",
            "readout_transpile_optimization_level",
        )
    for raw_value in args.phase_optimizer_transpile_seed or []:
        _set_phase_override(
            parser,
            phases_by_name,
            raw_value,
            "--phase-optimizer-transpile-seed",
            "optimizer_transpile_seed",
        )
    for raw_value in args.phase_readout_transpile_seed or []:
        _set_phase_override(
            parser,
            phases_by_name,
            raw_value,
            "--phase-readout-transpile-seed",
            "readout_transpile_seed",
        )
    for raw_value in args.phase_tol or []:
        _set_phase_override(parser, phases_by_name, raw_value, "--phase-tol", "tol", cast=float)
    for raw_value in args.phase_option or []:
        name, key, value = _parse_phase_option(parser, raw_value, "--phase-option")
        if name not in phases_by_name:
            parser.error(
                f"--phase-option references unknown phase {name!r}. "
                "Define it with --phase or in the selected recipe."
            )
        phases_by_name[name].setdefault("options", {})[key] = value
    for raw_value in args.phase_score_option or []:
        name, key, value = _parse_phase_option(parser, raw_value, "--phase-score-option")
        if name not in phases_by_name:
            parser.error(
                f"--phase-score-option references unknown phase {name!r}. "
                "Define it with --phase or in the selected recipe."
            )
        phases_by_name[name].setdefault("score_options", {})[key] = value
    for raw_value in args.phase_geometry_option or []:
        name, key, value = _parse_phase_option(parser, raw_value, "--phase-geometry-option")
        if name not in phases_by_name:
            parser.error(
                f"--phase-geometry-option references unknown phase {name!r}. "
                "Define it with --phase or in the selected recipe."
            )
        if key not in PHASE_GEOMETRY_CLI_OPTION_KEYS:
            allowed = ", ".join(sorted(PHASE_GEOMETRY_CLI_OPTION_KEYS))
            parser.error(f"--phase-geometry-option unknown geometry key {key!r}. Supported keys: {allowed}.")
        phases_by_name[name].setdefault("geometry", {})[key] = value

    if args.scouting_score is not None:
        scouting_raw["score_model"] = args.scouting_score
    if args.scouting_backend is not None:
        scouting_raw["backend"] = args.scouting_backend
    if args.scouting_shots is not None:
        scouting_raw["shots"] = args.scouting_shots
    if args.scouting_attempts is not None:
        scouting_raw["attempts"] = args.scouting_attempts
    if args.scouting_transpile_optimization_level is not None:
        scouting_raw["transpile_optimization_level"] = args.scouting_transpile_optimization_level
    if args.scouting_transpile_seed is not None:
        scouting_raw["transpile_seed"] = args.scouting_transpile_seed
    for raw_value in args.scouting_score_option or []:
        key, value = _parse_key_value_option(parser, raw_value, "--scouting-score-option")
        scouting_raw.setdefault("score_options", {})[key] = value
    if args.result_score is not None:
        result_raw["score_model"] = args.result_score
    elif result_raw.get("score_model") is None:
        result_raw["score_model"] = DEFAULT_RESULT_SCORE_MODEL
    for raw_value in args.result_score_option or []:
        key, value = _parse_key_value_option(parser, raw_value, "--result-score-option")
        result_raw.setdefault("score_options", {})[key] = value
    if args.primary_result is not None:
        result_raw["primary"] = args.primary_result
    if args.basis_circuit_batching is not None:
        basis_circuit_batching_raw = args.basis_circuit_batching
    if args.metric_atom_sets is not None:
        metrics_raw["atom_sets"] = args.metric_atom_sets
    if args.rmsd_alignment_atom_set is not None:
        metrics_raw["rmsd_alignment_atom_set"] = args.rmsd_alignment_atom_set
    if args.report_structure_domain is not None:
        report_raw["structure_domain"] = args.report_structure_domain

    if not isinstance(readouts_raw, list):
        parser.error("Selected recipe must define readouts as a list when present.")
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
            define_hint="Define it with --readout or in the selected recipe.",
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
            define_hint="Define it with --readout or in the selected recipe.",
        )
    for raw_value in args.readout_score or []:
        _set_phase_override(
            parser,
            readouts_by_name,
            raw_value,
            "--readout-score",
            "score_model",
            item_label="readout",
            define_hint="Define it with --readout or in the selected recipe.",
        )
    for raw_value in args.readout_transpile_optimization_level or []:
        _set_phase_override(
            parser,
            readouts_by_name,
            raw_value,
            "--readout-transpile-optimization-level",
            "transpile_optimization_level",
            item_label="readout",
            define_hint="Define it with --readout or in the selected recipe.",
        )
    for raw_value in args.readout_transpile_seed or []:
        _set_phase_override(
            parser,
            readouts_by_name,
            raw_value,
            "--readout-transpile-seed",
            "transpile_seed",
            item_label="readout",
            define_hint="Define it with --readout or in the selected recipe.",
        )

    if not phase_names:
        parser.error("The resolved phase schedule must contain at least one phase.")

    if args.transpile_optimization_level is not None:
        transpile_raw["optimization_level"] = args.transpile_optimization_level
    if args.transpile_seed is not None:
        transpile_raw["seed"] = args.transpile_seed
    default_transpile = _transpile_config(
        parser,
        optimization_level=transpile_raw.get("optimization_level"),
        seed=transpile_raw.get("seed"),
        context="default",
        description=transpile_raw.get("description"),
    )
    gate_estimate_levels_raw = _raw_config_value(
        transpile_raw,
        "gate_estimate_optimization_levels",
        DEFAULT_GATE_ESTIMATE_OPTIMIZATION_LEVELS,
    )
    gate_estimate_seed_raw = _raw_config_value(
        transpile_raw,
        "gate_estimate_seed",
        _raw_config_value(transpile_raw, "gate_estimate_transpile_seed", None),
    )
    if args.gate_estimate_optimization_levels is not None:
        gate_estimate_levels_raw = args.gate_estimate_optimization_levels
    if args.gate_estimate_transpile_seed is not None:
        gate_estimate_seed_raw = args.gate_estimate_transpile_seed
    gate_estimate_optimization_levels = _normalize_gate_estimate_optimization_levels(
        parser,
        gate_estimate_levels_raw,
        "gate estimate optimization levels",
    )
    gate_estimate_transpile_seed = _normalize_transpile_seed(
        parser,
        gate_estimate_seed_raw,
        "gate estimate transpile seed",
    )

    scouting_score = _normalize_score_model(
        parser,
        scouting_raw.get("score_model", DEFAULT_RESULT_SCORE_MODEL),
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
        score_options=_score_options(parser, scouting_raw.get("score_options"), "scouting"),
        transpile=_transpile_config(
            parser,
            optimization_level=_raw_config_value(
                scouting_raw,
                "transpile_optimization_level",
                default_transpile.optimization_level,
            ),
            seed=_raw_config_value(scouting_raw, "transpile_seed", default_transpile.seed),
            context="scouting",
        ),
        description=_optional_description(parser, scouting_raw.get("description"), "scouting"),
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
        phase_handoff_guard = _normalize_phase_handoff_guard_options(
            parser,
            phase_raw.get("handoff_guard"),
            phase_context,
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
                score_options=_score_options(parser, phase_raw.get("score_options"), phase_context),
                geometry=_phase_geometry_options(parser, phase_raw.get("geometry"), phase_context),
                handoff_guard=phase_handoff_guard,
                optimizer_transpile=_transpile_config(
                    parser,
                    optimization_level=_raw_config_value(
                        phase_raw,
                        "optimizer_transpile_optimization_level",
                        _raw_config_value(
                            phase_raw,
                            "transpile_optimization_level",
                            default_transpile.optimization_level,
                        ),
                    ),
                    seed=_raw_config_value(
                        phase_raw,
                        "optimizer_transpile_seed",
                        _raw_config_value(phase_raw, "transpile_seed", default_transpile.seed),
                    ),
                    context=f"{phase_context} optimizer",
                ),
                readout_transpile=_transpile_config(
                    parser,
                    optimization_level=_raw_config_value(
                        phase_raw,
                        "readout_transpile_optimization_level",
                        _raw_config_value(
                            phase_raw,
                            "transpile_optimization_level",
                            default_transpile.optimization_level,
                        ),
                    ),
                    seed=_raw_config_value(
                        phase_raw,
                        "readout_transpile_seed",
                        _raw_config_value(phase_raw, "transpile_seed", default_transpile.seed),
                    ),
                    context=f"{phase_context} readout",
                ),
                description=_optional_description(parser, phase_raw.get("description"), phase_context),
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
                transpile=_transpile_config(
                    parser,
                    optimization_level=_raw_config_value(
                        readout_raw,
                        "transpile_optimization_level",
                        default_transpile.optimization_level,
                    ),
                    seed=_raw_config_value(readout_raw, "transpile_seed", default_transpile.seed),
                    context=readout_context,
                ),
                description=_optional_description(parser, readout_raw.get("description"), readout_context),
            )
        )
    if primary_result == PRIMARY_LAST_PHASE and len(primary_flags) == 1 and args.primary_result is None:
        primary_result = primary_flags[0]
    if primary_result != PRIMARY_LAST_PHASE and primary_result not in {readout.name for readout in readouts}:
        parser.error(
            "--primary-result must be 'last_phase_structure' or the name of a configured readout. "
            f"Configured readouts: {', '.join(readout.name for readout in readouts) or 'none'}"
        )
    try:
        metric_atom_sets = normalize_metric_atom_sets(metrics_raw.get("atom_sets") or METRIC_ATOM_SETS)
        rmsd_alignment_atom_set = normalize_rmsd_alignment_atom_set(
            metrics_raw.get("rmsd_alignment_atom_set") or DEFAULT_RMSD_ALIGNMENT_ATOM_SET
        )
        report_structure_domain = _report_structure_domain(
            report_raw.get("structure_domain") or DEFAULT_REPORT_STRUCTURE_DOMAIN
        )
    except Exception as exc:
        parser.error(str(exc))
        raise AssertionError from exc
    evaluators = _normalize_evaluator_configs(parser, evaluators_raw)
    phase_comparisons = _normalize_phase_comparison_config(parser, phase_comparisons_raw, evaluators)
    reranking = _normalize_reranking_config(parser, reranking_raw, evaluators)
    phase_readiness = _normalize_phase_readiness_config(parser, phase_readiness_raw, evaluators)
    handoff_guard = _normalize_handoff_guard_config(parser, handoff_guard_raw, evaluators)
    validation = _normalize_validation_config(parser, validation_raw, evaluators)
    if phase_comparisons.enabled and not phase_comparisons.evaluators:
        parser.error("phase_comparisons enabled requires at least one evaluator.")
    if reranking.enabled and not reranking.evaluator:
        parser.error("reranking enabled requires evaluator.")
    if phase_readiness.enabled and not phase_readiness.evaluator:
        parser.error("phase_readiness enabled requires evaluator.")
    if phase_readiness.enabled and not phase_readiness.phases:
        parser.error("phase_readiness enabled requires at least one phase.")
    if handoff_guard.enabled and not handoff_guard.evaluator:
        parser.error("handoff_guard enabled requires evaluator.")
    unknown_readiness_phases = sorted(set(phase_readiness.phases) - set(phase_names))
    if unknown_readiness_phases:
        parser.error(f"phase_readiness references unknown phases: {', '.join(unknown_readiness_phases)}")
    unknown_guard_phases = sorted(set(handoff_guard.phases) - set(phase_names))
    if unknown_guard_phases:
        parser.error(f"handoff_guard references unknown phases: {', '.join(unknown_guard_phases)}")
    if validation.enabled and not validation.evaluators:
        parser.error("validation enabled requires at least one evaluator.")
    if validation.enabled and not validation.candidates:
        parser.error("validation enabled requires at least one candidate set.")
    return PhaseSchedule(
        preset=args.phase_preset,
        source="cli" if args.phase else "yaml",
        config_path=config_path,
        fold=dict(fold_raw),
        basis_circuit_batching=_normalize_basis_circuit_batching(
            parser,
            basis_circuit_batching_raw,
            "basis_circuit_batching",
        ),
        circuit_template=circuit_template,
        circuit=circuit,
        scouting=scouting,
        phases=phases,
        result=ResultConfig(
            primary=primary_result,
            score_model=result_score,
            score_options=_score_options(parser, result_raw.get("score_options"), "result"),
            description=_optional_description(parser, result_raw.get("description"), "result"),
        ),
        readouts=readouts,
        metrics=MetricsConfig(
            atom_sets=list(metric_atom_sets),
            rmsd_alignment_atom_set=rmsd_alignment_atom_set,
            description=_optional_description(parser, metrics_raw.get("description"), "metrics"),
        ),
        report=ReportConfig(
            structure_domain=report_structure_domain,
            description=_optional_description(parser, report_raw.get("description"), "report"),
        ),
        evaluators=evaluators,
        phase_comparisons=phase_comparisons,
        reranking=reranking,
        phase_readiness=phase_readiness,
        handoff_guard=handoff_guard,
        validation=validation,
        default_transpile=default_transpile,
        gate_estimate_optimization_levels=gate_estimate_optimization_levels,
        gate_estimate_transpile_seed=gate_estimate_transpile_seed,
        description=recipe_description,
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
        from pheat.molstar_assets import copy_molstar_assets

        return copy_molstar_assets(outdir / "vendor" / "molstar", warn=True)
    except Exception as exc:
        return None, str(exc)


def _format_optional_int(value) -> str:
    if value is None:
        return "n/a"
    return str(int(value))


def _format_optional_seed(value) -> str:
    if value is None:
        return "unset"
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


def _ibm_auth_service_kwargs(auth_config: Optional[IBMRuntimeAuthConfig]) -> dict[str, Any]:
    if auth_config is None:
        return {}

    kwargs: dict[str, Any] = {}
    if auth_config.token:
        kwargs["token"] = auth_config.token
        kwargs["channel"] = auth_config.channel or IBM_RUNTIME_DEFAULT_TOKEN_CHANNEL
    else:
        if auth_config.account_name:
            kwargs["name"] = auth_config.account_name
        if auth_config.channel:
            kwargs["channel"] = auth_config.channel
    if auth_config.instance_crn:
        kwargs["instance"] = auth_config.instance_crn
    if auth_config.url:
        kwargs["url"] = auth_config.url
    return kwargs


def _ibm_auth_source(auth_config: Optional[IBMRuntimeAuthConfig]) -> str:
    if auth_config is None:
        return "saved_account_default"
    if auth_config.token_source != "none":
        return f"token_{auth_config.token_source}"
    if auth_config.account_name:
        return "saved_account_named"
    if auth_config.channel or auth_config.instance_crn or auth_config.url:
        return "saved_account_default"
    return "saved_account_default"


def _ibm_auth_metadata(auth_config: Optional[IBMRuntimeAuthConfig]) -> dict[str, Any]:
    return {
        "ibm_auth_source": _ibm_auth_source(auth_config),
        "ibm_account_name": None if auth_config is None else auth_config.account_name,
        "ibm_account_name_provided": bool(auth_config and auth_config.account_name),
        "ibm_token_source": "none" if auth_config is None else auth_config.token_source,
        "ibm_token_provided": bool(auth_config and auth_config.token_source != "none"),
        "ibm_channel": None if auth_config is None else auth_config.channel,
        "ibm_instance_crn_provided": bool(auth_config and auth_config.instance_crn),
        "ibm_url_provided": bool(auth_config and auth_config.url),
    }


def _ibm_auth_context(auth_config: Optional[IBMRuntimeAuthConfig]) -> str:
    metadata = _ibm_auth_metadata(auth_config)
    parts = [f"auth_source={metadata['ibm_auth_source']}"]
    if metadata["ibm_account_name_provided"]:
        parts.append(f"account={metadata['ibm_account_name']!r}")
    if metadata["ibm_token_provided"]:
        parts.append(f"token_source={metadata['ibm_token_source']}")
    if metadata["ibm_channel"]:
        parts.append(f"channel={metadata['ibm_channel']}")
    if metadata["ibm_instance_crn_provided"]:
        parts.append("instance_crn=provided")
    if metadata["ibm_url_provided"]:
        parts.append("url=provided")
    return " (" + ", ".join(parts) + ")"


def _saved_ibm_accounts() -> dict[str, dict[str, Any]]:
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService

        accounts = QiskitRuntimeService.saved_accounts()
    except Exception:
        return {}
    if not isinstance(accounts, Mapping):
        return {}
    normalized: dict[str, dict[str, Any]] = {}
    for name, raw_config in accounts.items():
        if isinstance(raw_config, Mapping):
            normalized[str(name)] = dict(raw_config)
        else:
            normalized[str(name)] = {}
    return normalized


def _saved_ibm_account_names() -> list[str]:
    return sorted(_saved_ibm_accounts())


def _saved_account_auth_config_for_instance_override(
    *,
    context: str,
    instance_crn: str,
    ibm_auth_config: Optional[IBMRuntimeAuthConfig],
) -> IBMRuntimeAuthConfig:
    if ibm_auth_config is not None and ibm_auth_config.token:
        return replace(ibm_auth_config, account_name=None, instance_crn=instance_crn)

    accounts = _saved_ibm_accounts()
    selected_name: Optional[str] = None
    selected_config: Optional[dict[str, Any]] = None
    if ibm_auth_config is not None and ibm_auth_config.account_name:
        selected_name = ibm_auth_config.account_name
        selected_config = accounts.get(selected_name)
        if selected_config is None:
            raise ValueError(
                f"Gate-estimate backend {context} requested a backend-specific CRN, "
                f"but saved IBM account {selected_name!r} was not found."
            )
    else:
        preferred_channel = _optional_cli_text(None if ibm_auth_config is None else ibm_auth_config.channel)
        ordered_names = sorted(accounts)
        preferred_names: list[str] = []
        if preferred_channel is not None:
            preferred_names = [
                name
                for name in ordered_names
                if _optional_cli_text(accounts[name].get("channel")) == preferred_channel
            ]
        else:
            preferred_names = [
                name
                for name in ordered_names
                if _optional_cli_text(accounts[name].get("channel")) == "ibm_cloud"
            ]
        for candidate in [*preferred_names, *ordered_names]:
            if candidate in accounts:
                selected_name = candidate
                selected_config = accounts[candidate]
                break

    if selected_name is None or selected_config is None:
        raise ValueError(
            f"Gate-estimate backend {context} requested a backend-specific CRN, "
            "but no saved IBM account is available to supply a token. Provide --ibm-token-env or --ibm-token-file."
        )

    token = _optional_cli_text(selected_config.get("token"))
    if token is None:
        raise ValueError(
            f"Gate-estimate backend {context} requested a backend-specific CRN, "
            f"but saved IBM account {selected_name!r} has no token. Provide --ibm-token-env or --ibm-token-file."
        )

    channel = (
        _optional_cli_text(None if ibm_auth_config is None else ibm_auth_config.channel)
        or _optional_cli_text(selected_config.get("channel"))
        or IBM_RUNTIME_DEFAULT_TOKEN_CHANNEL
    )
    url = (
        _optional_cli_text(None if ibm_auth_config is None else ibm_auth_config.url)
        or _optional_cli_text(selected_config.get("url"))
    )
    return IBMRuntimeAuthConfig(
        account_name=selected_name,
        token=token,
        token_source="saved_account",
        channel=channel,
        instance_crn=instance_crn,
        url=url,
    )


def _gate_estimate_auth_config_for_backend(
    backend_name: str,
    ibm_auth_config: Optional[IBMRuntimeAuthConfig],
    backend_crn_map: Optional[Mapping[str, str]] = None,
) -> Optional[IBMRuntimeAuthConfig]:
    if not backend_crn_map:
        return ibm_auth_config
    key = str(backend_name).strip().lower()
    instance_crn = backend_crn_map.get(key)
    if instance_crn is None:
        return ibm_auth_config
    return _saved_account_auth_config_for_instance_override(
        context=f"'{backend_name}'",
        instance_crn=instance_crn,
        ibm_auth_config=ibm_auth_config,
    )


def _backend_lookup_error(
    context: str,
    exc: Exception,
    ibm_auth_config: Optional[IBMRuntimeAuthConfig] = None,
) -> ValueError:
    account_hint = ""
    if ibm_auth_config is None or ibm_auth_config.token_source == "none":
        account_names = _saved_ibm_account_names()
        if account_names:
            account_hint = f" Available saved IBM accounts: {', '.join(account_names)}."
    error = ValueError(f"Backend lookup failed for {context}{_ibm_auth_context(ibm_auth_config)}: {exc}{account_hint}")
    error.__cause__ = exc
    return error


def _least_busy_context(min_num_qubits: int) -> str:
    return (
        "'least_busy' with criteria "
        f"simulator=False, operational=True, min_num_qubits={int(min_num_qubits)}"
    )


def _ibm_runtime_service(
    context: str,
    ibm_auth_config: Optional[IBMRuntimeAuthConfig] = None,
):
    from qiskit_ibm_runtime import QiskitRuntimeService

    try:
        kwargs = _ibm_auth_service_kwargs(ibm_auth_config)
        if kwargs:
            return QiskitRuntimeService(**kwargs)
        return QiskitRuntimeService()
    except Exception as exc:
        raise _backend_lookup_error(context, exc, ibm_auth_config)


def _least_busy_backend(
    service,
    min_num_qubits: int,
    ibm_auth_config: Optional[IBMRuntimeAuthConfig] = None,
):
    min_qubits = max(1, int(min_num_qubits))
    criteria = _least_busy_context(min_qubits)
    try:
        candidates = service.backends(
            simulator=False,
            operational=True,
            min_num_qubits=min_qubits,
        )
    except Exception as exc:
        raise _backend_lookup_error(criteria, exc, ibm_auth_config)
    if not candidates:
        raise ValueError(
            f"Backend lookup failed for {criteria}{_ibm_auth_context(ibm_auth_config)}: "
            "no matching backends returned."
        )
    return sorted(candidates, key=lambda b: b.status().pending_jobs)[0]


def _parse_gate_estimate_backend_spec(estimate_arg, selected_backend: str) -> tuple[Optional[str], str]:
    if estimate_arg is None:
        return None, "not_requested"
    selected_key = str(selected_backend).strip().lower()
    if estimate_arg == GATE_ESTIMATE_SELECTED_BACKEND:
        if selected_key == "none":
            raise ValueError("--estimate-gates cannot infer a backend when --backend none.")
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


def _parse_gate_estimate_backend_crn_map(
    parser: argparse.ArgumentParser,
    values: Sequence[str],
) -> dict[str, str]:
    backend_crns: dict[str, str] = {}
    for raw_group in values or []:
        for raw in str(raw_group).split(","):
            item = raw.strip()
            if not item:
                continue
            if "=" not in item:
                parser.error(f"--gate-estimate-backend-crn entries must use BACKEND=CRN syntax: {raw!r}")
            backend, crn = item.split("=", 1)
            backend = backend.strip()
            crn = crn.strip()
            if not backend:
                parser.error("--gate-estimate-backend-crn backend name must not be blank.")
            if not crn:
                parser.error(f"--gate-estimate-backend-crn CRN must not be blank for backend {backend!r}.")
            key = backend.lower()
            if key in {"aer", "aer_simulator", "none", StatevectorShotsBackend.name}:
                parser.error(f"--gate-estimate-backend-crn only applies to IBM Runtime backends, not {backend!r}.")
            if key in backend_crns:
                parser.error(f"--gate-estimate-backend-crn was provided more than once for backend {backend!r}.")
            backend_crns[key] = crn
    return backend_crns


def _validate_gate_estimate_backend_crn_map(
    parser: argparse.ArgumentParser,
    backend_spec: Optional[str],
    backend_crn_map: Mapping[str, str],
) -> None:
    if not backend_crn_map:
        return
    if backend_spec is None:
        parser.error("--gate-estimate-backend-crn requires --estimate-gates.")
    requested = {entry.strip().lower() for entry in str(backend_spec).split(",") if entry.strip()}
    unknown = sorted(key for key in backend_crn_map if key not in requested)
    if unknown:
        parser.error(
            "--gate-estimate-backend-crn entries must match --estimate-gates backends; "
            f"unmatched: {', '.join(unknown)}"
        )


def _optional_cli_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _resolve_ibm_runtime_auth_config(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> IBMRuntimeAuthConfig:
    account_name = _optional_cli_text(args.ibm_account)
    instance_crn = _optional_cli_text(args.ibm_instance_crn)
    channel = _optional_cli_text(args.ibm_channel)
    url = _optional_cli_text(args.ibm_url)

    if args.ibm_account is not None and account_name is None:
        parser.error("--ibm-account must not be blank when provided.")
    if args.ibm_instance_crn is not None and instance_crn is None:
        parser.error("--ibm-instance-crn must not be blank when provided.")
    if args.ibm_channel is not None and channel is None:
        parser.error("--ibm-channel must not be blank when provided.")
    if args.ibm_url is not None and url is None:
        parser.error("--ibm-url must not be blank when provided.")

    token_candidates: list[tuple[str, str]] = []
    direct_token = _optional_cli_text(args.ibm_token)
    if args.ibm_token is not None:
        if direct_token is None:
            parser.error("--ibm-token must not be blank when provided.")
        token_candidates.append(("direct", direct_token))

    token_env = _optional_cli_text(args.ibm_token_env)
    if args.ibm_token_env is not None:
        if token_env is None:
            parser.error("--ibm-token-env must not be blank when provided.")
        env_token = os.environ.get(token_env, "").strip()
        if not env_token:
            parser.error(f"--ibm-token-env {token_env!r} is not set or is empty.")
        token_candidates.append(("env", env_token))

    token_file = _optional_cli_text(args.ibm_token_file)
    if args.ibm_token_file is not None:
        if token_file is None:
            parser.error("--ibm-token-file must not be blank when provided.")
        token_path = Path(token_file).expanduser()
        if not token_path.exists() or not token_path.is_file():
            parser.error(f"--ibm-token-file must be an existing file: {token_file}")
        try:
            file_token = token_path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            parser.error(f"--ibm-token-file could not be read: {exc}")
        if not file_token:
            parser.error(f"--ibm-token-file is empty: {token_file}")
        token_candidates.append(("file", file_token))

    if len(token_candidates) > 1:
        parser.error("Provide only one of --ibm-token, --ibm-token-env, or --ibm-token-file.")

    token_source = "none"
    token = None
    if token_candidates:
        token_source, token = token_candidates[0]
        if channel is None:
            channel = IBM_RUNTIME_DEFAULT_TOKEN_CHANNEL

    return IBMRuntimeAuthConfig(
        account_name=account_name,
        token=token,
        token_source=token_source,
        channel=channel,
        instance_crn=instance_crn,
        url=url,
    )


def _gate_estimate_service_cache_key(key: str, backend_crn_map: Optional[Mapping[str, str]]) -> str:
    if backend_crn_map and key in backend_crn_map:
        return f"crn:{key}"
    return "default"


def _verify_gate_estimate_backend_access(
    backend_spec: Optional[str],
    ibm_auth_config: Optional[IBMRuntimeAuthConfig] = None,
    backend_crn_map: Optional[Mapping[str, str]] = None,
) -> None:
    if backend_spec is None:
        return
    ibm_services = {}
    for raw_name in backend_spec.split(","):
        name = raw_name.strip()
        key = name.lower()
        if key in {"aer", "aer_simulator"}:
            continue
        estimate_auth_config = _gate_estimate_auth_config_for_backend(name, ibm_auth_config, backend_crn_map)
        service_key = _gate_estimate_service_cache_key(key, backend_crn_map)
        if key == "least_busy":
            criteria = _least_busy_context(1)
            if service_key not in ibm_services:
                ibm_services[service_key] = _ibm_runtime_service(criteria, estimate_auth_config)
            _least_busy_backend(ibm_services[service_key], 1, estimate_auth_config)
        else:
            context = f"'{name}'"
            if service_key not in ibm_services:
                ibm_services[service_key] = _ibm_runtime_service(context, estimate_auth_config)
            try:
                ibm_services[service_key].backend(name)
            except Exception as exc:
                raise _backend_lookup_error(context, exc, estimate_auth_config)


def _resolve_gate_estimate_backends(
    backend_spec: Optional[str],
    min_num_qubits: int,
    ibm_auth_config: Optional[IBMRuntimeAuthConfig] = None,
    backend_crn_map: Optional[Mapping[str, str]] = None,
):
    if backend_spec is None:
        return []
    backends = []
    ibm_services = {}
    for raw_name in backend_spec.split(","):
        name = raw_name.strip()
        key = name.lower()
        if key in {"aer", "aer_simulator"}:
            from qiskit_aer import AerSimulator

            backends.append({"requested": name, "backend": AerSimulator(), "source": "aer"})
            continue
        estimate_auth_config = _gate_estimate_auth_config_for_backend(name, ibm_auth_config, backend_crn_map)
        service_key = _gate_estimate_service_cache_key(key, backend_crn_map)
        crn_provided = bool(backend_crn_map and key in backend_crn_map)
        if key == "least_busy":
            min_qubits = max(1, int(min_num_qubits))
            criteria = _least_busy_context(min_qubits)
            if service_key not in ibm_services:
                ibm_services[service_key] = _ibm_runtime_service(criteria, estimate_auth_config)
            backend = _least_busy_backend(ibm_services[service_key], min_qubits, estimate_auth_config)
            backends.append({
                "requested": name,
                "backend": backend,
                "source": "least_busy",
                "instance_crn_provided": crn_provided,
            })
        else:
            context = f"'{name}'"
            if service_key not in ibm_services:
                ibm_services[service_key] = _ibm_runtime_service(context, estimate_auth_config)
            try:
                backend = ibm_services[service_key].backend(name)
            except Exception as exc:
                raise _backend_lookup_error(context, exc, estimate_auth_config)
            backends.append({
                "requested": name,
                "backend": backend,
                "source": "ibm",
                "instance_crn_provided": crn_provided,
            })
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
    filtered = QuantumCircuit(qc.num_qubits)
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
    base = QuantumCircuit(ansatz.num_qubits)
    base.append(ansatz, range(ansatz.num_qubits))
    base = base.decompose()

    circuits = {"base_circuit": base.copy()}
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


def _estimate_gate_costs(
    folder: QuantumBiophysicsFolder,
    backend_refs,
    *,
    optimization_levels: Sequence[int] = DEFAULT_GATE_ESTIMATE_OPTIMIZATION_LEVELS,
    transpile_seed: Optional[int] = None,
) -> list[dict]:
    if folder.ansatz is None:
        raise RuntimeError("Qiskit is required for --estimate-gates.")

    normalized_levels: list[int] = []
    for raw_level in optimization_levels:
        level = int(raw_level)
        if level not in TRANSPILE_OPTIMIZATION_LEVELS:
            raise ValueError("gate estimate optimization levels must be 0, 1, 2, or 3.")
        if level not in normalized_levels:
            normalized_levels.append(level)
    if 0 not in normalized_levels:
        normalized_levels.insert(0, 0)
    if not any(level > 0 for level in normalized_levels):
        normalized_levels.append(3)
    if transpile_seed is not None:
        transpile_seed = int(transpile_seed)
        if transpile_seed < 0:
            raise ValueError("gate estimate transpile seed must be non-negative.")

    estimates = []
    circuits = _build_gate_estimate_circuits(folder.ansatz)
    for backend_ref in backend_refs:
        backend = backend_ref["backend"]
        backend_name = _backend_display_name(backend)
        backend_metadata = _backend_processor_metadata(backend)
        for circuit_label, circuit in circuits.items():
            for optimization_level in normalized_levels:
                transpile_config = TranspileConfig(
                    optimization_level=int(optimization_level),
                    seed=transpile_seed,
                )
                kwargs = {"backend": backend, **_transpile_kwargs(transpile_config)}
                t0 = time.time()
                transpiled = transpile(circuit, **kwargs)
                transpile_time_s = time.time() - t0
                estimates.append(
                    {
                        "backend": backend_name,
                        "requested_backend": backend_ref["requested"],
                        "backend_source": backend_ref["source"],
                        **backend_metadata,
                        "circuit": circuit_label,
                        "optimization_level": optimization_level,
                        "seed_transpiler": transpile_seed,
                        "transpile_time_s": round(transpile_time_s, 3),
                        **_circuit_gate_metrics(transpiled),
                    }
                )
    return estimates


def _logical_gate_costs(ansatz) -> list[dict]:
    return [
        {
            "backend": "logical",
            "requested_backend": "logical",
            "backend_source": "untranspiled",
            "circuit": circuit_label,
            "optimization_level": None,
            "seed_transpiler": None,
            "transpile_time_s": 0.0,
            **_circuit_gate_metrics(circuit),
        }
        for circuit_label, circuit in _build_gate_estimate_circuits(ansatz).items()
    ]


def _estimate_phase_gate_costs(
    phase_results: Sequence[dict],
    *,
    circuit_template: Optional[dict[str, Any]],
    circuit: Optional[dict[str, Any]],
    backend_refs,
    optimization_levels: Sequence[int],
    transpile_seed: Optional[int],
) -> list[dict]:
    estimates: list[dict] = []
    for phase_result in phase_results:
        total_dofs = phase_result.get("total_dofs")
        if total_dofs is None:
            continue
        phase_context = {
            "phase_index": phase_result.get("index"),
            "phase_name": phase_result.get("name"),
            "phase_label": phase_result.get("label"),
            "total_dofs": phase_result.get("total_dofs"),
            "total_angle_dofs": phase_result.get("total_angle_dofs"),
            "total_length_dofs": phase_result.get("total_length_dofs"),
        }
        try:
            build = build_circuit(
                total_angles=int(total_dofs),
                circuit_template=circuit_template,
                circuit=circuit,
            )
            logical = _logical_gate_costs(build.circuit)
            transpiled = _estimate_gate_costs(
                SimpleNamespace(ansatz=build.circuit),
                backend_refs,
                optimization_levels=optimization_levels,
                transpile_seed=transpile_seed,
            )
            for item in [*logical, *transpiled]:
                estimates.append(
                    {
                        **phase_context,
                        "n_qubits": build.n_qubits,
                        "n_params": build.n_params,
                        "reps": build.reps,
                        **item,
                    }
                )
        except Exception as exc:
            estimates.append(
                {
                    **phase_context,
                    "status": "unavailable",
                    "error": str(exc),
                }
            )
    return estimates


def _candidate_objective_value(record: Mapping[str, Any]) -> float:
    try:
        value = record.get("objective", math.inf)
        if value is None:
            return math.inf
        resolved = float(value)
    except Exception:
        return math.inf
    return resolved if math.isfinite(resolved) else math.inf


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
        rows = '<tr><td colspan="22">No optimizer phases recorded.</td></tr>'
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
                f"<td>{html.escape(str(item.get('description') or ''))}</td>"
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
          <th>Description</th>
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
    report_domain = str(result.get("report_structure_domain") or DEFAULT_REPORT_STRUCTURE_DOMAIN)
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
    These PHEAT heavy-atom structures are captured after circuit initialization,
    after each optimizer phase, and after any optional readouts. Viewer PDBs are
    filtered to PHEAT domain <code>{html.escape(report_domain)}</code>. The Mol* viewer
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


def _report_domain_coverage_report_section(result: dict) -> str:
    domain = str(result.get("report_structure_domain") or DEFAULT_REPORT_STRUCTURE_DOMAIN)
    rows = []
    coverages = result.get("report_structure_domain_coverage") or {}
    for key, coverage in coverages.items():
        if not isinstance(coverage, dict):
            continue
        rows.append(_report_domain_coverage_row(str(key).replace("_", " ").title(), coverage))
        source_coverage = coverage.get("source_input_coverage")
        if isinstance(source_coverage, dict):
            rows.append(_report_domain_coverage_row(f"{str(key).replace('_', ' ').title()} source", source_coverage))
    for snapshot in result.get("structure_snapshots") or []:
        coverage = snapshot.get("report_domain_coverage")
        if isinstance(coverage, dict):
            label = snapshot.get("label") or snapshot.get("key") or "snapshot"
            rows.append(_report_domain_coverage_row(f"Snapshot: {label}", coverage))
    if not rows:
        return f'''
  <h2>Report Structure Domain</h2>
  <p class="muted">Report/viewer PDB artifacts use PHEAT domain <code>{html.escape(domain)}</code>. No domain coverage details were recorded.</p>
'''
    return f'''
  <h2>Report Structure Domain</h2>
  <p class="muted">
    Report/viewer PDB artifacts are filtered with PHEAT domain
    <code>{html.escape(domain)}</code>. Metric atom sets are applied separately
    to the structures used for diagnostics.
  </p>
  <div class="report-table-wrap">
    <table class="sortable-table">
      <thead>
        <tr>
          <th>Structure</th>
          <th>Domain</th>
          <th>Input atoms</th>
          <th>Reported atoms</th>
          <th>Ignored H</th>
          <th>Ignored nonprotein</th>
          <th>Unsupported residues</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
'''


def _report_domain_coverage_row(label: str, coverage: dict) -> str:
    unsupported = coverage.get("unsupported_residues") or []
    if isinstance(unsupported, (list, tuple)):
        unsupported_text = ", ".join(str(item) for item in unsupported) or "none"
    else:
        unsupported_text = str(unsupported) or "none"
    return (
        "<tr>"
        f"<td>{html.escape(str(label))}</td>"
        f"<td><code>{html.escape(str(coverage.get('domain') or 'n/a'))}</code></td>"
        f"<td>{html.escape(_format_optional_int(coverage.get('input_atom_count')))}</td>"
        f"<td>{html.escape(_format_optional_int(coverage.get('scored_atom_count')))}</td>"
        f"<td>{html.escape(_format_optional_int(coverage.get('ignored_hydrogen_atom_count')))}</td>"
        f"<td>{html.escape(_format_optional_int(coverage.get('ignored_nonprotein_atom_count')))}</td>"
        f"<td>{html.escape(unsupported_text)}</td>"
        "</tr>"
    )


def _landscape_report_section(result: dict) -> str:
    landscape_path = result.get("landscape_path")
    interactive_path = result.get("interactive_landscape_path")
    if not landscape_path and not interactive_path:
        return """
  <h2>Energy Landscape</h2>
  <p class="muted">No energy landscape artifact was recorded.</p>
"""
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
    if not landscape_path:
        return f"""
  <h2>Energy Landscape</h2>
  {interactive_block}
"""
    name = html.escape(Path(landscape_path).name)
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


def _software_status_badge(status: str) -> str:
    label = str(status or "n/a")
    css_category = "ok" if label == "available" else "warning" if label in {"missing", "not_selected"} else "neutral"
    return (
        f'<span class="phase-status-badge phase-status-{html.escape(css_category)}">'
        f'{html.escape(label)}</span>'
    )


def _software_report_section(result: dict) -> str:
    software = result.get("software_versions") or {}
    package_components = list(software.get("package_components") or [])
    if not package_components:
        package_components = [
            {
                "name": name,
                "version": version,
                "role": "recorded package",
                "required": False,
                "selected": True,
                "status": "available" if version else "missing",
            }
            for name, version in (software.get("packages") or {}).items()
        ]
    package_rows = "\n".join(
        "<tr>"
        f"<td><code>{html.escape(str(item.get('name') or ''))}</code></td>"
        f"<td>{html.escape(str(item.get('role') or 'n/a'))}</td>"
        f"<td>{'yes' if item.get('required') else 'no'}</td>"
        f"<td>{_software_status_badge(str(item.get('status') or 'n/a'))}</td>"
        f"<td>{html.escape(str(item.get('version') or 'not installed'))}</td>"
        "</tr>"
        for item in package_components
    ) or '<tr><td colspan="5">No run components recorded.</td></tr>'

    external_tools = list(software.get("external_tools") or [])
    tool_rows = "\n".join(
        "<tr>"
        f"<td><code>{html.escape(str(item.get('name') or ''))}</code></td>"
        f"<td>{html.escape(str(item.get('role') or 'n/a'))}</td>"
        f"<td>{'yes' if item.get('required') else 'no'}</td>"
        f"<td>{_software_status_badge(str(item.get('status') or 'n/a'))}</td>"
        f"<td><code>{html.escape(str(item.get('path') or 'not found'))}</code></td>"
        f"<td>{html.escape(str(item.get('version') or item.get('details') or 'n/a'))}</td>"
        "</tr>"
        for item in external_tools
    )
    external_tool_section = ""
    if tool_rows:
        external_tool_section = f"""
  <h3>Selected External Tools</h3>
  <div class="report-table-wrap">
    <table class="sortable-table">
      <thead><tr><th>Tool</th><th>Role</th><th>Required</th><th>Status</th><th>Path</th><th>Version/details</th></tr></thead>
      <tbody>{tool_rows}</tbody>
    </table>
  </div>
"""

    python_info = software.get("python") or {}
    platform_info = software.get("platform") or {}
    qtf_info = software.get("qtf") or {}
    pheat_info = software.get("pheat") or {}
    sidecar_link = _html_artifact_link(software.get("sidecar_path"), "Open full software version JSON")
    selected_models = ", ".join(str(model) for model in software.get("selected_score_models") or []) or "n/a"
    return f"""
  <h2>Software Versions</h2>
  <table>
    <tbody>
      <tr><th>Python</th><td>{html.escape(str(python_info.get("version") or "n/a"))} ({html.escape(str(python_info.get("implementation") or "n/a"))})</td></tr>
      <tr><th>Python executable</th><td><code>{html.escape(str(python_info.get("executable") or "n/a"))}</code></td></tr>
      <tr><th>Platform</th><td>{html.escape(str(platform_info.get("platform") or "n/a"))}</td></tr>
      <tr><th>QTF</th><td>{html.escape(_git_label((qtf_info.get("git") or {})))}<br><code>{html.escape(str(qtf_info.get("runner_path") or "n/a"))}</code></td></tr>
      <tr><th>PHEAT</th><td>version {html.escape(str(pheat_info.get("version") or "n/a"))}; {html.escape(_git_label((pheat_info.get("git") or {})))}<br><code>{html.escape(str(pheat_info.get("module_path") or "n/a"))}</code></td></tr>
      <tr><th>Selected score models</th><td><code>{html.escape(selected_models)}</code></td></tr>
      <tr><th>Installed distributions</th><td>{html.escape(str(software.get("installed_distribution_count") or 0))}; {sidecar_link}</td></tr>
    </tbody>
  </table>
  <h3>Run Components</h3>
  <p class="muted">This table highlights software used by the selected QTF run path. The sidecar JSON keeps the exhaustive installed-distribution inventory.</p>
  <div class="report-table-wrap">
    <table class="sortable-table">
      <thead><tr><th>Package</th><th>Role</th><th>Required</th><th>Status</th><th>Version</th></tr></thead>
      <tbody>{package_rows}</tbody>
    </table>
  </div>
  {external_tool_section}
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


def _pheat_score_capabilities_report_section(result: dict) -> str:
    capabilities = result.get("pheat_score_model_capabilities") or pheat_score_model_capabilities()
    rows = "\n".join(
        "<tr>"
        f"<td><code>{html.escape(str(item.get('public_model') or item.get('model') or ''))}</code></td>"
        f"<td><code>{html.escape(str(item.get('pheat_model') or item.get('raw_model') or ''))}</code></td>"
        f"<td>{'yes' if item.get('available') else 'no'}</td>"
        f"<td>{html.escape(str(item.get('units') or 'n/a'))}</td>"
        f"<td>{html.escape(_format_timing_metadata_value(item.get('requires') or []))}</td>"
        f"<td>{html.escape(_format_timing_metadata_value(item.get('optional_requires') or []))}</td>"
        f"<td>{html.escape(str(item.get('reason') or item.get('optional_status') or ''))}</td>"
        "</tr>"
        for item in capabilities
    ) or '<tr><td colspan="7">No PHEAT scoring models were reported by the installed PHEAT package.</td></tr>'
    return f"""
  <h2>PHEAT Score Models</h2>
  <p class="muted">
    Supported scoring models are loaded from the installed PHEAT package at runtime.
    QTF accepts only currently available models for optimization and readout scoring.
  </p>
  <div class="report-table-wrap">
    <table class="sortable-table">
      <thead>
        <tr>
          <th>QTF name</th>
          <th>PHEAT model</th>
          <th>Available</th>
          <th>Units</th>
          <th>Requires</th>
          <th>Optional requires</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
"""


def _simple_status_badge(status: str, detail: str = "") -> str:
    category = str(status or "n/a").lower()
    if category in {"ok", "available", "success", "ready", "accepted"}:
        css_category = "ok"
    elif category in {"skipped", "warning", "unavailable", "accepted_with_violations", "not_ready"}:
        css_category = "warning"
    else:
        css_category = "error"
    detail_markup = (
        f'<span class="phase-status-detail">{html.escape(str(detail))}</span>'
        if detail
        else ""
    )
    return (
        f'<span class="phase-status-badge phase-status-{html.escape(css_category)}">'
        f'{html.escape(str(status or "n/a"))}</span>{detail_markup}'
    )


def _external_evaluators_report_section(result: dict) -> str:
    statuses = result.get("external_evaluator_statuses") or []
    if not statuses:
        configured = result.get("external_evaluators") or []
        if configured:
            return f"""
  <h2>External Evaluators</h2>
  <p class="muted">{len(configured)} external evaluator definition(s) are present in the recipe, but none were active for this run.</p>
"""
        return """
  <h2>External Evaluators</h2>
  <p class="muted">No external evaluators were configured for this recipe.</p>
"""
    rows = []
    for item in statuses:
        detail = "; ".join(str(part) for part in (item.get("errors") or item.get("warnings") or []))
        capability = item.get("capability") or {}
        implementation = capability.get("implementation") or {}
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(str(item.get('name') or ''))}</code></td>"
            f"<td><code>{html.escape(str(item.get('score_model') or ''))}</code></td>"
            f"<td>{html.escape(str(implementation.get('backend') or 'n/a'))}</td>"
            f"<td>{'yes' if item.get('required') else 'no'}</td>"
            f"<td>{_simple_status_badge(str(item.get('status') or 'n/a'), detail)}</td>"
            f"<td>{html.escape(str(capability.get('units') or 'n/a'))}</td>"
            f"<td>{html.escape(_format_timing_metadata_value(capability.get('requires') or []))}</td>"
            "</tr>"
        )
    return f"""
  <h2>External Evaluators</h2>
  <p class="muted">
    External evaluator availability and option validation are checked before circuit construction.
    Optional unavailable evaluators are skipped; required unavailable evaluators fail the run.
  </p>
  <div class="report-table-wrap">
    <table class="sortable-table">
      <thead>
        <tr>
          <th>Evaluator</th>
          <th>Score model</th>
          <th>Backend</th>
          <th>Required</th>
          <th>Status</th>
          <th>Units</th>
          <th>Requires</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
"""


def _phase_comparisons_report_section(result: dict) -> str:
    comparisons = result.get("phase_comparison_results") or []
    if not comparisons:
        return """
  <h2>Phase Comparisons</h2>
  <p class="muted">No external phase comparisons were recorded.</p>
"""
    rows = []
    for item in comparisons:
        current_score = item.get("current_score") or {}
        previous_score = item.get("previous_score") or {}
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(str(item.get('evaluator') or ''))}</code></td>"
            f"<td>{html.escape(str(item.get('previous_label') or item.get('previous_snapshot_key') or 'n/a'))}</td>"
            f"<td>{html.escape(str(item.get('current_label') or item.get('current_snapshot_key') or 'n/a'))}</td>"
            f"<td>{html.escape(_format_optional_float(previous_score.get('total'), 6))}</td>"
            f"<td>{html.escape(_format_optional_float(current_score.get('total'), 6))}</td>"
            f"<td>{html.escape(_format_optional_float(item.get('delta_current_minus_previous'), 6))}</td>"
            f"<td>{html.escape(str(current_score.get('units') or previous_score.get('units') or 'n/a'))}</td>"
            f"<td>{_simple_status_badge(str(item.get('status') or 'n/a'), str(item.get('reason') or current_score.get('error') or previous_score.get('error') or ''))}</td>"
            "</tr>"
        )
    return f"""
  <h2>Phase Comparisons</h2>
  <p class="muted">
    External evaluator deltas compare consecutive phase-end structures. Lower evaluator scores are treated as better.
  </p>
  <div class="report-table-wrap">
    <table class="sortable-table">
      <thead>
        <tr>
          <th>Evaluator</th>
          <th>Previous</th>
          <th>Current</th>
          <th>Previous score</th>
          <th>Current score</th>
          <th>Delta</th>
          <th>Units</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
"""


def _reranking_report_section(result: dict) -> str:
    reranking = result.get("reranking_results") or []
    if not reranking:
        return """
  <h2>Reranking</h2>
  <p class="muted">No external reranking was recorded.</p>
"""
    rows = []
    for item in reranking:
        selected = item.get("selected") or {}
        counts = selected.get("physical_counts") or {}
        detail = str(item.get("reason") or item.get("selection_reason") or selected.get("selection_reason") or "")
        trigger = str(item.get("trigger") or "phase_end")
        interval = item.get("interval")
        trigger_label = trigger if interval in (None, "") else f"{trigger} ({interval})"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('phase_label') or item.get('phase_name') or ''))}</td>"
            f"<td>{html.escape(trigger_label)}</td>"
            f"<td>{html.escape(str(item.get('phase_evaluation') or 'n/a'))}</td>"
            f"<td><code>{html.escape(str(item.get('evaluator') or ''))}</code></td>"
            f"<td>{html.escape(str(item.get('candidate_count') or 0))}</td>"
            f"<td>{html.escape(str(selected.get('candidate_id') or 'n/a'))}</td>"
            f"<td>{html.escape(_format_optional_float(selected.get('score_total'), 6))}</td>"
            f"<td>{html.escape(str(selected.get('score_units') or 'n/a'))}</td>"
            f"<td>{html.escape(str(counts.get('clash_count', 'n/a')))}</td>"
            f"<td>{html.escape(str(counts.get('short_contact_count', 'n/a')))}</td>"
            f"<td>{html.escape(_format_optional_float(counts.get('min_nonlocal_distance'), 6))}</td>"
            f"<td>{html.escape(str(item.get('apply') or 'n/a'))}</td>"
            f"<td>{_simple_status_badge(str(item.get('status') or 'n/a'), detail)}</td>"
            "</tr>"
        )
    return f"""
  <h2>Reranking</h2>
  <p class="muted">
    Reranking scores a bounded pool of optimizer candidates with the configured evaluator.
    Periodic checkpoints are report-only during optimization and are eligible for phase-end handoff selection.
  </p>
  <div class="report-table-wrap">
    <table class="sortable-table">
      <thead>
        <tr>
          <th>Phase</th>
          <th>Trigger</th>
          <th>Eval</th>
          <th>Evaluator</th>
          <th>Candidates</th>
          <th>Selected</th>
          <th>Selected score</th>
          <th>Units</th>
          <th>Clashes</th>
          <th>Short contacts</th>
          <th>Min distance</th>
          <th>Apply</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
"""


def _phase_readiness_report_section(result: dict) -> str:
    readiness_results = result.get("phase_readiness_results") or []
    if not readiness_results:
        return """
  <h2>Phase Readiness Gates</h2>
  <p class="muted">No phase readiness gates were evaluated.</p>
"""
    rows = []
    for item in readiness_results:
        counts = item.get("counts") or {}
        thresholds = item.get("thresholds") or {}
        reasons = "; ".join(str(reason) for reason in item.get("reasons") or [])
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('phase_label') or item.get('phase_name') or ''))}</td>"
            f"<td><code>{html.escape(str(item.get('evaluator') or ''))}</code></td>"
            f"<td>{html.escape(str(item.get('decision') or 'n/a'))}</td>"
            f"<td>{html.escape(_format_optional_float(item.get('score_total'), 6))}</td>"
            f"<td>{html.escape(str(item.get('score_units') or 'n/a'))}</td>"
            f"<td>{html.escape(str(counts.get('clash_count', 'n/a')))} / max {html.escape(str(thresholds.get('max_clash_count', 'n/a')))}</td>"
            f"<td>{html.escape(str(counts.get('short_contact_count', 'n/a')))} / max {html.escape(str(thresholds.get('max_short_contact_count', 'n/a')))}</td>"
            f"<td>{html.escape(_format_optional_float(counts.get('min_nonlocal_distance'), 6))} / min {html.escape(str(thresholds.get('min_nonlocal_distance_a', 'n/a')))}</td>"
            f"<td>{html.escape(str(item.get('snapshot_key') or 'n/a'))}</td>"
            f"<td>{_simple_status_badge(str(item.get('status') or 'n/a'), reasons or str(item.get('reason') or ''))}</td>"
            "</tr>"
        )
    return f"""
  <h2>Phase Readiness Gates</h2>
  <p class="muted">
    Readiness gates score the incoming structure before selected phases run. RMSD is not used for these decisions.
  </p>
  <div class="report-table-wrap">
    <table class="sortable-table">
      <thead>
        <tr>
          <th>Phase</th>
          <th>Evaluator</th>
          <th>Decision</th>
          <th>Score</th>
          <th>Units</th>
          <th>Clashes</th>
          <th>Short contacts</th>
          <th>Min distance</th>
          <th>Snapshot</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
"""

def _handoff_guard_report_section(result: dict) -> str:
    guards = result.get("handoff_guard_results") or []
    if not guards:
        return """
  <h2>Handoff Guards</h2>
  <p class="muted">No handoff guard decisions were recorded.</p>
"""
    rows = []
    for item in guards:
        start_counts = item.get("phase_start_counts") or {}
        handoff_counts = item.get("handoff_counts") or {}
        reasons = "; ".join(str(reason) for reason in item.get("reasons") or [])
        start_clashes = start_counts.get("clash_count")
        handoff_clashes = handoff_counts.get("clash_count")
        clash_delta = "n/a"
        if start_clashes is not None and handoff_clashes is not None:
            try:
                clash_delta = str(int(handoff_clashes) - int(start_clashes))
            except Exception:
                clash_delta = "n/a"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('phase_label') or item.get('phase_name') or ''))}</td>"
            f"<td><code>{html.escape(str(item.get('evaluator') or ''))}</code></td>"
            f"<td>{html.escape(str(item.get('handoff_candidate_id') or 'n/a'))}</td>"
            f"<td>{html.escape(str(item.get('decision') or 'n/a'))}</td>"
            f"<td>{html.escape(str(start_counts.get('clash_count', 'n/a')))}</td>"
            f"<td>{html.escape(str(handoff_counts.get('clash_count', 'n/a')))}</td>"
            f"<td>{html.escape(clash_delta)}</td>"
            f"<td>{html.escape(str(start_counts.get('short_contact_count', 'n/a')))}</td>"
            f"<td>{html.escape(str(handoff_counts.get('short_contact_count', 'n/a')))}</td>"
            f"<td>{html.escape(_format_optional_float(start_counts.get('min_nonlocal_distance'), 6))}</td>"
            f"<td>{html.escape(_format_optional_float(handoff_counts.get('min_nonlocal_distance'), 6))}</td>"
            f"<td>{_simple_status_badge(str(item.get('status') or 'n/a'), reasons)}</td>"
            "</tr>"
        )
    return f"""
  <h2>Handoff Guards</h2>
  <p class="muted">
    Handoff guards compare the selected handoff candidate against the phase-start structure before the next phase.
  </p>
  <div class="report-table-wrap">
    <table class="sortable-table">
      <thead>
        <tr>
          <th>Phase</th>
          <th>Evaluator</th>
          <th>Candidate</th>
          <th>Decision</th>
          <th>Start clashes</th>
          <th>Candidate clashes</th>
          <th>Clash delta</th>
          <th>Start short contacts</th>
          <th>Candidate short contacts</th>
          <th>Start min distance</th>
          <th>Candidate min distance</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
"""


def _phase_gate_estimates_report_section(result: dict) -> str:
    estimates = result.get("phase_gate_estimates") or []
    if not estimates:
        return """
  <h2>Phase Circuit Cost Estimates</h2>
  <p class="muted">No per-phase circuit cost estimates were recorded.</p>
"""
    rows = []
    for item in estimates:
        status = str(item.get("status") or "ok")
        detail = str(item.get("error") or "")
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('phase_index') or ''))}</td>"
            f"<td>{html.escape(str(item.get('phase_label') or item.get('phase_name') or ''))}</td>"
            f"<td>{html.escape(_format_optional_int(item.get('total_dofs')))}</td>"
            f"<td>{html.escape(_format_optional_int(item.get('total_angle_dofs')))}</td>"
            f"<td>{html.escape(_format_optional_int(item.get('total_length_dofs')))}</td>"
            f"<td>{html.escape(_format_optional_int(item.get('n_qubits')))}</td>"
            f"<td><code>{html.escape(str(item.get('backend') or 'n/a'))}</code></td>"
            f"<td>{html.escape(str(item.get('processor_type') or 'n/a'))}</td>"
            f"<td>{html.escape(str(item.get('processor_revision') or 'n/a'))}</td>"
            f"<td><code>{html.escape(str(item.get('circuit') or 'n/a'))}</code></td>"
            f"<td>{html.escape(str(item.get('optimization_level')))}</td>"
            f"<td>{html.escape(_format_optional_seed(item.get('seed_transpiler')))}</td>"
            f"<td>{html.escape(_format_optional_int(item.get('all_gate_depth')))}</td>"
            f"<td>{html.escape(_format_optional_int(item.get('multi_qubit_gate_depth')))}</td>"
            f"<td>{html.escape(_format_optional_int(item.get('all_gate_count')))}</td>"
            f"<td>{html.escape(_format_optional_int(item.get('multi_qubit_gate_count')))}</td>"
            f"<td>{_simple_status_badge(status, detail)}</td>"
            "</tr>"
        )
    return f"""
  <h2>Phase Circuit Cost Estimates</h2>
  <p class="muted">
    Each phase is rebuilt with that phase's encoded angle and length DOFs. Logical rows are untranspiled;
    backend rows are transpiled at the configured optimization levels.
  </p>
  <div class="report-table-wrap">
    <table class="sortable-table">
      <thead>
        <tr>
          <th>#</th>
          <th>Phase</th>
          <th>DOFs</th>
          <th>Angle DOFs</th>
          <th>Length DOFs</th>
          <th>Qubits</th>
          <th>Backend</th>
          <th>Processor</th>
          <th>Revision</th>
          <th>Circuit</th>
          <th>Opt level</th>
          <th>Seed</th>
          <th>Depth</th>
          <th>Multi-Q depth</th>
          <th>Gates</th>
          <th>Multi-Q gates</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
"""


def _validation_report_section(result: dict) -> str:
    validations = result.get("validation_results") or []
    if not validations:
        return """
  <h2>Final Candidate Validation</h2>
  <p class="muted">No final external validation was recorded.</p>
"""
    rows = []
    for item in validations:
        score = item.get("score") or {}
        warnings = "; ".join(str(part) for part in item.get("warnings") or score.get("warnings") or [])
        detail_parts = [str(item.get("error") or score.get("error") or "")]
        if warnings:
            detail_parts.append(warnings)
        detail = "; ".join(part for part in detail_parts if part)
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('candidate_set') or ''))}</td>"
            f"<td>{html.escape(str(item.get('label') or item.get('candidate_key') or ''))}</td>"
            f"<td><code>{html.escape(str(item.get('evaluator') or ''))}</code></td>"
            f"<td><code>{html.escape(str(item.get('score_model') or ''))}</code></td>"
            f"<td>{html.escape(_format_optional_float(item.get('score_total'), 6))}</td>"
            f"<td>{html.escape(str(item.get('score_units') or 'n/a'))}</td>"
            f"<td>{html.escape(warnings or 'none')}</td>"
            f"<td>{_html_artifact_link(item.get('prepared_output'), 'prepared')}</td>"
            f"<td>{_simple_status_badge(str(item.get('status') or 'n/a'), detail)}</td>"
            "</tr>"
        )
    return f"""
  <h2>Final Candidate Validation</h2>
  <p class="muted">
    Final validation scores configured candidate structures with external PHEAT-backed evaluators.
    These values are matched-configuration ranking checks, not absolute folding free energies.
  </p>
  <div class="report-table-wrap">
    <table class="sortable-table">
      <thead>
        <tr>
          <th>Candidate set</th>
          <th>Candidate</th>
          <th>Evaluator</th>
          <th>Score model</th>
          <th>Score</th>
          <th>Units</th>
          <th>Warnings</th>
          <th>Prepared output</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
"""


def _physical_readiness_report_section(result: dict) -> str:
    readiness = result.get("physical_readiness") or {}
    if not readiness:
        return """
  <h2>Physical Readiness</h2>
  <p class="muted">No physical readiness summary was recorded.</p>
"""
    counts = readiness.get("counts") or {}
    thresholds = readiness.get("thresholds") or {}
    reasons = readiness.get("reasons") or []
    reason_text = "; ".join(str(reason) for reason in reasons) if reasons else "none"
    return f"""
  <h2>Physical Readiness</h2>
  <p class="muted">
    Length tuning should start only after the primary candidate passes the configured physical-integrity thresholds.
  </p>
  <div class="report-table-wrap">
    <table>
      <tbody>
        <tr><th>Status</th><td>{_simple_status_badge(str(readiness.get('status') or 'n/a'), reason_text if reasons else '')}</td></tr>
        <tr><th>Ready for length tuning</th><td>{'yes' if readiness.get('ready_for_length_tuning') else 'no'}</td></tr>
        <tr><th>Source</th><td><code>{html.escape(str(readiness.get('source') or 'n/a'))}</code></td></tr>
        <tr><th>Score</th><td>{html.escape(_format_optional_float(readiness.get('score_total'), 6))} {html.escape(str(readiness.get('score_units') or ''))}</td></tr>
        <tr><th>Clashes</th><td>{html.escape(str(counts.get('clash_count', 'n/a')))} / max {html.escape(str(thresholds.get('max_clash_count', 'n/a')))}</td></tr>
        <tr><th>Short contacts</th><td>{html.escape(str(counts.get('short_contact_count', 'n/a')))} / max {html.escape(str(thresholds.get('max_short_contact_count', 'n/a')))}</td></tr>
        <tr><th>Minimum nonlocal distance</th><td>{html.escape(_format_optional_float(counts.get('min_nonlocal_distance'), 6))} A / min {html.escape(str(thresholds.get('min_nonlocal_distance_a', 'n/a')))} A</td></tr>
        <tr><th>Non-finite atoms</th><td>{html.escape(str(counts.get('nonfinite_atom_count', 'n/a')))}</td></tr>
        <tr><th>Reasons</th><td>{html.escape(reason_text)}</td></tr>
      </tbody>
    </table>
  </div>
"""


def _write_qtf_html_report(
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
    metric_atom_sets = normalize_metric_atom_sets(result.get("metric_atom_sets") or METRIC_ATOM_SETS)
    final_rg = _rg_payload_for_atom_set(pheat_rg.get("final") or {}, PRIMARY_RMSD_ATOM_SET)
    rg_delta = _rg_payload_for_atom_set(pheat_rg.get("delta_final_minus_reference") or {}, PRIMARY_RMSD_ATOM_SET)
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
    fold_config = phase_schedule.get("fold") or {}
    readout_results = result.get("readout_results") or []
    basis_batching_stats = result.get("basis_circuit_batching_stats") or {}
    phase_alerts = _phase_status_alerts(result)
    phase_results_section = _phase_results_report_section(result)
    structure_snapshots_section = _structure_snapshots_report_section(result)
    report_domain_coverage_section = _report_domain_coverage_report_section(result)
    score_total = _format_optional_float(final_score.get("total"), 6)
    rg_rows = _pheat_rg_table_rows(result)
    structure_metrics_rows = _structure_metrics_table_rows(rmsd_details, metric_atom_sets)
    timings = result.get("timings") or {}
    timing_rows = _timing_table_rows(timings)
    circuit_timing = timings.get("circuit_construction") or {}
    circuit_metadata = circuit_timing.get("metadata") or {}
    circuit_build_metadata = circuit_metadata.get("circuit") or result.get("circuit") or {}
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
            f"<td>{html.escape(_format_optional_seed(item.get('seed_transpiler')))}</td>"
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
    Estimates are transpiled backend resources for the base circuit and the
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
          <th>Seed</th>
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
        "Mol* viewer assets loaded from PHEAT-managed runtime assets."
        if molstar_dir is not None
        else f"Mol* viewer assets unavailable: {molstar_error or 'unknown error'}"
    )
    landscape_section = _landscape_report_section(result)
    energy_trace_section = _energy_trace_report_sections(result)
    software_section = _software_report_section(result)
    execution_section = _execution_report_section(result)
    pheat_score_capabilities_section = _pheat_score_capabilities_report_section(result)
    external_evaluators_section = _external_evaluators_report_section(result)
    phase_comparisons_section = _phase_comparisons_report_section(result)
    reranking_section = _reranking_report_section(result)
    phase_readiness_section = _phase_readiness_report_section(result)
    handoff_guard_section = _handoff_guard_report_section(result)
    phase_gate_estimate_section = _phase_gate_estimates_report_section(result)
    validation_section = _validation_report_section(result)
    physical_readiness_section = _physical_readiness_report_section(result)
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
  <h2>PHEAT Structure Metrics</h2>
  <p class="muted">
    RMSD and atom matching are computed by PHEAT using keyed atom sets.
    Alignment selector: <code>{html.escape(str(result.get("rmsd_alignment_atom_set") or DEFAULT_RMSD_ALIGNMENT_ATOM_SET))}</code>.
  </p>
  <div class="report-table-wrap">
    <table class="sortable-table">
      <thead>
        <tr>
          <th>Atom set</th>
          <th>RMSD</th>
          <th>Alignment set</th>
          <th>Matched atoms</th>
          <th>Unmatched reference</th>
          <th>Unmatched folded</th>
          <th>Units</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>{structure_metrics_rows}</tbody>
    </table>
  </div>
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
    def _description_row(label: str, text) -> str:
        if text is None or not str(text).strip():
            return ""
        return f"<tr><th>{html.escape(label)}</th><td>{html.escape(str(text).strip())}</td></tr>"

    recipe_description_rows = (
        _description_row("Recipe description", result.get("recipe_description") or phase_schedule.get("description"))
        + _description_row("Fold description", fold_config.get("description"))
        + _description_row("Scouting description", scouting_config.get("description"))
        + _description_row("Result description", result_config.get("description"))
        + _description_row("Metrics description", (phase_schedule.get("metrics") or {}).get("description"))
        + _description_row("Report description", (phase_schedule.get("report") or {}).get("description"))
    )
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
    QTF fold report using PHEAT geometry. Sequence <code>{html.escape(sequence)}</code>,
    replica {result["replica_id"]}. {html.escape(report_summary)}
  </p>

  {pheat_citation_section}

  <section class="metric-grid" aria-label="Key metrics">
    {reference_metric_cards}
    <dl class="metric"><dt>Primary Rg</dt><dd>{html.escape(_format_optional_angstrom(_rg_value(final_rg, "unweighted"), 6))}</dd></dl>
    {readout_drift_card}
    {primary_vs_last_phase_card}
    <dl class="metric"><dt>Score</dt><dd>{html.escape(score_total)}</dd></dl>
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
      <tr><th>Recipe / source</th><td><code>{html.escape(str(result.get("recipe") or result.get("phase_preset") or phase_schedule.get("preset") or "n/a"))}</code> / {html.escape(str(result.get("recipe_source") or result.get("phase_source") or phase_schedule.get("source") or "n/a"))}</td></tr>
      {recipe_description_rows}
      <tr><th>Recipe file</th><td>{html.escape(str(result.get("phase_config_path") or "built-in assets"))}</td></tr>
      <tr><th>Basis circuit batching</th><td>{html.escape(str(result.get("basis_circuit_batching_requested") or phase_schedule.get("basis_circuit_batching") or "auto"))} / effective={html.escape(str(result.get("basis_circuit_batching_effective") or "n/a"))}, calls={html.escape(str(basis_batching_stats.get("calls", 0)))}, jobs={html.escape(str(basis_batching_stats.get("backend_jobs", 0)))}</td></tr>
      <tr><th>Basis batching fallback</th><td>{html.escape("; ".join(str(item) for item in basis_batching_stats.get("fallback_reasons") or []) or "none")}</td></tr>
      <tr><th>Scouting</th><td><code>{html.escape(str(scouting_config.get("score_model") or "n/a"))}</code>, shots={html.escape(str(scouting_config.get("shots") or "n/a"))}, attempts={html.escape(str(scouting_config.get("attempts") or "n/a"))}</td></tr>
      <tr><th>Result score model</th><td><code>{html.escape(str(result.get("result_score_model") or "n/a"))}</code></td></tr>
      <tr><th>Primary result</th><td><code>{html.escape(str(result_config.get("primary") or (result.get("primary_result") or {}).get("source") or "n/a"))}</code></td></tr>
      <tr><th>Optional readouts</th><td>{html.escape(str(len(readout_results)))}</td></tr>
      <tr><th>Global shots</th><td>{html.escape(str(result.get("shots")))}</td></tr>
      <tr><th>Seed mode</th><td>{html.escape(str(result.get("seed_mode")))}</td></tr>
      <tr><th>Run seed</th><td>{html.escape(_format_seed(result.get("run_seed")))} ({html.escape(str(result.get("seed_source")))})</td></tr>
      <tr><th>Shot seed</th><td>{html.escape(_format_seed(result.get("hw_shot_seed")))} ({html.escape(str(result.get("shot_seed_source")))})</td></tr>
      <tr><th>IBM auth source</th><td>{html.escape(str(result.get("ibm_auth_source") or "saved_account_default"))}</td></tr>
      <tr><th>IBM account</th><td>{html.escape(str(result.get("ibm_account_name") or ("provided" if result.get("ibm_account_name_provided") else "not provided")))}</td></tr>
      <tr><th>IBM token</th><td>{html.escape("provided via " + str(result.get("ibm_token_source")) if result.get("ibm_token_provided") else "not provided")}</td></tr>
      <tr><th>IBM channel</th><td>{html.escape(str(result.get("ibm_channel") or "default"))}</td></tr>
      <tr><th>IBM instance CRN</th><td>{'provided' if result.get("ibm_instance_crn_provided") else 'not provided'}</td></tr>
      <tr><th>Gate estimate CRNs</th><td>{html.escape(", ".join(result.get("gate_estimate_backend_crns_provided") or []) or "not provided")}</td></tr>
      <tr><th>IBM custom URL</th><td>{'provided' if result.get("ibm_url_provided") else 'not provided'}</td></tr>
      <tr><th>Optimizer angle mode</th><td>{html.escape(str(result.get("optimizer_angle_mode")))} ({html.escape(str(result.get("optimizer_angle_mode_requested")))})</td></tr>
      <tr><th>Angle extraction</th><td>{html.escape(str(result.get("angle_extraction_mode") or "n/a"))}: {html.escape(str(result.get("angle_extraction_description") or ""))}</td></tr>
      <tr><th>Readout / primary angle mode</th><td>{html.escape(str(result.get("rmsd_angle_mode")))} / {html.escape(str(result.get("primary_angle_mode")))}</td></tr>
      <tr><th>Stop on phase error</th><td>{'yes' if result.get("stop_on_phase_error") else 'no'}</td></tr>
      <tr><th>DOFs / qubits / reps / params</th><td>{result.get("total_dofs", result.get("total_angles"))} ({result.get("total_angle_dofs", result.get("total_angles"))} angle, {result.get("total_length_dofs", 0)} length) / {result.get("n_qubits")} / {result.get("reps")} / {result.get("n_params")}</td></tr>
      <tr><th>Stored angles</th><td>{html.escape(", ".join(result.get("stored_angles") or []) or "none")}</td></tr>
      <tr><th>Stored lengths</th><td>{html.escape(", ".join(result.get("stored_lengths") or []) or "none")}</td></tr>
      <tr><th>Length encoding</th><td>{html.escape(str(result.get("length_encoding_scope") or "n/a"))}</td></tr>
      <tr><th>Metric atom sets</th><td>{html.escape(", ".join(result.get("metric_atom_sets") or []) or "none")}</td></tr>
      <tr><th>RMSD alignment atom set</th><td>{html.escape(str(result.get("rmsd_alignment_atom_set") or "n/a"))}</td></tr>
      <tr><th>Report structure domain</th><td><code>{html.escape(str(result.get("report_structure_domain") or DEFAULT_REPORT_STRUCTURE_DOMAIN))}</code></td></tr>
      <tr><th>Geometry mode / profile</th><td>{html.escape(str(result.get("geometry_mode") or "default"))} / {html.escape(str(result.get("geometry_profile") or "default"))}</td></tr>
      <tr><th>Geometry table</th><td>{html.escape(str(result.get("geometry_table") or "default"))}</td></tr>
      <tr><th>Chi source / selection</th><td>{html.escape(str(result.get("chi_source")))} / {html.escape(str(result.get("chi_selection")))}</td></tr>
      <tr><th>Selective chi map</th><td>{html.escape(_format_selective_chi_map(result.get("selective_chi_map") or {}))}</td></tr>
      <tr><th>Max chi</th><td>{html.escape("all" if result.get("max_chi") is None else str(result.get("max_chi")))}</td></tr>
      <tr><th>Viewer status</th><td>{html.escape(molstar_status)}</td></tr>
    </tbody>
  </table>

  {pheat_score_capabilities_section}

  {external_evaluators_section}

  <h2>Circuit Construction</h2>
  <table>
    <tbody>
      <tr><th>Elapsed</th><td>{html.escape(_format_elapsed(circuit_timing.get("elapsed_s")))}</td></tr>
      <tr><th>Circuit source / name</th><td><code>{html.escape(str(circuit_build_metadata.get("source") or "n/a"))}:{html.escape(str(circuit_build_metadata.get("name") or "n/a"))}</code></td></tr>
      <tr><th>Circuit options</th><td>{html.escape(_format_timing_metadata_value(circuit_build_metadata.get("options") or {}))}</td></tr>
      <tr><th>DOFs / qubits / reps / params</th><td>{html.escape(str(circuit_metadata.get("total_dofs", result.get("total_dofs", result.get("total_angles")))))} ({html.escape(str(circuit_metadata.get("total_angle_dofs", result.get("total_angle_dofs", result.get("total_angles")))))} angle, {html.escape(str(circuit_metadata.get("total_length_dofs", result.get("total_length_dofs", 0))))} length) / {html.escape(str(circuit_metadata.get("n_qubits", result.get("n_qubits"))))} / {html.escape(str(circuit_metadata.get("reps", result.get("reps"))))} / {html.escape(str(circuit_metadata.get("n_params", result.get("n_params"))))}</td></tr>
      <tr><th>Stored angles</th><td>{html.escape(_format_timing_metadata_value(circuit_metadata.get("stored_angles", result.get("stored_angles"))))}</td></tr>
      <tr><th>Stored lengths</th><td>{html.escape(_format_timing_metadata_value(circuit_metadata.get("stored_lengths", result.get("stored_lengths"))))}</td></tr>
      <tr><th>Length encoding</th><td>{html.escape(str(circuit_metadata.get("length_encoding_scope", result.get("length_encoding_scope", "n/a"))))}</td></tr>
      <tr><th>Metric atom sets</th><td>{html.escape(_format_timing_metadata_value(circuit_metadata.get("metric_atom_sets", result.get("metric_atom_sets"))))}</td></tr>
      <tr><th>Chi selection / max chi</th><td>{html.escape(str(circuit_metadata.get("chi_selection", result.get("chi_selection"))))} / {html.escape(str(circuit_metadata.get("max_chi", result.get("max_chi"))))}</td></tr>
      <tr><th>Optimizer</th><td>{html.escape(str(circuit_metadata.get("optimizer_angle_mode", result.get("optimizer_angle_mode"))))} / {html.escape(str(circuit_metadata.get("optimizer_backend_mode", result.get("optimizer_backend_mode"))))}</td></tr>
      <tr><th>Recipe / phase count</th><td>{html.escape(str(circuit_metadata.get("recipe", circuit_metadata.get("phase_preset", result.get("recipe", result.get("phase_preset"))))))} / {html.escape(str(circuit_metadata.get("phase_count", len(phase_results))))}</td></tr>
      <tr><th>Basis circuit batching</th><td>{html.escape(str(circuit_metadata.get("basis_circuit_batching", result.get("basis_circuit_batching_requested"))))}</td></tr>
      <tr><th>Result score / primary</th><td>{html.escape(str(result_config.get("score_model") or result.get("result_score_model")))} / {html.escape(str(result_config.get("primary") or "n/a"))}</td></tr>
    </tbody>
  </table>

  {phase_results_section}

  {phase_comparisons_section}

  {reranking_section}

  {phase_readiness_section}

  {handoff_guard_section}

  {structure_snapshots_section}

  {report_domain_coverage_section}

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

  <h2>PHEAT Radius Of Gyration</h2>
  <div class="report-table-wrap">
    <table class="sortable-table">
      <thead><tr><th>Structure</th><th>Atom set</th><th>Unweighted</th><th>Mass-weighted</th><th>Atoms</th><th>Units</th><th>Status</th></tr></thead>
      <tbody>{rg_rows}</tbody>
    </table>
  </div>

  {diagnostic_rmsd_section}

  {validation_section}

  {physical_readiness_section}

  <h2>Selected PHEAT Score Terms</h2>
  <div class="report-table-wrap">
    <table class="sortable-table">
      <thead><tr><th>Term</th><th>Value</th></tr></thead>
      <tbody>{score_terms}</tbody>
    </table>
  </div>

  {gate_estimate_section}

  {phase_gate_estimate_section}

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
    ibm_auth_config: Optional[IBMRuntimeAuthConfig] = None,
):
    if hw_backend_name is None:
        return None
    backend_key = str(hw_backend_name).lower()
    if backend_key == "none":
        return None
    if backend_key == "statevector":
        raise ValueError(
            "statevector is not a backend name. Omit --backend to use exact local statevector extraction; "
            "use --backend statevector-shots for shot-based statevector sampling."
        )
    if backend_key == StatevectorShotsBackend.name:
        return StatevectorShotsBackend(seed=shot_seed)
    if backend_key == "aer":
        from qiskit_aer import AerSimulator

        if shot_seed is not None:
            return AerSimulator(seed_simulator=int(shot_seed))
        return AerSimulator()
    if backend_key == "least_busy":
        criteria = _least_busy_context(4)
        service = _ibm_runtime_service(criteria, ibm_auth_config)
        return _least_busy_backend(service, 4, ibm_auth_config)
    service = _ibm_runtime_service(f"'{hw_backend_name}'", ibm_auth_config)
    try:
        return service.backend(hw_backend_name)
    except Exception as exc:
        raise _backend_lookup_error(f"'{hw_backend_name}'", exc, ibm_auth_config)


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
    ibm_auth_config: Optional[IBMRuntimeAuthConfig],
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
                ibm_auth_config=ibm_auth_config,
            )
    return registry


def _backend_from_registry(registry: dict[str, object], spec: Optional[str], inherited_spec: str):
    effective = _effective_backend_spec(spec, inherited_spec)
    return registry[effective.lower()]


def _backend_label(backend) -> str:
    return "statevector" if backend is None else f"sampler:{_backend_display_name(backend)}"



def _run_optimization_phases(
    folder: QuantumBiophysicsFolder,
    *,
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
    pheat_reference_structure: Optional[HeavyAtomStructure] = None,
    metric_atom_sets: Sequence[str] = METRIC_ATOM_SETS,
    rmsd_alignment_atom_set: str = DEFAULT_RMSD_ALIGNMENT_ATOM_SET,
    stop_on_phase_error: bool = False,
    initial_snapshot_score_model: Optional[str] = None,
    initial_snapshot_shots: Optional[int] = None,
    initial_snapshot_backend_spec: Optional[str] = None,
    initial_snapshot_transpile: Optional[TranspileConfig] = None,
    evaluator_configs: Optional[dict[str, EvaluatorConfig]] = None,
    evaluator_statuses: Optional[Sequence[dict[str, Any]]] = None,
    phase_comparisons_config: Optional[PhaseComparisonConfig] = None,
    reranking_config: Optional[RerankingConfig] = None,
    phase_readiness_config: Optional[PhaseReadinessConfig] = None,
    handoff_guard_config: Optional[HandoffGuardConfig] = None,
    outdir: Optional[Path] = None,
    prefix: Optional[str] = None,
    status_writer: Optional[RunStatusWriter] = None,
):
    if workflow_progress is not None:
        workflow_progress.start("Optimization phases")
    print("--- STARTING OPTIMIZATION PHASES (HYBRID PIPELINE) ---")
    timings = timings or TimingRecorder()
    folder.timing_recorder = timings
    folder.tracker = PhaseLandscapeTracker()
    base_shots = int(shots)
    if base_shots <= 0:
        raise ValueError("shots must be positive.")
    phases = list(phase_schedule or [])
    if not phases:
        raise ValueError("phase_schedule must contain at least one phase.")
    if result_config is None:
        raise ValueError("result_config is required.")
    readouts = list(readout_schedule or [])
    evaluator_configs = dict(evaluator_configs or {})
    evaluator_status_by_name = _evaluator_status_map(evaluator_statuses or [])
    phase_comparisons_config = phase_comparisons_config or PhaseComparisonConfig(
        enabled=False,
        evaluators=[],
        compare="consecutive_phase_ends",
        affect_selection=False,
    )
    reranking_config = reranking_config or RerankingConfig(
        enabled=False,
        evaluator=None,
        triggers=[],
        candidate_pool={},
        apply="next_phase_start",
    )
    phase_readiness_config = phase_readiness_config or PhaseReadinessConfig(
        enabled=False,
        evaluator=None,
        phases=[],
        on_fail="continue",
        max_clash_count=None,
        max_short_contact_count=None,
        min_nonlocal_distance_a=None,
    )
    handoff_guard_config = handoff_guard_config or HandoffGuardConfig(
        enabled=False,
        evaluator=None,
        phases=[],
        fallback="phase_start",
        abort_on_reject=False,
        allow_improving_unsafe=True,
        max_clash_count=None,
        max_short_contact_count=None,
        min_nonlocal_distance_a=None,
        unsafe_transition_max_short_contact_count=None,
        unsafe_transition_min_nonlocal_distance_a=None,
        unsafe_transition_require_clash_count_decrease=False,
        reject_on_score_worse=True,
        reject_on_clash_count_increase=True,
        reject_on_short_contact_count_increase=True,
        reject_on_min_nonlocal_distance_decrease=True,
        reject_on_nonfinite=True,
    )
    backend_registry = backend_registry or {
        str(inherited_backend_spec).strip().lower(): hw_backend,
    }
    folder.phase_schedule = [asdict(phase) for phase in phases]
    folder.result_config = asdict(result_config)
    folder.readout_schedule = [asdict(readout) for readout in readouts]
    folder.readout_results = []
    folder.structure_snapshots = []
    folder.evaluator_statuses = list(evaluator_statuses or [])
    folder.phase_comparison_results = []
    folder.reranking_results = []
    folder.phase_readiness_results = []
    folder.handoff_guard_results = []
    folder.candidate_records = []
    metric_atom_sets = normalize_metric_atom_sets(metric_atom_sets)
    rmsd_alignment_atom_set = normalize_rmsd_alignment_atom_set(rmsd_alignment_atom_set)

    def _backend_for_spec(spec: Optional[str]):
        return _backend_from_registry(backend_registry, spec, inherited_backend_spec)

    folder.rmsd_angle_mode = "per_phase_readout"
    folder.primary_angle_mode = "primary_result"
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
            f"maxiter={phase.maxiter}, "
            f"optimizer_transpile={_transpile_config_dict(phase.optimizer_transpile)}, "
            f"readout_transpile={_transpile_config_dict(phase.readout_transpile)})"
        )
    print(f"  Primary result   : {result_config.primary}")
    print(f"  Result score     : {result_config.score_model}")
    print(f"  Optional readouts: {len(readouts)}")
    if phase_readiness_config.enabled:
        print(
            "  Phase readiness : "
            f"evaluator={phase_readiness_config.evaluator}, "
            f"phases={','.join(phase_readiness_config.phases)}, "
            f"on_fail={phase_readiness_config.on_fail}"
        )
    if status_writer is not None:
        status_writer.update(
            status="running",
            stage="optimization_phases",
            phase_count=len(phases),
            completed_phase_count=0,
            completed_phases=[],
            force=True,
            flush_console=True,
        )

    if initial_params is None:
        init_params = folder.get_smart_initialization()
    else:
        init_params = initial_params

    def _structure_from_params(
        params,
        *,
        angle_mode: str,
        backend,
        shots: int,
        transpile_config: Optional[TranspileConfig],
    ) -> HeavyAtomStructure:
        angles = folder._angle_vector_from_params(
            params,
            angle_mode=angle_mode,
            backend=backend,
            shots=shots,
            transpile_optimization_level=(
                None if transpile_config is None else transpile_config.optimization_level
            ),
            transpile_seed=None if transpile_config is None else transpile_config.seed,
        )
        return folder.structure_from_angle_vector(angles)

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
        transpile_config: Optional[TranspileConfig],
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
            "transpile": _transpile_config_dict(transpile_config),
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
                    transpile_config=transpile_config,
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
        folder.structure_snapshots.append(snapshot)
        return snapshot.get("structure")

    def _evaluator_is_runnable(name: str) -> bool:
        status = evaluator_status_by_name.get(name)
        if status is None:
            return name in evaluator_configs
        return status.get("status") == "ok"

    def _snapshot_by_key(key: str) -> Optional[dict]:
        return next(
            (snapshot for snapshot in folder.structure_snapshots if snapshot.get("key") == key),
            None,
        )

    def _score_snapshot_with_evaluator(
        snapshot: dict,
        evaluator_name: str,
        *,
        include_prepared_output: bool = False,
    ) -> dict[str, Any]:
        evaluator = evaluator_configs[evaluator_name]
        structure = snapshot.get("structure")
        if structure is None:
            return {
                "evaluator": evaluator_name,
                "score_model": evaluator.score_model,
                "status": "unavailable",
                "error": "snapshot structure is unavailable",
                "total": None,
                "units": None,
                "terms": {},
                "warnings": [],
                "metadata": {},
            }
        return _score_structure_with_evaluator(
            structure,
            evaluator,
            outdir=outdir,
            prefix=prefix,
            candidate_key=str(snapshot.get("key") or snapshot.get("label") or evaluator_name),
            include_prepared_output=include_prepared_output,
        )

    def _run_phase_comparisons_for_snapshot(current_snapshot: dict) -> None:
        if not phase_comparisons_config.enabled:
            return
        phase_snapshots = [
            snapshot
            for snapshot in folder.structure_snapshots
            if snapshot.get("role") == "phase"
            and snapshot.get("snapshot_status") == "ok"
            and snapshot.get("structure") is not None
        ]
        if len(phase_snapshots) < 2:
            return
        previous_snapshot = phase_snapshots[-2]
        for evaluator_name in phase_comparisons_config.evaluators:
            if evaluator_name not in evaluator_configs:
                continue
            if not _evaluator_is_runnable(evaluator_name):
                folder.phase_comparison_results.append(
                    {
                        "status": "skipped",
                        "evaluator": evaluator_name,
                        "previous_snapshot_key": previous_snapshot.get("key"),
                        "current_snapshot_key": current_snapshot.get("key"),
                        "reason": "evaluator is unavailable",
                    }
                )
                continue
            previous_score = _score_snapshot_with_evaluator(previous_snapshot, evaluator_name)
            current_score = _score_snapshot_with_evaluator(current_snapshot, evaluator_name)
            previous_total = previous_score.get("total")
            current_total = current_score.get("total")
            delta = None
            if previous_total is not None and current_total is not None:
                delta = float(current_total) - float(previous_total)
            status = "ok" if previous_score.get("status") == "ok" and current_score.get("status") == "ok" else "error"
            folder.phase_comparison_results.append(
                {
                    "status": status,
                    "evaluator": evaluator_name,
                    "score_model": evaluator_configs[evaluator_name].score_model,
                    "previous_snapshot_key": previous_snapshot.get("key"),
                    "previous_label": previous_snapshot.get("label"),
                    "current_snapshot_key": current_snapshot.get("key"),
                    "current_label": current_snapshot.get("label"),
                    "previous_score": previous_score,
                    "current_score": current_score,
                    "delta_current_minus_previous": delta,
                    "direction": None if delta is None else _rmsd_progress_label(float(current_total), float(previous_total)),
                }
            )

    def _reranking_triggers_for_phase(phase_start_iter: int, phase_end_iter: int) -> list[dict[str, Any]]:
        if not reranking_config.enabled:
            return []
        triggers = reranking_config.triggers or [{"when": "phase_end"}]
        active = []
        for trigger in triggers:
            when = trigger.get("when")
            if when == "phase_end":
                active.append(dict(trigger))
            elif when == "every_evaluations":
                interval = int(trigger.get("interval") or 0)
                if interval > 0 and phase_end_iter > phase_start_iter:
                    item = dict(trigger)
                    item["selection_boundary"] = "phase_end"
                    active.append(item)
        return active

    def _candidate_pool_for_phase(
        *,
        phase_index: int,
        phase_start_iter: int,
        phase_end_iter: int,
        phase_start_params,
        phase_final_params,
        phase_result: dict,
    ) -> list[dict[str, Any]]:
        records = [
            record
            for record in getattr(folder, "candidate_records", [])
            if record.get("phase_index") == phase_index
            and phase_start_iter <= int(record.get("iteration", -1)) < phase_end_iter
            and record.get("params") is not None
            and np.isfinite(float(record.get("objective", math.inf)))
        ]
        pool_config = dict(reranking_config.candidate_pool or {})
        top_k = int(pool_config.get("per_phase_top_k") or 5)
        top_k = max(1, top_k)
        include_latest_raw = pool_config.get("include_latest", True)
        include_latest = (
            include_latest_raw
            if isinstance(include_latest_raw, bool)
            else str(include_latest_raw).strip().lower() not in {"0", "false", "no", "off"}
        )
        include_phase_start_raw = pool_config.get("include_phase_start", False)
        include_phase_start = (
            include_phase_start_raw
            if isinstance(include_phase_start_raw, bool)
            else str(include_phase_start_raw).strip().lower() in {"1", "true", "yes", "on"}
        )
        triggers = _reranking_triggers_for_phase(phase_start_iter, phase_end_iter)
        interval_candidates: list[dict[str, Any]] = []
        intervals = [
            int(trigger.get("interval"))
            for trigger in triggers
            if trigger.get("when") == "every_evaluations" and int(trigger.get("interval") or 0) > 0
        ]
        for interval in intervals:
            interval_candidates.extend(
                record
                for record in records
                if ((int(record.get("iteration", 0)) - phase_start_iter) + 1) % interval == 0
            )
        ranked = sorted(records, key=_candidate_objective_value)[:top_k]
        candidates = [*ranked, *interval_candidates]
        if include_phase_start:
            candidates.append(
                {
                    "candidate_id": f"phase-{phase_index}-start",
                    "iteration": phase_start_iter,
                    "phase_index": phase_index,
                    "phase_name": phase_result.get("name"),
                    "phase_label": phase_result.get("label"),
                    "score_model": phase_result.get("score_model"),
                    "objective": phase_result.get("energy_first"),
                    "status": "ok",
                    "params": np.asarray(phase_start_params, dtype=float).copy(),
                }
            )
        if include_latest:
            candidates.append(
                {
                    "candidate_id": f"phase-{phase_index}-final",
                    "iteration": phase_end_iter,
                    "phase_index": phase_index,
                    "phase_name": phase_result.get("name"),
                    "phase_label": phase_result.get("label"),
                    "score_model": phase_result.get("score_model"),
                    "objective": phase_result.get("objective"),
                    "status": "ok",
                    "params": np.asarray(phase_final_params, dtype=float).copy(),
                }
            )
        unique: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            key = str(candidate.get("candidate_id"))
            unique[key] = candidate
        return sorted(unique.values(), key=_candidate_objective_value)

    def _run_reranking_for_phase(
        *,
        phase_index: int,
        phase: PhaseConfig,
        phase_result: dict,
        phase_start_iter: int,
        phase_end_iter: int,
        phase_start_params,
        phase_final_params,
        phase_readout_angle_mode: str,
        phase_readout_backend,
        phase_readout_transpile: Optional[TranspileConfig],
        apply_allowed: bool,
    ):
        if not reranking_config.enabled or not reranking_config.evaluator:
            return None
        evaluator_name = reranking_config.evaluator
        if evaluator_name not in evaluator_configs:
            return None
        result = {
            "phase_index": phase_index,
            "phase_name": phase.name,
            "phase_label": phase.label,
            "evaluator": evaluator_name,
            "score_model": evaluator_configs[evaluator_name].score_model,
            "status": "skipped",
            "trigger": "phase_end",
            "apply": reranking_config.apply,
            "candidate_count": 0,
            "candidates": [],
            "selected": None,
            "selection_reason": None,
        }
        if not _evaluator_is_runnable(evaluator_name):
            result["reason"] = "evaluator is unavailable"
            folder.reranking_results.append(result)
            return None
        candidate_records = _candidate_pool_for_phase(
            phase_index=phase_index,
            phase_start_iter=phase_start_iter,
            phase_end_iter=phase_end_iter,
            phase_start_params=phase_start_params,
            phase_final_params=phase_final_params,
            phase_result=phase_result,
        )
        result["candidate_count"] = len(candidate_records)
        scored = []
        for rank, candidate in enumerate(candidate_records, start=1):
            snapshot_key = f"rerank_phase_{phase_index:02d}_{rank:02d}"
            structure = _record_structure_snapshot(
                key=snapshot_key,
                role="rerank",
                label=f"Rerank candidate {rank} after phase {phase_index}",
                params=candidate.get("params"),
                angle_mode=phase_readout_angle_mode,
                backend=phase_readout_backend,
                shots=phase.readout_shots,
                transpile_config=phase_readout_transpile,
                score_model=evaluator_configs[evaluator_name].score_model,
                phase_index=phase_index,
                phase_name=phase.name,
                phase_label=phase.label,
                visible_default=False,
            )
            snapshot = _snapshot_by_key(snapshot_key) or {"key": snapshot_key, "structure": structure}
            score = _score_snapshot_with_evaluator(snapshot, evaluator_name)
            item = {
                "rank": rank,
                "candidate_id": candidate.get("candidate_id"),
                "iteration": candidate.get("iteration"),
                "objective": candidate.get("objective"),
                "snapshot_key": snapshot_key,
                "score": score,
                "score_total": score.get("total"),
                "score_units": score.get("units"),
                "physical_counts": _physical_counts_payload(score),
                "status": score.get("status"),
            }
            scored.append(item)
        ok_scored = [item for item in scored if item.get("score_total") is not None and item.get("status") == "ok"]
        selected = min(ok_scored, key=lambda item: float(item["score_total"])) if ok_scored else None
        result["status"] = "ok" if selected is not None else "error"
        result["candidates"] = scored
        result["selected"] = selected
        result["selection_reason"] = (
            f"lowest {evaluator_name} score among {len(ok_scored)} runnable candidates"
            if selected is not None
            else "no runnable candidates produced a finite evaluator score"
        )
        if selected is not None:
            selected["selection_reason"] = result["selection_reason"]
        folder.reranking_results.append(result)
        if selected is None or reranking_config.apply == "report_only" or not apply_allowed:
            return None
        selected_record = next(
            (record for record in candidate_records if str(record.get("candidate_id")) == str(selected.get("candidate_id"))),
            None,
        )
        if selected_record is None or selected_record.get("params") is None:
            return None
        print(
            " > Reranker selected candidate "
            f"{selected.get('candidate_id')} for next phase "
            f"({evaluator_name}, score={_format_optional_float(selected.get('score_total'), 4)} {selected.get('score_units') or ''})"
        )
        return {
            "params": np.asarray(selected_record["params"], dtype=float).copy(),
            "snapshot_key": selected.get("snapshot_key"),
            "candidate_id": selected.get("candidate_id"),
            "score": selected.get("score"),
            "score_total": selected.get("score_total"),
            "score_units": selected.get("score_units"),
        }

    def _periodic_reranking_intervals() -> list[int]:
        if not reranking_config.enabled or not reranking_config.evaluator:
            return []
        intervals = []
        for trigger in reranking_config.triggers or []:
            if trigger.get("when") != "every_evaluations":
                continue
            interval = int(trigger.get("interval") or 0)
            if interval > 0:
                intervals.append(interval)
        return sorted(set(intervals))

    def _run_periodic_reranking_checkpoint(
        *,
        phase_index: int,
        phase: PhaseConfig,
        phase_evaluations: int,
        interval: int,
        candidate: Mapping[str, Any],
        phase_readout_angle_mode: str,
        phase_readout_backend,
        phase_readout_transpile: Optional[TranspileConfig],
    ) -> None:
        evaluator_name = reranking_config.evaluator
        if not reranking_config.enabled or not evaluator_name or evaluator_name not in evaluator_configs:
            return
        result = {
            "phase_index": phase_index,
            "phase_name": phase.name,
            "phase_label": phase.label,
            "evaluator": evaluator_name,
            "score_model": evaluator_configs[evaluator_name].score_model,
            "status": "skipped",
            "trigger": "every_evaluations",
            "interval": int(interval),
            "phase_evaluation": int(phase_evaluations),
            "selection_boundary": "phase_end",
            "apply": "report_only",
            "candidate_count": 1,
            "candidates": [],
            "selected": None,
            "selection_reason": None,
        }
        if not _evaluator_is_runnable(evaluator_name):
            result["reason"] = "evaluator is unavailable"
            folder.reranking_results.append(result)
            return
        snapshot_key = f"rerank_phase_{phase_index:02d}_eval_{phase_evaluations:06d}"
        structure = _record_structure_snapshot(
            key=snapshot_key,
            role="rerank",
            label=f"Periodic rerank checkpoint after phase {phase_index} evaluation {phase_evaluations}",
            params=candidate.get("params"),
            angle_mode=phase_readout_angle_mode,
            backend=phase_readout_backend,
            shots=phase.readout_shots,
            transpile_config=phase_readout_transpile,
            score_model=evaluator_configs[evaluator_name].score_model,
            phase_index=phase_index,
            phase_name=phase.name,
            phase_label=phase.label,
            visible_default=False,
        )
        snapshot = _snapshot_by_key(snapshot_key) or {"key": snapshot_key, "structure": structure}
        score = _score_snapshot_with_evaluator(snapshot, evaluator_name)
        item = {
            "rank": 1,
            "candidate_id": candidate.get("candidate_id"),
            "iteration": candidate.get("iteration"),
            "phase_evaluation": int(phase_evaluations),
            "objective": candidate.get("objective"),
            "snapshot_key": snapshot_key,
            "score": score,
            "score_total": score.get("total"),
            "score_units": score.get("units"),
            "physical_counts": _physical_counts_payload(score),
            "status": score.get("status"),
        }
        result["candidates"] = [item]
        if item.get("status") == "ok" and item.get("score_total") is not None:
            result["status"] = "ok"
            result["selected"] = item
            result["selection_reason"] = (
                f"periodic {evaluator_name} checkpoint at evaluation {phase_evaluations}; "
                "eligible for phase-end handoff pool"
            )
            item["selection_reason"] = result["selection_reason"]
            print(
                " > Periodic reranker checkpoint "
                f"eval={phase_evaluations}, score="
                f"{_format_optional_float(item.get('score_total'), 4)} {item.get('score_units') or ''}"
            )
        else:
            result["status"] = "error"
            result["selection_reason"] = "periodic checkpoint score was unavailable"
        folder.reranking_results.append(result)

    def _run_phase_readiness_for_phase(
        *,
        phase_index: int,
        phase: PhaseConfig,
        params,
        angle_mode: str,
        backend,
        shots: int,
        transpile_config: Optional[TranspileConfig],
    ) -> Optional[dict[str, Any]]:
        if not phase_readiness_config.enabled or not phase_readiness_config.evaluator:
            return None
        if phase.name not in set(phase_readiness_config.phases):
            return None
        evaluator_name = phase_readiness_config.evaluator
        result = {
            "phase_index": phase_index,
            "phase_name": phase.name,
            "phase_label": phase.label,
            "evaluator": evaluator_name,
            "score_model": evaluator_configs.get(evaluator_name).score_model if evaluator_name in evaluator_configs else None,
            "on_fail": phase_readiness_config.on_fail,
            "effective_config": asdict(phase_readiness_config),
            "status": "skipped",
            "decision": "continue",
            "snapshot_key": None,
            "score": None,
            "score_total": None,
            "score_units": None,
            "counts": {},
            "thresholds": {},
            "reasons": [],
        }
        if evaluator_name not in evaluator_configs:
            result.update({"reason": "evaluator is not configured"})
            folder.phase_readiness_results.append(result)
            return result
        if not _evaluator_is_runnable(evaluator_name):
            result.update({"reason": "evaluator is unavailable"})
            folder.phase_readiness_results.append(result)
            return result
        snapshot_key = f"phase_{phase_index:02d}_{_slug(phase.name)}_readiness_input"
        structure = _record_structure_snapshot(
            key=snapshot_key,
            role="phase_readiness",
            label=f"Phase {phase_index}: {phase.label} readiness input",
            params=params,
            angle_mode=angle_mode,
            backend=backend,
            shots=int(shots),
            transpile_config=transpile_config,
            score_model=evaluator_configs[evaluator_name].score_model,
            phase_index=phase_index,
            phase_name=phase.name,
            phase_label=phase.label,
            visible_default=False,
        )
        snapshot = _snapshot_by_key(snapshot_key) or {"key": snapshot_key, "structure": structure}
        score = _score_snapshot_with_evaluator(snapshot, evaluator_name)
        decision = _phase_readiness_decision_payload(score, phase_readiness_config)
        result.update(decision)
        result.update(
            {
                "snapshot_key": snapshot_key,
                "score": score,
                "score_total": score.get("total"),
                "score_units": score.get("units"),
                "decision": "run" if decision["ready"] or phase_readiness_config.on_fail == "continue" else "skip_phase",
            }
        )
        if decision["ready"]:
            print(f" > Phase readiness passed for phase {phase_index} ({evaluator_name})")
        else:
            reason_text = "; ".join(decision["reasons"]) or "not ready"
            print(
                f" > Phase readiness not ready for phase {phase_index}; "
                f"decision={result['decision']}: {reason_text}"
            )
        folder.phase_readiness_results.append(result)
        return result

    def _run_handoff_guard_for_phase(
        *,
        phase_index: int,
        phase: PhaseConfig,
        phase_result: dict,
        phase_start_snapshot_key: str,
        handoff_snapshot_key: str,
        handoff_candidate_id: str,
        phase_start_params,
    ) -> Optional[dict[str, Any]]:
        if not handoff_guard_config.enabled or not handoff_guard_config.evaluator:
            return None
        if not _phase_handoff_guard_enabled(handoff_guard_config, phase):
            return None
        phase_guard_config = _effective_handoff_guard_config(handoff_guard_config, phase)
        evaluator_name = phase_guard_config.evaluator
        result = {
            "phase_index": phase_index,
            "phase_name": phase.name,
            "phase_label": phase.label,
            "evaluator": evaluator_name,
            "fallback": phase_guard_config.fallback,
            "abort_on_reject": bool(phase_guard_config.abort_on_reject),
            "phase_handoff_guard": dict(phase.handoff_guard or {}),
            "effective_config": asdict(phase_guard_config),
            "phase_start_snapshot_key": phase_start_snapshot_key,
            "handoff_snapshot_key": handoff_snapshot_key,
            "handoff_candidate_id": handoff_candidate_id,
            "status": "skipped",
            "decision": "not_applicable",
            "reasons": [],
            "phase_start_score": None,
            "handoff_score": None,
        }
        if evaluator_name not in evaluator_configs:
            result.update({"status": "skipped", "decision": "not_applicable", "reasons": ["evaluator is not configured"]})
            folder.handoff_guard_results.append(result)
            return result
        if not _evaluator_is_runnable(evaluator_name):
            result.update({"status": "skipped", "decision": "not_applicable", "reasons": ["evaluator is unavailable"]})
            folder.handoff_guard_results.append(result)
            return result
        start_snapshot = _snapshot_by_key(phase_start_snapshot_key)
        handoff_snapshot = _snapshot_by_key(handoff_snapshot_key)
        if start_snapshot is None or handoff_snapshot is None:
            result.update({"status": "error", "decision": "reject", "reasons": ["handoff guard snapshots are unavailable"]})
            folder.handoff_guard_results.append(result)
            return result

        start_score = _score_snapshot_with_evaluator(start_snapshot, evaluator_name)
        handoff_score = _score_snapshot_with_evaluator(handoff_snapshot, evaluator_name)
        result["phase_start_score"] = start_score
        result["handoff_score"] = handoff_score
        decision = _handoff_guard_decision_payload(
            start_score,
            handoff_score,
            phase_guard_config,
        )
        result.update(decision)
        reason_text = "; ".join(str(reason) for reason in decision.get("reasons") or [])
        start_counts = decision.get("phase_start_counts") or {}
        handoff_counts = decision.get("handoff_counts") or {}
        count_text = (
            f"start clashes={start_counts.get('clash_count', 'n/a')}, "
            f"candidate clashes={handoff_counts.get('clash_count', 'n/a')}, "
            f"start short={start_counts.get('short_contact_count', 'n/a')}, "
            f"candidate short={handoff_counts.get('short_contact_count', 'n/a')}, "
            f"candidate min distance={_format_optional_float(handoff_counts.get('min_nonlocal_distance'), 4)} A"
        )
        if decision["status"] == "rejected":
            fallback_key = f"phase_{phase_index:02d}_{_slug(phase.name)}_handoff_fallback"
            fallback_structure = _record_structure_snapshot(
                key=fallback_key,
                role="phase",
                label=f"Phase {phase_index}: {phase.label} handoff fallback",
                params=phase_start_params,
                angle_mode=phase_readout_angle_mode,
                backend=phase_readout_backend,
                shots=phase.readout_shots,
                transpile_config=phase.readout_transpile,
                score_model=phase.score_model,
                phase_index=phase_index,
                phase_name=phase.name,
                phase_label=phase.label,
                phase_status="warning",
                phase_status_label="handoff guard fallback",
                visible_default=False,
            )
            result.update(
                {
                    "status": "rejected",
                    "decision": "fallback",
                    "fallback_snapshot_key": fallback_key,
                    "fallback_atom_count": None if fallback_structure is None else len(fallback_structure.atoms),
                }
            )
            phase_result["handoff_guard_status"] = "rejected"
            phase_result["handoff_guard_decision"] = "fallback"
            phase_result["handoff_guard_reasons"] = list(decision["reasons"])
            phase_result["handoff_snapshot_key"] = fallback_key
            detail = f"; {reason_text}" if reason_text else ""
            print(
                f" > Handoff guard rejected phase {phase_index} candidate; "
                f"fallback=phase_start ({count_text}){detail}"
            )
        else:
            phase_result["handoff_guard_status"] = decision["status"]
            phase_result["handoff_guard_decision"] = "accept"
            phase_result["handoff_guard_reasons"] = list(decision["reasons"])
            phase_result["handoff_snapshot_key"] = handoff_snapshot_key
            if decision["status"] == "accepted_with_violations":
                detail = f"; {reason_text}" if reason_text else ""
                print(
                    f" > Handoff guard accepted phase {phase_index} candidate with threshold violations "
                    f"({count_text}){detail}"
                )
            else:
                print(f" > Handoff guard accepted phase {phase_index} candidate ({count_text})")
        folder.handoff_guard_results.append(result)
        return result

    initial_backend = _backend_for_spec(initial_snapshot_backend_spec)
    initial_angle_mode = folder._angle_mode_for_backend(initial_backend)
    _record_structure_snapshot(
        key="circuit_initial",
        role="circuit",
        label="Circuit initial structure",
        params=init_params,
        angle_mode=initial_angle_mode,
        backend=initial_backend,
        shots=int(initial_snapshot_shots or base_shots),
        transpile_config=initial_snapshot_transpile,
        score_model=initial_snapshot_score_model or getattr(folder, "active_score_model", folder.score_model),
        visible_default=False,
    )
    current_param_angle_mode = initial_angle_mode
    current_param_backend = initial_backend
    current_param_shots = int(initial_snapshot_shots or base_shots)
    current_param_transpile_config = initial_snapshot_transpile

    def _get_rmsd(
        params,
        phase: PhaseConfig,
        *,
        readout_backend,
        readout_angle_mode: str,
        transpile_config: Optional[TranspileConfig],
    ):
        if pheat_reference_structure is None:
            return None
        angles = folder._angle_vector_from_params(
            params,
            angle_mode=readout_angle_mode,
            backend=readout_backend,
            shots=phase.readout_shots,
            transpile_optimization_level=(
                None if transpile_config is None else transpile_config.optimization_level
            ),
            transpile_seed=None if transpile_config is None else transpile_config.seed,
        )
        structure = folder.structure_from_angle_vector(angles)
        return _pheat_alignment_details(
            pheat_reference_structure,
            structure,
            atom_sets=metric_atom_sets,
            alignment_atom_set=rmsd_alignment_atom_set,
        )

    current_params = init_params
    final_result = None
    phase_results = []
    previous_rmsd = None

    for phase_index, phase in enumerate(phases, start=1):
        phase_label = phase.label or phase.name
        timing_key = _phase_timing_key(phase_index, phase.name)
        phase_readiness_result = _run_phase_readiness_for_phase(
            phase_index=phase_index,
            phase=phase,
            params=np.asarray(current_params, dtype=float).copy(),
            angle_mode=current_param_angle_mode,
            backend=current_param_backend,
            shots=current_param_shots,
            transpile_config=current_param_transpile_config,
        )
        if phase_readiness_result is not None and phase_readiness_result.get("decision") == "skip_phase":
            phase_start_iter = int(folder.tracker.current_iter)
            timing_record = timings.skip(
                timing_key,
                label=phase_label,
                metadata={
                    "phase": phase.name,
                    "phase_index": phase_index,
                    "phase_label": phase_label,
                    "phase_count": len(phases),
                    "method": phase.optimizer,
                    "score_model": phase.score_model,
                    "phase_readiness_status": phase_readiness_result.get("status"),
                    "phase_readiness_decision": phase_readiness_result.get("decision"),
                    "phase_readiness_reasons": list(phase_readiness_result.get("reasons") or []),
                    "skipped_before_geometry": True,
                    "total_dofs": folder.total_dofs,
                    "total_angle_dofs": folder.total_angle_dofs,
                    "total_length_dofs": folder.total_length_dofs,
                    "length_encoding_scope": folder.length_encoding_scope,
                },
            )
            phase_result = {
                "index": phase_index,
                "name": phase.name,
                "label": phase_label,
                "description": phase.description,
                "timing_key": timing_key,
                "structure_snapshot_key": phase_readiness_result.get("snapshot_key"),
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
                "score_options": dict(phase.score_options),
                "geometry": dict(phase.geometry),
                "geometry_handoff_status": "skipped_before_geometry",
                "geometry_handoff_error": None,
                "optimizer_transpile": _transpile_config_dict(phase.optimizer_transpile),
                "readout_transpile": _transpile_config_dict(phase.readout_transpile),
                "total_dofs": folder.total_dofs,
                "total_angle_dofs": folder.total_angle_dofs,
                "total_length_dofs": folder.total_length_dofs,
                "n_qubits": folder.n_qubits,
                "n_params": folder.n_params,
                "length_encoding_scope": folder.length_encoding_scope,
                "objective": None,
                "success": False,
                "status": None,
                "message": "phase readiness gate skipped phase",
                "phase_status": "warning",
                "phase_status_label": "readiness gate skipped phase",
                "phase_readiness_status": phase_readiness_result.get("status"),
                "phase_readiness_decision": phase_readiness_result.get("decision"),
                "phase_readiness_reasons": list(phase_readiness_result.get("reasons") or []),
                "phase_readiness_snapshot_key": phase_readiness_result.get("snapshot_key"),
                "rmsd": None,
                "rmsd_atom_count": None,
                "rmsd_details": None,
                "elapsed_s": timing_record.get("elapsed_s"),
                "energy_start_index": phase_start_iter,
                "energy_end_index": phase_start_iter,
                "energy_evaluations": 0,
                "energy_first": None,
                "energy_last": None,
                "energy_min": None,
                "energy_max": None,
                "rmsd_progress": "n/a",
            }
            phase_results.append(phase_result)
            if status_writer is not None:
                status_writer.update(
                    status="running",
                    stage="optimization_phases",
                    completed_phase_count=len(phase_results),
                    completed_phases=[
                        {
                            "index": result.get("index"),
                            "name": result.get("name"),
                            "label": result.get("label"),
                            "description": result.get("description"),
                            "status": result.get("phase_status"),
                            "objective": result.get("objective"),
                            "evaluations": result.get("energy_evaluations"),
                            "phase_readiness_status": result.get("phase_readiness_status"),
                            "phase_readiness_decision": result.get("phase_readiness_decision"),
                        }
                        for result in phase_results
                    ],
                    current_phase={
                        "index": phase_index,
                        "total": len(phases),
                        "name": phase.name,
                        "label": phase_label,
                        "description": phase.description,
                        "status": "warning",
                        "status_label": "readiness gate skipped phase",
                        "phase_readiness_status": phase_readiness_result.get("status"),
                        "phase_readiness_decision": phase_readiness_result.get("decision"),
                    },
                    optimization={
                        "total_evaluations": int(folder.tracker.current_iter),
                        "phase_evaluations": 0,
                        "latest_objective": None,
                        "best_objective": None,
                    },
                    force=True,
                    flush_console=True,
                )
            continue

        geometry_updates = _phase_geometry_update_kwargs(phase.geometry)
        geometry_handoff_status = "unchanged"
        geometry_handoff_error = None
        if geometry_updates:
            handoff_structure = None
            if phase_index > 1:
                try:
                    handoff_structure = _structure_from_params(
                        current_params,
                        angle_mode=current_param_angle_mode,
                        backend=current_param_backend,
                        shots=current_param_shots,
                        transpile_config=current_param_transpile_config,
                    )
                except Exception as exc:
                    geometry_handoff_error = str(exc)
            changed = folder.update_geometry_encoding(**geometry_updates)
            if changed:
                if handoff_structure is not None:
                    try:
                        handoff_geometry = structure_to_residue_geometry(
                            handoff_structure,
                            angle_units=folder.angle_units,
                            stored_angles=folder.stored_angles,
                            stored_lengths=folder.stored_lengths,
                            max_chi=folder.max_chi,
                        )
                        folder.set_base_residue_geometry(handoff_geometry)
                        geometry_handoff_status = "base_geometry_from_previous_best"
                    except Exception as exc:
                        folder.set_base_residue_geometry(None)
                        geometry_handoff_status = "base_geometry_unavailable"
                        geometry_handoff_error = str(exc)
                else:
                    geometry_handoff_status = "new_geometry_without_previous_best"
                current_params = np.zeros(folder.n_params)
                print(
                    " > Geometry DOFs : "
                    f"{folder.total_dofs} total "
                    f"({folder.total_angle_dofs} angles, {folder.total_length_dofs} lengths), "
                    f"{folder.n_qubits} qubits, handoff={geometry_handoff_status}"
                )
            elif len(current_params) != folder.n_params:
                current_params = np.zeros(folder.n_params)
                geometry_handoff_status = "parameter_shape_reset"

        phase_optimizer_backend = _backend_for_spec(phase.optimizer_backend)
        phase_readout_backend = _backend_for_spec(phase.readout_backend)
        phase_optimizer_angle_mode = folder._angle_mode_for_backend(phase_optimizer_backend)
        phase_readout_angle_mode = folder._angle_mode_for_backend(phase_readout_backend)
        phase_optimizer_label = _backend_label(phase_optimizer_backend)
        phase_readout_label = _backend_label(phase_readout_backend)
        phase_start_params = np.asarray(current_params, dtype=float).copy()
        folder.active_score_model = phase.score_model
        folder.active_score_options = dict(phase.score_options)
        folder.optimizer_angle_mode = phase_optimizer_angle_mode
        folder.optimizer_backend = phase_optimizer_backend
        folder.optimizer_shots = phase.optimizer_shots
        folder.current_stage = phase_index
        folder.current_phase_name = phase.name
        folder.current_phase_label = phase_label
        minimize_options = _phase_minimize_options(phase)
        phase_maxfev = minimize_options.get("maxfev")
        phase_maxfev = int(phase_maxfev) if phase_maxfev is not None else None
        minimize_kwargs = {
            "method": phase.optimizer,
            "options": minimize_options,
        }
        if phase.tol is not None:
            minimize_kwargs["tol"] = phase.tol

        phase_start_iter = int(folder.tracker.current_iter)
        phase_end_iter = phase_start_iter
        phase_start_snapshot_key = f"phase_{phase_index:02d}_{_slug(phase.name)}_start"
        _record_structure_snapshot(
            key=phase_start_snapshot_key,
            role="phase_start",
            label=f"Phase {phase_index}: {phase_label} start",
            params=phase_start_params,
            angle_mode=phase_readout_angle_mode,
            backend=phase_readout_backend,
            shots=phase.readout_shots,
            transpile_config=phase.readout_transpile,
            score_model=phase.score_model,
            phase_index=phase_index,
            phase_name=phase.name,
            phase_label=phase_label,
            visible_default=False,
        )
        if status_writer is not None:
            status_writer.update(
                status="running",
                stage="optimization_phases",
                completed_phase_count=len(phase_results),
                completed_phases=[
                    {
                        "index": result.get("index"),
                        "name": result.get("name"),
                        "label": result.get("label"),
                        "description": result.get("description"),
                        "status": result.get("phase_status"),
                        "objective": result.get("objective"),
                        "evaluations": result.get("energy_evaluations"),
                        "phase_readiness_status": result.get("phase_readiness_status"),
                        "phase_readiness_decision": result.get("phase_readiness_decision"),
                        "handoff_guard_status": result.get("handoff_guard_status"),
                        "handoff_guard_decision": result.get("handoff_guard_decision"),
                    }
                    for result in phase_results
                ],
                current_phase={
                    "index": phase_index,
                    "total": len(phases),
                    "name": phase.name,
                    "label": phase_label,
                    "description": phase.description,
                    "status": "starting",
                    "optimizer": phase.optimizer,
                    "score_model": phase.score_model,
                    "maxiter": phase.maxiter,
                    "maxfev": phase_maxfev,
                    "optimizer_options": dict(minimize_options),
                    "total_dofs": folder.total_dofs,
                    "total_angle_dofs": folder.total_angle_dofs,
                    "total_length_dofs": folder.total_length_dofs,
                    "n_qubits": folder.n_qubits,
                },
                optimization={
                    "total_evaluations": int(folder.tracker.current_iter),
                    "phase_evaluations": 0,
                    "latest_objective": None,
                    "best_objective": None,
                },
                force=True,
                flush_console=True,
            )
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
                "score_options": dict(phase.score_options),
                "geometry": dict(phase.geometry),
                "geometry_handoff_status": geometry_handoff_status,
                "geometry_handoff_error": geometry_handoff_error,
                "optimizer_transpile": _transpile_config_dict(phase.optimizer_transpile),
                "readout_transpile": _transpile_config_dict(phase.readout_transpile),
                "total_dofs": folder.total_dofs,
                "total_angle_dofs": folder.total_angle_dofs,
                "total_length_dofs": folder.total_length_dofs,
                "length_encoding_scope": folder.length_encoding_scope,
            },
        ):
            print(
                f"Phase {phase_index}/{len(phases)}: {phase_label} "
                f"({phase.optimizer}, score={phase.score_model}, "
                f"optimizer={phase_optimizer_label}, optimizer_shots={folder.optimizer_shots}, "
                f"readout={phase_readout_label}, readout_shots={phase.readout_shots}, "
                f"optimizer_transpile={_transpile_config_dict(phase.optimizer_transpile)}, "
                f"readout_transpile={_transpile_config_dict(phase.readout_transpile)})..."
            )
            folder.tracker.mark_phase(phase_label)
            folder.transpile_optimization_level = phase.optimizer_transpile.optimization_level
            folder.transpile_seed = phase.optimizer_transpile.seed
            phase_wall_start = time.perf_counter()
            phase_best_objective: Optional[float] = None
            last_status_eval = 0
            last_status_time = time.time()
            last_console_heartbeat_time = time.time()
            periodic_rerank_intervals = _periodic_reranking_intervals()
            periodic_rerank_fired: set[tuple[int, int]] = set()

            def _phase_objective(params):
                nonlocal phase_best_objective, last_status_eval, last_status_time, last_console_heartbeat_time
                value = folder.energy_function(params)
                latest_objective = float(value)
                phase_evaluations = int(folder.tracker.current_iter) - phase_start_iter
                if periodic_rerank_intervals and phase_evaluations > 0:
                    latest_candidate = next(
                        (
                            record
                            for record in reversed(getattr(folder, "candidate_records", []) or [])
                            if record.get("phase_index") == phase_index
                            and int(record.get("iteration", -1)) == phase_start_iter + phase_evaluations - 1
                        ),
                        None,
                    )
                    for interval in periodic_rerank_intervals:
                        key = (interval, phase_evaluations)
                        if phase_evaluations % interval == 0 and key not in periodic_rerank_fired and latest_candidate is not None:
                            periodic_rerank_fired.add(key)
                            _run_periodic_reranking_checkpoint(
                                phase_index=phase_index,
                                phase=phase,
                                phase_evaluations=phase_evaluations,
                                interval=interval,
                                candidate=latest_candidate,
                                phase_readout_angle_mode=phase_readout_angle_mode,
                                phase_readout_backend=phase_readout_backend,
                                phase_readout_transpile=phase.readout_transpile,
                            )
                if math.isfinite(latest_objective) and (
                    phase_best_objective is None or latest_objective < phase_best_objective
                ):
                    phase_best_objective = latest_objective
                if status_writer is not None:
                    now = time.time()
                    status_due = (
                        phase_evaluations <= 1
                        or phase_evaluations - last_status_eval >= STATUS_EVAL_INTERVAL
                        or now - last_status_time >= STATUS_UPDATE_INTERVAL_S
                    )
                    console_due = now - last_console_heartbeat_time >= STATUS_CONSOLE_HEARTBEAT_INTERVAL_S
                    if console_due:
                        latest_text = "nan" if not math.isfinite(latest_objective) else f"{latest_objective:.4f}"
                        best_text = (
                            "n/a"
                            if phase_best_objective is None
                            else f"{phase_best_objective:.4f}"
                        )
                        print(
                            " > Phase Progress: "
                            f"eval={phase_evaluations}, latest={latest_text}, best={best_text}, "
                            f"elapsed={_format_elapsed(time.perf_counter() - phase_wall_start)}"
                        )
                        last_console_heartbeat_time = now
                    if status_due or console_due:
                        last_status_eval = phase_evaluations
                        last_status_time = now
                        remaining_evaluations = (
                            max(phase_maxfev - phase_evaluations, 0)
                            if phase_maxfev is not None
                            else None
                        )
                        status_writer.update(
                            status="running",
                            stage="optimization_phases",
                            current_phase={
                                "index": phase_index,
                                "total": len(phases),
                                "name": phase.name,
                                "label": phase_label,
                                "description": phase.description,
                                "status": "optimizing",
                                "optimizer": phase.optimizer,
                                "score_model": phase.score_model,
                                "maxiter": phase.maxiter,
                                "maxfev": phase_maxfev,
                                "optimizer_options": dict(minimize_options),
                                "total_dofs": folder.total_dofs,
                                "total_angle_dofs": folder.total_angle_dofs,
                                "total_length_dofs": folder.total_length_dofs,
                                "n_qubits": folder.n_qubits,
                            },
                            optimization={
                                "total_evaluations": int(folder.tracker.current_iter),
                                "phase_evaluations": phase_evaluations,
                                "latest_objective": latest_objective,
                                "best_objective": phase_best_objective,
                                "phase_elapsed_s": _round_seconds(time.perf_counter() - phase_wall_start),
                                "remaining_evaluations_to_maxfev": remaining_evaluations,
                            },
                            force=True,
                            flush_console=console_due,
                        )
                return value

            final_result = minimize(
                _phase_objective,
                current_params,
                **minimize_kwargs,
            )
            details = _get_rmsd(
                final_result.x,
                phase,
                readout_backend=phase_readout_backend,
                readout_angle_mode=phase_readout_angle_mode,
                transpile_config=phase.readout_transpile,
            )
            rmsd_value = details.get("all_heavy_rmsd") if details is not None else None
            rmsd_atoms = details.get("matched_heavy_atoms") if details is not None else None
            print(f" > Phase Energy : {final_result.fun:.2f}")
            if rmsd_value is not None:
                print(
                    f" > Phase RMSD   : {rmsd_value:.4f} Å "
                    f"({rmsd_atoms} PHEAT heavy atoms, readout_shots={phase.readout_shots})"
                )
            phase_end_iter = int(folder.tracker.current_iter)

        timing_record = timings.sections.get(timing_key) or {}
        phase_energy_values = folder.tracker.history[phase_start_iter:phase_end_iter]
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
        readiness_warning = (
            phase_readiness_result is not None
            and phase_readiness_result.get("status") == "not_ready"
            and phase_readiness_result.get("decision") == "run"
        )
        if phase_status == "ok" and readiness_warning:
            phase_status = "warning"
            phase_status_label = "readiness gate allowed not-ready phase"
        if phase_status != "ok":
            timing_record["status"] = phase_status
        snapshot_key = f"phase_{phase_index:02d}_{_slug(phase.name)}"
        phase_result = {
            "index": phase_index,
            "name": phase.name,
            "label": phase_label,
            "description": phase.description,
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
            "score_options": dict(phase.score_options),
            "geometry": dict(phase.geometry),
            "geometry_handoff_status": geometry_handoff_status,
            "geometry_handoff_error": geometry_handoff_error,
            "optimizer_transpile": _transpile_config_dict(phase.optimizer_transpile),
            "readout_transpile": _transpile_config_dict(phase.readout_transpile),
            "total_dofs": folder.total_dofs,
            "total_angle_dofs": folder.total_angle_dofs,
            "total_length_dofs": folder.total_length_dofs,
            "n_qubits": folder.n_qubits,
            "n_params": folder.n_params,
            "length_encoding_scope": folder.length_encoding_scope,
            "objective": float(final_result.fun),
            "success": optimizer_success,
            "status": optimizer_status,
            "message": optimizer_message,
            "phase_status": phase_status,
            "phase_status_label": phase_status_label,
            "phase_readiness_status": None if phase_readiness_result is None else phase_readiness_result.get("status"),
            "phase_readiness_decision": None if phase_readiness_result is None else phase_readiness_result.get("decision"),
            "phase_readiness_reasons": [] if phase_readiness_result is None else list(phase_readiness_result.get("reasons") or []),
            "phase_readiness_snapshot_key": None if phase_readiness_result is None else phase_readiness_result.get("snapshot_key"),
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
        phase_structure = _record_structure_snapshot(
            key=snapshot_key,
            role="phase",
            label=f"Phase {phase_index}: {phase_label}",
            params=final_result.x,
            angle_mode=phase_readout_angle_mode,
            backend=phase_readout_backend,
            shots=phase.readout_shots,
            transpile_config=phase.readout_transpile,
            score_model=phase.score_model,
            phase_index=phase_index,
            phase_name=phase.name,
            phase_label=phase_label,
            phase_status=phase_status,
            phase_status_label=phase_status_label,
            visible_default=False,
        )
        phase_snapshot = _snapshot_by_key(snapshot_key)
        if phase_snapshot is not None and phase_structure is not None:
            _run_phase_comparisons_for_snapshot(phase_snapshot)
        if phase_status != "ok":
            print(f" > Phase Status : {phase_status_label}")
        if rmsd_value is not None:
            previous_rmsd = rmsd_value
        reranking_selection = _run_reranking_for_phase(
            phase_index=phase_index,
            phase=phase,
            phase_result=phase_result,
            phase_start_iter=phase_start_iter,
            phase_end_iter=phase_end_iter,
            phase_start_params=phase_start_params,
            phase_final_params=final_result.x,
            phase_readout_angle_mode=phase_readout_angle_mode,
            phase_readout_backend=phase_readout_backend,
            phase_readout_transpile=phase.readout_transpile,
            apply_allowed=True,
        )
        handoff_params = (
            np.asarray(reranking_selection["params"], dtype=float).copy()
            if reranking_selection is not None
            else np.asarray(final_result.x, dtype=float).copy()
        )
        handoff_snapshot_key = (
            str(reranking_selection.get("snapshot_key"))
            if reranking_selection is not None and reranking_selection.get("snapshot_key")
            else snapshot_key
        )
        handoff_candidate_id = (
            str(reranking_selection.get("candidate_id"))
            if reranking_selection is not None and reranking_selection.get("candidate_id")
            else f"phase-{phase_index}-final"
        )
        handoff_guard_result = _run_handoff_guard_for_phase(
            phase_index=phase_index,
            phase=phase,
            phase_result=phase_result,
            phase_start_snapshot_key=phase_start_snapshot_key,
            handoff_snapshot_key=handoff_snapshot_key,
            handoff_candidate_id=handoff_candidate_id,
            phase_start_params=phase_start_params,
        )
        if handoff_guard_result is not None and handoff_guard_result.get("status") == "rejected":
            handoff_params = np.asarray(phase_start_params, dtype=float).copy()
            if handoff_guard_result.get("abort_on_reject"):
                phase_result["phase_status"] = "error"
                phase_result["phase_status_label"] = "handoff guard rejected candidate"
                folder.phase_results = phase_results
                folder.phase_rmsds = {result["name"]: result["rmsd"] for result in phase_results}
                folder.phase_rmsd_details = {
                    result["name"]: result["rmsd_details"] for result in phase_results
                }
                raise PhaseOptimizationError(phase_result)
        elif handoff_snapshot_key != snapshot_key:
            selected_key = f"phase_{phase_index:02d}_{_slug(phase.name)}_handoff_selected"
            selected_label = f"Phase {phase_index}: {phase.label} selected handoff"
            selected_status = phase_result.get("handoff_guard_status") or "selected"
            selected_status_label = (
                "handoff accepted with threshold violations"
                if selected_status == "accepted_with_violations"
                else "reranked handoff selected"
            )
            selected_structure = _record_structure_snapshot(
                key=selected_key,
                role="phase",
                label=selected_label,
                params=handoff_params,
                angle_mode=phase_readout_angle_mode,
                backend=phase_readout_backend,
                shots=phase.readout_shots,
                transpile_config=phase.readout_transpile,
                score_model=phase.score_model,
                phase_index=phase_index,
                phase_name=phase.name,
                phase_label=phase.label,
                phase_status="warning" if selected_status == "accepted_with_violations" else "ok",
                phase_status_label=selected_status_label,
                visible_default=False,
            )
            phase_result["selected_snapshot_key"] = selected_key
            phase_result["selected_source_snapshot_key"] = handoff_snapshot_key
            phase_result["selected_candidate_id"] = handoff_candidate_id
            phase_result["selected_atom_count"] = None if selected_structure is None else len(selected_structure.atoms)
        current_params = handoff_params
        current_param_angle_mode = phase_readout_angle_mode
        current_param_backend = phase_readout_backend
        current_param_shots = phase.readout_shots
        current_param_transpile_config = phase.readout_transpile
        if status_writer is not None:
            status_writer.update(
                status="running",
                stage="optimization_phases",
                completed_phase_count=len(phase_results),
                completed_phases=[
                    {
                        "index": result.get("index"),
                        "name": result.get("name"),
                        "label": result.get("label"),
                        "description": result.get("description"),
                        "status": result.get("phase_status"),
                        "objective": result.get("objective"),
                        "evaluations": result.get("energy_evaluations"),
                        "phase_readiness_status": result.get("phase_readiness_status"),
                        "phase_readiness_decision": result.get("phase_readiness_decision"),
                        "handoff_guard_status": result.get("handoff_guard_status"),
                        "handoff_guard_decision": result.get("handoff_guard_decision"),
                    }
                    for result in phase_results
                ],
                current_phase={
                    "index": phase_index,
                    "total": len(phases),
                    "name": phase.name,
                    "label": phase_label,
                    "description": phase.description,
                    "status": phase_status,
                    "status_label": phase_status_label,
                    "objective": float(final_result.fun),
                    "evaluations": phase_energy_summary["count"],
                    "elapsed_s": timing_record.get("elapsed_s"),
                    "handoff_guard_status": phase_result.get("handoff_guard_status"),
                    "handoff_guard_decision": phase_result.get("handoff_guard_decision"),
                },
                optimization={
                    "total_evaluations": int(folder.tracker.current_iter),
                    "phase_evaluations": phase_energy_summary["count"],
                    "latest_objective": float(final_result.fun),
                    "best_objective": phase_energy_summary["min"],
                },
                force=True,
                flush_console=True,
            )
        if stop_on_phase_error and phase_status == "error":
            folder.phase_results = phase_results
            folder.phase_rmsds = {result["name"]: result["rmsd"] for result in phase_results}
            folder.phase_rmsd_details = {
                result["name"]: result["rmsd_details"] for result in phase_results
            }
            raise PhaseOptimizationError(phase_result)

    if final_result is None:
        raise RuntimeError("No optimizer phases ran.")

    if pheat_reference_structure is not None:
        print("\n── RMSD Progress by Phase (PHEAT heavy atoms) ──────")
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

    folder.phase_results = phase_results
    folder.phase_rmsds = {result["name"]: result["rmsd"] for result in phase_results}
    folder.phase_rmsd_details = {
        result["name"]: result["rmsd_details"] for result in phase_results
    }
    folder.phase_comparison_config = asdict(phase_comparisons_config)
    folder.reranking_config = asdict(reranking_config)
    folder.phase_readiness_config = asdict(phase_readiness_config)
    folder.handoff_guard_config = asdict(handoff_guard_config)
    folder.active_score_options = {}

    optimal_params = np.asarray(current_params, dtype=float).copy()

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
                readout_angle_mode = folder._angle_mode_for_backend(readout_backend)
                readout_key = f"readout_{_slug(readout.name)}"
                print(
                    "  "
                    f"{readout.name}: source={readout.source}, "
                    f"backend={_backend_label(readout_backend)}, shots={readout.shots}, "
                    f"transpile={_transpile_config_dict(readout.transpile)}"
                )
                structure = _record_structure_snapshot(
                    key=readout_key,
                    role="readout",
                    label=f"Readout: {readout.name}",
                    params=optimal_params,
                    angle_mode=readout_angle_mode,
                    backend=readout_backend,
                    shots=readout.shots,
                    transpile_config=readout.transpile,
                    score_model=readout.score_model,
                    visible_default=(
                        readout.primary
                        or result_config.primary == readout.name
                    ),
                )
                details = None
                if pheat_reference_structure is not None and structure is not None:
                    details = _pheat_alignment_details(
                        pheat_reference_structure,
                        structure,
                        atom_sets=metric_atom_sets,
                        alignment_atom_set=rmsd_alignment_atom_set,
                    )
                folder.readout_results.append(
                    {
                        "name": readout.name,
                        "key": readout_key,
                        "source": readout.source,
                        "backend": _effective_backend_spec(readout.backend, inherited_backend_spec),
                        "angle_mode": readout_angle_mode,
                        "shots": readout.shots,
                        "transpile": _transpile_config_dict(readout.transpile),
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
                for snapshot in reversed(folder.structure_snapshots)
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
                    for snapshot in folder.structure_snapshots
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
        folder.primary_angle_mode = primary_snapshot.get("angle_mode")
        folder.primary_backend_mode = primary_snapshot.get("backend")
        folder.primary_shots = primary_snapshot.get("shots")
        folder.primary_result = {
            "source": result_config.primary,
            "snapshot_key": primary_snapshot.get("key"),
            "label": primary_snapshot.get("label"),
            "score_model": result_config.score_model,
            "angle_mode": folder.primary_angle_mode,
            "backend": folder.primary_backend_mode,
            "shots": folder.primary_shots,
            "transpile": primary_snapshot.get("transpile"),
            "atom_count": len(primary_structure.atoms),
        }
        primary_score = _safe_score_payload_for_folder(
            folder,
            optimal_params,
            result_config.score_model,
            fallback_structure=primary_structure,
            options=result_config.score_options,
        )
        folder.primary_result["score"] = primary_score
        folder.primary_result["score_total"] = primary_score.get("total")
        folder.primary_result["score_units"] = primary_score.get("units")
        folder.primary_score = primary_score
        folder.optimizer_objective = float(final_result.fun)
        coords, labels, bonds = folder._structure_to_arrays(primary_structure)
        print(
            "\n[PRIMARY RESULT] "
            f"{folder.primary_result['label']} "
            f"({folder.primary_result['source']}, {folder.primary_result['atom_count']} atoms)"
        )
    primary_total = (getattr(folder, "primary_score", {}) or {}).get("total")
    print(f"\n[RESULT] Optimizer objective: {final_result.fun:.4f}")
    if primary_total is not None:
        print(f"[RESULT] Primary score: {float(primary_total):.4f}")
    returned_objective = float(primary_total) if primary_total is not None else float(final_result.fun)
    return coords, labels, bonds, folder.tracker, optimal_params, returned_objective

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predict", required=True)
    parser.add_argument(
        "--reference-structure",
        dest="reference_structure",
        default=None,
        help="Reference PDB, PHEAT atom-structure JSON, or PHEAT residue-geometry JSON for RMSD/reporting.",
    )
    parser.add_argument(
        "--reference-geometry-mode",
        choices=("metrics-only", "seed"),
        default="metrics-only",
        help=(
            "How --reference-structure affects folding geometry. The default metrics-only mode uses it "
            "only for RMSD/reporting; seed mode also seeds residue geometry templates from the reference."
        ),
    )
    parser.add_argument(
        "--metric-atom-sets",
        default=None,
        help="Comma-separated PHEAT atom sets for structure metrics: ca,backbone,all-heavy,all-atoms.",
    )
    parser.add_argument(
        "--rmsd-alignment-atom-set",
        default=None,
        help="PHEAT atom set used for RMSD alignment, or same-as-rmsd.",
    )
    parser.add_argument(
        "--report-structure-domain",
        choices=SCORING_DOMAINS,
        default=None,
        help="PHEAT domain used for PDB/report/viewer structure artifacts.",
    )
    parser.add_argument("--forcefield", default="protein-coarse-charge-v1")
    parser.add_argument("--replica-id", dest="replica_id", type=int, required=True)
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
        "--backend",
        dest="hw_backend",
        default=None,
        help=(
            "Default simulator or hardware backend inherited by scouting, optimizer phases, readouts, "
            "and gate estimates unless overridden. Omit for exact local statevector extraction."
        ),
    )
    parser.add_argument(
        "--ibm-account",
        default=None,
        help="Optional saved Qiskit Runtime account name from ~/.qiskit/qiskit-ibm.json.",
    )
    parser.add_argument(
        "--ibm-token",
        default=None,
        help="Optional IBM Quantum token. Prefer --ibm-token-env or --ibm-token-file to avoid shell history exposure.",
    )
    parser.add_argument(
        "--ibm-token-env",
        default=None,
        help="Environment variable containing the IBM Quantum token.",
    )
    parser.add_argument(
        "--ibm-token-file",
        default=None,
        help="Path to a file containing the IBM Quantum token.",
    )
    parser.add_argument(
        "--ibm-channel",
        choices=IBM_RUNTIME_CHANNELS,
        default=None,
        help="Qiskit Runtime channel for explicit IBM credentials; defaults to ibm_quantum_platform when a token is provided.",
    )
    parser.add_argument(
        "--ibm-url",
        default=None,
        help="Optional custom IBM/Qiskit Runtime API URL.",
    )
    parser.add_argument(
        "--ibm-instance-crn",
        dest="ibm_instance_crn",
        default=None,
        help="Optional IBM Cloud quantum instance CRN for QiskitRuntimeService IBM backend lookup.",
    )
    parser.add_argument("--shots", type=int, default=4096, help="Default shot count for sampled angle decoding.")
    parser.add_argument("--recipe", dest="phase_preset", default="qtf-heavy-atom-phased", help="QTF recipe name from YAML assets or --recipe-file.")
    parser.add_argument("--recipe-file", dest="phase_config", default=None, help="Optional YAML file containing additional recipes.")
    parser.add_argument("--circuit-template", default=None, help="Parameterized circuit template name, e.g. EfficientSU2 or brickwork-ryrz-nearest-neighbor.")
    parser.add_argument("--circuit-template-source", choices=["qiskit-library", "qtf"], default=None)
    parser.add_argument("--circuit-template-option", action="append", default=[], metavar="key=value")
    parser.add_argument("--circuit-source", choices=["qpy", "qasm2", "qasm3"], default=None)
    parser.add_argument("--circuit-path", default=None, help="QPY/QASM circuit file to use instead of a template.")
    parser.add_argument("--circuit-index", type=int, default=None, help="QPY circuit index; default is 0.")
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
    parser.add_argument(
        "--phase-optimizer-transpile-optimization-level",
        action="append",
        default=[],
        metavar="NAME=LEVEL",
    )
    parser.add_argument(
        "--phase-readout-transpile-optimization-level",
        action="append",
        default=[],
        metavar="NAME=LEVEL",
    )
    parser.add_argument("--phase-optimizer-transpile-seed", action="append", default=[], metavar="NAME=SEED")
    parser.add_argument("--phase-readout-transpile-seed", action="append", default=[], metavar="NAME=SEED")
    parser.add_argument("--phase-tol", action="append", default=[], metavar="NAME=TOL")
    parser.add_argument("--phase-option", action="append", default=[], metavar="NAME:key=value")
    parser.add_argument("--phase-score-option", action="append", default=[], metavar="NAME:key=value")
    parser.add_argument("--phase-geometry-option", action="append", default=[], metavar="NAME:key=value")
    parser.add_argument("--scouting-score", default=None)
    parser.add_argument("--scouting-backend", default=None)
    parser.add_argument("--scouting-shots", type=int, default=None)
    parser.add_argument("--scouting-attempts", type=int, default=None)
    parser.add_argument("--scouting-transpile-optimization-level", default=None)
    parser.add_argument("--scouting-transpile-seed", default=None)
    parser.add_argument("--scouting-score-option", action="append", default=[], metavar="key=value")
    parser.add_argument("--result-score", default=None)
    parser.add_argument("--result-score-option", action="append", default=[], metavar="key=value")
    parser.add_argument(
        "--primary-result",
        default=None,
        help="Primary result source: last_phase_structure or a configured readout name.",
    )
    parser.add_argument("--readout", action="append", default=[], help="Define an optional post-optimization readout name.")
    parser.add_argument("--readout-backend", action="append", default=[], metavar="NAME=BACKEND")
    parser.add_argument("--readout-shots", action="append", default=[], metavar="NAME=SHOTS")
    parser.add_argument("--readout-score", action="append", default=[], metavar="NAME=MODEL")
    parser.add_argument("--readout-transpile-optimization-level", action="append", default=[], metavar="NAME=LEVEL")
    parser.add_argument("--readout-transpile-seed", action="append", default=[], metavar="NAME=SEED")
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
        "--transpile-optimization-level",
        default=None,
        help="Default sampler transpile optimization level: none, 0, 1, 2, or 3.",
    )
    parser.add_argument(
        "--transpile-seed",
        default=None,
        help="Default optional Qiskit seed_transpiler for sampled circuits.",
    )
    parser.add_argument(
        "--estimate-gates",
        nargs="?",
        const=GATE_ESTIMATE_SELECTED_BACKEND,
        default=None,
        metavar="BACKENDS",
        help=(
            "Estimate transpiled gate counts/depths for BACKENDS, a comma-separated "
            "list such as aer,ibm_brisbane. If BACKENDS is omitted, use --backend; "
            "statevector-shots implicitly estimates against aer."
        ),
    )
    parser.add_argument(
        "--gate-estimate-optimization-levels",
        default=None,
        help="Comma-separated gate-estimate transpile optimization levels; level 0 is always included.",
    )
    parser.add_argument(
        "--gate-estimate-transpile-seed",
        default=None,
        help="Optional Qiskit seed_transpiler for gate-estimate transpilation.",
    )
    parser.add_argument(
        "--gate-estimate-backend-crn",
        action="append",
        default=[],
        metavar="BACKEND=CRN",
        help=(
            "Runtime instance CRN to use only for a named IBM gate-estimate backend; "
            "repeat for backends that live under different CRNs."
        ),
    )
    parser.add_argument("--outdir", default="outputs/replica")
    parser.add_argument(
        "--report-command-line",
        default=None,
        help="Exact shell command to show in the report; defaults to a reconstructed Python command.",
    )
    parser.add_argument("--average-reference-backbone", dest="average_reference_backbone", default=False, type=bool)

    parser.add_argument("--angle-units", choices=ANGLE_UNITS, default="radians")
    parser.add_argument(
        "--store-angles",
        default="all",
        help="Comma-separated PHEAT optional angles to optimize/store: omega,tau,theta,all, or blank.",
    )
    parser.add_argument(
        "--store-lengths",
        default=None,
        help="Comma-separated PHEAT bond-length selectors to store/export: all,backbone,sidechain, or ATOM-ATOM keys.",
    )
    parser.add_argument("--max-chi", default=None, help="Maximum chi angles per residue; blank/all/none keeps all.")
    parser.add_argument(
        "--selective-chi",
        action="append",
        default=[],
        metavar="RES=CHI1,CHI2",
        help="Restrict optimized side-chain chis for a residue; repeat for multiple residues.",
    )
    parser.add_argument("--include-terminal-oxt", action="store_true", default=False)
    parser.add_argument("--geometry-mode", default=None, help="Optional PHEAT geometry lookup mode.")
    parser.add_argument("--geometry-table", default=None, help="Optional PHEAT geometry table path or packaged table name.")
    parser.add_argument("--geometry-profile", default=None, help="Optional PHEAT geometry profile name.")
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
    parser.add_argument(
        "--length-encoding-scope",
        choices=["shared-by-type", "per-residue"],
        default="shared-by-type",
        help="How selected PHEAT bond lengths become optimizer DOFs.",
    )
    parser.add_argument("--backbone-length-span", type=float, default=0.08)
    parser.add_argument("--sidechain-length-span", type=float, default=0.12)

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
    status_path = outdir / f"{prefix}_status.json"
    software_versions_path = outdir / f"{prefix}_software_versions.json"
    command_line = _command_line(argv, args.report_command_line)

    def _flush_console_log() -> None:
        console_log_path.write_text(console_capture.getvalue(), encoding="utf-8")

    status_writer = RunStatusWriter(
        status_path,
        replica_id=args.replica_id,
        run_label=run_label,
        command_line=command_line,
        console_output_path=console_log_path,
        flush_console=_flush_console_log,
    )
    execution_context = {
        "command_line": command_line,
        "working_directory": str(Path.cwd()),
        "environment": _environment_snapshot(),
        "console_output_path": str(console_log_path),
        "status_path": str(status_path),
        "software_versions_path": str(software_versions_path),
    }
    stored_angles = normalize_stored_angles(args.store_angles)
    try:
        stored_lengths = _normalize_stored_lengths_for_cli(args.store_lengths)
    except Exception as exc:
        parser.error(str(exc))
    max_chi = normalize_max_chi(args.max_chi)
    selective_chi_map = _parse_selective_chi_map(parser, args.selective_chi)
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
    try:
        initial_geometry = _effective_initial_geometry(
            stored_angles=stored_angles,
            stored_lengths=stored_lengths,
            max_chi=max_chi,
            selective_chi_map=selective_chi_map,
            length_encoding_scope=args.length_encoding_scope,
            backbone_length_span=args.backbone_length_span,
            sidechain_length_span=args.sidechain_length_span,
            phase=phase_schedule.phases[0] if phase_schedule.phases else None,
        )
    except Exception as exc:
        parser.error(str(exc))
    evaluator_statuses = _validate_recipe_evaluators_for_run(
        parser,
        phase_schedule,
        outdir=outdir,
    )
    evaluator_status_by_name = _evaluator_status_map(evaluator_statuses)
    metric_atom_sets = tuple(phase_schedule.metrics.atom_sets)
    rmsd_alignment_atom_set = phase_schedule.metrics.rmsd_alignment_atom_set
    report_structure_domain = phase_schedule.report.structure_domain
    timings = TimingRecorder()
    workflow_progress = WorkflowProgress(args.replica_id, status_writer=status_writer)
    status_writer.update(
        status="starting",
        recipe=phase_schedule.preset,
        output_directory=str(outdir),
        result_path=str(outdir / f"{prefix}_result.json"),
        tracker_path=str(outdir / f"{prefix}_tracker.json"),
        force=True,
        flush_console=True,
    )
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
    ibm_auth_config = _resolve_ibm_runtime_auth_config(args, parser)
    ibm_auth_metadata = _ibm_auth_metadata(ibm_auth_config)
    gate_estimate_requested = args.estimate_gates is not None
    gate_estimate_backend_spec = None
    gate_estimate_source = "not_requested"
    gate_estimate_backends = []
    gate_estimates = []
    gate_estimate_backend_refs = []
    gate_estimate_backend_crn_map = _parse_gate_estimate_backend_crn_map(
        parser,
        args.gate_estimate_backend_crn,
    )
    phase_gate_estimates = []
    workflow_backend_registry = {}

    try:
        gate_estimate_backend_spec, gate_estimate_source = _parse_gate_estimate_backend_spec(
            args.estimate_gates,
            args.hw_backend,
        )
    except ValueError as exc:
        parser.error(str(exc))
    _validate_gate_estimate_backend_crn_map(
        parser,
        gate_estimate_backend_spec,
        gate_estimate_backend_crn_map,
    )

    try:
        print(f"{_console_prefix(args.replica_id)} Starting...")
        workflow_progress.start("Backend access")
        with timings.section(
            "backend_access",
            label="Backend access",
            metadata={
                "hw_backend": args.hw_backend,
                "gate_estimate_requested": gate_estimate_requested,
                **ibm_auth_metadata,
            },
        ):
            inherited_backend_spec = _effective_backend_spec("inherit", args.hw_backend)
            hw_backend = get_hw_backend(
                inherited_backend_spec,
                shot_seed=resolved_shot_seed,
                ibm_auth_config=ibm_auth_config,
            )
            workflow_backend_registry = _resolve_workflow_backends(
                phase_schedule,
                inherited_spec=inherited_backend_spec,
                inherited_backend=hw_backend,
                shot_seed=resolved_shot_seed,
                ibm_auth_config=ibm_auth_config,
            )
            if gate_estimate_requested:
                _verify_gate_estimate_backend_access(
                    gate_estimate_backend_spec,
                    ibm_auth_config,
                    gate_estimate_backend_crn_map,
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
    print(f"  Native profile : {args.forcefield}")
    selected_backend_spec = _effective_backend_spec("inherit", args.hw_backend)
    selected_backend_display = _backend_spec_display(args.hw_backend)
    print(f"  Backend        : {selected_backend_display}")
    if ibm_auth_metadata["ibm_account_name_provided"]:
        print(f"  IBM account    : {ibm_auth_metadata['ibm_account_name']}")
    if ibm_auth_metadata["ibm_token_provided"]:
        print(f"  IBM token      : provided via {ibm_auth_metadata['ibm_token_source']}")
    if ibm_auth_metadata["ibm_channel"]:
        print(f"  IBM channel    : {ibm_auth_metadata['ibm_channel']}")
    if ibm_auth_metadata["ibm_instance_crn_provided"]:
        print("  IBM instance   : CRN provided")
    if ibm_auth_metadata["ibm_url_provided"]:
        print("  IBM URL        : provided")
    print(f"  Derived seed   : {derived_seed}")
    print(f"  Seed mode      : {args.seed_mode}")
    print(f"  Run seed       : {_format_seed(run_seed)} ({seed_source})")
    print(f"  Shot seed      : {_format_seed(resolved_shot_seed)} ({shot_seed_source})")
    print(f"  Optimizer mode : {args.optimizer_angle_mode} -> {optimizer_angle_mode}")
    print(f"  Stop on error  : {'yes' if args.stop_on_phase_error else 'no'}")
    print(f"  Shots          : {shots}")
    print(f"  Max iter       : {args.maxiter}")
    print(f"  Recipe         : {phase_schedule.preset}")
    print(f"  Recipe source  : {phase_schedule.source}")
    if phase_schedule.description:
        print(f"  Recipe desc    : {phase_schedule.description}")
    if phase_schedule.circuit is not None:
        print(
            "  Circuit        : "
            f"{phase_schedule.circuit.get('source')}:{phase_schedule.circuit.get('path')}"
        )
    else:
        circuit_template = phase_schedule.circuit_template or {}
        print(
            "  Circuit        : "
            f"{circuit_template.get('source')}:{circuit_template.get('name')}"
        )
    print(f"  Basis batching : {phase_schedule.basis_circuit_batching}")
    print(
        "  Transpile      : "
        f"optimization_level={phase_schedule.default_transpile.optimization_level}, "
        f"seed={phase_schedule.default_transpile.seed}"
    )
    print(
        f"  Scouting       : score={phase_schedule.scouting.score_model}, "
        f"backend={phase_schedule.scouting.backend}, shots={phase_schedule.scouting.shots}, "
        f"attempts={phase_schedule.scouting.attempts}, "
        f"transpile={_transpile_config_dict(phase_schedule.scouting.transpile)}"
    )
    print("  Phases         :")
    for phase in phase_schedule.phases:
        print(
            "    "
            f"{phase.name}: optimizer={phase.optimizer}, score={phase.score_model}, "
            f"optimizer_backend={phase.optimizer_backend}, readout_backend={phase.readout_backend}, "
            f"optimizer_shots={phase.optimizer_shots}, readout_shots={phase.readout_shots}, "
            f"maxiter={phase.maxiter}, "
            f"optimizer_transpile={_transpile_config_dict(phase.optimizer_transpile)}, "
            f"readout_transpile={_transpile_config_dict(phase.readout_transpile)}"
        )
        if phase.description:
            print(f"      description={phase.description}")
    print(f"  Primary result : {phase_schedule.result.primary}")
    print(f"  Result score   : {phase_schedule.result.score_model}")
    print(f"  Readouts       : {len(phase_schedule.readouts)}")
    if phase_schedule.phase_readiness.enabled:
        print(
            "  Readiness     : "
            f"evaluator={phase_schedule.phase_readiness.evaluator}, "
            f"phases={','.join(phase_schedule.phase_readiness.phases)}, "
            f"on_fail={phase_schedule.phase_readiness.on_fail}"
        )
    if evaluator_statuses:
        print("  Evaluators     :")
        for status in evaluator_statuses:
            detail = status.get("errors") or status.get("warnings") or []
            suffix = f" ({'; '.join(str(item) for item in detail)})" if detail else ""
            print(
                "    "
                f"{status['name']}: score={status['score_model']}, "
                f"status={status['status']}, required={status['required']}{suffix}"
            )
    else:
        inactive_count = len(phase_schedule.evaluators)
        suffix = f" ({inactive_count} configured inactive)" if inactive_count else ""
        print(f"  Evaluators     : none{suffix}")
    print(f"  Metric atoms   : {','.join(metric_atom_sets)}")
    print(f"  RMSD alignment : {rmsd_alignment_atom_set}")
    print(f"  Report domain  : {report_structure_domain}")
    print(f"  Stored angles  : {','.join(initial_geometry['stored_angles']) if initial_geometry['stored_angles'] else 'none'}")
    print(f"  Stored lengths : {','.join(initial_geometry['stored_lengths']) if initial_geometry['stored_lengths'] else 'none'}")
    print(f"  Length encoding: {initial_geometry['length_encoding_scope']}")
    print(f"  Angle units    : {args.angle_units}")
    if args.geometry_mode or args.geometry_table or args.geometry_profile:
        print(
            "  Geometry       : "
            f"mode={args.geometry_mode or 'default'}, "
            f"table={args.geometry_table or 'default'}, "
            f"profile={args.geometry_profile or 'default'}"
        )
    print(f"  Chi selection  : {_chi_selection_summary(initial_geometry['max_chi'], initial_geometry['selective_chi_map'])}")
    print(f"  Selective chis : {_format_selective_chi_map(initial_geometry['selective_chi_map'])}")
    if gate_estimate_requested:
        print(f"  Gate estimates : {gate_estimate_backend_spec} ({gate_estimate_source})")
        if gate_estimate_backend_crn_map:
            print(f"  Gate CRNs      : {','.join(sorted(gate_estimate_backend_crn_map))} (provided)")
        print(
            "  Gate transpile : "
            f"levels={','.join(str(level) for level in phase_schedule.gate_estimate_optimization_levels)}, "
            f"seed={phase_schedule.gate_estimate_transpile_seed}"
        )

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
                    stored_angles=initial_geometry["stored_angles"],
                    stored_lengths=initial_geometry["stored_lengths"],
                    max_chi=initial_geometry["max_chi"],
                    include_terminal_oxt=args.include_terminal_oxt,
                    report_structure_domain=report_structure_domain,
                    geometry_mode=args.geometry_mode,
                    geometry_table=args.geometry_table,
                    geometry_profile=args.geometry_profile,
                )
                if pheat_reference.sequence != args.predict:
                    parser.error(
                        "--reference-structure sequence does not match --predict: "
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

    workflow_progress.start("Circuit construction")
    timings.start("circuit_construction")
    try:
        folder = QuantumBiophysicsFolder(
            args.predict,
            force_field=args.forcefield,
            selective_chi_map=initial_geometry["selective_chi_map"],
            angle_units=args.angle_units,
            stored_angles=initial_geometry["stored_angles"],
            stored_lengths=initial_geometry["stored_lengths"],
            max_chi=initial_geometry["max_chi"],
            include_terminal_oxt=args.include_terminal_oxt,
            geometry_mode=args.geometry_mode,
            geometry_table=args.geometry_table,
            geometry_profile=args.geometry_profile,
            score_model=phase_schedule.result.score_model,
            bond_angle_encoding=args.bond_angle_encoding,
            tau_center_deg=args.tau_center_deg,
            tau_span_deg=args.tau_span_deg,
            theta_center_deg=args.theta_center_deg,
            theta_span_deg=args.theta_span_deg,
            length_encoding_scope=initial_geometry["length_encoding_scope"],
            backbone_length_span=initial_geometry["backbone_length_span"],
            sidechain_length_span=initial_geometry["sidechain_length_span"],
            optimizer_angle_mode=optimizer_angle_mode,
            optimizer_backend=hw_backend,
            optimizer_shots=shots,
            basis_circuit_batching=phase_schedule.basis_circuit_batching,
            transpile_optimization_level=phase_schedule.default_transpile.optimization_level,
            transpile_seed=phase_schedule.default_transpile.seed,
            reference_residue_geometry=(
                pheat_reference.residue_geometry
                if pheat_reference is not None and args.reference_geometry_mode == "seed"
                else None
            ),
            circuit_template=phase_schedule.circuit_template,
            circuit=phase_schedule.circuit,
        )
    except Exception:
        timings.stop("circuit_construction", label="Circuit construction", status="error")
        raise
    circuit_metadata = {
        "total_angles": folder.total_angles,
        "n_qubits": folder.n_qubits,
        "reps": folder.reps,
        "n_params": folder.n_params,
        "circuit": dict(folder.circuit_metadata),
        "stored_angles": list(initial_geometry["stored_angles"]),
        "stored_lengths": list(initial_geometry["stored_lengths"]),
        "requested_stored_angles": list(stored_angles),
        "requested_stored_lengths": list(stored_lengths),
        "length_encoding_scope": initial_geometry["length_encoding_scope"],
        "backbone_length_span": initial_geometry["backbone_length_span"],
        "sidechain_length_span": initial_geometry["sidechain_length_span"],
        "total_angle_dofs": folder.total_angle_dofs,
        "total_length_dofs": folder.total_length_dofs,
        "total_dofs": folder.total_dofs,
        "metric_atom_sets": list(metric_atom_sets),
        "rmsd_alignment_atom_set": rmsd_alignment_atom_set,
        "report_structure_domain": report_structure_domain,
        "geometry_mode": args.geometry_mode,
        "geometry_table": args.geometry_table,
        "geometry_profile": args.geometry_profile,
        "chi_selection": _chi_selection_summary(initial_geometry["max_chi"], initial_geometry["selective_chi_map"]),
        "selective_chi_map": {
            str(key): list(value)
            for key, value in initial_geometry["selective_chi_map"].items()
        },
        "max_chi": "all" if initial_geometry["max_chi"] is None else initial_geometry["max_chi"],
        "optimizer_angle_mode": optimizer_angle_mode,
        "optimizer_backend_mode": None if hw_backend is None else _backend_display_name(hw_backend),
        "recipe": phase_schedule.preset,
        "recipe_source": phase_schedule.source,
        "phase_preset": phase_schedule.preset,
        "phase_source": phase_schedule.source,
        "phase_count": len(phase_schedule.phases),
        "external_evaluator_count": len(phase_schedule.evaluators),
        "phase_comparisons_enabled": phase_schedule.phase_comparisons.enabled,
        "reranking_enabled": phase_schedule.reranking.enabled,
        "phase_readiness_enabled": phase_schedule.phase_readiness.enabled,
        "validation_enabled": phase_schedule.validation.enabled,
        "basis_circuit_batching": phase_schedule.basis_circuit_batching,
        "transpile": _transpile_config_dict(phase_schedule.default_transpile),
        "scouting_score_model": phase_schedule.scouting.score_model,
        "result_score_model": phase_schedule.result.score_model,
        "primary_result": phase_schedule.result.primary,
        "readout_count": len(phase_schedule.readouts),
    }
    circuit_record = timings.stop(
        "circuit_construction",
        label="Circuit construction",
        metadata=circuit_metadata,
        print_elapsed=False,
    )
    folder.current_stage = 3
    print(f"{_console_prefix(args.replica_id)} Circuit construction complete")
    print(
        f"  Optimized DOFs : {folder.total_dofs} total "
        f"({folder.total_angle_dofs} angles, {folder.total_length_dofs} lengths), "
        f"{folder.n_qubits} qubits, "
        f"{folder.reps} reps, {folder.n_params} params"
    )
    print(
        "  Circuit template: "
        f"{folder.circuit_metadata.get('source')}:{folder.circuit_metadata.get('name')}"
    )
    print(f"  Circuit elapsed : {_format_elapsed(circuit_record['elapsed_s'])}")

    workflow_progress.start("Gate estimates")
    if gate_estimate_requested:
        try:
            with timings.section(
                "gate_estimates",
                label="Gate estimates",
                metadata={
                    "backend_spec": gate_estimate_backend_spec,
                    "backend_crns_provided": sorted(gate_estimate_backend_crn_map),
                    "optimization_levels": list(phase_schedule.gate_estimate_optimization_levels),
                    "transpile_seed": phase_schedule.gate_estimate_transpile_seed,
                },
            ):
                backend_refs = _resolve_gate_estimate_backends(
                    gate_estimate_backend_spec,
                    folder.n_qubits,
                    ibm_auth_config,
                    gate_estimate_backend_crn_map,
                )
                gate_estimate_backend_refs = list(backend_refs)
                gate_estimates = _estimate_gate_costs(
                    folder,
                    backend_refs,
                    optimization_levels=phase_schedule.gate_estimate_optimization_levels,
                    transpile_seed=phase_schedule.gate_estimate_transpile_seed,
                )
                gate_estimate_backends = sorted(
                    {_backend_display_name(ref["backend"]) for ref in backend_refs}
                )
                print(
                    f"{_console_prefix(args.replica_id)} Gate estimates computed for "
                    f"{', '.join(gate_estimate_backends)}"
                )
                for estimate in gate_estimates:
                    if estimate["circuit"] != "measurement_z":
                        continue
                    level = estimate["optimization_level"]
                    print(
                        "  "
                        f"{estimate['backend']} opt{level} measurement_z: "
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
        elif selected_backend_spec.lower() in {"aer", "aer_simulator"}:
            print(f"{_console_prefix(args.replica_id)} Optimizer uses Aer shot-based sampling; this is slower than statevector.")
        else:
            print(
                f"{_console_prefix(args.replica_id)} WARNING: optimizer objective uses the selected backend; "
                "this may submit many sampled circuit jobs."
            )
    else:
        print(
            f"{_console_prefix(args.replica_id)} Optimizer uses exact statevector phase extraction; "
            "shots are not used for optimization readout."
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
            "transpile": _transpile_config_dict(phase_schedule.scouting.transpile),
            "score_options": dict(phase_schedule.scouting.score_options),
        },
    ):
        scouting_backend = _backend_from_registry(
            workflow_backend_registry,
            phase_schedule.scouting.backend,
            args.hw_backend,
        )
        folder.active_score_model = phase_schedule.scouting.score_model
        folder.active_score_options = dict(phase_schedule.scouting.score_options)
        folder.optimizer_backend = scouting_backend
        folder.optimizer_angle_mode = folder._angle_mode_for_backend(scouting_backend)
        folder.optimizer_shots = phase_schedule.scouting.shots
        folder.transpile_optimization_level = phase_schedule.scouting.transpile.optimization_level
        folder.transpile_seed = phase_schedule.scouting.transpile.seed
        start_params = folder.get_smart_initialization(
            n_attempts=phase_schedule.scouting.attempts,
            seed=run_seed,
        )
    print(f"{_console_prefix(args.replica_id)} Init strategy: {strat}, seed: {_format_seed(run_seed)}")

    try:
        coords, labels, bonds, tracker, opt_params, final_energy = _run_optimization_phases(
            folder,
            max_iter=args.maxiter,
            initial_params=start_params,
            hw_backend=hw_backend,
            shots=shots,
            phase_schedule=phase_schedule.phases,
            result_config=phase_schedule.result,
            readout_schedule=phase_schedule.readouts,
            backend_registry=workflow_backend_registry,
            inherited_backend_spec=selected_backend_spec,
            timings=timings,
            workflow_progress=workflow_progress,
            pheat_reference_structure=pheat_reference.metric_structure if pheat_reference is not None else None,
            metric_atom_sets=metric_atom_sets,
            rmsd_alignment_atom_set=rmsd_alignment_atom_set,
            stop_on_phase_error=args.stop_on_phase_error,
            initial_snapshot_score_model=phase_schedule.scouting.score_model,
            initial_snapshot_shots=phase_schedule.scouting.shots,
            initial_snapshot_backend_spec=phase_schedule.scouting.backend,
            initial_snapshot_transpile=phase_schedule.scouting.transpile,
            evaluator_configs=phase_schedule.evaluators,
            evaluator_statuses=evaluator_statuses,
            phase_comparisons_config=phase_schedule.phase_comparisons,
            reranking_config=phase_schedule.reranking,
            phase_readiness_config=phase_schedule.phase_readiness,
            handoff_guard_config=phase_schedule.handoff_guard,
            outdir=outdir,
            prefix=prefix,
            status_writer=status_writer,
        )
    except PhaseOptimizationError as exc:
        print(f"{_console_prefix(args.replica_id)} {exc}")
        execution_context["console_output"] = console_capture.getvalue()
        console_log_path.write_text(execution_context["console_output"], encoding="utf-8")
        status_writer.update(
            status="failed",
            stage="optimization_phases",
            error=str(exc),
            failed_phase=exc.phase_result,
            force=True,
            flush_console=True,
        )
        return 2

    final_structure = folder.structure_from_coords_labels(coords, labels)
    output_stored_angles = folder.stored_angles
    output_stored_lengths = folder.stored_lengths
    output_max_chi = folder.max_chi
    final_score = _safe_score_payload_for_folder(
        folder,
        opt_params,
        phase_schedule.result.score_model,
        fallback_structure=final_structure,
        options=phase_schedule.result.score_options,
    )
    final_residue_geometry = structure_to_residue_geometry(
        final_structure,
        angle_units=args.angle_units,
        stored_angles=output_stored_angles,
        stored_lengths=output_stored_lengths,
        max_chi=output_max_chi,
    )

    pdb_path = outdir / f"{prefix}.pdb"
    ca_pdb_path = outdir / f"{prefix}_ca.pdb"
    heavy_json_path = outdir / f"{prefix}_heavy.json"
    residue_geometry_path = outdir / f"{prefix}_residue_geometry.json"
    score_path = outdir / f"{prefix}_score.json"
    tracker_path = outdir / f"{prefix}_tracker.json"
    result_path = outdir / f"{prefix}_result.json"
    reference_pdb_path = None
    reference_residue_geometry_path = None
    reference_metric_residue_geometry_path = None
    structure_snapshots = list(getattr(folder, "structure_snapshots", []) or [])
    structure_snapshot_payloads = []
    snapshot_structures = {}
    report_structure_domain_coverages: dict[str, dict[str, Any]] = {}

    workflow_progress.start("Artifact generation")
    with timings.section(
        "artifact_writes",
        label="Artifact writes",
        metadata={"outdir": str(outdir)},
    ):
        _final_report_structure, final_report_coverage = _write_report_pdb(
            final_structure,
            pdb_path,
            domain=report_structure_domain,
        )
        report_structure_domain_coverages["final"] = final_report_coverage
        write_structure_json(final_structure, heavy_json_path)
        write_residue_geometry_json(
            final_residue_geometry,
            residue_geometry_path,
            stored_angles=output_stored_angles,
            stored_lengths=output_stored_lengths,
            max_chi=output_max_chi,
        )
        _write_json(score_path, final_score)

        if pheat_reference is not None:
            reference_pdb_path = outdir / f"{prefix}_reference_pheat.pdb"
            reference_residue_geometry_path = outdir / f"{prefix}_reference_residue_geometry.json"
            reference_metric_residue_geometry_path = outdir / f"{prefix}_reference_metric_residue_geometry.json"
            _reference_report_structure, reference_report_coverage = _write_report_pdb(
                pheat_reference.structure,
                reference_pdb_path,
                domain=report_structure_domain,
            )
            if pheat_reference.source_domain_coverage is not None:
                reference_report_coverage = {
                    **reference_report_coverage,
                    "source_input_coverage": dict(pheat_reference.source_domain_coverage),
                }
            report_structure_domain_coverages["reference"] = reference_report_coverage
            write_residue_geometry_json(
                pheat_reference.residue_geometry,
                reference_residue_geometry_path,
                stored_angles=output_stored_angles,
                stored_lengths=output_stored_lengths,
                max_chi=output_max_chi,
            )
            write_residue_geometry_json(
                pheat_reference.metric_residue_geometry,
                reference_metric_residue_geometry_path,
                stored_angles=output_stored_angles,
                stored_lengths=output_stored_lengths,
                max_chi=output_max_chi,
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
                report_structure, report_coverage = _write_report_pdb(
                    structure,
                    raw_pdb_path,
                    domain=report_structure_domain,
                )
                payload["source_atom_count"] = len(structure.atoms)
                payload["atom_count"] = len(report_structure.atoms)
                payload["report_structure_domain"] = report_structure_domain
                payload["report_domain_coverage"] = report_coverage
            if raw_pdb_path is not None:
                payload["pdb_path"] = str(raw_pdb_path)
                payload["viewer_pdb_path"] = str(raw_pdb_path)
            if structure is not None:
                snapshot_structures[key] = structure
                payload.setdefault("source_atom_count", len(structure.atoms))
                payload.setdefault("atom_count", len(structure.atoms))
            structure_snapshot_payloads.append(payload)

    phase_results = getattr(folder, "phase_results", [])
    phase_comparison_results = list(getattr(folder, "phase_comparison_results", []) or [])
    reranking_results = list(getattr(folder, "reranking_results", []) or [])
    phase_readiness_results = list(getattr(folder, "phase_readiness_results", []) or [])
    handoff_guard_results = list(getattr(folder, "handoff_guard_results", []) or [])
    readout_results = list(getattr(folder, "readout_results", []) or [])
    workflow_progress.start("Phase gate estimates")
    if gate_estimate_requested:
        try:
            with timings.section(
                "phase_gate_estimates",
                label="Phase gate estimates",
                metadata={
                    "backend_spec": gate_estimate_backend_spec,
                    "backend_crns_provided": sorted(gate_estimate_backend_crn_map),
                    "optimization_levels": list(phase_schedule.gate_estimate_optimization_levels),
                    "transpile_seed": phase_schedule.gate_estimate_transpile_seed,
                    "phase_count": len(phase_results),
                },
            ):
                phase_gate_estimates = _estimate_phase_gate_costs(
                    phase_results,
                    circuit_template=phase_schedule.circuit_template,
                    circuit=phase_schedule.circuit,
                    backend_refs=gate_estimate_backend_refs,
                    optimization_levels=phase_schedule.gate_estimate_optimization_levels,
                    transpile_seed=phase_schedule.gate_estimate_transpile_seed,
                )
        except Exception as exc:
            print(f"{_console_prefix(args.replica_id)} Phase gate estimates failed: {exc}")
            phase_gate_estimates = [
                {
                    "status": "unavailable",
                    "error": str(exc),
                }
            ]
    else:
        timings.skip(
            "phase_gate_estimates",
            label="Phase gate estimates",
            metadata={"enabled": False},
        )
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
                atom_sets=metric_atom_sets,
                alignment_atom_set=rmsd_alignment_atom_set,
            )
            primary_vs_last_phase_rmsd = primary_vs_last_phase_details["all_heavy_rmsd"]
            primary_vs_last_phase_atom_count = primary_vs_last_phase_details["matched_heavy_atoms"]
            print(
                f"{_console_prefix(args.replica_id)} Primary vs last phase RMSD: "
                f"{_format_optional_angstrom(primary_vs_last_phase_rmsd, 3)} "
                f"({primary_vs_last_phase_atom_count} PHEAT heavy atoms)"
            )
        except Exception as exc:
            primary_vs_last_phase_details = {"status": "unavailable", "error": str(exc)}
    for readout_result in readout_results:
        readout_structure = snapshot_structures.get(readout_result.get("snapshot_key"))
        if readout_structure is None:
            continue
        try:
            drift_details = _pheat_alignment_details(
                final_structure,
                readout_structure,
                atom_sets=metric_atom_sets,
                alignment_atom_set=rmsd_alignment_atom_set,
            )
            readout_result["primary_drift_rmsd"] = drift_details["all_heavy_rmsd"]
            readout_result["primary_drift_atom_count"] = drift_details["matched_heavy_atoms"]
            readout_result["primary_drift_details"] = drift_details
        except Exception as exc:
            readout_result["primary_drift_details"] = {"status": "unavailable", "error": str(exc)}
    validation_results = []
    workflow_progress.start("External validation")
    if phase_schedule.validation.enabled:
        with timings.section(
            "external_validation",
            label="External validation",
            metadata={
                "candidate_sets": list(phase_schedule.validation.candidates),
                "evaluators": list(phase_schedule.validation.evaluators),
            },
        ):
            print("\n[VALIDATION] External evaluator checks")
            validation_results = _run_validation_evaluators(
                phase_schedule.validation,
                phase_schedule.evaluators,
                evaluator_status_by_name,
                primary_structure=final_structure,
                primary_snapshot_key=(getattr(folder, "primary_result", {}) or {}).get("snapshot_key"),
                structure_snapshot_payloads=structure_snapshot_payloads,
                snapshot_structures=snapshot_structures,
                reranking_results=reranking_results,
                outdir=outdir,
                prefix=prefix,
            )
            for item in validation_results:
                warning_text = "; ".join(str(part) for part in item.get("warnings") or [])
                warning_suffix = f" [warnings: {warning_text}]" if warning_text else ""
                if item.get("status") == "ok":
                    print(
                        "  "
                        f"{item.get('label')} / {item.get('evaluator')}: "
                        f"{_format_optional_float(item.get('score_total'), 4)} {item.get('score_units') or ''}"
                        f"{warning_suffix}"
                    )
                else:
                    print(
                        "  "
                        f"{item.get('label')} / {item.get('evaluator')}: "
                        f"{item.get('status')} {item.get('error') or ''}"
                        f"{warning_suffix}"
                    )
    else:
        timings.skip(
            "external_validation",
            label="External validation",
            metadata={"enabled": False},
        )
    physical_readiness = _physical_readiness_payload(
        final_score,
        validation_results,
        max_clash_count=(
            phase_schedule.handoff_guard.max_clash_count
            if phase_schedule.handoff_guard.max_clash_count is not None
            else 0
        ),
        max_short_contact_count=(
            phase_schedule.handoff_guard.max_short_contact_count
            if phase_schedule.handoff_guard.max_short_contact_count is not None
            else 0
        ),
        min_nonlocal_distance_a=(
            phase_schedule.handoff_guard.min_nonlocal_distance_a
            if phase_schedule.handoff_guard.min_nonlocal_distance_a is not None
            else 0.7
        ),
    )
    landscape_path = None
    interactive_landscape_path = outdir / f"{prefix}_landscape_interactive.html"
    timings.skip("landscape_plot", label="Landscape plot")

    timings.start("interactive_landscape_plot")
    try:
        from qtf.visualization import plot_tracker_energy_landscape

        plot_tracker_energy_landscape(
            tracker,
            sequence=args.predict,
            forcefield=args.forcefield,
            save_path=interactive_landscape_path,
            title=f"Optimization Energy Landscape | {run_label} | {phase_schedule.result.score_model}",
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
                rmsd_details = _pheat_alignment_details(
                    pheat_reference.metric_structure,
                    final_structure,
                    atom_sets=metric_atom_sets,
                    alignment_atom_set=rmsd_alignment_atom_set,
                )
                rmsd = rmsd_details["all_heavy_rmsd"]
                rmsd_atom_count = rmsd_details["matched_heavy_atoms"]
                rmsd_reference_atom_count = rmsd_details["reference_atom_count"]
                rmsd_predicted_atom_count = rmsd_details["target_atom_count"]
                reference_ca = _ca_coords_from_structure(pheat_reference.metric_structure)
                t_e2e, t_rg = _physics_metrics(reference_ca)
                print(
                    f"{_console_prefix(args.replica_id)} Primary RMSD: {_format_optional_angstrom(rmsd, 3)} "
                    f"({rmsd_atom_count} PHEAT heavy atoms)"
                )
            except Exception as exc:
                print(f"{_console_prefix(args.replica_id)} Primary RMSD failed: {exc}")
            reference_rg = _safe_pheat_radius_of_gyration(pheat_reference.metric_structure, atom_sets=metric_atom_sets)
            final_rg = _safe_pheat_radius_of_gyration(final_structure, atom_sets=metric_atom_sets)
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
        final_rg = _safe_pheat_radius_of_gyration(final_structure, atom_sets=metric_atom_sets)
        unavailable_reference_rg = _safe_pheat_radius_of_gyration(None, atom_sets=metric_atom_sets)
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
    selected_score_models = _selected_score_models_for_schedule(phase_schedule)
    software_versions_full = _collect_software_versions(
        selected_score_models=selected_score_models,
        evaluator_statuses=evaluator_statuses,
        selected_quantum_packages=_selected_quantum_package_names(args, gate_estimate_backend_spec),
    )
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
                    pheat_reference.metric_structure,
                    final_structure,
                )
                _aligned_reference_report, aligned_reference_coverage = _write_report_pdb(
                    aligned_reference,
                    reference_aligned_pdb_path,
                    domain=report_structure_domain,
                )
                _aligned_folded_report, aligned_folded_coverage = _write_report_pdb(
                    aligned_folded,
                    folded_aligned_pdb_path,
                    domain=report_structure_domain,
                )
                report_structure_domain_coverages["aligned_reference"] = aligned_reference_coverage
                report_structure_domain_coverages["aligned_folded"] = aligned_folded_coverage
                for snapshot in structure_snapshot_payloads:
                    if snapshot.get("snapshot_status") != "ok":
                        continue
                    key = snapshot.get("key")
                    structure = snapshot_structures.get(key)
                    if structure is None:
                        continue
                    aligned_snapshot_path = outdir / f"{_snapshot_file_stem(prefix, snapshot)}_aligned.pdb"
                    _snapshot_reference, aligned_snapshot, aligned_atom_count = _aligned_pheat_structures(
                        pheat_reference.metric_structure,
                        structure,
                    )
                    aligned_snapshot_report, aligned_snapshot_coverage = _write_report_pdb(
                        aligned_snapshot,
                        aligned_snapshot_path,
                        domain=report_structure_domain,
                    )
                    snapshot["aligned_pdb_path"] = str(aligned_snapshot_path)
                    snapshot["viewer_pdb_path"] = str(aligned_snapshot_path)
                    snapshot["aligned_matched_heavy_atoms"] = aligned_atom_count
                    snapshot["aligned_atom_count"] = len(aligned_snapshot_report.atoms)
                    snapshot["aligned_report_domain_coverage"] = aligned_snapshot_coverage
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
    angle_extraction_mode = (
        "direct_statevector"
        if optimizer_angle_mode == "statevector" and hw_backend is None
        else "sampled_readout"
    )
    angle_extraction_description = (
        "Exact statevector phases are read directly; shot counts are reported for compatibility and circuit estimates only."
        if angle_extraction_mode == "direct_statevector"
        else "Angles are decoded from sampled measurement readout using the configured backend and shot counts."
    )
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
        "optimizer_objective": getattr(folder, "optimizer_objective", None),
        "primary_score_total": final_score.get("total"),
        "primary_score_units": final_score.get("units"),
        "recipe": phase_schedule.preset,
        "recipe_source": phase_schedule.source,
        "recipe_description": phase_schedule.description,
        "phase_preset": phase_schedule.preset,
        "phase_source": phase_schedule.source,
        "phase_config_path": phase_schedule.config_path,
        "phase_schedule": _phase_schedule_payload(phase_schedule),
        "transpile_config": _transpile_config_dict(phase_schedule.default_transpile),
        "basis_circuit_batching_requested": phase_schedule.basis_circuit_batching,
        "basis_circuit_batching_effective": basis_circuit_batching_stats.get("last_effective"),
        "basis_circuit_batching_stats": basis_circuit_batching_stats,
        "phase_results": phase_results,
        "external_evaluators": [asdict(evaluator) for evaluator in phase_schedule.evaluators.values()],
        "external_evaluator_statuses": evaluator_statuses,
        "phase_comparisons_config": asdict(phase_schedule.phase_comparisons),
        "phase_comparison_results": phase_comparison_results,
        "reranking_config": asdict(phase_schedule.reranking),
        "reranking_results": reranking_results,
        "phase_readiness_config": asdict(phase_schedule.phase_readiness),
        "phase_readiness_results": phase_readiness_results,
        "handoff_guard_config": asdict(phase_schedule.handoff_guard),
        "handoff_guard_results": handoff_guard_results,
        "validation_config": asdict(phase_schedule.validation),
        "validation_results": validation_results,
        "physical_readiness": physical_readiness,
        "ready_for_length_tuning": physical_readiness.get("ready_for_length_tuning"),
        "phase_rmsds": phase_rmsds,
        "phase_rmsd_details": phase_rmsd_details,
        "result_config": asdict(phase_schedule.result),
        "report_config": asdict(phase_schedule.report),
        "report_structure_domain": report_structure_domain,
        "report_structure_domain_coverage": report_structure_domain_coverages,
        "primary_result": getattr(folder, "primary_result", None),
        "circuit": dict(folder.circuit_metadata),
        "circuit_template": phase_schedule.circuit_template,
        "loaded_circuit": phase_schedule.circuit,
        "readout_schedule": [asdict(readout) for readout in phase_schedule.readouts],
        "readout_results": readout_results,
        "structure_snapshots": structure_snapshot_payloads,
        "primary_vs_last_phase_rmsd": primary_vs_last_phase_rmsd,
        "primary_vs_last_phase_atom_count": primary_vs_last_phase_atom_count,
        "primary_vs_last_phase_details": primary_vs_last_phase_details,
        "energy_trace": energy_trace,
        "scouting_config": asdict(phase_schedule.scouting),
        "score_model": phase_schedule.result.score_model,
        "score_status": final_score.get("status"),
        "score_total": final_score.get("total"),
        "score_units": final_score.get("units"),
        "score_terms": final_score.get("terms"),
        "result_score_model": phase_schedule.result.score_model,
        "pheat_score_model_capabilities": pheat_score_model_capabilities(),
        "angle_units": args.angle_units,
        "stored_angles": list(output_stored_angles),
        "stored_lengths": list(output_stored_lengths),
        "requested_stored_angles": list(stored_angles),
        "requested_stored_lengths": list(stored_lengths),
        "reference_geometry_mode": args.reference_geometry_mode,
        "length_encoding_scope": folder.length_encoding_scope,
        "backbone_length_span": folder.backbone_length_span,
        "sidechain_length_span": folder.sidechain_length_span,
        "metric_atom_sets": list(metric_atom_sets),
        "rmsd_alignment_atom_set": rmsd_alignment_atom_set,
        "chi_source": "pheat.residue_angle_specs",
        "chi_selection": _chi_selection_summary(output_max_chi, folder.selective_chi_map),
        "selective_chi_map": {
            str(key): list(value)
            for key, value in folder.selective_chi_map.items()
        },
        "max_chi": output_max_chi,
        "include_terminal_oxt": args.include_terminal_oxt,
        "geometry_mode": args.geometry_mode,
        "geometry_table": args.geometry_table,
        "geometry_profile": args.geometry_profile,
        "bond_angle_encoding": args.bond_angle_encoding,
        "tau_center_deg": args.tau_center_deg,
        "tau_span_deg": args.tau_span_deg,
        "theta_center_deg": args.theta_center_deg,
        "theta_span_deg": args.theta_span_deg,
        "total_dofs": folder.total_dofs,
        "total_angle_dofs": folder.total_angle_dofs,
        "total_length_dofs": folder.total_length_dofs,
        "total_angles": folder.total_angles,
        "n_qubits": folder.n_qubits,
        "reps": folder.reps,
        "n_params": folder.n_params,
        "rmsd_to_reference": rmsd,
        "reference_available": pheat_reference is not None and reference_aligned_pdb_path is not None,
        "rmsd_mode": "pheat_structure_metrics" if pheat_reference is not None else None,
        "rmsd_reference_source": str(pheat_reference.source_path) if pheat_reference is not None else None,
        "rmsd_reference_source_type": pheat_reference.source_type if pheat_reference is not None else None,
        "rmsd_reference_numbering": "sequence-indexed" if pheat_reference is not None else None,
        "reference_geometry_mode": args.reference_geometry_mode,
        "rmsd_reference_geometry_path": (
            str(reference_metric_residue_geometry_path)
            if reference_metric_residue_geometry_path is not None
            else None
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
        "software_provenance": software_versions_summary,
        "software_versions_path": str(software_versions_path),
        "hw_backend": selected_backend_display,
        "hw_backend_resolved": backend_display,
        "shots": shots,
        "hw_shot_seed": resolved_shot_seed,
        "shot_seed_source": shot_seed_source,
        "optimizer_angle_mode_requested": args.optimizer_angle_mode,
        "optimizer_angle_mode": optimizer_angle_mode,
        "optimization_angle_mode": optimizer_angle_mode,
        "optimizer_backend_mode": backend_display if optimizer_angle_mode == "sampler" else None,
        "angle_extraction_mode": angle_extraction_mode,
        "angle_extraction_description": angle_extraction_description,
        "rmsd_angle_mode": rmsd_angle_mode,
        "primary_angle_mode": primary_angle_mode,
        "primary_backend_mode": primary_backend_mode,
        "rmsd_backend_mode": backend_display if rmsd_angle_mode == "sampler" else None,
        **ibm_auth_metadata,
        "gate_estimate_requested": gate_estimate_requested,
        "gate_estimate_backend_spec": gate_estimate_backend_spec,
        "gate_estimate_source": gate_estimate_source,
        "gate_estimate_backends": gate_estimate_backends,
        "gate_estimate_backend_crns_provided": sorted(gate_estimate_backend_crn_map),
        "gate_estimate_optimization_levels": list(phase_schedule.gate_estimate_optimization_levels),
        "gate_estimate_transpile_seed": phase_schedule.gate_estimate_transpile_seed,
        "gate_estimates": gate_estimates,
        "phase_gate_estimates": phase_gate_estimates,
        "pdb_path": str(pdb_path),
        "ca_pdb_path": str(ca_pdb_path),
        "heavy_json_path": str(heavy_json_path),
        "residue_geometry_path": str(residue_geometry_path),
        "score_path": str(score_path),
        "reference_pheat_pdb_path": str(reference_pdb_path) if reference_pdb_path is not None else None,
        "reference_residue_geometry_path": (
            str(reference_residue_geometry_path) if reference_residue_geometry_path is not None else None
        ),
        "reference_metric_residue_geometry_path": (
            str(reference_metric_residue_geometry_path) if reference_metric_residue_geometry_path is not None else None
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
        "landscape_path": None,
        "interactive_landscape_path": (
            str(interactive_landscape_path) if interactive_landscape_path is not None else None
        ),
    }

    print(f"\n{_console_prefix(args.replica_id)} Done")
    print(f"  Objective  : {final_energy:.4f}")
    print(f"  Score      : {final_score.get('total')} {final_score.get('units')}")
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
            _write_qtf_html_report(
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
    status_writer.update(
        status="completed",
        stage="complete",
        result_path=str(result_path),
        tracker_path=str(tracker_path),
        report_path=None if report_path is None else str(report_path),
        physical_readiness=physical_readiness,
        final_score_total=final_score.get("total"),
        final_score_units=final_score.get("units"),
        runtime_s=runtime_s,
        force=True,
        flush_console=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
