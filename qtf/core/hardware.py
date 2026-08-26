#!/usr/bin/env python3
"""Single-shot hardware folding from random or saved circuit parameters.

``qtf fold-hardware`` initializes random parameters for the requested sequence
by default. Passing ``--params-json`` instead loads parameters saved by
``qtf fold-simulation`` (format ``qtf.circuit_parameters.v1``). It then:

  1. Reconstructs a ``QuantumBiophysicsFolder`` with matching configuration
     (sequence, geometry settings, and ansatz layout).
  2. Binds the parameters onto the ansatz and executes the circuit once on
     real quantum hardware (sampler mode: measure_all + shots).
  3. Converts the empirical shot distribution to torsion angles via the
     same CDF mapping used by ``QuantumBiophysicsFolder._get_angles_sampler``.
  4. Rebuilds the full heavy-atom structure via NERF
     (``folder.build_full_structure``) and applies the same physical-range
     mapping used in statevector runs.
  5. Writes a PDB file, and — when a reference structure is provided —
     rigid-aligns to it and reports RMSD.

Unless ``--backend-name`` is supplied, IBM Runtime's ``least_busy()`` chooses
an operational, non-simulator backend with enough qubits. Use
``--local-simulator`` for an explicit offline Aer run.

Usage
-----
    qtf fold-hardware --sequence YYDPETGTWY --shots 8192

    qtf fold-hardware \\
        --params-json run_outputs/fold/circuit_parameters \\
        --replica-id 0 --backend-name ibm_torino --shots 8192
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import os
import sys
from datetime import datetime
from time import perf_counter
from typing import Optional, Tuple

import numpy as np

from qtf.core.folder import QuantumBiophysicsFolder
from qtf.utils import save_pdb
from qtf.utils import workflow as utils
from qtf.utils.paths import relativize_absolute_paths

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_call(obj, attr: str, default=None):
    value = getattr(obj, attr, default)
    if callable(value):
        try:
            return value()
        except Exception:
            return default
    return value


def _backend_name(backend) -> str:
    name = _safe_call(backend, "name", None)
    if name is None:
        name = getattr(backend, "name_str", None)
    return str(name if name is not None else backend)


def _package_versions() -> dict:
    versions = {}
    for package in ("qiskit", "qiskit-aer", "qiskit-ibm-runtime"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _coupling_map_summary(backend) -> dict:
    coupling_map = getattr(backend, "coupling_map", None)
    edges = []
    if coupling_map is not None:
        try:
            edges = [tuple(map(int, edge)) for edge in coupling_map.get_edges()]
        except Exception:
            edges = []
    if not edges:
        config = _safe_call(backend, "configuration", None)
        raw_edges = getattr(config, "coupling_map", None) if config is not None else None
        if raw_edges:
            edges = [tuple(map(int, edge)) for edge in raw_edges]

    degree = {}
    for a, b in edges:
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1
    degrees = list(degree.values())
    return {
        "num_edges": int(len(edges)),
        "num_connected_qubits": int(len(degree)),
        "min_degree": int(min(degrees)) if degrees else None,
        "max_degree": int(max(degrees)) if degrees else None,
        "mean_degree": float(np.mean(degrees)) if degrees else None,
    }


def _backend_metadata(backend) -> dict:
    config = _safe_call(backend, "configuration", None)
    status = _safe_call(backend, "status", None)
    target = getattr(backend, "target", None)
    target_operation_names = []
    if target is not None:
        try:
            target_operation_names = sorted(str(name) for name in target.operation_names)
        except Exception:
            target_operation_names = []
    config_basis_gates = getattr(config, "basis_gates", None) if config is not None else None
    config_coupling_map = getattr(config, "coupling_map", None) if config is not None else None
    return {
        "name": _backend_name(backend),
        "version": getattr(backend, "version", None),
        "num_qubits": getattr(backend, "num_qubits", None),
        "basis_gates": list(config_basis_gates or []),
        "coupling_map_summary": _coupling_map_summary(backend),
        "configuration": {
            "backend_name": getattr(config, "backend_name", None) if config is not None else None,
            "backend_version": getattr(config, "backend_version", None) if config is not None else None,
            "n_qubits": getattr(config, "n_qubits", None) if config is not None else None,
            "simulator": getattr(config, "simulator", None) if config is not None else None,
            "local": getattr(config, "local", None) if config is not None else None,
            "conditional": getattr(config, "conditional", None) if config is not None else None,
            "open_pulse": getattr(config, "open_pulse", None) if config is not None else None,
            "memory": getattr(config, "memory", None) if config is not None else None,
            "max_shots": getattr(config, "max_shots", None) if config is not None else None,
            "coupling_map_edge_count": len(config_coupling_map or []) if config_coupling_map is not None else None,
        },
        "status": {
            "operational": getattr(status, "operational", None) if status is not None else None,
            "pending_jobs": getattr(status, "pending_jobs", None) if status is not None else None,
            "status_msg": getattr(status, "status_msg", None) if status is not None else None,
        },
        "target": {
            "num_qubits": getattr(target, "num_qubits", None) if target is not None else None,
            "operation_names": target_operation_names,
            "dt": getattr(target, "dt", None) if target is not None else None,
            "granularity": getattr(target, "granularity", None) if target is not None else None,
            "min_length": getattr(target, "min_length", None) if target is not None else None,
            "pulse_alignment": getattr(target, "pulse_alignment", None) if target is not None else None,
            "acquire_alignment": getattr(target, "acquire_alignment", None) if target is not None else None,
        },
    }


# ---------------------------------------------------------------------------
# Hardware backend selection
# ---------------------------------------------------------------------------
def get_hardware_backend(args, *, min_num_qubits: int) -> Tuple[object, str]:
    """Select an IBM backend, using Runtime's least-busy device by default."""

    if args.local_simulator:
        try:
            from qiskit_aer import AerSimulator
        except ImportError as exc:
            raise SystemExit("--local-simulator requires qiskit-aer") from exc
        return AerSimulator(), "aer"

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError as exc:
        raise SystemExit(
            "qiskit-ibm-runtime is required for hardware execution.\n"
            "Install with: pip install qiskit-ibm-runtime"
        ) from exc

    service_kwargs = {}
    if args.channel:
        service_kwargs["channel"] = args.channel
    if args.instance:
        service_kwargs["instance"] = args.instance
    if args.token:
        service_kwargs["token"] = args.token
    service = QiskitRuntimeService(**service_kwargs)
    if args.backend_name:
        return service.backend(args.backend_name), "ibm_runtime"
    backend = service.least_busy(
        min_num_qubits=int(min_num_qubits),
        operational=True,
        simulator=False,
    )
    logger.info("Runtime least_busy selected backend: %s", _backend_name(backend))
    return backend, "ibm_runtime"


# ---------------------------------------------------------------------------
# One hardware execution: bind → transpile → sample → CDF → angles
# ---------------------------------------------------------------------------
def _counts_from_backend_run(backend, tqc, shots):
    """Legacy ``backend.run(...)`` path (Aer, older IBM backends)."""
    submitted_at = _now_iso()
    t0 = perf_counter()
    job = backend.run(tqc, shots=shots)
    job_id = _safe_call(job, "job_id", None)
    result = job.result()
    completed_at = _now_iso()
    elapsed_s = perf_counter() - t0
    return result.get_counts(), {
        "job_id": job_id,
        "job_status": str(_safe_call(job, "status", "")),
        "queue_position": _safe_call(job, "queue_position", None),
        "submitted_at": submitted_at,
        "completed_at": completed_at,
        "elapsed_seconds": float(elapsed_s),
        "primitive": "backend.run",
    }


def _sampler_max_mitigation_options(shots: int) -> dict:
    return {
        "default_shots": int(shots),
        "dynamical_decoupling": {
            "enable": True,
            "sequence_type": "XY4",
            "extra_slack_distribution": "middle",
            "scheduling_method": "alap",
            "skip_reset_qubits": False,
        },
        "twirling": {
            "enable_gates": True,
            "enable_measure": True,
            "num_randomizations": "auto",
            "shots_per_randomization": "auto",
            "strategy": "all",
        },
        "execution": {
            "meas_type": "classified",
        },
    }


def _counts_from_sampler_v2(backend, tqc, shots, *, sampler_options: Optional[dict] = None):
    """Modern ``SamplerV2`` path (qiskit-ibm-runtime ≥ 0.23)."""
    from qiskit_ibm_runtime import SamplerV2  # type: ignore

    sampler = SamplerV2(mode=backend, options=sampler_options)
    submitted_at = _now_iso()
    t0 = perf_counter()
    job = sampler.run([tqc], shots=shots)
    job_id = _safe_call(job, "job_id", None)
    result = job.result()
    completed_at = _now_iso()
    elapsed_s = perf_counter() - t0
    pub_result = result[0]
    # Prefer the 'meas' classical register created by measure_all(); fall
    # back to whichever single register is present.
    data = pub_result.data
    if hasattr(data, "meas"):
        bit_array = data.meas
    else:
        # pick the first classical register present on the pub result
        (only_name,) = [name for name in dir(data)
                        if not name.startswith("_") and hasattr(getattr(data, name), "get_counts")][:1]
        bit_array = getattr(data, only_name)
    job_meta = {
        "job_id": job_id,
        "job_status": str(_safe_call(job, "status", "")),
        "queue_position": _safe_call(job, "queue_position", None),
        "submitted_at": submitted_at,
        "completed_at": completed_at,
        "elapsed_seconds": float(elapsed_s),
        "primitive": "SamplerV2",
        "sampler_options": sampler_options or {},
    }
    try:
        job_meta["result_metadata"] = dict(getattr(pub_result, "metadata", {}) or {})
    except Exception:
        job_meta["result_metadata"] = {}
    return bit_array.get_counts(), job_meta


def _bitstring_counts_to_pvec(counts: dict, n_qubits: int) -> np.ndarray:
    n_states = 2 ** n_qubits
    pvec = np.zeros(n_states, dtype=float)
    total = float(sum(counts.values()))
    if total <= 0.0:
        raise RuntimeError("Backend returned zero total shots.")
    for bitstring, c in counts.items():
        # Qiskit little-endian: qubit 0 is the rightmost bit.
        bs = str(bitstring).replace(" ", "")[::-1]
        # Strip any '0x' prefix defensively.
        if bs.startswith("x0"):  # already reversed 0x
            bs = bs[2:]
        idx = int(bs, 2)
        pvec[idx] += c / total
    return pvec


def _instruction_num_qubits(instruction) -> int:
    """Return how many qubits an instruction acts on across Qiskit versions."""
    qubits = getattr(instruction, "qubits", None)
    if qubits is not None:
        return len(qubits)
    try:
        return len(instruction[1])
    except Exception:
        operation = getattr(instruction, "operation", None)
        return int(getattr(operation, "num_qubits", 0) or 0)


def _circuit_depth(circuit, *, n_qubits: Optional[int] = None) -> int:
    if n_qubits is None:
        depth = circuit.depth()
    else:
        depth = circuit.depth(filter_function=lambda inst: _instruction_num_qubits(inst) == n_qubits)
    return int(depth or 0)


def _circuit_metrics(circuit) -> dict:
    """Compact circuit resource metrics for persisted hardware metadata."""
    op_counts = {str(name): int(count) for name, count in circuit.count_ops().items()}
    non_gate_names = {"measure", "barrier", "delay"}
    one_qubit_gate_count = 0
    two_qubit_gate_count = 0
    multi_qubit_gate_count = 0
    two_qubit_gate_types = {}
    for inst in circuit.data:
        n_q = _instruction_num_qubits(inst)
        operation = getattr(inst, "operation", None)
        if operation is None:
            operation = inst[0]
        name = str(getattr(operation, "name", ""))
        if name in non_gate_names:
            continue
        if n_q == 1:
            one_qubit_gate_count += 1
        elif n_q == 2:
            two_qubit_gate_count += 1
            two_qubit_gate_types[name] = two_qubit_gate_types.get(name, 0) + 1
        elif n_q > 2:
            multi_qubit_gate_count += 1

    return {
        "num_qubits": int(circuit.num_qubits),
        "num_clbits": int(circuit.num_clbits),
        "depth": _circuit_depth(circuit),
        "size": int(circuit.size()),
        "total_operation_count": int(sum(op_counts.values())),
        "total_gate_count": int(sum(count for name, count in op_counts.items() if name not in non_gate_names)),
        "measurement_count": int(op_counts.get("measure", 0)),
        "barrier_count": int(op_counts.get("barrier", 0)),
        "delay_count": int(op_counts.get("delay", 0)),
        "one_qubit_gate_count": int(one_qubit_gate_count),
        "two_qubit_gate_count": int(two_qubit_gate_count),
        "two_qubit_gate_depth": _circuit_depth(circuit, n_qubits=2),
        "multi_qubit_gate_count": int(multi_qubit_gate_count),
        "operation_counts": op_counts,
        "two_qubit_gate_counts_by_name": {k: int(v) for k, v in sorted(two_qubit_gate_types.items())},
    }


def _qubit_index(circuit, qubit) -> Optional[int]:
    try:
        return int(circuit.find_bit(qubit).index)
    except Exception:
        return None


def _serialize_layout_mapping(mapping, source_circuit=None) -> list[dict]:
    rows = []
    if not mapping:
        return rows
    try:
        items = mapping.items()
    except Exception:
        return rows
    for key, value in items:
        if isinstance(key, int):
            physical = int(key)
            virtual = value
        elif isinstance(value, int):
            physical = int(value)
            virtual = key
        else:
            physical = None
            virtual = key
        rows.append({
            "virtual": str(virtual),
            "virtual_index": _qubit_index(source_circuit, virtual) if source_circuit is not None else None,
            "physical": physical,
        })
    return sorted(rows, key=lambda row: (row["physical"] is None, row["physical"] or -1, str(row["virtual"])))


def _layout_metadata(transpiled_circuit, source_circuit=None) -> dict:
    layout = getattr(transpiled_circuit, "layout", None)
    meta = {
        "transpiled_qubit_count": int(transpiled_circuit.num_qubits),
        "transpiled_qubits": [str(q) for q in getattr(transpiled_circuit, "qubits", [])],
        "initial_layout": [],
        "final_layout": [],
        "input_qubit_mapping": [],
    }
    if layout is None:
        return meta

    initial_layout = getattr(layout, "initial_layout", None)
    final_layout = getattr(layout, "final_layout", None)
    input_qubit_mapping = getattr(layout, "input_qubit_mapping", None)

    if initial_layout is not None:
        try:
            meta["initial_layout"] = _serialize_layout_mapping(initial_layout.get_virtual_bits(), source_circuit)
        except Exception:
            meta["initial_layout"] = []
    if final_layout is not None:
        try:
            meta["final_layout"] = _serialize_layout_mapping(final_layout.get_virtual_bits(), transpiled_circuit)
        except Exception:
            meta["final_layout"] = []
    if input_qubit_mapping:
        meta["input_qubit_mapping"] = _serialize_layout_mapping(input_qubit_mapping, source_circuit)

    try:
        final_index_layout = layout.final_index_layout()
        meta["final_index_layout"] = [
            int(value) if value is not None else None
            for value in final_index_layout
        ]
    except Exception:
        meta["final_index_layout"] = []

    return meta


def _counts_metadata(counts: dict) -> dict:
    normalized_counts = {str(bitstring).replace(" ", ""): int(count) for bitstring, count in counts.items()}
    total = int(sum(normalized_counts.values()))
    probabilities = {
        bitstring: (float(count) / float(total) if total > 0 else 0.0)
        for bitstring, count in normalized_counts.items()
    }
    sorted_counts = sorted(normalized_counts.items(), key=lambda item: (-item[1], item[0]))
    return {
        "counts": dict(sorted_counts),
        "probabilities": {bitstring: probabilities[bitstring] for bitstring, _ in sorted_counts},
        "top_bitstrings": [
            {
                "bitstring": bitstring,
                "count": int(count),
                "probability": probabilities[bitstring],
            }
            for bitstring, count in sorted_counts[:10]
        ],
    }


def hardware_angles(
    folder: QuantumBiophysicsFolder,
    params: np.ndarray,
    backend,
    shots: int,
    *,
    use_sampler_v2: bool,
    optimization_level: int,
    seed_transpiler: Optional[int],
    sampler_max_mitigation: bool,
) -> Tuple[np.ndarray, dict]:
    """Run one hardware folding execution and return physical torsion angles.

    Also returns a small ``meta`` dict with the raw counts summary and
    backend identifier for logging.
    """
    from qiskit import transpile

    if params.size != folder.n_params:
        raise ValueError(
            f"Saved params length {params.size} != ansatz n_params {folder.n_params}"
        )

    param_dict = dict(zip(folder.ansatz.parameters, params))
    bound = folder.ansatz.assign_parameters(param_dict)

    qc = bound.copy()
    qc.measure_all()
    transpile_started_at = _now_iso()
    t0 = perf_counter()
    tqc = transpile(
        qc,
        backend=backend,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
    )
    transpile_completed_at = _now_iso()
    transpile_elapsed_s = perf_counter() - t0
    transpiled_metrics = _circuit_metrics(tqc)
    transpiled_metrics["layout"] = _layout_metadata(tqc, qc)
    circuit_meta = {
        "ansatz": _circuit_metrics(folder.ansatz),
        "bound_measured": _circuit_metrics(qc),
        "transpiled": transpiled_metrics,
        "transpile_optimization_level": int(optimization_level),
        "transpile_seed": seed_transpiler,
        "transpile_started_at": transpile_started_at,
        "transpile_completed_at": transpile_completed_at,
        "transpile_elapsed_seconds": float(transpile_elapsed_s),
    }

    if use_sampler_v2:
        sampler_options = _sampler_max_mitigation_options(shots) if sampler_max_mitigation else None
        counts, job_meta = _counts_from_sampler_v2(backend, tqc, shots, sampler_options=sampler_options)
    else:
        counts, job_meta = _counts_from_backend_run(backend, tqc, shots)

    pvec = _bitstring_counts_to_pvec(counts, folder.n_qubits)

    # CDF-to-angle mapping — identical to folder._get_angles_sampler.
    cdf = np.cumsum(pvec)
    angles = 2.0 * np.pi * cdf - np.pi
    angles = angles[: folder.total_angles]
    # The rebuilt folder applies each DOF's physical convention while converting
    # this decoded vector into residue geometry.
    phys = angles

    meta = {
        "shots_total": int(sum(counts.values())),
        "unique_bitstrings": int(len(counts)),
        "bitstring_counts": _counts_metadata(counts),
        "n_qubits": int(folder.n_qubits),
        "n_states": int(2 ** folder.n_qubits),
        "backend_name": _backend_name(backend),
        "backend": _backend_metadata(backend),
        "job": job_meta,
        "software_versions": _package_versions(),
        "circuit": circuit_meta,
        "sampler_max_mitigation": bool(sampler_max_mitigation and use_sampler_v2),
    }
    return phys, meta


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def _resolve_params_json(path: str, replica_id: Optional[int]) -> str:
    """Resolve a replica JSON from a file, manifest, or fold directory."""
    candidate = os.path.abspath(os.path.expanduser(path))
    if os.path.isdir(candidate):
        direct = os.path.join(candidate, "circuit_parameters.json")
        nested = os.path.join(candidate, "circuit_parameters", "circuit_parameters.json")
        candidate = direct if os.path.isfile(direct) else nested
    if not os.path.isfile(candidate):
        raise FileNotFoundError(f"circuit parameter input not found: {path}")
    if os.path.basename(candidate) != "circuit_parameters.json":
        return candidate

    with open(candidate, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    entries = list(manifest.get("replicas") or [])
    if replica_id is None:
        if len(entries) != 1:
            raise ValueError(
                f"{candidate} contains {len(entries)} replicas; pass --replica-id to select one"
            )
        entry = entries[0]
    else:
        matches = [entry for entry in entries if int(entry.get("replica_id", -1)) == replica_id]
        if not matches:
            raise ValueError(f"replica_id {replica_id} is not present in {candidate}")
        entry = matches[0]
    raw = entry.get("json_path")
    if not raw:
        raise ValueError(f"selected manifest entry has no json_path: {candidate}")
    selected = raw if os.path.isabs(str(raw)) else os.path.join(os.path.dirname(candidate), str(raw))
    if not os.path.isfile(selected):
        raise FileNotFoundError(f"saved replica parameter JSON not found: {selected}")
    return selected


def _load_params_manifest(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or "params" not in payload:
        raise ValueError(
            f"{path} does not look like a qtf.circuit_parameters.v1 replica JSON "
            f"(no 'params' key)."
        )
    fmt = payload.get("format", "")
    if fmt and not fmt.startswith("qtf.circuit_parameters"):
        logger.warning("Unexpected params format: %r; proceeding anyway.", fmt)
    return payload


def _build_folder_from_manifest(payload: dict, args) -> QuantumBiophysicsFolder:
    sequence = payload.get("sequence") or args.sequence
    if not sequence:
        raise ValueError("params JSON has no 'sequence' and --sequence not given")

    from qtf.recipes import resolve_recipe

    recipe_name = payload.get("recipe") or args.recipe
    recipe = resolve_recipe(recipe_name) if recipe_name else {}
    geometry = dict(recipe.get("geometry") or {})
    fold = dict(recipe.get("fold") or {})
    result = dict(recipe.get("result") or {})
    folder_kwargs = dict(
        sequence=sequence,
        force_field=str(fold.get("force_field") or "protein-coarse-charge-v1"),
        stored_angles=geometry.get("stored_angles") or [],
        stored_lengths=geometry.get("stored_lengths") or [],
        max_chi=geometry.get("max_chi"),
        include_terminal_oxt=bool(geometry.get("include_terminal_oxt", False)),
        geometry_mode=geometry.get("geometry_mode"),
        geometry_table=geometry.get("geometry_table"),
        geometry_profile=geometry.get("geometry_profile"),
        rebuild_method=str(payload.get("rebuild_method") or args.rebuild_method or "pheat"),
        score_model=str(result.get("score_model") or "pheat-custom-energy-v1"),
        circuit_template=recipe.get("circuit_template"),
        circuit=recipe.get("circuit"),
        optimizer_angle_mode="sampler",
        optimizer_shots=int(args.shots),
    )
    folder = QuantumBiophysicsFolder(**folder_kwargs)

    n_qubits_saved = payload.get("n_qubits")
    n_params_saved = payload.get("n_params")
    if n_qubits_saved is not None and int(n_qubits_saved) != int(folder.n_qubits):
        raise ValueError(
            f"n_qubits mismatch: saved={n_qubits_saved} vs rebuilt={folder.n_qubits}. "
            f"Check that sequence and recipe match the original run."
        )
    if n_params_saved is not None and int(n_params_saved) != int(folder.n_params):
        raise ValueError(
            f"n_params mismatch: saved={n_params_saved} vs rebuilt={folder.n_params}."
        )
    return folder


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="qtf fold-hardware",
        description="Execute a QTF circuit on IBM hardware and rebuild the measured structure. "
                    "Saved parameters are optional; random parameters are used by default.",
    )
    ap.add_argument("--params-json", "--params_json", dest="params_json", default=None,
                    help="Optional replica JSON, circuit_parameters manifest/directory, or simulation output directory.")
    ap.add_argument("--replica-id", type=int, default=None,
                    help="Zero-based replica ID when selecting from a manifest or directory.")
    ap.add_argument("--outdir", default="run_outputs/hardware_fold",
                    help="Output directory used when explicit output files are not supplied.")
    ap.add_argument("--out-pdb", "--out_pdb", dest="out_pdb", default=None,
                    help="Output PDB path; defaults to OUTDIR/hardware_model.pdb.")
    ap.add_argument("--out-json", "--out_json", dest="out_json", default=None,
                    help="Metadata JSON path; defaults to OUTDIR/hardware_result.json.")
    ap.add_argument("--reference_pdb", default=None,
                    help="Optional local reference PDB path for RMSD.")
    ap.add_argument("--reference_structure", default=None,
                    help="Optional reference PDB ID (fetched via qtf reference utils).")
    ap.add_argument("--rmsd_mode", default="ca", choices=["ca", "heavy"])
    ap.add_argument("--rmsd_residue_scope", default="core", choices=["core", "all"])
    ap.add_argument("--average_reference_backbone", action="store_true")

    ap.add_argument("--backend-name", "--backend_name", dest="backend_name", default=None,
                    help="Specific IBM backend. If omitted, Runtime least_busy() selects an operational device.")
    ap.add_argument("--local-simulator", action="store_true",
                    help=("Testing/development mode: use Aer locally instead of submitting an IBM job. "
                          "Use it for offline validation or continued parameter experimentation."))
    ap.add_argument("--channel", default=None, help="qiskit_ibm_runtime channel.")
    ap.add_argument("--instance", default=None, help="qiskit_ibm_runtime instance.")
    ap.add_argument("--token", default=None, help="qiskit_ibm_runtime API token.")
    ap.add_argument("--use-sampler-v2", "--use_sampler_v2", dest="use_sampler_v2", action="store_true", default=True,
                    help="Use the Runtime SamplerV2 primitive (default for IBM hardware).")
    ap.add_argument("--no-sampler-v2", dest="use_sampler_v2", action="store_false",
                    help="Use backend.run instead; mainly intended for local compatibility tests.")
    ap.add_argument("--shots", type=int, default=8192)
    ap.add_argument("--optimization-level", type=int, default=3,
                    help="Transpiler optimization level for the hardware circuit.")
    ap.add_argument("--seed-transpiler", "--seed_transpiler", dest="seed_transpiler", type=int, default=None,
                    help="Optional Qiskit transpiler seed for reproducible circuit layout/optimization.")
    ap.add_argument("--sampler-max-mitigation", "--sampler_max_mitigation", dest="sampler_max_mitigation", action="store_true", default=True,
                    help="Enable strongest available SamplerV2 controls: DD plus gate/measurement twirling.")
    ap.add_argument("--no-sampler-max-mitigation", "--no_sampler_max_mitigation", dest="sampler_max_mitigation", action="store_false",
                    help="Disable SamplerV2 DD/twirling mitigation options.")
    ap.add_argument("--gromacs", dest="gromacs", action="store_true", default=True,
                    help="GROMACS-refine the rebuilt hardware structure (default).")
    ap.add_argument("--no-gromacs", dest="gromacs", action="store_false",
                    help="Skip post-run GROMACS refinement.")
    ap.add_argument("--gromacs-outdir", default=None,
                    help="GROMACS artifact directory; defaults to OUTDIR/gromacs_minimized.")
    ap.add_argument("--gromacs-forcefield", default="amber99sb-ildn")
    ap.add_argument("--gromacs-water", default="tip3p")
    ap.add_argument("--gromacs-nsteps", type=int, default=5000)
    ap.add_argument("--gromacs-emtol", type=float, default=100.0)
    ap.add_argument("--gromacs-maxwarn", type=int, default=2)

    # Folder-construction knobs that are NOT recorded in the params JSON.
    # Defaults match qtf-fold-simulation defaults so a plain rerun of a fold-produced
    # replica reconstructs the same folder identity.
    ap.add_argument("--sequence", default=None,
                    help="Protein sequence; required when --params-json is omitted.")
    ap.add_argument("--recipe", default="qtf-default-config-snapshots",
                    help="Recipe defining the circuit and geometry when random parameters are used.")
    ap.add_argument("--rebuild-method", choices=["pheat", "nerf"], default=None)
    ap.add_argument("--seed", type=int, default=None,
                    help="Optional seed for random circuit-parameter initialization.")
    ap.add_argument("--chi_mode", default="all",
                    choices=["beam", "selective", "all"],
                    help="Only used if the params JSON does not record chi_mode.")
    ap.add_argument("--omega_mode", default="window",
                    choices=["free", "fixed", "window"])
    ap.add_argument("--use_e2e_constraint", type=int, default=1)
    ap.add_argument("--e2e_scale", type=float, default=1.0)

    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    output_dir = os.path.abspath(os.path.expanduser(args.outdir))
    args.out_pdb = args.out_pdb or os.path.join(output_dir, "hardware_model.pdb")
    args.out_json = args.out_json or os.path.join(output_dir, "hardware_result.json")
    args.gromacs_outdir = args.gromacs_outdir or os.path.join(output_dir, "gromacs_minimized")

    # ---- Construct the circuit and initialize its parameters ----
    params_json = None
    if args.params_json:
        params_json = _resolve_params_json(args.params_json, args.replica_id)
        payload = _load_params_manifest(params_json)
    else:
        if not args.sequence:
            ap.error("--sequence is required when --params-json is not supplied")
        payload = {
            "format": "qtf.hardware_parameters.random.v1",
            "sequence": args.sequence,
            "recipe": args.recipe,
            "replica_id": None,
            "parameter_source": "random",
        }
    folder = _build_folder_from_manifest(payload, args)
    if params_json is not None:
        params = np.asarray(payload["params"], dtype=float).reshape(-1)
        logger.info("Loaded %d circuit parameters from %s", params.size, params_json)
        parameter_source = "saved"
    else:
        params = np.random.default_rng(args.seed).uniform(-np.pi, np.pi, size=folder.n_params)
        payload["params"] = params.tolist()
        payload["n_qubits"] = int(folder.n_qubits)
        payload["n_params"] = int(folder.n_params)
        payload["random_seed"] = args.seed
        parameter_source = "random"
        logger.info("Initialized %d random circuit parameters (seed=%s)", params.size, args.seed)
    logger.info("Rebuilt folder: n_qubits=%d n_params=%d total_angles=%d",
                folder.n_qubits, folder.n_params, folder.total_angles)

    # ---- Pick backend & wire it into the folder ----
    backend, backend_kind = get_hardware_backend(args, min_num_qubits=folder.n_qubits)
    folder.backend = backend
    folder.shots = int(args.shots)
    logger.info("Using backend: %s (%s), shots=%d, sampler_v2=%s",
                getattr(backend, "name", str(backend)), backend_kind,
                args.shots, bool(args.use_sampler_v2 and backend_kind == "ibm_runtime"))

    # ---- Single hardware folding execution → angles ----
    angles, run_meta = hardware_angles(
        folder,
        params,
        backend,
        shots=int(args.shots),
        use_sampler_v2=bool(args.use_sampler_v2 and backend_kind == "ibm_runtime"),
        optimization_level=int(args.optimization_level),
        seed_transpiler=args.seed_transpiler,
        sampler_max_mitigation=bool(args.sampler_max_mitigation),
    )
    logger.info("Hardware sampling complete: %d unique bitstrings out of %d shots.",
                run_meta["unique_bitstrings"], run_meta["shots_total"])

    # ---- NERF rebuild ----
    coords, labels, _bonds = folder.build_full_structure(angles)
    coords = np.asarray(coords, dtype=float)

    # ---- Optional RMSD to reference ----
    rmsd_value = float("nan")
    rmsd_meta = None
    if args.reference_pdb or args.reference_structure:
        reference_source = args.reference_pdb or args.reference_structure
        true_rmsd_coords, true_rmsd_labels, _ref_meta = utils.load_reference_rmsd_coords(
            reference_source,
            args.rmsd_mode,
            average_backbone=bool(args.average_reference_backbone),
        )
        aligned_coords, rmsd_value, rmsd_meta, _alignment = utils.align_structure_to_reference(
            coords,
            labels,
            true_rmsd_coords,
            true_rmsd_labels,
            args.rmsd_mode,
            args.rmsd_residue_scope,
        )
        coords = aligned_coords
        logger.info("RMSD to reference (%s, %s): %.4f Å",
                    args.rmsd_mode, args.rmsd_residue_scope, rmsd_value)

    # ---- Save PDB ----
    out_pdb = os.path.abspath(args.out_pdb)
    os.makedirs(os.path.dirname(out_pdb) or ".", exist_ok=True)
    remarks = [
        "QTF SINGLE-SHOT HARDWARE FOLD REBUILD",
        f"BACKEND {run_meta['backend_name']}  KIND {backend_kind}",
        f"SHOTS {run_meta['shots_total']}  UNIQUE_BITSTRINGS {run_meta['unique_bitstrings']}",
        f"SEQUENCE {folder.sequence}",
        f"REPLICA_ID {payload.get('replica_id', 'NA')}  ENSEMBLE_ID {payload.get('ensemble_id', 'NA')}",
        f"TIMESTAMP {datetime.now().isoformat(timespec='seconds')}",
    ]
    if rmsd_meta is not None:
        remarks.append(f"RMSD_TO_REFERENCE_A {rmsd_value:.4f} MODE {args.rmsd_mode} SCOPE {args.rmsd_residue_scope}")
    save_pdb(
        coords,
        labels,
        filename=out_pdb,
        energy=float(payload.get("energy", 0.0)),
        remarks=remarks,
        include_hydrogens=False,
        sequence=folder.sequence,
    )
    logger.info("Wrote structure PDB: %s", out_pdb)

    # ---- Optional metadata dump ----
    if args.out_json:
        out_meta = {
            "params_json": os.path.abspath(params_json) if params_json is not None else None,
            "parameter_source": parameter_source,
            "random_seed": args.seed if parameter_source == "random" else None,
            "circuit_parameters": params.tolist(),
            "sequence": folder.sequence,
            "rebuild_method": folder.rebuild_method,
            "replica_id": payload.get("replica_id"),
            "replica_number": payload.get("replica_number", payload.get("replica")),
            "ensemble_id": payload.get("ensemble_id"),
            "backend_name": run_meta["backend_name"],
            "backend_kind": backend_kind,
            "backend": run_meta.get("backend"),
            "job": run_meta.get("job"),
            "software_versions": run_meta.get("software_versions"),
            "shots": run_meta["shots_total"],
            "unique_bitstrings": run_meta["unique_bitstrings"],
            "bitstring_counts": run_meta.get("bitstring_counts"),
            "n_qubits": folder.n_qubits,
            "n_params": folder.n_params,
            "total_angles": folder.total_angles,
            "circuit": run_meta.get("circuit"),
            "sampler_max_mitigation": run_meta.get("sampler_max_mitigation"),
            "rmsd_to_reference_A": rmsd_value,
            "rmsd_mode": args.rmsd_mode,
            "rmsd_residue_scope": args.rmsd_residue_scope,
            "rmsd_meta": rmsd_meta,
            "reference_source": args.reference_pdb or args.reference_structure,
            "out_pdb": out_pdb,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        if args.gromacs:
            from qtf.core.hardware_gromacs import refine_hardware_structure

            logger.info("Running GROMACS refinement: %s", args.gromacs_outdir)
            try:
                gromacs_meta = refine_hardware_structure(
                    out_pdb,
                    args.gromacs_outdir,
                    reference_source=args.reference_pdb or args.reference_structure,
                    rmsd_mode=args.rmsd_mode,
                    rmsd_residue_scope=args.rmsd_residue_scope,
                    forcefield=args.gromacs_forcefield,
                    water=args.gromacs_water,
                    nsteps=args.gromacs_nsteps,
                    emtol=args.gromacs_emtol,
                    maxwarn=args.gromacs_maxwarn,
                )
                out_meta.update(gromacs_meta)
                effective_rmsd = gromacs_meta.get("hardware_effective_rmsd_to_reference_A")
                if effective_rmsd is not None:
                    out_meta["rmsd_to_reference_A"] = effective_rmsd
                logger.info(
                    "GROMACS refinement status: %s",
                    gromacs_meta.get("gromacs_status", "unknown"),
                )
            except Exception as exc:
                logger.warning("GROMACS refinement failed: %s", exc)
                out_meta.update(
                    {
                        "hardware_gromacs_enabled": True,
                        "gromacs_status": "error",
                        "gromacs_message": str(exc),
                    }
                )
        else:
            out_meta["hardware_gromacs_enabled"] = False
        out_json = os.path.abspath(args.out_json)
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as handle:
            json.dump(relativize_absolute_paths(_jsonify(out_meta)), handle, indent=2)
        logger.info("Wrote run metadata: %s", out_json)

    return 0


def _jsonify(x):
    if isinstance(x, dict):
        return {str(k): _jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonify(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


if __name__ == "__main__":
    sys.exit(main())
