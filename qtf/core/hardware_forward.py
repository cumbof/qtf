#!/usr/bin/env python3
"""
Single-shot hardware forward pass for a saved QTF circuit-parameter vector.

Given a ``replica_<k>_params.json`` produced by ``qtf-fold`` (format
``qtf.circuit_parameters.v1``), this script:

  1. Reconstructs a ``QuantumBiophysicsFolder`` with matching configuration
     (sequence, chi_mode, omega_mode, energy_backend, ansatz layout).
  2. Binds the saved parameters onto the ansatz and executes **one forward
     pass on real quantum hardware** (sampler mode: measure_all + shots).
  3. Converts the empirical shot distribution to torsion angles via the
     same CDF mapping used by ``QuantumBiophysicsFolder._get_angles_sampler``.
  4. Rebuilds the full heavy-atom structure via NERF
     (``folder.build_full_structure``) and applies the same physical-range
     mapping used in statevector runs.
  5. Writes a PDB file, and — when a reference structure is provided —
     rigid-aligns to it and reports RMSD.

The hardware backend is selected in ``get_hardware_backend()``. Edit that
function (or use the CLI flags) to plug in your IBM/other credentials.

Usage
-----
    python -m qtf.core.hardware_forward \\
        --params_json /path/to/circuit_parameters/replica_1_params.json \\
        --backend_name ibm_torino \\
        --channel ibm_quantum \\
        --instance ibm-q/open/main \\
        --shots 4096 \\
        --out_pdb  ./hw_replica_1.pdb \\
        --reference_pdb /path/to/native.pdb   # optional

If ``--backend_name`` is omitted, an ``AerSimulator`` is used so the same
script can be dry-run on a laptop before touching real hardware.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Optional, Tuple

import numpy as np

from qtf.core.folder import QuantumBiophysicsFolder
from qtf.utils import workflow as utils

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hardware backend selection
# ---------------------------------------------------------------------------
def get_hardware_backend(args) -> Tuple[object, str]:
    """Return ``(backend, kind)`` where ``kind`` is 'ibm_runtime' or 'aer'.

    EDIT THIS FUNCTION to hard-code your hardware credentials / instance if
    you do not want to pass them on the command line. The default behaviour
    is:

      * If ``--backend_name`` is given → use ``qiskit_ibm_runtime``
        (``QiskitRuntimeService``) with the given channel/instance/token
        (falling back to a saved account if these are omitted).
      * Otherwise → return a local ``AerSimulator`` so the script is
        directly runnable without a hardware connection (useful for
        dry-runs / debugging).
    """
    if args.backend_name:
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
        backend = service.backend(args.backend_name)
        return backend, "ibm_runtime"

    try:
        from qiskit_aer import AerSimulator
    except ImportError as exc:
        raise SystemExit(
            "No --backend_name given and qiskit-aer is not installed; "
            "cannot fall back to local simulation."
        ) from exc
    return AerSimulator(), "aer"


# ---------------------------------------------------------------------------
# One forward pass: bind → transpile → sample → CDF → angles
# ---------------------------------------------------------------------------
def _counts_from_backend_run(backend, tqc, shots):
    """Legacy ``backend.run(...)`` path (Aer, older IBM backends)."""
    result = backend.run(tqc, shots=shots).result()
    return result.get_counts()


def _counts_from_sampler_v2(backend, tqc, shots):
    """Modern ``SamplerV2`` path (qiskit-ibm-runtime ≥ 0.23)."""
    from qiskit_ibm_runtime import SamplerV2  # type: ignore

    sampler = SamplerV2(mode=backend)
    job = sampler.run([tqc], shots=shots)
    result = job.result()
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
    return bit_array.get_counts()


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


def hardware_forward_angles(
    folder: QuantumBiophysicsFolder,
    params: np.ndarray,
    backend,
    shots: int,
    *,
    use_sampler_v2: bool,
    optimization_level: int,
) -> Tuple[np.ndarray, dict]:
    """Run one hardware forward pass and return physical torsion angles.

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
    tqc = transpile(qc, backend=backend, optimization_level=optimization_level)

    if use_sampler_v2:
        counts = _counts_from_sampler_v2(backend, tqc, shots)
    else:
        counts = _counts_from_backend_run(backend, tqc, shots)

    pvec = _bitstring_counts_to_pvec(counts, folder.n_qubits)

    # CDF-to-angle mapping — identical to folder._get_angles_sampler.
    cdf = np.cumsum(pvec)
    angles = 2.0 * np.pi * cdf - np.pi
    angles = angles[: folder.total_angles]
    phys = folder._map_angle_vector_to_physical_ranges(angles)

    meta = {
        "shots_total": int(sum(counts.values())),
        "unique_bitstrings": int(len(counts)),
        "n_qubits": int(folder.n_qubits),
        "n_states": int(2 ** folder.n_qubits),
        "backend_name": getattr(backend, "name", None) or getattr(backend, "name_str", None) or str(backend),
    }
    return phys, meta


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
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

    folder_kwargs = dict(
        sequence=sequence,
        chi_mode=payload.get("chi_mode", args.chi_mode),
        omega_mode=payload.get("omega_mode", args.omega_mode),
        energy_backend=payload.get("energy_backend", args.energy_backend),
        use_e2e_constraint=bool(args.use_e2e_constraint),
        e2e_scale=float(args.e2e_scale),
        rosetta_repack=bool(args.rosetta_repack),
        rosetta_fa_min=bool(args.rosetta_fa_min),
        rosetta_cen_min=bool(args.rosetta_cen_min),
        # sampler-mode wiring (backend/shots are attached after construction
        # so callers can freely swap them later).
        mode="sampler",
        shots=int(args.shots),
    )
    folder = QuantumBiophysicsFolder(**folder_kwargs)

    n_qubits_saved = payload.get("n_qubits")
    n_params_saved = payload.get("n_params")
    if n_qubits_saved is not None and int(n_qubits_saved) != int(folder.n_qubits):
        raise ValueError(
            f"n_qubits mismatch: saved={n_qubits_saved} vs rebuilt={folder.n_qubits}. "
            f"Check that sequence/omega_mode/chi_mode/energy_backend match the original run."
        )
    if n_params_saved is not None and int(n_params_saved) != int(folder.n_params):
        raise ValueError(
            f"n_params mismatch: saved={n_params_saved} vs rebuilt={folder.n_params}."
        )
    return folder


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="qtf-hw-forward",
        description="Run one hardware forward pass from a saved circuit-parameter JSON, "
                    "NERF-rebuild the structure, and (optionally) compute RMSD.",
    )
    ap.add_argument("--params_json", required=True,
                    help="Path to replica_<k>_params.json (qtf.circuit_parameters.v1).")
    ap.add_argument("--out_pdb", required=True,
                    help="Output PDB path for the rebuilt heavy-atom structure.")
    ap.add_argument("--out_json", default=None,
                    help="Optional JSON path for run metadata (backend, shots, RMSD, ...).")
    ap.add_argument("--reference_pdb", default=None,
                    help="Optional local reference PDB path for RMSD.")
    ap.add_argument("--reference_structure", default=None,
                    help="Optional reference PDB ID (fetched via qtf reference utils).")
    ap.add_argument("--rmsd_mode", default="ca", choices=["ca", "heavy"])
    ap.add_argument("--rmsd_residue_scope", default="core", choices=["core", "all"])
    ap.add_argument("--average_reference_backbone", action="store_true")

    # Hardware backend flags — leave empty to fall back to AerSimulator.
    ap.add_argument("--backend_name", default=None,
                    help="Hardware backend name (e.g. 'ibm_torino'). "
                         "If omitted, AerSimulator is used.")
    ap.add_argument("--channel", default=None, help="qiskit_ibm_runtime channel.")
    ap.add_argument("--instance", default=None, help="qiskit_ibm_runtime instance.")
    ap.add_argument("--token", default=None, help="qiskit_ibm_runtime API token.")
    ap.add_argument("--use_sampler_v2", action="store_true",
                    help="Use SamplerV2 primitive instead of backend.run(). "
                         "Required for qiskit-ibm-runtime ≥ 0.23.")
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--optimization_level", type=int, default=1,
                    help="Transpiler optimization level for the hardware circuit.")

    # Folder-construction knobs that are NOT recorded in the params JSON.
    # Defaults match qtf-fold defaults so a plain rerun of a fold-produced
    # replica reconstructs the same folder identity.
    ap.add_argument("--sequence", default=None,
                    help="Override sequence (usually taken from the params JSON).")
    ap.add_argument("--chi_mode", default="all",
                    choices=["beam", "selective", "all"],
                    help="Only used if the params JSON does not record chi_mode.")
    ap.add_argument("--omega_mode", default="window",
                    choices=["free", "fixed", "window"])
    ap.add_argument("--energy_backend", default="custom",
                    choices=["custom", "rosetta", "openmm"])
    ap.add_argument("--use_e2e_constraint", type=int, default=1)
    ap.add_argument("--e2e_scale", type=float, default=1.0)
    ap.add_argument("--rosetta_repack", type=int, default=0)
    ap.add_argument("--rosetta_fa_min", type=int, default=0)
    ap.add_argument("--rosetta_cen_min", type=int, default=0)

    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # ---- Load parameter manifest & construct folder ----
    payload = _load_params_manifest(args.params_json)
    params = np.asarray(payload["params"], dtype=float).reshape(-1)
    logger.info("Loaded %d circuit parameters from %s (sequence=%s)",
                params.size, args.params_json, payload.get("sequence"))

    folder = _build_folder_from_manifest(payload, args)
    logger.info("Rebuilt folder: n_qubits=%d n_params=%d total_angles=%d",
                folder.n_qubits, folder.n_params, folder.total_angles)

    # ---- Pick backend & wire it into the folder ----
    backend, backend_kind = get_hardware_backend(args)
    folder.backend = backend
    folder.shots = int(args.shots)
    logger.info("Using backend: %s (%s), shots=%d, sampler_v2=%s",
                getattr(backend, "name", str(backend)), backend_kind,
                args.shots, bool(args.use_sampler_v2))

    # ---- Single hardware forward pass → angles ----
    angles, run_meta = hardware_forward_angles(
        folder,
        params,
        backend,
        shots=int(args.shots),
        use_sampler_v2=bool(args.use_sampler_v2),
        optimization_level=int(args.optimization_level),
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
        "QTF SINGLE-SHOT HARDWARE FORWARD REBUILD",
        f"BACKEND {run_meta['backend_name']}  KIND {backend_kind}",
        f"SHOTS {run_meta['shots_total']}  UNIQUE_BITSTRINGS {run_meta['unique_bitstrings']}",
        f"SEQUENCE {folder.sequence}",
        f"REPLICA {payload.get('replica', 'NA')}  ENSEMBLE_ID {payload.get('ensemble_id', 'NA')}",
        f"TIMESTAMP {datetime.now().isoformat(timespec='seconds')}",
    ]
    if rmsd_meta is not None:
        remarks.append(f"RMSD_TO_REFERENCE_A {rmsd_value:.4f} MODE {args.rmsd_mode} SCOPE {args.rmsd_residue_scope}")
    folder.save_pdb(
        coords,
        labels,
        filename=out_pdb,
        energy=float(payload.get("energy", 0.0)),
        remarks=remarks,
        include_hydrogens=False,
    )
    logger.info("Wrote structure PDB: %s", out_pdb)

    # ---- Optional metadata dump ----
    if args.out_json:
        out_meta = {
            "params_json": os.path.abspath(args.params_json),
            "sequence": folder.sequence,
            "replica": payload.get("replica"),
            "ensemble_id": payload.get("ensemble_id"),
            "backend_name": run_meta["backend_name"],
            "backend_kind": backend_kind,
            "shots": run_meta["shots_total"],
            "unique_bitstrings": run_meta["unique_bitstrings"],
            "n_qubits": folder.n_qubits,
            "n_params": folder.n_params,
            "total_angles": folder.total_angles,
            "rmsd_to_reference_A": rmsd_value,
            "rmsd_mode": args.rmsd_mode,
            "rmsd_residue_scope": args.rmsd_residue_scope,
            "rmsd_meta": rmsd_meta,
            "reference_source": args.reference_pdb or args.reference_structure,
            "out_pdb": out_pdb,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        out_json = os.path.abspath(args.out_json)
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as handle:
            json.dump(_jsonify(out_meta), handle, indent=2)
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
