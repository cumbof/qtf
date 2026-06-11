"""Circuit construction and loading for QTF folding."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


QTF_BRICKWORK_TEMPLATE = "brickwork-ryrz-nearest-neighbor"
DEFAULT_CIRCUIT_TEMPLATE = {
    "source": "qiskit-library",
    "name": "EfficientSU2",
    "options": {
        "reps": "auto",
        "entanglement": "circular",
    },
}


@dataclass
class CircuitBuildResult:
    circuit: QuantumCircuit
    n_qubits: int
    n_params: int
    reps: Optional[int]
    source: str
    name: str
    options: dict[str, Any] = field(default_factory=dict)
    path: Optional[str] = None
    index: Optional[int] = None

    def metadata(self) -> dict[str, Any]:
        payload = {
            "source": self.source,
            "name": self.name,
            "options": dict(self.options),
            "n_qubits": self.n_qubits,
            "n_params": self.n_params,
            "reps": self.reps,
        }
        if self.path is not None:
            payload["path"] = self.path
        if self.index is not None:
            payload["index"] = self.index
        return payload


def default_n_qubits(total_angles: int) -> int:
    return max(2, int(math.ceil(math.log2(max(int(total_angles), 1)))))


def efficient_su2_auto_reps(total_angles: int, n_qubits: int) -> int:
    return int(math.ceil(max(int(total_angles), 1) / max(int(n_qubits), 1))) + 2


def brickwork_auto_reps(total_angles: int, n_qubits: int) -> int:
    min_reps = int(math.ceil(max(int(total_angles), 1) / (2 * max(int(n_qubits), 1))))
    return max(3, min(min_reps, 6))


def build_circuit(
    *,
    total_angles: int,
    circuit_template: Optional[dict[str, Any]] = None,
    circuit: Optional[dict[str, Any]] = None,
) -> CircuitBuildResult:
    n_qubits = default_n_qubits(total_angles)
    if circuit_template and circuit:
        raise ValueError("Specify either circuit_template or circuit, not both.")
    if circuit:
        return load_circuit(circuit, n_qubits=n_qubits)
    return build_circuit_template(circuit_template or DEFAULT_CIRCUIT_TEMPLATE, total_angles=total_angles, n_qubits=n_qubits)


def build_circuit_template(
    config: dict[str, Any],
    *,
    total_angles: int,
    n_qubits: int,
) -> CircuitBuildResult:
    source = str(config.get("source") or "qiskit-library").strip().lower()
    name = str(config.get("name") or "EfficientSU2").strip()
    options = dict(config.get("options") or {})
    if source == "qiskit-library":
        return _build_qiskit_library_template(name, options, total_angles=total_angles, n_qubits=n_qubits)
    if source == "qtf":
        return _build_qtf_template(name, options, total_angles=total_angles, n_qubits=n_qubits)
    raise ValueError("circuit_template source must be one of qiskit-library or qtf.")


def load_circuit(config: dict[str, Any], *, n_qubits: int) -> CircuitBuildResult:
    source = str(config.get("source") or "").strip().lower()
    path = config.get("path")
    if not path:
        raise ValueError("circuit path is required.")
    if source == "qpy":
        index = int(config.get("index", config.get("circuit_index", 0)) or 0)
        circuit = _load_qpy(Path(path), index)
        name = str(config.get("name") or f"qpy[{index}]")
        return _validated_loaded_circuit(circuit, source=source, name=name, path=str(path), index=index, n_qubits=n_qubits)
    if source == "qasm2":
        from qiskit import qasm2

        circuit = qasm2.load(path)
        name = str(config.get("name") or Path(path).name)
        return _validated_loaded_circuit(circuit, source=source, name=name, path=str(path), index=None, n_qubits=n_qubits)
    if source == "qasm3":
        from qiskit import qasm3

        circuit = qasm3.load(str(path), num_qubits=n_qubits)
        name = str(config.get("name") or Path(path).name)
        return _validated_loaded_circuit(circuit, source=source, name=name, path=str(path), index=None, n_qubits=n_qubits)
    raise ValueError("circuit source must be one of qpy, qasm2, or qasm3.")


def _build_qiskit_library_template(
    name: str,
    options: dict[str, Any],
    *,
    total_angles: int,
    n_qubits: int,
) -> CircuitBuildResult:
    library = importlib.import_module("qiskit.circuit.library")
    factory_name = "efficient_su2" if name == "EfficientSU2" and hasattr(library, "efficient_su2") else name
    factory = getattr(library, factory_name, None)
    if factory is None or not callable(factory):
        raise ValueError(f"Unknown Qiskit circuit template: {name}")
    resolved = _resolve_template_options(options, total_angles=total_angles, n_qubits=n_qubits, auto_reps=efficient_su2_auto_reps)
    if factory_name == "efficient_su2":
        resolved.setdefault("parameter_prefix", "theta")
    circuit = factory(num_qubits=n_qubits, **resolved)
    return _validated_template_circuit(
        circuit,
        source="qiskit-library",
        name=name,
        options=resolved,
        reps=resolved.get("reps"),
        n_qubits=n_qubits,
    )


def _build_qtf_template(
    name: str,
    options: dict[str, Any],
    *,
    total_angles: int,
    n_qubits: int,
) -> CircuitBuildResult:
    normalized = name.strip().lower()
    if normalized != QTF_BRICKWORK_TEMPLATE:
        raise ValueError(f"Unknown QTF circuit template: {name}")
    resolved = _resolve_template_options(options, total_angles=total_angles, n_qubits=n_qubits, auto_reps=brickwork_auto_reps)
    reps = int(resolved.get("reps"))
    rotation_gates = [str(item).lower() for item in resolved.get("rotation_gates", ["ry", "rz"])]
    entangler = str(resolved.get("entangler", "cx")).lower()
    pattern = str(resolved.get("entanglement_pattern", "even_odd_linear")).lower()
    final_rotation_layer = bool(resolved.get("final_rotation_layer", True))
    circuit = _brickwork_ryrz_nearest_neighbor(
        n_qubits,
        reps=reps,
        rotation_gates=rotation_gates,
        entangler=entangler,
        entanglement_pattern=pattern,
        final_rotation_layer=final_rotation_layer,
    )
    return _validated_template_circuit(
        circuit,
        source="qtf",
        name=QTF_BRICKWORK_TEMPLATE,
        options=resolved,
        reps=reps,
        n_qubits=n_qubits,
    )


def _resolve_template_options(
    options: dict[str, Any],
    *,
    total_angles: int,
    n_qubits: int,
    auto_reps,
) -> dict[str, Any]:
    resolved = dict(options)
    if str(resolved.get("reps", "auto")).strip().lower() == "auto":
        resolved["reps"] = auto_reps(total_angles, n_qubits)
    elif "reps" in resolved:
        resolved["reps"] = int(resolved["reps"])
    return resolved


def _brickwork_ryrz_nearest_neighbor(
    n_qubits: int,
    *,
    reps: int,
    rotation_gates: list[str],
    entangler: str,
    entanglement_pattern: str,
    final_rotation_layer: bool,
) -> QuantumCircuit:
    if rotation_gates != ["ry", "rz"]:
        raise ValueError("QTF brickwork currently supports rotation_gates=[ry,rz].")
    if entangler != "cx":
        raise ValueError("QTF brickwork currently supports entangler=cx.")
    if entanglement_pattern != "even_odd_linear":
        raise ValueError("QTF brickwork currently supports entanglement_pattern=even_odd_linear.")

    n_params_total = 2 * n_qubits * (reps + (1 if final_rotation_layer else 0))
    params = ParameterVector("theta", n_params_total)
    qc = QuantumCircuit(n_qubits, name=QTF_BRICKWORK_TEMPLATE)
    p_idx = 0
    for _rep in range(reps):
        p_idx = _append_ryrz_layer(qc, params, p_idx)
        _append_even_odd_cx(qc, n_qubits)
    if final_rotation_layer:
        _append_ryrz_layer(qc, params, p_idx)
    return qc


def _append_ryrz_layer(qc: QuantumCircuit, params: ParameterVector, p_idx: int) -> int:
    for qubit in range(qc.num_qubits):
        qc.ry(params[p_idx], qubit)
        p_idx += 1
        qc.rz(params[p_idx], qubit)
        p_idx += 1
    return p_idx


def _append_even_odd_cx(qc: QuantumCircuit, n_qubits: int) -> None:
    for qubit in range(0, n_qubits - 1, 2):
        qc.cx(qubit, qubit + 1)
    for qubit in range(1, n_qubits - 1, 2):
        qc.cx(qubit, qubit + 1)


def _load_qpy(path: Path, index: int) -> QuantumCircuit:
    from qiskit import qpy

    if index < 0:
        raise ValueError(f"circuit_index must be non-negative, got {index}.")
    with path.open("rb") as handle:
        payload = qpy.load(handle)
    try:
        circuit = payload[index]
    except IndexError as exc:
        raise ValueError(f"QPY circuit_index {index} is out of range for {path}.") from exc
    if not isinstance(circuit, QuantumCircuit):
        raise ValueError(f"QPY entry {index} in {path} is not a QuantumCircuit.")
    return circuit


def _validated_template_circuit(
    circuit,
    *,
    source: str,
    name: str,
    options: dict[str, Any],
    reps,
    n_qubits: int,
) -> CircuitBuildResult:
    if not isinstance(circuit, QuantumCircuit):
        if hasattr(circuit, "decompose"):
            circuit = circuit.decompose()
        if not isinstance(circuit, QuantumCircuit):
            raise ValueError(f"Circuit template {source}:{name} did not return a QuantumCircuit.")
    _validate_circuit(circuit, n_qubits=n_qubits)
    return CircuitBuildResult(
        circuit=circuit,
        n_qubits=n_qubits,
        n_params=int(circuit.num_parameters),
        reps=None if reps is None else int(reps),
        source=source,
        name=name,
        options=options,
    )


def _validated_loaded_circuit(
    circuit: QuantumCircuit,
    *,
    source: str,
    name: str,
    path: str,
    index: Optional[int],
    n_qubits: int,
) -> CircuitBuildResult:
    _validate_circuit(circuit, n_qubits=n_qubits)
    return CircuitBuildResult(
        circuit=circuit,
        n_qubits=n_qubits,
        n_params=int(circuit.num_parameters),
        reps=None,
        source=source,
        name=name,
        options={},
        path=path,
        index=index,
    )


def _validate_circuit(circuit: QuantumCircuit, *, n_qubits: int) -> None:
    if int(circuit.num_qubits) != int(n_qubits):
        raise ValueError(f"Circuit has {circuit.num_qubits} qubits, but QTF requires {n_qubits}.")
    if int(circuit.num_parameters) <= 0:
        raise ValueError("Circuit must contain at least one trainable parameter.")
