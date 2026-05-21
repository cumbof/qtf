"""Tests for QTF circuit construction."""

import pytest

from qtf.core.circuits import QTF_BRICKWORK_TEMPLATE, build_circuit


def test_default_circuit_uses_efficient_su2_auto_reps():
    result = build_circuit(total_angles=10)
    assert result.source == "qiskit-library"
    assert result.name == "EfficientSU2"
    assert result.n_qubits == 4
    assert result.reps == 5
    assert result.n_params > 0


def test_qtf_brickwork_template_builds_parameterized_circuit():
    result = build_circuit(
        total_angles=20,
        circuit_template={
            "source": "qtf",
            "name": QTF_BRICKWORK_TEMPLATE,
            "options": {"reps": "auto"},
        },
    )
    assert result.source == "qtf"
    assert result.name == QTF_BRICKWORK_TEMPLATE
    assert result.reps == 3
    assert result.n_qubits == 5
    assert result.n_params == 2 * result.n_qubits * (result.reps + 1)


def test_loaded_qpy_circuit_is_validated(tmp_path):
    qiskit = pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit, qpy
    from qiskit.circuit import Parameter

    theta = Parameter("theta")
    circuit = QuantumCircuit(2)
    circuit.ry(theta, 0)
    circuit.cx(0, 1)
    qpy_path = tmp_path / "circuit.qpy"
    with qpy_path.open("wb") as handle:
        qpy.dump(circuit, handle)

    result = build_circuit(
        total_angles=4,
        circuit={"source": "qpy", "path": str(qpy_path), "index": 0},
    )
    assert result.source == "qpy"
    assert result.n_qubits == 2
    assert result.n_params == 1


def test_loaded_circuit_qubit_count_must_match(tmp_path):
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit, qpy
    from qiskit.circuit import Parameter

    theta = Parameter("theta")
    circuit = QuantumCircuit(3)
    circuit.ry(theta, 0)
    qpy_path = tmp_path / "wrong.qpy"
    with qpy_path.open("wb") as handle:
        qpy.dump(circuit, handle)

    with pytest.raises(ValueError, match="requires 2"):
        build_circuit(total_angles=4, circuit={"source": "qpy", "path": str(qpy_path)})
