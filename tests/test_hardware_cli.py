from argparse import Namespace

import numpy as np

import qtf.core.hardware as hardware
from qtf.utils import workflow


class _FakeService:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.backend_calls = []
        self.least_busy_calls = []
        self.__class__.instances.append(self)

    def backend(self, name):
        self.backend_calls.append(name)
        return f"named:{name}"

    def least_busy(self, **kwargs):
        self.least_busy_calls.append(kwargs)
        return "least-busy-backend"


def _backend_args(**overrides):
    values = {
        "local_simulator": False,
        "channel": None,
        "instance": None,
        "token": None,
        "backend_name": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_hardware_backend_defaults_to_runtime_least_busy(monkeypatch):
    import qiskit_ibm_runtime

    _FakeService.instances.clear()
    monkeypatch.setattr(qiskit_ibm_runtime, "QiskitRuntimeService", _FakeService)
    backend, kind = hardware.get_hardware_backend(_backend_args(), min_num_qubits=6)

    assert backend == "least-busy-backend"
    assert kind == "ibm_runtime"
    service = _FakeService.instances[-1]
    assert service.backend_calls == []
    assert service.least_busy_calls == [
        {"min_num_qubits": 6, "operational": True, "simulator": False}
    ]


def test_hardware_backend_honors_explicit_name(monkeypatch):
    import qiskit_ibm_runtime

    _FakeService.instances.clear()
    monkeypatch.setattr(qiskit_ibm_runtime, "QiskitRuntimeService", _FakeService)
    backend, kind = hardware.get_hardware_backend(
        _backend_args(backend_name="ibm_example"), min_num_qubits=6
    )

    assert backend == "named:ibm_example"
    assert kind == "ibm_runtime"
    service = _FakeService.instances[-1]
    assert service.backend_calls == ["ibm_example"]
    assert service.least_busy_calls == []


def test_hardware_alignment_restores_complete_structure_coordinates():
    labels = [(0, "CA", "C"), (1, "CA", "C"), (2, "CA", "C"), (1, "CB", "C")]
    model = np.asarray([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 0, 1]], dtype=float)
    rotation = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    translation = np.asarray([4, 3, 2], dtype=float)
    reference = model @ rotation + translation

    aligned, rmsd, metadata, transform = workflow.align_structure_to_reference(
        model, labels, reference, labels, "ca", "all"
    )

    np.testing.assert_allclose(aligned, reference, atol=1e-12)
    assert rmsd < 1e-12
    assert metadata["rmsd_n_matched"] == 3
    assert transform["rotation"].shape == (3, 3)
