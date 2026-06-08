"""Optional GPU-resident statevector simulation via qiskit-aer.

Provides a drop-in replacement for ``qiskit.quantum_info.Statevector``
that transparently delegates to an ``AerSimulator`` backend (GPU when
available, CPU otherwise) and falls back to the in-process
``Statevector`` when Aer is not installed.

Environment variables
---------------------
``QTF_AER_DEVICE``
    One of ``"auto"`` (default), ``"gpu"``, or ``"cpu"``.  Forces the
    target device when Aer *is* present.  ``"auto"`` tries GPU first,
    then CPU.
"""

from __future__ import annotations

import os
import logging

import numpy as np

logger = logging.getLogger(__name__)

_HAS_AER: bool = False
_GPU_AVAILABLE: bool = False
_backend = None


def _init_backend() -> None:
    """Lazily initialise the Aer simulator (called at most once)."""
    global _HAS_AER, _GPU_AVAILABLE, _backend

    if _backend is not None:
        return

    try:
        from qiskit_aer import AerSimulator
        _HAS_AER = True
    except ImportError:
        logger.debug("qiskit-aer not installed; using in-process Statevector")
        return

    device = os.getenv("QTF_AER_DEVICE", "auto").strip().lower()

    if device == "cpu":
        try:
            _backend = AerSimulator(method="statevector", device="CPU")
            logger.info("Aer CPU simulator ready")
        except Exception as exc:
            logger.warning("Failed to create Aer CPU simulator: %s", exc)
        return

    candidates = ["GPU", "CPU"] if device in ("auto", "gpu") else [device]
    for dev in candidates:
        try:
            _backend = AerSimulator(method="statevector", device=dev)
            if dev == "GPU":
                _GPU_AVAILABLE = True
                logger.info("Aer GPU simulator ready (CUDA)")
            else:
                logger.info("Aer CPU simulator ready (fallback)")
            return
        except Exception as exc:
            logger.debug("Aer %s simulator failed: %s", dev, exc)

    logger.warning("Aer available but no viable device; falling back to Statevector")


def statevector_data(circuit) -> np.ndarray:
    """Return the complex statevector amplitudes of *circuit*.

    Uses, in order of preference:
    1. Aer GPU simulator
    2. Aer CPU simulator
    3. In-process ``qiskit.quantum_info.Statevector``
    """
    _init_backend()

    if _backend is not None:
        try:
            qc = circuit.copy()
            qc.save_statevector()
            result = _backend.run(qc).result()
            return _extract(result, qc)
        except Exception as exc:
            logger.warning("Aer run failed (%s); falling back to Statevector", exc)

    from qiskit.quantum_info import Statevector
    return Statevector(circuit).data


def _extract(result, circuit) -> np.ndarray:
    """Extract the statevector from an Aer ``Result`` object."""
    try:
        sv = result.get_statevector(circuit)
        return np.asarray(sv.data)
    except Exception:
        try:
            sv = result.get_statevector(0)
            return np.asarray(sv.data)
        except Exception:
            data = result.data(0)
            sv = data.get("statevector", data.get("sv", None))
            if sv is not None:
                return np.asarray(sv)
            raise


def aer_available() -> bool:
    """Whether ``qiskit-aer`` is importable."""
    _init_backend()
    return _HAS_AER


def gpu_available() -> bool:
    """Whether an Aer GPU backend was successfully initialised."""
    _init_backend()
    return _GPU_AVAILABLE


def backend_info() -> dict:
    """Return a dict with backend status."""
    _init_backend()
    if _backend is None:
        return {"available": False, "device": None, "aer": _HAS_AER}
    return {
        "available": True,
        "device": "GPU" if _GPU_AVAILABLE else "CPU",
        "aer": True,
    }
