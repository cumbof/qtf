# Installation

## Requirements

| Package | Minimum version | Purpose |
|---------|-----------------|---------|
| Python  | 3.9             | Runtime |
| `numpy` | 1.24            | Array mathematics, distance matrices, linear algebra |
| `scipy` | 1.10            | COBYLA and SLSQP optimisers |
| `qiskit`| 1.0             | `EfficientSU2` ansatz; exact statevector simulation |
| `qiskit-aer` | 0.14       | GPU/CPU statevector simulation backend |
| `plotly`| 5.18            | Interactive 3-D and 2-D figures |
| `pandas`| 2.0             | Per-replica statistics DataFrame |

---

## From PyPI

Once published, install with:

```bash
pip install qtf
```

---

## From source

Clone the repository and install in editable mode so that any local changes are
immediately reflected without re-installing:

```bash
git clone https://github.com/cumbof/QTF.git
cd QTF
pip install -e .
```

---

## Optional extras

| Extra | Dependencies included | Use case |
|-------|-----------------------|----------|
| `[dev]` | `pytest`, `pytest-cov`, `ruff`, `mypy` | Development, linting, testing |
| `[notebook]` | `jupyter`, `nbformat` | Running `QTF.ipynb` |
| `[workflows]` | `matplotlib`, `mdtraj`, `biopython`, `openmm` | Energy backends, PDB I/O, minimisation |
| `[gpu]` | `numba`, `qiskit-aer-gpu` | GPU-accelerated classical & quantum simulation (Linux x86_64) |
| `[docs]` | `mkdocs`, `mkdocs-material`, `mkdocstrings[python]` | Building this documentation site |

Install any combination with:

```bash
pip install -e ".[dev,workflows]"
```

---

## Verifying the installation

```python
import qtf
print(qtf.__version__)

from qtf import QuantumBiophysicsFolder
folder = QuantumBiophysicsFolder("GA")
print(f"Qubits : {folder.n_qubits}")
print(f"Params : {folder.n_params}")
```

---

## Running the tests

```bash
pytest -q
```

The full test suite runs

--8<-- "includes/test_count.md"

individual tests covering the tracker,
stability analysis, PDB utilities, the folder, ensemble manager, ranking,
ansatz construction, GPU sim backend, Numba acceleration, parallel ensemble,
circular statistics, and all three visualisation functions.

---

!!! note "Qiskit version"
    QTF requires Qiskit **1.0 or later**, which introduced the `qiskit.circuit.library.efficient_su2`
    function used to build the ansatz. Versions prior to 1.0 are not supported.
