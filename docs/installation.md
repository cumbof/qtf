# Installation

## Requirements

| Package | Minimum version | Purpose |
|---------|-----------------|---------|
| Python  | 3.9             | Runtime |
| `numpy` | 1.24            | Array mathematics, distance matrices, linear algebra |
| `scipy` | 1.10            | COBYLA and SLSQP optimisers |
| `qiskit`| 1.0             | `EfficientSU2` ansatz; exact statevector simulation |
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

=== "Development"

    Includes `pytest`, `pytest-cov`, `ruff`, and `mypy`:

    ```bash
    pip install -e ".[dev]"
    ```

=== "Notebook"

    Adds `jupyter` and `nbformat` for running `QTF.ipynb`:

    ```bash
    pip install -e ".[notebook]"
    ```

=== "Documentation"

    Installs MkDocs and plugins used to build these pages:

    ```bash
    pip install -e ".[docs]"
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

All 116 tests should pass. The test suite covers the tracker, stability analysis, PDB
utilities, the folder, ensemble manager, ranking, and all three visualisation functions.

---

!!! note "Qiskit version"
    QTF requires Qiskit **1.0 or later**, which introduced the `qiskit.circuit.library.efficient_su2`
    function used to build the ansatz. Versions prior to 1.0 are not supported.
