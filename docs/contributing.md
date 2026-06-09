# Contributing

Contributions are welcome! This page explains how to set up a development environment,
run the test suite, and follow the project's code conventions.

---

## Development setup

1. **Fork and clone** the repository:

    ```bash
    git clone https://github.com/<your-username>/QTF.git
    cd QTF
    ```

2. **Create a virtual environment** (Python 3.9+):

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3. **Install with development extras**:

    ```bash
    pip install -e ".[dev]"
    ```

    This installs the package in editable mode along with `pytest`, `pytest-cov`,
    `ruff`, and `mypy`.

---

## Running tests

```bash
# Run the full suite
pytest -q

# Run with coverage report
pytest --cov=qtf --cov-report=term-missing

# Run all tests in a specific file
pytest tests/test_folder.py -v

# Run a single test by its node id
pytest tests/test_folder.py::TestAnsatz::test_custom_quantum_circuit -v

# Run all tests in a class
pytest tests/test_folder.py::TestAnsatz -v

# Run only tests matching a keyword expression
pytest -k "ansatz or rmsd" -v
```

The full test suite runs

--8<-- "includes/test_count.md"

individual tests. All must pass before opening a pull request.

---

## Code style

QTF uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
ruff check qtf/         # lint
ruff format qtf/        # auto-format
```

Type hints are encouraged for all public functions. You can verify with:

```bash
mypy qtf/
```

---

## Project structure

```
QTF/
├── qtf/
│   ├── __init__.py          # public API exports; __version__
│   ├── core/
│   │   ├── folder.py        # QuantumBiophysicsFolder — main class
│   │   ├── ensemble.py      # EnsembleFoldingManager
│   │   └── tracker.py       # LandscapeTracker
│   ├── analysis/
│   │   ├── ranking.py       # EnsembleRanking
│   │   ├── stability.py     # kabsch_rmsd, StabilityAnalyzer
│   │   └── panel.py         # Panel/grid analysis pipeline
│   ├── visualization/
│   │   └── plots.py         # plot_structure, plot_energy_landscape, plot_ranking
│   ├── utils/
│   │   ├── pdb.py           # save_pdb, get_ground_truth_backbone, calculate_physics_metrics
│   │   ├── workflow.py      # make_folder, score_native_structure, gromacs postprocess, RMSD utils
│   │   ├── aer_sim.py       # GPU/CPU statevector simulation backend
│   │   ├── accelerate.py    # Numba-jitted distance matrix, electrostatic, VDW, SASA kernels
│   │   └── gromacs.py       # GROMACS minimisation wrapper and PDB utilities
│   └── cli/
│       ├── fold.py          # qtf-fold: quantum folding prediction
│       ├── bench.py         # qtf-bench: beam-search benchmark
│       ├── eval.py          # qtf-eval: score experimental structures
│       ├── grid_search.py   # qtf-grid-search: parameter grid sweep
│       └── relax.py         # qtf-relax: GROMACS relaxation
├── tests/                   # pytest test suite
├── docs/                    # MkDocs documentation source
├── .github/
│   └── workflows/
│       ├── tests.yml        # CI: pytest on push/PR to main
│       ├── publish.yml      # PyPI publish (manual dispatch)
│       └── docs.yml         # GitHub Pages deploy on push to main
├── QTF.ipynb                # Demo notebook (Chignolin 5AWL)
├── pyproject.toml
├── mkdocs.yml
└── README.md
```

---

## Submitting a pull request

1. Create a feature branch: `git checkout -b my-feature`
2. Write tests for any new functionality
3. Ensure `pytest -q` passes with no failures
4. Run `ruff check qtf/` and fix any issues
5. Commit and push, then open a PR against `main`

Please include a clear description of what the PR changes and why.

---

## Adding a new energy term

1. Implement the term as a private method `_energy_<name>(self, coords, ...)` in
   `qtf/core/folder.py`, returning a scalar `float`.
2. Add the term (with a weight) to the `_compute_energy` dispatcher.
3. Add unit tests in `tests/test_folder.py`.
4. Document the new term in `docs/physics.md`.

---

## Reporting bugs

Please [open an issue](https://github.com/cumbof/QTF/issues) with:

- Python version and OS
- QTF version (`python -c "import qtf; print(qtf.__version__)"`)
- Minimal reproducible example
- Full traceback
