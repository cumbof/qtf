"""Regression checks for the generated QTF workflow notebook."""

import json
from pathlib import Path


def test_qtf_notebook_has_valid_three_mode_workflow() -> None:
    notebook_path = Path(__file__).resolve().parents[1] / "QTF.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook:{cell['id']}", "exec")

    required_contracts = (
        'EXECUTION_MODE = "simulator"',
        "SIMULATOR_MAX_ITER = 2000",
        "GROMACS_EMTOL = 100.0",
        "collapse, refine, relax",
        "pheat-custom-energy-v1",
        '"hardware_warm"',
        '"hardware_cold"',
        '"--params-json"',
        '"--gromacs"',
        'shutil.which("gmx")',
        '"circuit_parameters.json"',
    )
    for contract in required_contracts:
        assert contract in source
