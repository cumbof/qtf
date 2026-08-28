"""Generate the self-contained QTF simulator and hardware workflow notebook."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "QTF.ipynb"


def lines(source: str) -> list[str]:
    return dedent(source).strip("\n").splitlines(keepends=True)


def markdown(cell_id: str, source: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": lines(source)}


def code(cell_id: str, source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": lines(source),
    }


cells = [
    markdown(
        "overview",
        """
        # Quantum Torsion Folding: simulation and IBM hardware

        This notebook exposes three complete workflows through one configuration
        cell: local simulation, IBM hardware with simulator-optimized parameters
        (warm start), and IBM hardware with random parameters (cold start).

        All modes rebuild an all-atom structure with PHEAT. GROMACS minimization
        is enabled by default and writes a refined PDB plus run metadata. Run
        `simulator` first when you intend to use `hardware_warm`.
        """,
    ),
    code(
        "configuration",
        """
        # Primary mode: "simulator", "hardware_warm", or "hardware_cold".
        EXECUTION_MODE = "simulator"

        # Protein and workflow. This recipe runs three optimization stages
        # (collapse, refine, relax), each using pheat-custom-energy-v1. It then
        # performs PHEAT reconstruction and validates structures with GROMACS.
        SEQUENCE = "YYDPETGTWY"
        REFERENCE_STRUCTURE = "5AWL"  # PDB ID/path, or None to skip RMSD
        RECIPE = "qtf-default-config-snapshots"
        OUTPUT_ROOT = "qtf_notebook_outputs"
        SIMULATOR_RUNS = 1
        # Applied to each of the three optimization stages.
        SIMULATOR_MAX_ITER = 2000
        SCOUT_ATTEMPTS = 20
        RANDOM_SEED = 23

        # Warm hardware mode reads the simulation output directory or a specific
        # circuit-parameter manifest/JSON/NPZ file. For multi-replica simulation
        # outputs, select the desired zero-based replica ID.
        WARM_START_SOURCE = f"{OUTPUT_ROOT}/simulator"
        WARM_START_REPLICA_ID = 0

        # IBM Runtime. None uses the saved default account and least-busy QPU.
        HARDWARE_BACKEND = None
        IBM_CHANNEL = None
        IBM_INSTANCE = None
        IBM_TOKEN = None  # Avoid saving secrets in shared notebooks.
        HARDWARE_SHOTS = 8192
        HARDWARE_OPTIMIZATION_LEVEL = 3
        HARDWARE_TRANSPILE_SEED = 23
        HARDWARE_MAX_MITIGATION = True

        # Set True to exercise either hardware path locally with Aer without
        # submitting IBM jobs. This does not change warm/cold parameter behavior.
        LOCAL_HARDWARE_SIMULATOR = False

        # PHEAT rebuild and GROMACS energy minimization are enabled in all modes.
        REQUIRE_GROMACS = True
        GROMACS_FORCEFIELD = "amber99sb-ildn"
        GROMACS_WATER = "tip3p"
        GROMACS_NSTEPS = 5000
        # Stop when the maximum force falls below this value (kJ mol^-1 nm^-1).
        # This preserves the stricter threshold used by the established
        # chignolin workflow; increase GROMACS_NSTEPS rather than weakening it.
        GROMACS_EMTOL = 100.0
        GROMACS_MAXWARN = 2
        """,
    ),
    code(
        "imports-validation",
        """
        import json
        import shutil
        from pathlib import Path

        import plotly.graph_objects as go

        from qtf.cli import main as qtf_main
        from qtf.utils.gromacs import parse_pdb_atoms
        from qtf.utils.pdb import get_ground_truth_backbone

        valid_modes = {"simulator", "hardware_warm", "hardware_cold"}
        if EXECUTION_MODE not in valid_modes:
            raise ValueError(f"EXECUTION_MODE must be one of {sorted(valid_modes)}")
        if REQUIRE_GROMACS and shutil.which("gmx") is None:
            raise RuntimeError(
                "GROMACS executable 'gmx' is missing. Install conda-forge::gromacs "
                "in the image before running this notebook."
            )

        output_root = Path(OUTPUT_ROOT)
        output_root.mkdir(parents=True, exist_ok=True)

        reference_path = None
        if REFERENCE_STRUCTURE:
            reference_candidate = Path(str(REFERENCE_STRUCTURE)).expanduser()
            if reference_candidate.is_file():
                reference_path = reference_candidate.resolve()
            elif len(str(REFERENCE_STRUCTURE)) == 4 and str(REFERENCE_STRUCTURE).isalnum():
                reference_cache = output_root / "reference"
                get_ground_truth_backbone(str(REFERENCE_STRUCTURE), cache_dir=str(reference_cache))
                reference_path = (reference_cache / f"{str(REFERENCE_STRUCTURE).upper()}.pdb").resolve()
            else:
                raise FileNotFoundError(
                    "REFERENCE_STRUCTURE must be an existing local structure or a four-character PDB ID"
                )
        """,
    ),
    markdown(
        "arguments-heading",
        """
        ## Build the selected workflow

        The notebook calls QTF's supported command API directly. The displayed
        argument list is reproducible from a terminal by prefixing it with `qtf`.
        """,
    ),
    code(
        "build-arguments",
        """
        if EXECUTION_MODE == "simulator":
            run_output = output_root / "simulator"
            workflow_args = [
                "fold-simulation", RECIPE,
                "--sequence", SEQUENCE,
                "--outdir", str(run_output),
                "--n-runs", str(SIMULATOR_RUNS),
                "--max-iter", str(SIMULATOR_MAX_ITER),
                "--scout-attempts", str(SCOUT_ATTEMPTS),
                "--seed", str(RANDOM_SEED),
                "--rebuild-method", "pheat",
            ]
            if reference_path:
                workflow_args += ["--reference-structure", str(reference_path)]
        else:
            run_output = output_root / EXECUTION_MODE
            workflow_args = [
                "fold-hardware",
                "--outdir", str(run_output),
                "--shots", str(HARDWARE_SHOTS),
                "--optimization-level", str(HARDWARE_OPTIMIZATION_LEVEL),
                "--seed-transpiler", str(HARDWARE_TRANSPILE_SEED),
                "--rebuild-method", "pheat",
                "--gromacs",
                "--gromacs-forcefield", GROMACS_FORCEFIELD,
                "--gromacs-water", GROMACS_WATER,
                "--gromacs-nsteps", str(GROMACS_NSTEPS),
                "--gromacs-emtol", str(GROMACS_EMTOL),
                "--gromacs-maxwarn", str(GROMACS_MAXWARN),
            ]
            if EXECUTION_MODE == "hardware_warm":
                warm_source = Path(WARM_START_SOURCE)
                if not warm_source.exists():
                    raise FileNotFoundError(
                        f"Warm-start source is missing: {warm_source}. Run simulator mode first."
                    )
                workflow_args += [
                    "--params-json", str(warm_source),
                    "--replica-id", str(WARM_START_REPLICA_ID),
                ]
            else:
                workflow_args += [
                    "--sequence", SEQUENCE,
                    "--recipe", RECIPE,
                    "--seed", str(RANDOM_SEED),
                ]
            if reference_path:
                workflow_args += ["--reference_structure", str(reference_path)]
            if HARDWARE_BACKEND:
                workflow_args += ["--backend-name", HARDWARE_BACKEND]
            if IBM_CHANNEL:
                workflow_args += ["--channel", IBM_CHANNEL]
            if IBM_INSTANCE:
                workflow_args += ["--instance", IBM_INSTANCE]
            if IBM_TOKEN:
                workflow_args += ["--token", IBM_TOKEN]
            if LOCAL_HARDWARE_SIMULATOR:
                workflow_args += ["--local-simulator"]
            if not HARDWARE_MAX_MITIGATION:
                workflow_args += ["--no-sampler-max-mitigation"]

        print("Selected mode:", EXECUTION_MODE)
        print("Equivalent command:")
        print("qtf " + " ".join(workflow_args))
        """,
    ),
    code(
        "execute",
        """
        exit_status = qtf_main(workflow_args)
        if exit_status not in (None, 0):
            raise RuntimeError(f"QTF workflow exited with status {exit_status}")
        print(f"Completed {EXECUTION_MODE}; outputs: {run_output.resolve()}")
        """,
    ),
    markdown(
        "outputs-heading",
        """
        ## Inspect structures and metadata

        Simulator mode stores a circuit-parameter manifest for warm hardware
        runs. Hardware modes store the raw PHEAT rebuild, GROMACS-minimized PDB,
        IBM job metadata, and structural metrics.
        """,
    ),
    code(
        "inspect-outputs",
        """
        pdb_files = sorted(run_output.rglob("*.pdb"))
        json_files = sorted(run_output.rglob("*.json"))
        print(f"PDB files ({len(pdb_files)}):")
        for path in pdb_files:
            print(" ", path)
        print(f"JSON files ({len(json_files)}):")
        for path in json_files:
            print(" ", path)

        manifests = [path for path in json_files if path.name == "circuit_parameters.json"]
        if EXECUTION_MODE == "simulator":
            if not manifests:
                raise RuntimeError("Simulator completed without a circuit-parameter manifest")
            print("Warm-start manifest:", manifests[0].resolve())

        minimized = [path for path in pdb_files if "gromacs" in path.name.lower() or "minimized" in path.name.lower()]
        if REQUIRE_GROMACS and not minimized:
            raise RuntimeError("Workflow completed without a GROMACS-minimized PDB")
        structure_path = minimized[0] if minimized else (pdb_files[0] if pdb_files else None)
        if structure_path is None:
            raise RuntimeError("Workflow completed without a PDB structure")
        print("Displayed structure:", structure_path.resolve())
        """,
    ),
    code(
        "plot-structure",
        """
        coords, labels = parse_pdb_atoms(str(structure_path))
        ca_indices = [index for index, label in enumerate(labels) if label[1] == "CA"]
        if not ca_indices:
            raise RuntimeError(f"No C-alpha atoms found in {structure_path}")
        ca = coords[ca_indices]
        figure = go.Figure(
            data=[go.Scatter3d(
                x=ca[:, 0], y=ca[:, 1], z=ca[:, 2],
                mode="lines+markers",
                marker={"size": 5},
                line={"width": 5},
                text=[f"Residue {labels[index][0] + 1}" for index in ca_indices],
            )]
        )
        figure.update_layout(
            title=f"QTF {EXECUTION_MODE}: {SEQUENCE}",
            scene={"xaxis_title": "x (Å)", "yaxis_title": "y (Å)", "zaxis_title": "z (Å)"},
        )
        figure.show()
        """,
    ),
    code(
        "versions",
        """
        from importlib.metadata import version
        import subprocess

        for package in ("qtf", "pheat", "numpy", "qiskit", "qiskit-aer", "qiskit-ibm-runtime"):
            print(f"{package}: {version(package)}")
        print(subprocess.run(["gmx", "--version"], capture_output=True, text=True, check=True).stdout.splitlines()[0])
        """,
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
TARGET.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(TARGET)
