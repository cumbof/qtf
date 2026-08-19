# QTF — Quantum Torsion Folding

> **Logarithmic-scale variational quantum eigensolver for torsion-space, off-lattice protein structure prediction.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/cumbof/QTF/blob/main/LICENSE)
[![Tests](https://github.com/cumbof/QTF/actions/workflows/tests.yml/badge.svg)](https://github.com/cumbof/QTF/actions/workflows/tests.yml)

---

## What is QTF?

**QTF** (Quantum Torsion Folder) is a hybrid quantum-classical protein structure prediction
package. Unlike lattice-based quantum folding approaches, QTF works entirely in **continuous
torsion space**: backbone dihedral angles (φ, ψ) and side-chain rotamer angles (χ₁–χ₅) are
the fundamental degrees of freedom.

Rather than assigning one qubit per degree of
freedom — which would require hundreds of qubits — QTF uses only ⌈log₂ N⌉ qubits to
represent N continuous angles. The phases of complex amplitudes in the quantum statevector
are extracted and mapped directly to torsion angles in [−π, π].

## Architecture

```
        ┌──────────────────────────────────┐
        │          QUANTUM ACTOR           │
        │                                  │
        │   θ (params) → EfficientSU2  →  │
        │   |ψ⟩  →  extract phases  →  φ,ψ,χ
        └────────────────┬─────────────────┘
                         │  torsion angles
                         ▼
        ┌──────────────────────────────────┐
        │       NERF GEOMETRY BUILDER      │
        │  angles → 3-D Cartesian coords  │
        │  (N, Cα, C, O + full side chains)│
        └────────────────┬─────────────────┘
                         │  all-atom coords
                         ▼
        ┌──────────────────────────────────┐
        │      CLASSICAL CRITIC (energy)   │
        │                                  │
        │  Hydrophobicity · H-bonds        │
        │  Electrostatics · Sterics        │
        │  Ramachandran · Rotamers         │
        │  π–π stacking · Geometry         │
        └────────────────┬─────────────────┘
                         │  E(θ)
                         ▼
        ┌──────────────────────────────────┐
        │    CLASSICAL OPTIMISER (loop)    │
        │                                  │
        │  Stage 1 – Collapse   (COBYLA)   │
        │  Stage 2 – Refine     (SLSQP)    │
        │  Stage 3 – Relax      (SLSQP)    │
        └──────────────────────────────────┘
```

## Key features

- **Logarithmic qubit scaling** — a 10-residue protein (~30 DoF) uses only 5 qubits
- **Physics-based energy function** with 10 distinct terms (no neural network required)
- **Native custom scoring** plus optional OpenMM and external force-field validation
- **Ensemble folding** with reproducible random initialisation and full provenance
- **Comprehensive ranking** — energy, RMSD vs ground truth, radius of gyration, convergence
- **Interactive Plotly visualisations** — 3-D backbone overlay, energy landscape, ranking dashboard
- **PDB I/O** — export predictions and download ground-truth structures from RCSB

## Quick example

```python
from qtf import QuantumBiophysicsFolder, EnsembleFoldingManager
from qtf.analysis import EnsembleRanking
from qtf.visualization import plot_structure, plot_energy_landscape, plot_ranking
from qtf.utils import get_ground_truth_backbone

folder  = QuantumBiophysicsFolder("YYDPETGTWY")
manager = EnsembleFoldingManager(folder)
manager.run_ensemble(n_runs=5, max_iter=2000, scout_attempts=50)

true_ca = get_ground_truth_backbone("5AWL")
ranking = EnsembleRanking.from_ensemble(manager.get_results(), ground_truth_ca=true_ca)
print(ranking.summary())

plot_structure(ranking, ground_truth_ca=true_ca).show()
plot_energy_landscape(ranking).show()
plot_ranking(ranking).show()
```

---

[:octicons-rocket-16: Get started](installation.md){ .md-button .md-button--primary }
[:octicons-book-16: Quick Start](quickstart.md){ .md-button }
[:octicons-code-16: API Reference](api.md){ .md-button }
