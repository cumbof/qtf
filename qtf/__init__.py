"""QTF — Quantum Torsion Folding.

A hybrid quantum-classical protein structure prediction package built on
Qiskit's EfficientSU2 ansatz and a physics-based energy critic.

Quick start
-----------
>>> from qtf import QuantumBiophysicsFolder, EnsembleFoldingManager
>>> from qtf.analysis import EnsembleRanking
>>> from qtf.visualization import plot_structure, plot_energy_landscape, plot_ranking
>>> from qtf.utils import get_ground_truth_backbone
>>>
>>> folder = QuantumBiophysicsFolder("YYDPETGTWY", force_field="amber")
>>> manager = EnsembleFoldingManager(folder)
>>> manager.run_ensemble(n_runs=3)
>>>
>>> true_ca = get_ground_truth_backbone("5AWL")
>>> ranking = EnsembleRanking.from_ensemble(manager.get_results(), ground_truth_ca=true_ca)
>>> print(ranking.summary())
>>>
>>> plot_structure(ranking, ground_truth_ca=true_ca).show()
>>> plot_energy_landscape(ranking).show()
>>> plot_ranking(ranking).show()
"""

from qtf.core.folder import QuantumBiophysicsFolder
from qtf.core.ensemble import EnsembleFoldingManager
from qtf.core.tracker import LandscapeTracker

__all__ = [
    "QuantumBiophysicsFolder",
    "EnsembleFoldingManager",
    "LandscapeTracker",
]

__version__ = "0.1.8"
