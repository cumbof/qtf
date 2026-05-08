"""EnsembleFoldingManager — orchestrates multiple independent folding replicas.

Only **random** initialisation is supported: circuit parameters are drawn from
a uniform distribution and the lowest-energy starting point is selected via
basin-hopping before each full optimisation run.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

import numpy as np

from qtf.core.folder import QuantumBiophysicsFolder

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class EnsembleFoldingManager:
    """Manages multiple independent folding runs with random initialisation."""

    def __init__(self, folder: QuantumBiophysicsFolder) -> None:
        self.folder = folder
        self.results: list[dict] = []

    # ------------------------------------------------------------------
    # Ensemble run
    # ------------------------------------------------------------------

    def run_ensemble(
        self,
        n_runs: int = 5,
        max_iter: int = 2000,
        scout_attempts: int = 50,
    ) -> None:
        """Run *n_runs* independent folding trajectories with random initialisation.

        Parameters
        ----------
        n_runs:
            Number of independent replicas.
        max_iter:
            Maximum optimiser iterations per replica.
        scout_attempts:
            Number of random parameter sets evaluated during basin-hopping
            to find a good starting point for each replica.
        """
        logger.info("Starting ensemble run: %d trajectories", n_runs)
        self.results = []

        # Deterministic base seed derived from protein sequence
        base_seed = int(
            hashlib.sha256(self.folder.sequence.encode()).hexdigest(), 16
        ) % (2 ** 32)

        for i in range(n_runs):
            replica_seed = base_seed + i
            logger.info("Replica %d/%d (seed=%d)", i + 1, n_runs, replica_seed)

            start_params = self.folder.get_smart_initialization(
                n_attempts=scout_attempts, seed=replica_seed
            )

            coords, labels, bonds, tracker, final_params, final_energy = self.folder.fold(
                max_iter=max_iter, initial_params=start_params
            )

            logger.info("  Replica %d final energy: %.4f", i + 1, final_energy)
            self.results.append(
                {
                    "id": i,
                    "seed": replica_seed,
                    "energy": final_energy,
                    "coords": coords,
                    "labels": labels,
                    "bonds": bonds,
                    "params": final_params,
                    "tracker": tracker,
                }
            )

    # ------------------------------------------------------------------
    # Result access
    # ------------------------------------------------------------------

    def get_results(self) -> list[dict]:
        """Return all replica results, sorted by ascending energy."""
        return sorted(self.results, key=lambda x: x["energy"])
