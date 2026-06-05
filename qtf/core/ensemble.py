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
from scipy.optimize import minimize

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

    def prime_circuit(self, target_type='helix', seed=42):
        """
        Smart Initialization: Pre-optimizes circuit to output Secondary Structure angles.
        """
        print(f"--- PRIMING CIRCUIT FOR {target_type.upper()} ---")
        
        rng = np.random.default_rng(seed)
        
        if target_type == 'helix':
            t_phi, t_psi = np.deg2rad(-60.0), np.deg2rad(-45.0)
        elif target_type == 'sheet':
            t_phi, t_psi = np.deg2rad(-135.0), np.deg2rad(135.0)
        else:
            return rng.uniform(-0.8, 0.8, self.folder.n_params)

        targets = np.zeros(self.folder.total_angles)
        masks = np.zeros(self.folder.total_angles)
        
        for i, dof in enumerate(self.folder.dof_map):
            if dof['type'] == 'phi': targets[i] = t_phi; masks[i] = 1.0
            elif dof['type'] == 'psi': targets[i] = t_psi; masks[i] = 1.0
            
        def priming_cost(params):
            curr = self.folder._get_angles(params)
            diff = (curr - targets + np.pi) % (2 * np.pi) - np.pi
            return np.sum((diff * masks)**2)

        init_guess = rng.uniform(-0.1, 0.1, self.folder.n_params)
        res = minimize(priming_cost, init_guess, method='COBYLA', options={'maxiter': 200})
        print(f" > Priming Error: {res.fun:.4f}")
        return res.x

    def run_ensemble(
        self,
        n_runs: int = 5,
        max_iter: int = 2000,
        scout_attempts: int = 50,
        prime_strategy: str = "random",
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

            strat = prime_strategy
            if prime_strategy == "mixed":
                if i % 3 == 0:
                    strat = "helix"
                elif i % 3 == 1:
                    strat = "sheet"
                else:
                    strat = "random"

            if strat != "random":
                start_params = self.prime_circuit(target_type=strat, seed=replica_seed)
            else:
                start_params = self.folder.get_smart_initialization(
                    n_attempts=scout_attempts, seed=replica_seed
                )

            coords, labels, bonds, tracker, final_params, final_energy = self.folder.fold(
                max_iter=max_iter, initial_params=start_params
            )

            logger.info("  Replica %d final energy: %.4f", i + 1, final_energy)
            energy_terms = dict(getattr(self.folder, "last_energy_terms", {}) or {})
            self.results.append(
                {
                    "id": i,
                    "seed": replica_seed,
                    "type": strat,
                    "energy": final_energy,
                    "energy_terms": energy_terms,
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

    def get_ranked_results(self):
        """Return all ensemble results sorted by energy (ascending)."""
        if not self.results:
            return []
        return sorted(self.results, key=lambda x: x['energy'])

    def select_top(self, top_k=None, top_frac=None):
        """
        Select top low-energy structures.

        Parameters
        ----------
        top_k : int | None
            Keep the top_k lowest-energy structures.
        top_frac : float | None
            Keep the top fraction (0<top_frac<=1) of lowest-energy structures.
            If provided, top_frac takes precedence over top_k.

        Returns
        -------
        list[dict]
            Ranked subset of self.results.
        """
        ranked = self.get_ranked_results()
        if not ranked:
            return []
        if top_frac is not None:
            k = max(1, int(np.ceil(len(ranked) * float(top_frac))))
            return ranked[:k]
        if top_k is not None:
            k = max(1, min(int(top_k), len(ranked)))
            return ranked[:k]
        return ranked
