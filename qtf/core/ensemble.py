"""EnsembleFoldingManager — orchestrates multiple independent folding replicas.

Only **random** initialisation is supported: circuit parameters are drawn from
a uniform distribution and the lowest-energy starting point is selected via
basin-hopping before each full optimisation run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
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
        self._last_error: Exception | None = None
        self._checkpoint_path: str | None = None

    @property
    def last_error(self) -> Exception | None:
        """Return the most recent per-replica exception, or ``None`` if the
        last ensemble run completed without any per-replica failure."""
        return self._last_error

    # ------------------------------------------------------------------
    # Ensemble run
    # ------------------------------------------------------------------

    def prime_circuit(self, target_type='helix', seed=42, overwrite: bool = False):
        """Smart initialization: pre-optimize the circuit to output secondary
        structure angles.

        Parameters
        ----------
        target_type:
            ``'helix'`` (default, ``phi=-60``, ``psi=-45``),
            ``'sheet'`` (``phi=-135``, ``psi=135``), or anything else in
            which case a uniform random parameter vector is returned
            directly (no COBYLA priming cost is incurred).
        seed:
            Seed for the priming optimiser's random initial guess.
        overwrite:
            If ``False`` (the default), the call is a no-op when the
            folder already has a non-``None`` ``circuit_parameters``
            attribute — the existing value is returned untouched. Pass
            ``overwrite=True`` to force re-priming.

        Returns
        -------
        numpy.ndarray
            The primed (or pre-existing) circuit-parameter vector of shape
            ``(self.folder.n_params,)``.

        Notes
        -----
        ``QuantumBiophysicsFolder`` does not declare a
        ``circuit_parameters`` attribute by default; the idempotency
        guard is therefore a soft opt-in for users who set it manually
        (or who subclass the folder to provide a persistent store).
        This matches the convention in the rest of the code base where
        the folder is treated as the single source of truth for a
        given sequence, and priming is a one-shot transformation that
        should never silently destroy a user's prior work.
        """
        existing = getattr(self.folder, "circuit_parameters", None)
        if existing is not None and not overwrite:
            logger.info(
                "Skipping prime_circuit: folder.circuit_parameters is already set; "
                "pass overwrite=True to force re-priming"
            )
            return existing

        logger.info("--- PRIMING CIRCUIT FOR %s ---", target_type.upper())

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
        logger.info("  > Priming Error: %.4f", res.fun)
        return res.x

    def run_ensemble(
        self,
        n_runs: int = 5,
        max_iter: int = 2000,
        scout_attempts: int = 50,
        prime_strategy: str = "random",
        checkpoint_path: str | None = None,
    ) -> None:
        """Run *n_runs* independent folding trajectories with random initialisation.

        A failure in a single replica (any ``Exception``) is logged, recorded in
        :attr:`last_error`, and the loop proceeds to the next replica. A
        ``KeyboardInterrupt`` / ``SystemExit`` (e.g. ``Ctrl-C``) is *not*
        swallowed: it is re-raised after writing a final checkpoint so the
        caller still gets control, but the results already collected are
        preserved on disk.

        Parameters
        ----------
        n_runs:
            Number of independent replicas.
        max_iter:
            Maximum optimiser iterations per replica.
        scout_attempts:
            Number of random parameter sets evaluated during basin-hopping
            to find a good starting point for each replica.
        checkpoint_path:
            Optional path to a JSON file. When provided, a JSON-safe snapshot
            of the successfully completed replicas is written to disk after
            every successful replica, and again before any re-raised
            interrupt. The file is created if missing and overwritten if it
            exists. The snapshot contains metadata only (id, seed, type,
            energy, energy_terms) so it stays small; the heavy arrays
            (coords, params) are deliberately not serialised.
        """
        logger.info("Starting ensemble run: %d trajectories", n_runs)
        self.results = []
        self._last_error = None
        self._checkpoint_path = checkpoint_path

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

            try:
                if strat != "random":
                    start_params = self.prime_circuit(
                        target_type=strat, seed=replica_seed
                    )
                else:
                    start_params = self.folder.get_smart_initialization(
                        n_attempts=scout_attempts, seed=replica_seed
                    )

                coords, labels, bonds, tracker, final_params, final_energy = (
                    self.folder.fold(
                        max_iter=max_iter, initial_params=start_params
                    )
                )
            except (KeyboardInterrupt, SystemExit):
                # User-initiated abort: preserve whatever we have, checkpoint
                # if requested, and let the exception propagate so the caller
                # can decide what to do.
                logger.warning(
                    "Replica %d aborted by user; preserving %d prior result(s)",
                    i + 1,
                    len(self.results),
                )
                self._write_checkpoint()
                raise
            except Exception as exc:        # noqa: BLE001 — last-ditch recovery
                # Per-replica failure: log, record, and continue with the next
                # replica. This is the only way to keep an ensemble of N
                # replicas alive when one of them hits a numerical issue, a
                # COBYLA MAXFUN error, or a misconfigured force field.
                logger.error(
                    "Replica %d failed: %s",
                    i + 1,
                    exc,
                    exc_info=True,
                )
                self._last_error = exc
                continue

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
            self._write_checkpoint()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _write_checkpoint(self) -> None:
        """Atomically write a JSON snapshot of completed replicas to disk.

        The snapshot is metadata-only: ``id``, ``seed``, ``type``, ``energy``,
        and ``energy_terms``. Heavy arrays (``coords``, ``labels``, ``bonds``,
        ``params``, ``tracker``) are intentionally omitted to keep the file
        small and JSON-safe. The file is written atomically (via a sibling
        temp file + ``os.replace``) so a crash mid-write cannot leave a
        half-written checkpoint.

        No-op if ``self._checkpoint_path`` is ``None``.
        """
        if self._checkpoint_path is None:
            return
        snapshot = {
            "sequence": self.folder.sequence,
            "replicas": [
                {
                    "id": r["id"],
                    "seed": r["seed"],
                    "type": r["type"],
                    "energy": float(r["energy"]),
                    "energy_terms": {
                        k: (float(v) if isinstance(v, (int, float)) else v)
                        for k, v in r.get("energy_terms", {}).items()
                    },
                }
                for r in self.results
            ],
        }
        target_dir = os.path.dirname(os.path.abspath(self._checkpoint_path)) or "."
        try:
            fd, tmp_path = tempfile.mkstemp(
                prefix=".qtf_ckpt_", suffix=".json", dir=target_dir
            )
            try:
                with os.fdopen(fd, "w") as fh:
                    json.dump(snapshot, fh, indent=2)
                os.replace(tmp_path, self._checkpoint_path)
            except Exception:
                # Best-effort cleanup of the temp file; the exception itself
                # is intentionally swallowed because a failed checkpoint
                # write must not abort the ensemble.
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                logger.warning(
                    "Failed to write checkpoint to %s",
                    self._checkpoint_path,
                    exc_info=True,
                )
        except Exception:
            logger.warning(
                "Failed to prepare checkpoint at %s",
                self._checkpoint_path,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Result access
    # ------------------------------------------------------------------

    def get_results(self, ranked: bool = False) -> list[dict]:
        """Return all replica results.

        Parameters
        ----------
        ranked:
            If *True*, results are sorted by ascending energy (lowest first).
            Default is *False* — results are returned in insertion order.

        Returns
        -------
        list[dict]
            The full list of result dicts.
        """
        if ranked:
            return sorted(self.results, key=lambda x: x["energy"])
        return self.results

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
        ranked = self.get_results(ranked=True)
        if not ranked:
            return []
        if top_frac is not None:
            k = max(1, int(np.ceil(len(ranked) * float(top_frac))))
            return ranked[:k]
        if top_k is not None:
            k = max(1, min(int(top_k), len(ranked)))
            return ranked[:k]
        return ranked
