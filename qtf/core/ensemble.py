"""EnsembleFoldingManager — orchestrates multiple independent folding replicas.

Replicas are executed in parallel via :class:`concurrent.futures.ProcessPoolExecutor`.
Each worker process creates its own :class:`~qtf.core.folder.QuantumBiophysicsFolder`
instance from the sequence and configuration, so there is no shared mutable state
and no need for shared-memory synchronisation of the topology cache.
"""

from __future__ import annotations

import concurrent.futures
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


# ---------------------------------------------------------------------------
# Standalone helpers for worker subprocesses
# ---------------------------------------------------------------------------


def _prime_circuit(
    folder: QuantumBiophysicsFolder,
    target_type: str = "helix",
    seed: int = 42,
) -> np.ndarray:
    """Pre-optimise *folder*'s circuit to produce secondary-structure angles.

    This is the same logic as :meth:`EnsembleFoldingManager.prime_circuit`
    but operates on an arbitrary folder instance (needed because the method
    form is tied to the manager's ``self.folder``).

    Returns the optimised parameter vector.
    """
    rng = np.random.default_rng(seed)

    if target_type == "helix":
        t_phi, t_psi = np.deg2rad(-60.0), np.deg2rad(-45.0)
    elif target_type == "sheet":
        t_phi, t_psi = np.deg2rad(-135.0), np.deg2rad(135.0)
    else:
        return rng.uniform(-0.8, 0.8, folder.n_params)

    targets = np.zeros(folder.total_angles)
    masks = np.zeros(folder.total_angles)

    for i, dof in enumerate(folder.dof_map):
        if dof["type"] == "phi":
            targets[i] = t_phi
            masks[i] = 1.0
        elif dof["type"] == "psi":
            targets[i] = t_psi
            masks[i] = 1.0

    def priming_cost(params: np.ndarray) -> float:
        curr = folder._get_angles(params)
        diff = (curr - targets + np.pi) % (2 * np.pi) - np.pi
        return float(np.sum((diff * masks) ** 2))

    init_guess = rng.uniform(-0.1, 0.1, folder.n_params)
    res = minimize(priming_cost, init_guess, method="COBYLA", options={"maxiter": 200})
    return res.x


def _run_one_replica(
    folder_kwargs: dict,
    replica_seed: int,
    index: int,
    strat: str,
    max_iter: int,
    scout_attempts: int,
) -> dict:
    """Execute a single folding replica in a subprocess.

    Each call creates a fresh :class:`QuantumBiophysicsFolder` from
    *folder_kwargs* so workers never share mutable state.
    """
    folder = QuantumBiophysicsFolder(**folder_kwargs)

    if strat != "random":
        start_params = _prime_circuit(folder, target_type=strat, seed=replica_seed)
    else:
        start_params = folder.get_smart_initialization(
            n_attempts=scout_attempts, seed=replica_seed
        )

    coords, labels, bonds, tracker, final_params, final_energy = folder.fold(
        max_iter=max_iter, initial_params=start_params
    )

    energy_terms = dict(getattr(folder, "last_energy_terms", {}) or {})

    return {
        "id": index,
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


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class EnsembleFoldingManager:
    """Manages multiple independent folding runs with random initialisation.

    Replicas are executed in parallel across ``max_workers`` subprocesses.
    Each worker builds its own :class:`~qtf.core.folder.QuantumBiophysicsFolder`
    from the configuration provided by the manager, eliminating shared-memory
    contention on the topology cache.
    """

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

    def prime_circuit(self, target_type="helix", seed=42, overwrite: bool = False):
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
            return float(np.sum((diff * masks)**2))

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
        max_workers: int | None = None,
    ) -> None:
        """Run *n_runs* independent folding trajectories in parallel.

        Replicas are distributed across a :class:`~concurrent.futures.ProcessPoolExecutor`.
        Each subprocess creates its own :class:`~qtf.core.folder.QuantumBiophysicsFolder`
        from the same configuration, so the topology cache is independently
        rebuilt in every worker — no shared-memory synchronisation is needed.

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
        prime_strategy:
            ``"random"`` (default), ``"mixed"``, ``"helix"``, or ``"sheet"``.
            ``"mixed"`` cycles through ``helix`` / ``sheet`` / ``random``
            every 3 replicas.
        checkpoint_path:
            Optional path to a JSON file. When provided, a JSON-safe snapshot
            of the successfully completed replicas is written to disk after
            every successful replica, and again before any re-raised
            interrupt. The file is created if missing and overwritten if it
            exists. The snapshot contains metadata only (id, seed, type,
            energy, energy_terms) so it stays small; the heavy arrays
            (coords, params) are deliberately not serialised.
        max_workers:
            Maximum number of subprocesses.  When ``None`` (the default) the
            pool size is ``min(os.cpu_count(), n_runs)``.
        """
        logger.info("Starting ensemble run: %d trajectories", n_runs)
        self.results = []
        self._last_error = None
        self._checkpoint_path = checkpoint_path

        # Deterministic base seed derived from protein sequence
        base_seed = int(
            hashlib.sha256(self.folder.sequence.encode()).hexdigest(), 16
        ) % (2 ** 32)

        # Build task list
        tasks: list[tuple[int, int, str]] = []
        for i in range(n_runs):
            replica_seed = base_seed + i

            strat = prime_strategy
            if prime_strategy == "mixed":
                if i % 3 == 0:
                    strat = "helix"
                elif i % 3 == 1:
                    strat = "sheet"
                else:
                    strat = "random"

            tasks.append((i, replica_seed, strat))

        # Extract folder kwargs so each worker can build its own folder
        folder_kwargs = {
            "sequence": self.folder.sequence,
            "chi_mode": self.folder.chi_mode,
            "selective_chi_map": self.folder.selective_chi_map,
            "energy_backend": self.folder.stage_backend,
            "use_e2e_constraint": self.folder.use_e2e_constraint,
            "e2e_scale": self.folder.e2e_scale,
            "rosetta_repack": self.folder.rosetta_do_repack,
            "rosetta_fa_min": self.folder.rosetta_do_fullatom_min,
            "rosetta_cen_min": self.folder.rosetta_do_centroid_min,
        }

        n_workers = (
            max_workers if max_workers is not None
            else min(os.cpu_count() or 1, n_runs)
        )

        if n_workers <= 1:
            self._run_sequential(
                tasks, folder_kwargs, max_iter, scout_attempts, n_runs,
            )
        else:
            self._run_parallel(
                tasks, folder_kwargs, max_iter, scout_attempts, n_runs, n_workers,
            )

        # Restore deterministic insertion order (by replica id)
        self.results.sort(key=lambda r: r["id"])

    # ------------------------------------------------------------------
    # Sequential path  (used when max_workers ≤ 1, also the test-friendly path)
    # ------------------------------------------------------------------

    def _run_sequential(
        self,
        tasks: list[tuple[int, int, str]],
        folder_kwargs: dict,
        max_iter: int,
        scout_attempts: int,
        n_runs: int,
    ) -> None:
        """Run replicas in-process using ``self.folder`` (no subprocess).

        This path calls methods on the manager's folder instance directly,
        so patches and mocks applied in tests are visible.  It is also
        marginally faster for single-replica runs.
        """
        for i, replica_seed, strat in tasks:
            logger.info("Replica %d/%d (seed=%d)", i + 1, n_runs, replica_seed)

            try:
                if strat != "random":
                    start_params = self.prime_circuit(
                        target_type=strat, seed=replica_seed,
                    )
                else:
                    start_params = self.folder.get_smart_initialization(
                        n_attempts=scout_attempts, seed=replica_seed,
                    )

                coords, labels, bonds, tracker, final_params, final_energy = (
                    self.folder.fold(
                        max_iter=max_iter, initial_params=start_params,
                    )
                )
            except (KeyboardInterrupt, SystemExit):
                logger.warning(
                    "Replica %d aborted by user; preserving %d prior result(s)",
                    i + 1,
                    len(self.results),
                )
                self._write_checkpoint()
                raise
            except Exception as exc:        # noqa: BLE001
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
    # Parallel path
    # ------------------------------------------------------------------

    def _run_parallel(
        self,
        tasks: list[tuple[int, int, str]],
        folder_kwargs: dict,
        max_iter: int,
        scout_attempts: int,
        n_runs: int,
        n_workers: int,
    ) -> None:
        """Distribute replicas across *n_workers* subprocesses."""
        from concurrent.futures import ProcessPoolExecutor, as_completed

        fut_to_idx: dict[concurrent.futures.Future, int] = {}
        executor = ProcessPoolExecutor(max_workers=n_workers)
        try:
            for i, replica_seed, strat in tasks:
                fut = executor.submit(
                    _run_one_replica,
                    folder_kwargs,
                    replica_seed,
                    i,
                    strat,
                    max_iter,
                    scout_attempts,
                )
                fut_to_idx[fut] = i

            done_iter = as_completed(fut_to_idx)
            while True:
                try:
                    fut = next(done_iter)
                except StopIteration:
                    break
                idx = fut_to_idx[fut]
                try:
                    result = fut.result()
                    self.results.append(result)
                    logger.info(
                        "  Replica %d/%d final energy: %.4f",
                        idx + 1, n_runs, result["energy"],
                    )
                    self._write_checkpoint()
                except Exception as exc:        # noqa: BLE001
                    self._last_error = exc
                    logger.error(
                        "Replica %d failed: %s",
                        idx + 1,
                        exc,
                        exc_info=True,
                    )

        except (KeyboardInterrupt, SystemExit):
            logger.warning(
                "Ensemble aborted; preserving %d prior result(s)",
                len(self.results),
            )
            for fut in fut_to_idx:
                fut.cancel()
            self._write_checkpoint()
            raise
        finally:
            executor.shutdown(wait=False)

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
