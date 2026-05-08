"""EnsembleRanking — comprehensive ranking and statistics for folding ensembles.

Ranking strategy
----------------
* **All** predicted structures are retained and ranked.
* If a ground-truth CA trace is provided:
  - Each structure is compared against the ground truth via Kabsch RMSD.
  - Two "best" picks are reported: lowest energy and lowest RMSD vs ground truth.
* Without a ground truth, the best pick is the lowest-energy structure only.
* Full statistics are exposed as a :class:`pandas.DataFrame` for easy
  downstream analysis or export.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from qtf.analysis.stability import StabilityAnalyzer, kabsch_rmsd
from qtf.utils.pdb import calculate_physics_metrics

logger = logging.getLogger(__name__)


@dataclass
class EnsembleRanking:
    """Ranked collection of all predicted structures with per-structure statistics.

    Attributes
    ----------
    stats_df:
        DataFrame with one row per replica, columns:
        ``rank_energy``, ``replica_id``, ``seed``, ``energy``,
        ``radius_of_gyration``, ``end_to_end_dist``,
        ``rmsd_vs_gt`` (NaN if no ground truth),
        ``rank_rmsd`` (NaN if no ground truth),
        ``is_best_energy``, ``is_best_rmsd``.
    best_by_energy:
        The replica dict for the structure with the lowest energy.
    best_by_rmsd:
        The replica dict for the structure with the lowest RMSD vs ground
        truth, or ``None`` if no ground truth was provided.
    pairwise_rmsd_matrix:
        All-vs-all RMSD matrix between predicted structures.
    convergence:
        Convergence summary dict from :func:`~qtf.analysis.stability.StabilityAnalyzer.convergence_summary`.
    _results:
        Full list of replica dicts (private; used by visualisation helpers).
    """

    stats_df: pd.DataFrame
    best_by_energy: dict
    best_by_rmsd: Optional[dict]
    pairwise_rmsd_matrix: np.ndarray
    convergence: dict
    _results: list = field(default_factory=list, repr=False)

    @classmethod
    def from_ensemble(
        cls,
        results: list[dict],
        ground_truth_ca: Optional[np.ndarray] = None,
        ca_label: str = "CA",
    ) -> "EnsembleRanking":
        """Build an :class:`EnsembleRanking` from ensemble manager results.

        Parameters
        ----------
        results:
            List of replica dicts as returned by
            :meth:`~qtf.core.ensemble.EnsembleFoldingManager.get_results`.
            Each dict must contain ``id``, ``seed``, ``energy``, ``coords``,
            and ``labels``.
        ground_truth_ca:
            Optional ``(N_residues, 3)`` array of ground-truth Cα coordinates.
            When provided, RMSD vs ground truth is computed for every replica.
        ca_label:
            Atom name identifying Cα atoms in ``labels`` (default ``"CA"``).

        Returns
        -------
        EnsembleRanking
        """
        if not results:
            raise ValueError("results list is empty — run the ensemble first.")

        # ------------------------------------------------------------------
        # Extract Cα traces
        # ------------------------------------------------------------------
        def _extract_ca(result: dict) -> np.ndarray:
            coords = result["coords"]
            labels = result["labels"]
            return np.array([coords[i] for i, lbl in enumerate(labels) if lbl[1] == ca_label])

        ca_traces = [_extract_ca(r) for r in results]

        # ------------------------------------------------------------------
        # Per-replica statistics
        # ------------------------------------------------------------------
        rows = []
        for r, ca in zip(results, ca_traces):
            e2e, rg = calculate_physics_metrics(ca)
            row: dict = {
                "replica_id": r["id"],
                "seed": r["seed"],
                "energy": r["energy"],
                "end_to_end_dist": e2e,
                "radius_of_gyration": rg,
                "rmsd_vs_gt": np.nan,
            }
            if ground_truth_ca is not None:
                n = min(len(ca), len(ground_truth_ca))
                rmsd, _ = kabsch_rmsd(ca[:n], ground_truth_ca[:n])
                row["rmsd_vs_gt"] = rmsd
            rows.append(row)

        df = pd.DataFrame(rows)

        # ------------------------------------------------------------------
        # Rankings
        # ------------------------------------------------------------------
        df["rank_energy"] = df["energy"].rank(method="min").astype(int)
        if ground_truth_ca is not None:
            df["rank_rmsd"] = df["rmsd_vs_gt"].rank(method="min").astype(int)
        else:
            df["rank_rmsd"] = pd.Series(dtype="Int64")

        best_energy_idx = int(df["energy"].idxmin())
        df["is_best_energy"] = df.index == best_energy_idx

        if ground_truth_ca is not None:
            best_rmsd_idx = int(df["rmsd_vs_gt"].idxmin())
            df["is_best_rmsd"] = df.index == best_rmsd_idx
        else:
            df["is_best_rmsd"] = False

        # Sort by energy for display
        df = df.sort_values("rank_energy").reset_index(drop=True)

        # ------------------------------------------------------------------
        # Pairwise RMSD between all predicted structures
        # ------------------------------------------------------------------
        pwrmsd = StabilityAnalyzer.pairwise_rmsd_matrix(ca_traces)
        convergence = StabilityAnalyzer.convergence_summary(pwrmsd)

        # Additional aggregate columns derived from pairwise matrix
        # Mean RMSD of each structure vs all others
        n = len(ca_traces)
        mean_pairwise = (pwrmsd.sum(axis=1) / max(n - 1, 1))
        df["mean_rmsd_vs_ensemble"] = mean_pairwise[df["replica_id"].values]

        # Which structure minimises total pairwise distance (ensemble centroid)
        df["is_ensemble_centroid"] = df["mean_rmsd_vs_ensemble"] == df["mean_rmsd_vs_ensemble"].min()

        # ------------------------------------------------------------------
        # Retrieve best replica dicts
        # ------------------------------------------------------------------
        best_energy_replica = results[best_energy_idx]
        best_rmsd_replica = results[best_rmsd_idx] if ground_truth_ca is not None else None

        return cls(
            stats_df=df,
            best_by_energy=best_energy_replica,
            best_by_rmsd=best_rmsd_replica,
            pairwise_rmsd_matrix=pwrmsd,
            convergence=convergence,
            _results=results,
        )

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = ["=== Ensemble Ranking Summary ==="]
        lines.append(f"Total replicas : {len(self.stats_df)}")
        lines.append(f"Convergence    : {self.convergence['verdict']}")
        lines.append(f"  avg pairwise RMSD : {self.convergence['avg_pairwise_rmsd']:.3f} Å")
        lines.append(f"  max pairwise RMSD : {self.convergence['max_pairwise_rmsd']:.3f} Å")

        best_e = self.stats_df[self.stats_df["is_best_energy"]].iloc[0]
        lines.append(f"\nBest by energy  : replica {int(best_e['replica_id'])}  "
                     f"(E = {best_e['energy']:.4f})")

        if self.best_by_rmsd is not None:
            best_r = self.stats_df[self.stats_df["is_best_rmsd"]].iloc[0]
            lines.append(f"Best by RMSD    : replica {int(best_r['replica_id'])}  "
                         f"(RMSD = {best_r['rmsd_vs_gt']:.3f} Å, "
                         f"E = {best_r['energy']:.4f})")

        lines.append("\nFull rankings:")
        display_cols = [
            "rank_energy", "replica_id", "energy",
            "rmsd_vs_gt", "rank_rmsd",
            "radius_of_gyration", "end_to_end_dist",
            "mean_rmsd_vs_ensemble", "is_best_energy", "is_best_rmsd", "is_ensemble_centroid",
        ]
        present = [c for c in display_cols if c in self.stats_df.columns]
        lines.append(self.stats_df[present].to_string(index=False))
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"EnsembleRanking("
            f"n_replicas={len(self.stats_df)}, "
            f"best_energy={self.stats_df['energy'].min():.4f}, "
            f"has_gt={'yes' if self.best_by_rmsd is not None else 'no'})"
        )
