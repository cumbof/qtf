"""Ensemble convergence helpers layered on PHEAT primitives.

Kabsch alignment, pairwise RMSD matrices, and ensemble statistics live in
PHEAT (see :func:`pheat.geometry.kabsch_rmsd`,
:func:`pheat.metrics.pairwise_rmsd_matrix`, and
:func:`pheat.metrics.ensemble_rmsd_stats`).  This module keeps only the
QTF-specific verdict layer that classifies an ensemble as STABLE/FLEXIBLE/
UNSTABLE based on protein-Cα RMSD thresholds.
"""

from __future__ import annotations

import numpy as np

from pheat.metrics import ensemble_rmsd_stats


class StabilityAnalyzer:
    """QTF-specific convergence verdict layered on PHEAT ensemble statistics."""

    @staticmethod
    def convergence_summary(rmsd_matrix: np.ndarray) -> dict:
        """Return convergence statistics plus a STABLE/FLEXIBLE/UNSTABLE verdict.

        Raw statistics (``avg_pairwise_rmsd``, ``max_pairwise_rmsd``,
        ``min_nonzero_rmsd``) are produced by
        :func:`pheat.metrics.ensemble_rmsd_stats`.  This wrapper adds the
        protein-Cα verdict thresholds (2.0 Å, 4.5 Å) on top.

        Returns
        -------
        dict with keys: avg_pairwise_rmsd, max_pairwise_rmsd,
        min_nonzero_rmsd, verdict
        """

        stats = ensemble_rmsd_stats(rmsd_matrix)
        if stats["ensemble_size"] < 2:
            return {
                "avg_pairwise_rmsd": 0.0,
                "max_pairwise_rmsd": 0.0,
                "min_nonzero_rmsd": 0.0,
                "verdict": "SINGLE_STRUCTURE",
            }
        avg = stats["avg_pairwise_rmsd"]
        if avg < 2.0:
            verdict = "STABLE"
        elif avg < 4.5:
            verdict = "FLEXIBLE"
        else:
            verdict = "UNSTABLE"
        return {
            "avg_pairwise_rmsd": avg,
            "max_pairwise_rmsd": stats["max_pairwise_rmsd"],
            "min_nonzero_rmsd": stats["min_nonzero_rmsd"],
            "verdict": verdict,
        }
