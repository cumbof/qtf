"""Structural analysis utilities: Kabsch algorithm and convergence metrics."""

from __future__ import annotations

import numpy as np


def kabsch_rmsd(
    P: np.ndarray, Q: np.ndarray
) -> tuple[float, np.ndarray]:
    """Compute the minimum RMSD between two point sets using the Kabsch algorithm.

    Both *P* and *Q* must have shape ``(N, 3)``.  *P* is rotated to best
    align with *Q*.

    Parameters
    ----------
    P, Q:
        Coordinate matrices to compare.

    Returns
    -------
    rmsd : float
        Root-mean-square deviation after optimal superposition.
    P_aligned : ndarray, shape (N, 3)
        *P* after centering, rotation, and re-translation onto *Q*.
    """
    P_c = P - P.mean(axis=0)
    Q_c = Q - Q.mean(axis=0)

    H = P_c.T @ Q_c
    V, S, Wt = np.linalg.svd(H)

    # Ensure a proper rotation (det = +1), not a reflection
    if (np.linalg.det(V) * np.linalg.det(Wt)) < 0.0:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    R = V @ Wt
    P_rotated = P_c @ R
    diff = P_rotated - Q_c
    rms = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
    return rms, P_rotated + Q.mean(axis=0)


class StabilityAnalyzer:
    """Tools for structural consistency analysis of folding ensembles."""

    @staticmethod
    def pairwise_rmsd_matrix(structures: list[np.ndarray]) -> np.ndarray:
        """Compute the all-vs-all RMSD matrix for a list of CA coordinate arrays.

        Parameters
        ----------
        structures:
            List of ``(N_residues, 3)`` CA coordinate arrays.

        Returns
        -------
        matrix : ndarray, shape (M, M)
            Symmetric pairwise RMSD matrix (diagonal = 0).
        """
        n = len(structures)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                rmsd, _ = kabsch_rmsd(structures[i], structures[j])
                matrix[i, j] = matrix[j, i] = rmsd
        return matrix

    @staticmethod
    def convergence_summary(rmsd_matrix: np.ndarray) -> dict:
        """Return convergence statistics derived from a pairwise RMSD matrix.

        Returns
        -------
        dict with keys: avg_pairwise_rmsd, max_pairwise_rmsd,
        min_nonzero_rmsd, verdict
        """
        n = len(rmsd_matrix)
        if n < 2:
            return {"avg_pairwise_rmsd": 0.0, "max_pairwise_rmsd": 0.0,
                    "min_nonzero_rmsd": 0.0, "verdict": "SINGLE_STRUCTURE"}
        upper = rmsd_matrix[np.triu_indices(n, k=1)]
        avg = float(np.mean(upper))
        mx = float(np.max(upper))
        mn = float(np.min(upper))
        if avg < 2.0:
            verdict = "STABLE"
        elif avg < 4.5:
            verdict = "FLEXIBLE"
        else:
            verdict = "UNSTABLE"
        return {
            "avg_pairwise_rmsd": avg,
            "max_pairwise_rmsd": mx,
            "min_nonzero_rmsd": mn,
            "verdict": verdict,
        }
