"""Numba-accelerated computational kernels for QTF energy evaluation.

Gating
------
Importing this module never raises: when `numba` is not installed the
decorated functions remain pure-Python fallbacks so the rest of QTF
works without change.  Install with::

    pip install "qtf[gpu]"

to enable 5-10× speedup on the custom energy function.
"""

from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def _maybe_jit(func):
    """Apply ``@njit`` when numba is available, otherwise a no-op."""
    if _HAS_NUMBA:
        return njit(func)
    return func


# ---------------------------------------------------------------------------
# Distance matrix
# ---------------------------------------------------------------------------


@_maybe_jit
def distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance matrix with 1e-9 floor (no NaNs)."""
    n = coords.shape[0]
    D = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            dz = coords[i, 2] - coords[j, 2]
            D[i, j] = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-9
    return D


# ---------------------------------------------------------------------------
# Electrostatic energy
# ---------------------------------------------------------------------------


@_maybe_jit
def electrostatic_energy(
    q_vector: np.ndarray,
    D: np.ndarray,
    mask_non_bonded: np.ndarray,
    prefactor: float,
    dielectric: float,
) -> float:
    """Coulomb term as a double loop (avoids large Q_mat allocation)."""
    total = 0.0
    n = q_vector.shape[0]
    for i in range(n):
        qi = q_vector[i]
        if abs(qi) < 1e-4:
            continue
        for j in range(i + 1, n):
            qj = q_vector[j]
            if abs(qj) < 1e-4:
                continue
            if not mask_non_bonded[i, j]:
                continue
            r = D[i, j]
            if r < 1.0:
                r = 1.0
            total += prefactor * qi * qj / (dielectric * r)
    return total


# ---------------------------------------------------------------------------
# Van der Waals repulsion
# ---------------------------------------------------------------------------


@_maybe_jit
def vdw_repulsion(
    D: np.ndarray,
    sigma_mat: np.ndarray,
    mask_vdw: np.ndarray,
    mask_vdw_14: np.ndarray,
    scale_14: float,
) -> float:
    """Lennard-Jones repulsion, short-range + 1-4 scaled."""
    total = 0.0
    n = D.shape[0]

    for i in range(n):
        for j in range(i + 1, n):
            r = D[i, j]
            s = sigma_mat[i, j]

            if mask_vdw[i, j] and r < s:
                term = (s / (r + 0.1)) ** 12
                if term > 50.0:
                    term = 50.0 + math.log(term - 49.0)
                total += 0.1 * term

            if mask_vdw_14[i, j] and r < s:
                term = (s / (r + 0.1)) ** 12
                if term > 50.0:
                    term = 50.0 + math.log(term - 49.0)
                total += scale_14 * 0.1 * term

    return total


# ---------------------------------------------------------------------------
# SASA / hydrophobic burial
# ---------------------------------------------------------------------------


@_maybe_jit
def sasa_energy(
    D: np.ndarray,
    mask_hydrophobic: np.ndarray,
    gamma: float,
) -> float:
    """Solvent-accessible-surface-area proxy."""
    n_hydro = 0
    for i in range(mask_hydrophobic.shape[0]):
        if mask_hydrophobic[i]:
            n_hydro += 1
    if n_hydro == 0:
        return 0.0

    total = 0.0
    for i in range(mask_hydrophobic.shape[0]):
        if not mask_hydrophobic[i]:
            continue
        neighbor_count = 0.0
        for j in range(D.shape[1]):
            neighbor_count += 1.0 / (1.0 + math.exp(1.0 * (D[i, j] - 6.0)))
        neighbor_count -= 1.0
        burial = max(0.0, min(neighbor_count / 15.0, 1.0))
        total += gamma * 30.0 * (1.0 - burial)
    return total
