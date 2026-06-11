"""Tests for QTF's StabilityAnalyzer convergence verdict layer.

Kabsch alignment and the pairwise RMSD matrix primitives now live in
PHEAT (see ``pheat.geometry.kabsch_rmsd`` and
``pheat.metrics.pairwise_rmsd_matrix``).  These tests cover only the
QTF-specific verdict thresholds layered on top of
``pheat.metrics.ensemble_rmsd_stats``.
"""

import numpy as np
import pytest

from qtf.analysis.stability import StabilityAnalyzer


def test_convergence_single_structure():
    mat = np.zeros((1, 1))
    result = StabilityAnalyzer.convergence_summary(mat)
    assert result["verdict"] == "SINGLE_STRUCTURE"


def test_convergence_stable():
    mat = np.array([[0.0, 1.0], [1.0, 0.0]])
    result = StabilityAnalyzer.convergence_summary(mat)
    assert result["verdict"] == "STABLE"
    assert result["avg_pairwise_rmsd"] == pytest.approx(1.0)


def test_convergence_flexible():
    mat = np.array([[0.0, 3.0], [3.0, 0.0]])
    result = StabilityAnalyzer.convergence_summary(mat)
    assert result["verdict"] == "FLEXIBLE"


def test_convergence_unstable():
    mat = np.array([[0.0, 5.0], [5.0, 0.0]])
    result = StabilityAnalyzer.convergence_summary(mat)
    assert result["verdict"] == "UNSTABLE"


def test_convergence_keys():
    mat = np.array([[0.0, 2.0], [2.0, 0.0]])
    result = StabilityAnalyzer.convergence_summary(mat)
    assert set(result.keys()) == {
        "avg_pairwise_rmsd",
        "max_pairwise_rmsd",
        "min_nonzero_rmsd",
        "verdict",
    }


def test_convergence_max_and_min():
    mat = np.array([[0.0, 1.0, 3.0], [1.0, 0.0, 2.0], [3.0, 2.0, 0.0]])
    result = StabilityAnalyzer.convergence_summary(mat)
    # Upper triangle values: 1.0, 3.0, 2.0 → avg=2.0, max=3.0, min=1.0
    assert result["max_pairwise_rmsd"] == pytest.approx(3.0)
    assert result["min_nonzero_rmsd"] == pytest.approx(1.0)
    assert result["avg_pairwise_rmsd"] == pytest.approx(2.0)
