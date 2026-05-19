"""Tests for kabsch_rmsd and StabilityAnalyzer."""

import numpy as np
import pytest

from qtf.analysis.stability import StabilityAnalyzer, kabsch_rmsd


# ---------------------------------------------------------------------------
# kabsch_rmsd
# ---------------------------------------------------------------------------


def test_kabsch_identical_structures():
    P = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    rmsd, P_aligned = kabsch_rmsd(P, P.copy())
    assert rmsd == pytest.approx(0.0, abs=1e-9)


def test_kabsch_translated_structure():
    """Pure translation should not change RMSD after alignment."""
    P = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    Q = P + np.array([3.0, 5.0, -2.0])
    rmsd, _ = kabsch_rmsd(P, Q)
    assert rmsd == pytest.approx(0.0, abs=1e-9)


def test_kabsch_rotated_structure():
    """90° rotation around Z should give RMSD ≈ 0."""
    P = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    theta = np.pi / 2
    R = np.array(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    )
    Q = P @ R.T
    rmsd, _ = kabsch_rmsd(P, Q)
    assert rmsd == pytest.approx(0.0, abs=1e-6)


def test_kabsch_nonzero_rmsd():
    P = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    Q = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    rmsd, _ = kabsch_rmsd(P, Q)
    assert rmsd > 0.0


def test_kabsch_aligned_shape():
    P = np.random.default_rng(0).random((5, 3))
    Q = np.random.default_rng(1).random((5, 3))
    rmsd, P_aligned = kabsch_rmsd(P, Q)
    assert P_aligned.shape == P.shape


def test_kabsch_rmsd_nonnegative():
    P = np.random.default_rng(42).random((8, 3))
    Q = np.random.default_rng(99).random((8, 3))
    rmsd, _ = kabsch_rmsd(P, Q)
    assert rmsd >= 0.0


def test_kabsch_symmetric():
    """RMSD(P, Q) should equal RMSD(Q, P)."""
    P = np.random.default_rng(7).random((6, 3))
    Q = np.random.default_rng(8).random((6, 3))
    rmsd_pq, _ = kabsch_rmsd(P, Q)
    rmsd_qp, _ = kabsch_rmsd(Q, P)
    assert rmsd_pq == pytest.approx(rmsd_qp, rel=1e-5)


# ---------------------------------------------------------------------------
# StabilityAnalyzer.pairwise_rmsd_matrix
# ---------------------------------------------------------------------------


def test_pairwise_rmsd_matrix_shape():
    structures = [np.random.default_rng(i).random((4, 3)) for i in range(3)]
    mat = StabilityAnalyzer.pairwise_rmsd_matrix(structures)
    assert mat.shape == (3, 3)


def test_pairwise_rmsd_matrix_symmetric():
    structures = [np.random.default_rng(i).random((4, 3)) for i in range(3)]
    mat = StabilityAnalyzer.pairwise_rmsd_matrix(structures)
    np.testing.assert_array_almost_equal(mat, mat.T)


def test_pairwise_rmsd_matrix_diagonal_zero():
    structures = [np.random.default_rng(i).random((4, 3)) for i in range(3)]
    mat = StabilityAnalyzer.pairwise_rmsd_matrix(structures)
    np.testing.assert_array_almost_equal(np.diag(mat), np.zeros(3))


def test_pairwise_rmsd_single_structure():
    structures = [np.random.default_rng(0).random((4, 3))]
    mat = StabilityAnalyzer.pairwise_rmsd_matrix(structures)
    assert mat.shape == (1, 1)
    assert mat[0, 0] == 0.0


def test_pairwise_rmsd_two_identical():
    s = np.random.default_rng(5).random((4, 3))
    mat = StabilityAnalyzer.pairwise_rmsd_matrix([s, s.copy()])
    assert mat[0, 1] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# StabilityAnalyzer.convergence_summary
# ---------------------------------------------------------------------------


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
