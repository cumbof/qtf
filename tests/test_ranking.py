"""Tests for EnsembleRanking."""

import numpy as np
import pytest

from qtf.analysis.ranking import EnsembleRanking


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def three_results(results_factory):
    return results_factory([5.0, 2.0, 8.0])


@pytest.fixture(scope="module")
def ranking_no_gt(two_results):
    return EnsembleRanking.from_ensemble(two_results)


@pytest.fixture(scope="module")
def ranking_with_gt(two_results, zero_structure):
    coords, labels, _ = zero_structure
    ca_coords = np.array([coords[i] for i, lbl in enumerate(labels) if lbl[1] == "CA"])
    return EnsembleRanking.from_ensemble(two_results, ground_truth_ca=ca_coords)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_from_ensemble_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        EnsembleRanking.from_ensemble([])


# ---------------------------------------------------------------------------
# No ground truth
# ---------------------------------------------------------------------------


def test_from_ensemble_single_result(results_factory):
    results = results_factory([3.0])
    ranking = EnsembleRanking.from_ensemble(results)
    assert len(ranking.stats_df) == 1
    assert ranking.best_by_rmsd is None


def test_from_ensemble_no_gt_best_by_rmsd_none(ranking_no_gt):
    assert ranking_no_gt.best_by_rmsd is None


def test_from_ensemble_no_gt_row_count(ranking_no_gt):
    assert len(ranking_no_gt.stats_df) == 2


def test_from_ensemble_best_energy_correct(three_results):
    ranking = EnsembleRanking.from_ensemble(three_results)
    assert ranking.best_by_energy["energy"] == pytest.approx(2.0)


def test_from_ensemble_rank_energy_column(ranking_no_gt):
    assert "rank_energy" in ranking_no_gt.stats_df.columns


def test_from_ensemble_is_best_energy_flag(three_results):
    ranking = EnsembleRanking.from_ensemble(three_results)
    best_rows = ranking.stats_df[ranking.stats_df["is_best_energy"]]
    assert len(best_rows) == 1
    assert best_rows.iloc[0]["energy"] == pytest.approx(2.0)


def test_from_ensemble_rmsd_vs_gt_nan_when_no_gt(ranking_no_gt):
    assert ranking_no_gt.stats_df["rmsd_vs_gt"].isna().all()


# ---------------------------------------------------------------------------
# With ground truth
# ---------------------------------------------------------------------------


def test_from_ensemble_best_by_rmsd_not_none(ranking_with_gt):
    assert ranking_with_gt.best_by_rmsd is not None


def test_from_ensemble_rmsd_vs_gt_populated(ranking_with_gt):
    assert not ranking_with_gt.stats_df["rmsd_vs_gt"].isna().all()


def test_from_ensemble_rank_rmsd_column(ranking_with_gt):
    assert "rank_rmsd" in ranking_with_gt.stats_df.columns


def test_from_ensemble_rmsd_nonnegative(ranking_with_gt):
    rmsd_vals = ranking_with_gt.stats_df["rmsd_vs_gt"].dropna()
    assert (rmsd_vals >= 0.0).all()


# ---------------------------------------------------------------------------
# DataFrame columns and statistics
# ---------------------------------------------------------------------------


def test_stats_df_required_columns(ranking_no_gt):
    required = [
        "replica_id",
        "energy",
        "rank_energy",
        "is_best_energy",
        "radius_of_gyration",
        "end_to_end_dist",
        "mean_rmsd_vs_ensemble",
        "is_ensemble_centroid",
    ]
    for col in required:
        assert col in ranking_no_gt.stats_df.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# Pairwise RMSD matrix
# ---------------------------------------------------------------------------


def test_pairwise_rmsd_matrix_shape(three_results):
    ranking = EnsembleRanking.from_ensemble(three_results)
    assert ranking.pairwise_rmsd_matrix.shape == (3, 3)


def test_pairwise_rmsd_matrix_diagonal_zero(three_results):
    ranking = EnsembleRanking.from_ensemble(three_results)
    np.testing.assert_array_almost_equal(
        np.diag(ranking.pairwise_rmsd_matrix), np.zeros(3)
    )


def test_pairwise_rmsd_matrix_symmetric(three_results):
    ranking = EnsembleRanking.from_ensemble(three_results)
    mat = ranking.pairwise_rmsd_matrix
    np.testing.assert_array_almost_equal(mat, mat.T)


# ---------------------------------------------------------------------------
# Convergence dict
# ---------------------------------------------------------------------------


def test_convergence_keys_present(ranking_no_gt):
    assert "verdict" in ranking_no_gt.convergence
    assert "avg_pairwise_rmsd" in ranking_no_gt.convergence
    assert "max_pairwise_rmsd" in ranking_no_gt.convergence


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------


def test_summary_returns_string(ranking_no_gt):
    s = ranking_no_gt.summary()
    assert isinstance(s, str)


def test_summary_contains_header(ranking_no_gt):
    s = ranking_no_gt.summary()
    assert "Ensemble Ranking" in s


def test_summary_contains_replica_count(ranking_no_gt):
    s = ranking_no_gt.summary()
    assert "2" in s


def test_summary_with_gt_mentions_rmsd(ranking_with_gt):
    s = ranking_with_gt.summary()
    assert "RMSD" in s


# ---------------------------------------------------------------------------
# Internal _results attribute (used by visualization)
# ---------------------------------------------------------------------------


def test_private_results_attr(ranking_no_gt, two_results):
    assert hasattr(ranking_no_gt, "_results")
    assert len(ranking_no_gt._results) == len(two_results)
