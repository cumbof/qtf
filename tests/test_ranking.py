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


# ---------------------------------------------------------------------------
# B3: tied-energy consistency between best_by_energy, is_best_energy,
# and best_replica_id
# ---------------------------------------------------------------------------


def test_ranking_tie_returns_first(results_factory):
    """When two or more replicas share the minimum energy, the ranking
    must pick the *first* one in the input list (np.argmin is a strict
    first-occurrence tie-breaker). The same index must be used for
    `best_by_energy`, `stats_df['is_best_energy']`, and
    `best_replica_id`."""
    results = results_factory([3.0, 3.0, 5.0])  # replicas 0 and 1 tied at 3.0
    ranking = EnsembleRanking.from_ensemble(results)

    # The lowest-energy replica is the first one in the list
    assert ranking.best_by_energy["id"] == 0
    assert ranking.best_by_energy["energy"] == pytest.approx(3.0)
    # The boolean flag points at the same replica
    best_row = ranking.stats_df[ranking.stats_df["is_best_energy"]]
    assert len(best_row) == 1
    assert int(best_row.iloc[0]["replica_id"]) == 0
    # best_replica_id matches too
    assert ranking.best_replica_id == 0


def test_best_by_energy_matches_flag_across_reorderings(results_factory):
    """Three replicas with energies [5.0, 3.0, 4.0]. Reorder the list
    (simulating a manager that skipped a failed replica, see B1) and
    verify that `best_by_energy`, `stats_df['is_best_energy']`, and
    `best_replica_id` are still mutually consistent for every
    ordering.

    This is the direct regression test for B3 from QTF-plan-2.md: a
    single source of truth must drive all three so the visualisation
    layer (e.g. plot_ranking) highlights and renders the same
    replica.
    """
    base = results_factory([5.0, 3.0, 4.0])  # ids 0, 1, 2 in that order
    for perm in (
        (0, 1, 2),
        (1, 0, 2),
        (2, 1, 0),
        (0, 2, 1),
    ):
        reordered = [base[i] for i in perm]
        ranking = EnsembleRanking.from_ensemble(reordered)

        # The lowest energy in this permutation is 3.0, which is at
        # perm.index(1) in the input list, and corresponds to the
        # original id 1.
        expected_position = perm.index(1)
        expected_id = reordered[expected_position]["id"]

        assert ranking.best_by_energy["id"] == expected_id
        assert ranking.best_replica_id == expected_id
        best_row = ranking.stats_df[ranking.stats_df["is_best_energy"]]
        assert len(best_row) == 1
        assert int(best_row.iloc[0]["replica_id"]) == expected_id
        assert int(best_row.iloc[0]["replica_id"]) == ranking.best_by_energy["id"]


def test_best_replica_id_attribute_exposed(results_factory):
    """`best_replica_id` is the public, single-source-of-truth handle
    for the best-by-energy replica. It must be present on every
    ranking and equal to `best_by_energy['id']`."""
    results = results_factory([2.0, 7.0, 5.0])
    ranking = EnsembleRanking.from_ensemble(results)
    assert hasattr(ranking, "best_replica_id")
    assert isinstance(ranking.best_replica_id, int)
    assert ranking.best_replica_id == ranking.best_by_energy["id"]


def test_is_best_energy_unique_per_ranking(results_factory):
    """Exactly one row in `stats_df` must have `is_best_energy=True`."""
    results = results_factory([5.0, 3.0, 3.0, 3.0, 9.0])  # triple tie
    ranking = EnsembleRanking.from_ensemble(results)
    assert ranking.stats_df["is_best_energy"].sum() == 1


def test_summary_reports_consistent_best_replica(results_factory):
    """`summary()` must report the same replica id as `best_by_energy`
    and `best_replica_id`. This is the user-facing contract that the
    visualisation layer relies on."""
    results = results_factory([4.0, 4.0, 1.0, 4.0])
    ranking = EnsembleRanking.from_ensemble(results)
    s = ranking.summary()
    # The id of the best-by-energy replica (2) must appear in the
    # 'Best by energy' line.
    assert f"replica {ranking.best_replica_id}" in s
    assert f"replica {ranking.best_by_energy['id']}" in s
