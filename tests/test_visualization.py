"""Tests for Plotly visualization functions."""

import numpy as np
import plotly.graph_objects as go
import pytest

from qtf.visualization import plot_energy_landscape, plot_ranking, plot_structure


@pytest.fixture(scope="module")
def ranking_for_plots(two_results):
    from qtf.analysis.ranking import EnsembleRanking

    return EnsembleRanking.from_ensemble(two_results)


# ---------------------------------------------------------------------------
# plot_structure
# ---------------------------------------------------------------------------


def test_plot_structure_returns_figure(ranking_for_plots):
    fig = plot_structure(ranking_for_plots)
    assert isinstance(fig, go.Figure)


def test_plot_structure_has_traces(ranking_for_plots):
    fig = plot_structure(ranking_for_plots)
    assert len(fig.data) >= 1


def test_plot_structure_with_ground_truth(ranking_for_plots, zero_structure):
    coords, labels, _ = zero_structure
    ca = np.array([coords[i] for i, lbl in enumerate(labels) if lbl[1] == "CA"])
    fig = plot_structure(ranking_for_plots, ground_truth_ca=ca)
    assert isinstance(fig, go.Figure)
    # Ground truth trace should be added
    assert len(fig.data) >= 3  # 2 replicas + GT


def test_plot_structure_custom_title(ranking_for_plots):
    fig = plot_structure(ranking_for_plots, title="My Title")
    assert fig.layout.title.text == "My Title"


# ---------------------------------------------------------------------------
# plot_energy_landscape
# ---------------------------------------------------------------------------


def test_plot_energy_landscape_returns_figure(ranking_for_plots):
    fig = plot_energy_landscape(ranking_for_plots)
    assert isinstance(fig, go.Figure)


def test_plot_energy_landscape_has_traces(ranking_for_plots):
    fig = plot_energy_landscape(ranking_for_plots)
    assert len(fig.data) >= 1


def test_plot_energy_landscape_custom_title(ranking_for_plots):
    fig = plot_energy_landscape(ranking_for_plots, title="Energy Test")
    assert fig.layout.title.text == "Energy Test"


# ---------------------------------------------------------------------------
# plot_ranking
# ---------------------------------------------------------------------------


def test_plot_ranking_returns_figure(ranking_for_plots):
    fig = plot_ranking(ranking_for_plots)
    assert isinstance(fig, go.Figure)


def test_plot_ranking_has_bar_and_table(ranking_for_plots):
    fig = plot_ranking(ranking_for_plots)
    trace_types = {type(t).__name__ for t in fig.data}
    assert "Bar" in trace_types
    assert "Table" in trace_types


def test_plot_ranking_custom_title(ranking_for_plots):
    fig = plot_ranking(ranking_for_plots, title="My Ranking")
    assert fig.layout.title.text == "My Ranking"


# ---------------------------------------------------------------------------
# _collect_results error handling
# ---------------------------------------------------------------------------


class TestCollectResults:
    """Verify that _collect_results raises clearly instead of returning broken data."""

    def test_returns_full_list_for_valid_ranking(self, ranking_for_plots):
        from qtf.visualization.plots import _collect_results

        results = _collect_results(ranking_for_plots)
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_every_result_has_required_keys(self, ranking_for_plots):
        from qtf.visualization.plots import _collect_results

        for r in _collect_results(ranking_for_plots):
            for key in ("id", "coords", "labels", "tracker"):
                assert key in r, f"result dict missing key '{key}'"

    def test_raises_for_object_without_results(self):
        from qtf.visualization.plots import _collect_results

        class FakeRanking:
            pass  # no _results attribute

        with pytest.raises(ValueError, match="_collect_results"):
            _collect_results(FakeRanking())

    def test_raises_for_empty_results_list(self):
        from qtf.visualization.plots import _collect_results

        class FakeRanking:
            _results: list = []

        with pytest.raises(ValueError, match="EnsembleRanking.from_ensemble"):
            _collect_results(FakeRanking())

    def test_result_count_matches_dataframe(self, ranking_for_plots):
        """Number of result dicts must equal number of rows in stats_df."""
        from qtf.visualization.plots import _collect_results

        results = _collect_results(ranking_for_plots)
        assert len(results) == len(ranking_for_plots.stats_df)
