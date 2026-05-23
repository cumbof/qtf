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


# ---------------------------------------------------------------------------
# Stage marker colour consistency
# ---------------------------------------------------------------------------


class TestStageMarkerColours:
    """Verify stage markers are coloured by name, not by per-replica index."""

    def _make_ranking(self, markers_per_replica):
        """Build a minimal EnsembleRanking from synthetic tracker data.

        Parameters
        ----------
        markers_per_replica:
            list of lists of stage-name strings, one inner list per replica.
        """
        from qtf.analysis.ranking import EnsembleRanking
        from qtf.core.tracker import LandscapeTracker
        from qtf.core.folder import QuantumBiophysicsFolder

        f = QuantumBiophysicsFolder("GA")
        results = []
        for i, stage_names in enumerate(markers_per_replica):
            tracker = LandscapeTracker()
            for step, name in enumerate(stage_names, start=1):
                for _ in range(step * 5):
                    tracker.log(-float(step))
                tracker.mark_stage(name)
            angles = np.full(f.total_angles, 0.1)
            coords, labels, _ = f.build_full_structure(angles)
            results.append({
                "id": i,
                "seed": i,
                "energy": -float(i + 1),
                "coords": coords,
                "labels": labels,
                "tracker": tracker,
            })
        return EnsembleRanking.from_ensemble(results)

    def test_same_stage_name_same_colour_across_replicas(self):
        """Stage1 must get the same hex colour whether it is the first or only marker."""
        from qtf.visualization.plots import _PALETTE
        # replica 0: Stage1, Stage2, Stage3
        # replica 1: Stage1, Stage3 (skips Stage2)
        ranking = self._make_ranking([
            ["Stage1", "Stage2", "Stage3"],
            ["Stage1", "Stage3"],
        ])
        fig = plot_energy_landscape(ranking)
        # Collect annotation texts and their font colours from vlines
        colour_by_name: dict[str, str] = {}
        for annotation in fig.layout.annotations:
            txt = annotation.text
            colour = annotation.font.color
            colour_by_name[txt] = colour

        # Stage1 → stage1 palette colour
        assert colour_by_name.get("Stage1") == _PALETTE["stage1"]
        # Stage2 → stage2 palette colour
        assert colour_by_name.get("Stage2") == _PALETTE["stage2"]
        # Stage3 → stage3 palette colour (NOT stage2, which would be the bug)
        assert colour_by_name.get("Stage3") == _PALETTE["stage3"]

    def test_skipped_stage_does_not_shift_colours(self):
        """A replica missing Stage2 must not promote Stage3 to stage2 colour."""
        from qtf.visualization.plots import _PALETTE
        ranking = self._make_ranking([
            ["Stage1", "Stage3"],  # only replica; Stage2 is never emitted
        ])
        fig = plot_energy_landscape(ranking)
        colour_by_name: dict[str, str] = {}
        for annotation in fig.layout.annotations:
            colour_by_name[annotation.text] = annotation.font.color

        # Stage1 → stage1 (index 0)
        assert colour_by_name.get("Stage1") == _PALETTE["stage1"]
        # Stage3 → stage2 colour because Stage2 was never seen globally
        # (only two unique names exist, so Stage3 maps to index 1 = stage2)
        assert colour_by_name.get("Stage3") == _PALETTE["stage2"]

    def test_each_stage_name_drawn_at_most_once(self):
        """Duplicate stage names across replicas must produce only one vline."""
        ranking = self._make_ranking([
            ["Stage1", "Stage2"],
            ["Stage1", "Stage2"],
        ])
        fig = plot_energy_landscape(ranking)
        texts = [a.text for a in fig.layout.annotations]
        assert texts.count("Stage1") == 1
        assert texts.count("Stage2") == 1
