"""Tests for EnsembleFoldingManager."""

from unittest.mock import patch

import numpy as np
import pytest

from qtf.core.ensemble import EnsembleFoldingManager
from qtf.core.tracker import LandscapeTracker


@pytest.fixture
def mock_folder_ga(folder_ga, zero_structure):
    """Wrap the real folder_ga with patched fold/get_smart_initialization."""
    return folder_ga


def _fake_fold_result(folder_ga, zero_structure, energy=-10.0):
    """Build a valid fold() return value using the real folder."""
    coords, labels, bonds = zero_structure
    tracker = LandscapeTracker()
    tracker.log(energy)
    params = np.zeros(folder_ga.n_params)
    return coords, labels, bonds, tracker, params, energy


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------


def test_init_stores_folder(folder_ga):
    manager = EnsembleFoldingManager(folder_ga)
    assert manager.folder is folder_ga


def test_init_empty_results(folder_ga):
    manager = EnsembleFoldingManager(folder_ga)
    assert manager.results == []


# ---------------------------------------------------------------------------
# get_results
# ---------------------------------------------------------------------------


def test_get_results_sorted_ascending(folder_ga):
    manager = EnsembleFoldingManager(folder_ga)
    manager.results = [
        {"id": 0, "energy": 5.0},
        {"id": 1, "energy": 2.0},
        {"id": 2, "energy": 8.0},
    ]
    sorted_results = manager.get_results()
    energies = [r["energy"] for r in sorted_results]
    assert energies == sorted(energies)


def test_get_results_does_not_mutate_original(folder_ga):
    manager = EnsembleFoldingManager(folder_ga)
    manager.results = [{"id": 0, "energy": 5.0}, {"id": 1, "energy": 2.0}]
    _ = manager.get_results()
    # Original list still unsorted
    assert manager.results[0]["energy"] == 5.0


def test_get_results_empty(folder_ga):
    manager = EnsembleFoldingManager(folder_ga)
    assert manager.get_results() == []


# ---------------------------------------------------------------------------
# run_ensemble (with mocked fold)
# ---------------------------------------------------------------------------


def test_run_ensemble_populates_results(folder_ga, zero_structure):
    coords, labels, bonds = zero_structure
    tracker = LandscapeTracker()
    fake_params = np.zeros(folder_ga.n_params)
    fake_result = (coords, labels, bonds, tracker, fake_params, -10.0)

    with patch.object(folder_ga, "fold", return_value=fake_result) as mock_fold:
        with patch.object(folder_ga, "get_smart_initialization", return_value=fake_params):
            manager = EnsembleFoldingManager(folder_ga)
            manager.run_ensemble(n_runs=2)

    assert len(manager.results) == 2
    assert mock_fold.call_count == 2


def test_run_ensemble_result_keys(folder_ga, zero_structure):
    coords, labels, bonds = zero_structure
    tracker = LandscapeTracker()
    fake_params = np.zeros(folder_ga.n_params)
    fake_result = (coords, labels, bonds, tracker, fake_params, -5.0)

    with patch.object(folder_ga, "fold", return_value=fake_result):
        with patch.object(folder_ga, "get_smart_initialization", return_value=fake_params):
            manager = EnsembleFoldingManager(folder_ga)
            manager.run_ensemble(n_runs=1)

    result = manager.results[0]
    for key in ("id", "seed", "energy", "coords", "labels", "bonds", "params", "tracker"):
        assert key in result, f"Key '{key}' missing from result dict"


def test_run_ensemble_stores_energy(folder_ga, zero_structure):
    coords, labels, bonds = zero_structure
    tracker = LandscapeTracker()
    fake_params = np.zeros(folder_ga.n_params)
    fake_result = (coords, labels, bonds, tracker, fake_params, 99.5)

    with patch.object(folder_ga, "fold", return_value=fake_result):
        with patch.object(folder_ga, "get_smart_initialization", return_value=fake_params):
            manager = EnsembleFoldingManager(folder_ga)
            manager.run_ensemble(n_runs=1)

    assert manager.results[0]["energy"] == pytest.approx(99.5)


def test_run_ensemble_resets_results_each_call(folder_ga, zero_structure):
    """Calling run_ensemble twice should replace, not append, results."""
    coords, labels, bonds = zero_structure
    tracker = LandscapeTracker()
    fake_params = np.zeros(folder_ga.n_params)
    fake_result = (coords, labels, bonds, tracker, fake_params, 1.0)

    with patch.object(folder_ga, "fold", return_value=fake_result):
        with patch.object(folder_ga, "get_smart_initialization", return_value=fake_params):
            manager = EnsembleFoldingManager(folder_ga)
            manager.run_ensemble(n_runs=3)
            manager.run_ensemble(n_runs=1)

    assert len(manager.results) == 1
