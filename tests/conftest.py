"""Shared pytest fixtures for the QTF test suite."""

import numpy as np
import pytest

from qtf.core.folder import QuantumBiophysicsFolder
from qtf.core.tracker import LandscapeTracker


@pytest.fixture(scope="session")
def folder_ga():
    """QuantumBiophysicsFolder for the 2-residue sequence 'GA' (fast to run)."""
    return QuantumBiophysicsFolder("GA")


@pytest.fixture(scope="session")
def zero_structure(folder_ga):
    """3-D structure built from an all-zero angle vector."""
    coords, labels, bonds = folder_ga.build_full_structure(np.zeros(folder_ga.total_angles))
    return coords, labels, bonds


@pytest.fixture(scope="session")
def results_factory(folder_ga, zero_structure):
    """Return a callable that creates a list of fake ensemble result dicts."""
    coords, labels, bonds = zero_structure

    def _make(energies):
        results = []
        for i, energy in enumerate(energies):
            tracker = LandscapeTracker()
            tracker.log(float(energy))
            tracker.mark_stage("Stage1")
            results.append(
                {
                    "id": i,
                    "seed": 100 + i,
                    "energy": float(energy),
                    "coords": coords.copy(),
                    "labels": labels,
                    "bonds": bonds,
                    "params": np.zeros(folder_ga.n_params),
                    "tracker": tracker,
                }
            )
        return results

    return _make


@pytest.fixture(scope="session")
def two_results(results_factory):
    """Two fake replica results with energies [5.0, 3.0]."""
    return results_factory([5.0, 3.0])
