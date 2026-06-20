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
    sorted_results = manager.get_results(ranked=True)
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


def test_get_results_ranked_false_returns_insertion_order(folder_ga):
    manager = EnsembleFoldingManager(folder_ga)
    manager.results = [
        {"id": 0, "energy": 5.0},
        {"id": 1, "energy": 2.0},
    ]
    assert manager.get_results(ranked=False)[0]["id"] == 0



# ---------------------------------------------------------------------------
# run_ensemble (with mocked fold)
# ---------------------------------------------------------------------------


def test_run_ensemble_populates_results(folder_ga, zero_structure):
    coords, labels, bonds = zero_structure
    tracker = LandscapeTracker()
    fake_params = np.zeros(folder_ga.n_params)
    fake_result = (coords, labels, bonds, tracker, fake_params, -10.0, [])

    with patch.object(folder_ga, "fold", return_value=fake_result) as mock_fold:
        with patch.object(folder_ga, "get_smart_initialization", return_value=fake_params):
            manager = EnsembleFoldingManager(folder_ga)
            manager.run_ensemble(n_runs=2, max_workers=1)

    assert len(manager.results) == 2
    assert mock_fold.call_count == 2


def test_run_ensemble_result_keys(folder_ga, zero_structure):
    coords, labels, bonds = zero_structure
    tracker = LandscapeTracker()
    fake_params = np.zeros(folder_ga.n_params)
    fake_result = (coords, labels, bonds, tracker, fake_params, -5.0, [])

    with patch.object(folder_ga, "fold", return_value=fake_result):
        with patch.object(folder_ga, "get_smart_initialization", return_value=fake_params):
            manager = EnsembleFoldingManager(folder_ga)
            manager.run_ensemble(n_runs=1, max_workers=1)

    result = manager.results[0]
    for key in ("id", "seed", "energy", "coords", "labels", "bonds", "params", "tracker", "best_snapshots"):
        assert key in result, f"Key '{key}' missing from result dict"


def test_run_ensemble_stores_energy(folder_ga, zero_structure):
    coords, labels, bonds = zero_structure
    tracker = LandscapeTracker()
    fake_params = np.zeros(folder_ga.n_params)
    fake_result = (coords, labels, bonds, tracker, fake_params, 99.5, [])

    with patch.object(folder_ga, "fold", return_value=fake_result):
        with patch.object(folder_ga, "get_smart_initialization", return_value=fake_params):
            manager = EnsembleFoldingManager(folder_ga)
            manager.run_ensemble(n_runs=1, max_workers=1)

    assert manager.results[0]["energy"] == pytest.approx(99.5)


def test_run_ensemble_resets_results_each_call(folder_ga, zero_structure):
    """Calling run_ensemble twice should replace, not append, results."""
    coords, labels, bonds = zero_structure
    tracker = LandscapeTracker()
    fake_params = np.zeros(folder_ga.n_params)
    fake_result = (coords, labels, bonds, tracker, fake_params, 1.0, [])

    with patch.object(folder_ga, "fold", return_value=fake_result):
        with patch.object(folder_ga, "get_smart_initialization", return_value=fake_params):
            manager = EnsembleFoldingManager(folder_ga)
            manager.run_ensemble(n_runs=3, max_workers=1)
            manager.run_ensemble(n_runs=1, max_workers=1)

    assert len(manager.results) == 1


# ---------------------------------------------------------------------------
# run_ensemble: per-replica failure handling (B1)
# ---------------------------------------------------------------------------


def test_run_ensemble_continues_after_failure(folder_ga, zero_structure, tmp_path):
    """A single replica raising must NOT abort the whole ensemble.

    Reproduces the bug originally tracked as B1 in QTF-plan-2.md: a
    `RuntimeError` from `self.folder.fold()` on replica 2 used to bubble
    out of the loop and discard the partial results from replicas 0 and 1.
    After the fix the loop catches the exception, records it in
    `last_error`, and proceeds so `len(m.results) == n_runs - 1`.
    """
    coords, labels, bonds = zero_structure
    tracker = LandscapeTracker()
    fake_params = np.zeros(folder_ga.n_params)
    ok_result = (coords, labels, bonds, tracker, fake_params, -3.0, [])

    # Replica 0 ok, replica 1 ok, replica 2 raises, replica 3 ok.
    # We use n_runs=4 so the loop has to demonstrably continue past the
    # failure rather than just terminating on the last replica.
    side_effects = [ok_result, ok_result, RuntimeError("COBYLA blew up"), ok_result]

    with patch.object(folder_ga, "fold", side_effect=side_effects):
        with patch.object(folder_ga, "get_smart_initialization", return_value=fake_params):
            manager = EnsembleFoldingManager(folder_ga)
            ckpt = tmp_path / "ckpt.json"
            manager.run_ensemble(n_runs=4, checkpoint_path=str(ckpt), max_workers=1)

    # Three of four replicas succeeded
    assert len(manager.results) == 3
    # The failed replica's id (2) is not in the surviving results
    surviving_ids = {r["id"] for r in manager.results}
    assert surviving_ids == {0, 1, 3}
    # last_error is set and is the RuntimeError we injected
    assert isinstance(manager.last_error, RuntimeError)
    assert str(manager.last_error) == "COBYLA blew up"
    # Checkpoint reflects the 3 surviving replicas
    assert ckpt.exists()
    import json as _json
    payload = _json.loads(ckpt.read_text())
    assert payload["sequence"] == folder_ga.sequence
    assert [r["id"] for r in payload["replicas"]] == [0, 1, 3]


def test_run_ensemble_keyboard_interrupt_propagates_after_preserving_results(
    folder_ga, zero_structure, tmp_path
):
    """KeyboardInterrupt / SystemExit must re-raise (Ctrl-C still aborts)
    but partial results must NOT be discarded, and a final checkpoint must
    be written so the user can resume from disk."""
    coords, labels, bonds = zero_structure
    tracker = LandscapeTracker()
    fake_params = np.zeros(folder_ga.n_params)
    ok_result = (coords, labels, bonds, tracker, fake_params, -1.0, [])

    # Replica 0 ok, replica 1 raises KeyboardInterrupt.
    side_effects = [ok_result, KeyboardInterrupt()]

    with patch.object(folder_ga, "fold", side_effect=side_effects):
        with patch.object(folder_ga, "get_smart_initialization", return_value=fake_params):
            manager = EnsembleFoldingManager(folder_ga)
            ckpt = tmp_path / "ckpt.json"
            with pytest.raises(KeyboardInterrupt):
                manager.run_ensemble(n_runs=5, checkpoint_path=str(ckpt), max_workers=1)

    # The first replica's result survived the abort
    assert len(manager.results) == 1
    assert manager.results[0]["id"] == 0
    # last_error is NOT set for user-initiated interrupts (we re-raise
    # rather than swallow), so it remains None from the call's start.
    assert manager.last_error is None
    # A final checkpoint was still written before the exception propagated
    assert ckpt.exists()
    import json as _json
    payload = _json.loads(ckpt.read_text())
    assert [r["id"] for r in payload["replicas"]] == [0]


def test_run_ensemble_last_error_reset_at_start(folder_ga, zero_structure):
    """`last_error` must be reset to None at the beginning of every
    run_ensemble call, so a previously failed run does not leave stale
    state visible to the caller."""
    coords, labels, bonds = zero_structure
    tracker = LandscapeTracker()
    fake_params = np.zeros(folder_ga.n_params)
    ok_result = (coords, labels, bonds, tracker, fake_params, -2.0, [])
    fail = [ok_result, RuntimeError("boom"), ok_result]

    manager = EnsembleFoldingManager(folder_ga)
    with patch.object(folder_ga, "fold", side_effect=fail):
        with patch.object(folder_ga, "get_smart_initialization", return_value=fake_params):
            manager.run_ensemble(n_runs=3, max_workers=1)
    assert isinstance(manager.last_error, RuntimeError)

    # Second call: no failure -> last_error should be cleared
    with patch.object(folder_ga, "fold", return_value=ok_result):
        with patch.object(folder_ga, "get_smart_initialization", return_value=fake_params):
            manager.run_ensemble(n_runs=1, max_workers=1)
    assert manager.last_error is None


def test_run_ensemble_last_error_none_initially(folder_ga):
    """A freshly constructed manager must report `last_error is None`,
    not raise AttributeError."""
    manager = EnsembleFoldingManager(folder_ga)
    assert manager.last_error is None


def test_run_ensemble_checkpoint_is_metadata_only(folder_ga, zero_structure, tmp_path):
    """The checkpoint JSON must NOT contain heavy arrays (coords, params,
    tracker, labels, bonds) — only the metadata needed to resume."""
    coords, labels, bonds = zero_structure
    tracker = LandscapeTracker()
    fake_params = np.zeros(folder_ga.n_params)
    ok_result = (coords, labels, bonds, tracker, fake_params, -7.0, [])

    with patch.object(folder_ga, "fold", return_value=ok_result):
        with patch.object(folder_ga, "get_smart_initialization", return_value=fake_params):
            manager = EnsembleFoldingManager(folder_ga)
            ckpt = tmp_path / "ckpt.json"
            manager.run_ensemble(n_runs=1, checkpoint_path=str(ckpt), max_workers=1)

    payload_text = ckpt.read_text()
    for forbidden in ("\"coords\"", "\"params\"", "\"tracker\"", "\"labels\"", "\"bonds\""):
        assert forbidden not in payload_text, (
            f"checkpoint must be metadata-only; found {forbidden}"
        )
    import json as _json
    payload = _json.loads(payload_text)
    assert payload["replicas"][0]["energy"] == pytest.approx(-7.0)
    assert payload["replicas"][0]["id"] == 0


# ---------------------------------------------------------------------------
# max_workers / parallel path
# ---------------------------------------------------------------------------


def test_run_ensemble_accepts_max_workers(folder_ga, zero_structure):
    """The run_ensemble method must accept a max_workers parameter."""
    coords, labels, bonds = zero_structure
    tracker = LandscapeTracker()
    fake_params = np.zeros(folder_ga.n_params)
    fake_result = (coords, labels, bonds, tracker, fake_params, -10.0, [])

    with patch.object(folder_ga, "fold", return_value=fake_result):
        with patch.object(folder_ga, "get_smart_initialization", return_value=fake_params):
            manager = EnsembleFoldingManager(folder_ga)
            manager.run_ensemble(n_runs=2, max_iter=5, scout_attempts=2, max_workers=1)

    assert len(manager.results) == 2


def test_max_workers_defaults_to_1():
    """The signature must expose max_workers=1 as default (sequential)."""
    import inspect
    sig = inspect.signature(EnsembleFoldingManager.run_ensemble)
    assert sig.parameters["max_workers"].default == 1


def test_worker_exists_and_accepts_expected_args():
    """Verify _run_one_replica signature is correct."""
    import inspect
    from qtf.core.ensemble import _run_one_replica
    sig = inspect.signature(_run_one_replica)
    for param in ("folder_kwargs", "replica_seed", "index", "max_iter", "scout_attempts", "top_k_snapshots"):
        assert param in sig.parameters, f"Missing parameter: {param}"
