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


def test_get_ranked_results_emits_deprecation_warning(folder_ga):
    manager = EnsembleFoldingManager(folder_ga)
    manager.results = [{"id": 0, "energy": 1.0}]
    with pytest.warns(DeprecationWarning, match="get_results.*ranked=True"):
        _ = manager.get_ranked_results()


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
    ok_result = (coords, labels, bonds, tracker, fake_params, -3.0)

    # Replica 0 ok, replica 1 ok, replica 2 raises, replica 3 ok.
    # We use n_runs=4 so the loop has to demonstrably continue past the
    # failure rather than just terminating on the last replica.
    side_effects = [ok_result, ok_result, RuntimeError("COBYLA blew up"), ok_result]

    with patch.object(folder_ga, "fold", side_effect=side_effects):
        with patch.object(folder_ga, "get_smart_initialization", return_value=fake_params):
            manager = EnsembleFoldingManager(folder_ga)
            ckpt = tmp_path / "ckpt.json"
            manager.run_ensemble(n_runs=4, checkpoint_path=str(ckpt))

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
    ok_result = (coords, labels, bonds, tracker, fake_params, -1.0)

    # Replica 0 ok, replica 1 raises KeyboardInterrupt.
    side_effects = [ok_result, KeyboardInterrupt()]

    with patch.object(folder_ga, "fold", side_effect=side_effects):
        with patch.object(folder_ga, "get_smart_initialization", return_value=fake_params):
            manager = EnsembleFoldingManager(folder_ga)
            ckpt = tmp_path / "ckpt.json"
            with pytest.raises(KeyboardInterrupt):
                manager.run_ensemble(n_runs=5, checkpoint_path=str(ckpt))

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
    ok_result = (coords, labels, bonds, tracker, fake_params, -2.0)
    fail = [ok_result, RuntimeError("boom"), ok_result]

    manager = EnsembleFoldingManager(folder_ga)
    with patch.object(folder_ga, "fold", side_effect=fail):
        with patch.object(folder_ga, "get_smart_initialization", return_value=fake_params):
            manager.run_ensemble(n_runs=3)
    assert isinstance(manager.last_error, RuntimeError)

    # Second call: no failure -> last_error should be cleared
    with patch.object(folder_ga, "fold", return_value=ok_result):
        with patch.object(folder_ga, "get_smart_initialization", return_value=fake_params):
            manager.run_ensemble(n_runs=1)
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
    ok_result = (coords, labels, bonds, tracker, fake_params, -7.0)

    with patch.object(folder_ga, "fold", return_value=ok_result):
        with patch.object(folder_ga, "get_smart_initialization", return_value=fake_params):
            manager = EnsembleFoldingManager(folder_ga)
            ckpt = tmp_path / "ckpt.json"
            manager.run_ensemble(n_runs=1, checkpoint_path=str(ckpt))

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
# prime_circuit: idempotency and logging (B2)
# ---------------------------------------------------------------------------


def test_prime_circuit_is_idempotent_when_circuit_parameters_set(folder_ga):
    """When folder.circuit_parameters is set, a second prime_circuit call
    with overwrite=False must return the SAME array object — it must not
    silently destroy the user's prior work, and it must not re-run COBYLA.

    This is the regression test for B2 (QTF-plan-2.md): prior to the fix,
    a second call would re-run the priming optimisation and overwrite
    whatever had been stored on the folder.
    """
    from unittest.mock import patch as _patch

    manager = EnsembleFoldingManager(folder_ga)
    preset = np.linspace(-0.5, 0.5, folder_ga.n_params)
    had_attr_before = hasattr(folder_ga, "circuit_parameters")
    prev_value = getattr(folder_ga, "circuit_parameters", None)
    folder_ga.circuit_parameters = preset
    try:
        with _patch("qtf.core.ensemble.minimize") as mock_min:
            out1 = manager.prime_circuit(target_type="helix", seed=7)
            out2 = manager.prime_circuit(target_type="helix", seed=7)

        # COBYLA was NOT invoked (the guard short-circuited)
        assert mock_min.call_count == 0
        # Both calls returned the exact same object the user set
        assert out1 is preset
        assert out2 is preset
        # And the folder's stored value is still the user's array
        assert folder_ga.circuit_parameters is preset
    finally:
        # Restore the session-scoped fixture so we do not leak state to
        # later tests in the session.
        if had_attr_before:
            folder_ga.circuit_parameters = prev_value
        else:
            del folder_ga.circuit_parameters


def test_prime_circuit_overwrite_true_reprimes(folder_ga):
    """When overwrite=True, the guard must NOT fire and the priming
    optimisation must run, returning a fresh array (not the preset one)."""
    from unittest.mock import patch as _patch

    manager = EnsembleFoldingManager(folder_ga)
    preset = np.full(folder_ga.n_params, 99.0)  # sentinel value
    had_attr_before = hasattr(folder_ga, "circuit_parameters")
    prev_value = getattr(folder_ga, "circuit_parameters", None)
    folder_ga.circuit_parameters = preset
    try:
        # Patch minimize so we don't actually run COBYLA, and so we can
        # control the return value.
        from scipy.optimize import OptimizeResult
        fresh = np.zeros(folder_ga.n_params)
        fake_res = OptimizeResult(x=fresh, fun=0.123)
        with _patch("qtf.core.ensemble.minimize", return_value=fake_res) as mock_min:
            out = manager.prime_circuit(target_type="helix", seed=11, overwrite=True)

        assert mock_min.call_count == 1
        assert out is fresh              # a new array, not the preset
        assert out is not preset
    finally:
        if had_attr_before:
            folder_ga.circuit_parameters = prev_value
        else:
            del folder_ga.circuit_parameters


def test_prime_circuit_uses_logger_not_print(folder_ga, capsys):
    """prime_circuit must not write to stdout. Both the banner and the
    priming-error line used to be `print(...)` calls (B2) — after the
    fix they go through `logger.info`, so stdout stays empty in the
    default logging configuration.

    We also assert that the messages are reachable via the standard
    logging facility, by capturing the 'qtf.core.ensemble' logger at
    INFO level.
    """
    import logging as _logging
    from qtf.core.folder import QuantumBiophysicsFolder

    # Use a fresh folder (session-scoped folder_ga may have a stray
    # circuit_parameters set by an earlier test in this class).
    fresh_folder = QuantumBiophysicsFolder("GA")
    manager = EnsembleFoldingManager(fresh_folder)

    # Capture logging output from this module only
    records = []

    class _ListHandler(_logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _ListHandler(level=_logging.INFO)
    logger_under_test = _logging.getLogger("qtf.core.ensemble")
    logger_under_test.addHandler(handler)
    prev_level = logger_under_test.level
    logger_under_test.setLevel(_logging.INFO)
    try:
        out = manager.prime_circuit(target_type="helix", seed=1)
    finally:
        logger_under_test.removeHandler(handler)
        logger_under_test.setLevel(prev_level)

    captured = capsys.readouterr()
    assert captured.out == "", f"prime_circuit must not print; got {captured.out!r}"
    assert captured.err == ""

    # And the two log records are present with the right content
    msgs = [r.getMessage() for r in records]
    assert any("PRIMING CIRCUIT" in m for m in msgs), msgs
    assert any("Priming Error" in m for m in msgs), msgs

    # Sanity: it actually produced a parameter vector
    assert out.shape == (fresh_folder.n_params,)


def test_prime_circuit_default_overwrite_is_false(folder_ga):
    """The signature must expose `overwrite=False` as the default so the
    idempotency behaviour is opt-out (not opt-in)."""
    import inspect as _inspect
    sig = _inspect.signature(EnsembleFoldingManager.prime_circuit)
    assert "overwrite" in sig.parameters
    assert sig.parameters["overwrite"].default is False
