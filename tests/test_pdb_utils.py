"""Tests for PDB utility functions (save_pdb, calculate_physics_metrics)."""

import os
import tempfile
import urllib.error
import urllib.request

import numpy as np
import pytest

from qtf.utils.pdb import (
    calculate_physics_metrics,
    get_ground_truth_backbone,
    save_pdb,
)


# ---------------------------------------------------------------------------
# calculate_physics_metrics
# ---------------------------------------------------------------------------


def test_e2e_collinear():
    """Three atoms in a line: end-to-end = 2.0."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    result = calculate_physics_metrics(coords)
    assert result["end_to_end"] == pytest.approx(2.0)


def test_rg_collinear():
    """Three atoms in a line: Rg = sqrt(2/3)."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    result = calculate_physics_metrics(coords)
    expected_rg = np.sqrt(np.mean([1.0, 0.0, 1.0]))
    assert result["radius_of_gyration"] == pytest.approx(expected_rg)


def test_single_atom():
    """A single atom: all three metrics should be zero."""
    coords = np.array([[5.0, 3.0, 1.0]])
    result = calculate_physics_metrics(coords)
    assert result["end_to_end"] == pytest.approx(0.0)
    assert result["radius_of_gyration"] == pytest.approx(0.0)
    assert result["root_mean_square_bond_length"] == pytest.approx(0.0)


def test_e2e_returns_euclidean():
    coords = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    result = calculate_physics_metrics(coords)
    assert result["end_to_end"] == pytest.approx(5.0)


def test_rg_symmetric_square():
    """4 atoms at corners of a square: Rg = distance from corner to centroid."""
    coords = np.array([[1.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0]])
    result = calculate_physics_metrics(coords)
    # centroid = (0,0,0), each distance = sqrt(2)
    assert result["radius_of_gyration"] == pytest.approx(np.sqrt(2.0))


def test_returns_floats():
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    result = calculate_physics_metrics(coords)
    for key in ("end_to_end", "radius_of_gyration", "root_mean_square_bond_length"):
        assert isinstance(result[key], float), f"{key} should be a float"


# ---------------------------------------------------------------------------
# B7: standard Rg definition (regression tests for the textbook formula)
# ---------------------------------------------------------------------------


def test_rg_known_value_tetrahedron():
    """Rg for a regular tetrahedron centred at the origin = sqrt(3).

    The 4 vertices ``(1, 1, 1)``, ``(1, -1, -1)``, ``(-1, 1, -1)``,
    ``(-1, -1, 1)`` are equidistant from the origin with
    ``|r_i|^2 = 3`` for every i, so ``Rg^2 = mean(|r_i|^2) = 3`` and
    ``Rg = sqrt(3)``. This is a hand-computed, parametrization-free
    reference value that pins the textbook Rg definition against
    regressions to the older ``np.diff``-based implementation.
    """
    coords = np.array([
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ])
    result = calculate_physics_metrics(coords)
    assert result["radius_of_gyration"] == pytest.approx(np.sqrt(3.0))


def test_rg_matches_textbook_definition():
    """Rg must equal ``sqrt(mean(|r_i - centroid|^2))`` exactly."""
    rng = np.random.default_rng(seed=20240605)
    coords = rng.normal(loc=(1.0, -2.0, 0.5), scale=2.5, size=(37, 3))
    result = calculate_physics_metrics(coords)
    centroid = coords.mean(axis=0)
    expected = float(np.sqrt(np.mean(np.sum((coords - centroid) ** 2, axis=1))))
    assert result["radius_of_gyration"] == pytest.approx(expected)


def test_rg_is_not_rms_bond_length():
    """Sanity: the textbook Rg must not collapse to the RMS bond length.

    A 3-point chain ``(0,0,0) -> (1,0,0) -> (0,2,0)`` has centroid
    ``(1/3, 2/3, 0)`` and squared distances to the centroid
    ``5/9, 8/9, 17/9`` summing to ``30/9 = 10/3``, so
    ``Rg = sqrt(10/3) / 3 = sqrt(10)/3``. The RMS bond length is
    ``sqrt((1^2 + 5^2) / 2) = sqrt(13/2)``. The two values differ,
    and the standard Rg is the smaller one. This guards against a
    regression where the function silently reverts to the buggy
    ``np.diff`` formula.
    """
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    result = calculate_physics_metrics(coords)
    expected_rg = np.sqrt(10.0) / 3.0
    rms_bond = np.sqrt((1.0 ** 2 + np.sqrt(5.0) ** 2) / 2.0)
    assert result["radius_of_gyration"] == pytest.approx(expected_rg)
    assert not np.isclose(result["radius_of_gyration"], rms_bond)
    # The dict exposes both quantities, so the two should be reported
    # separately rather than collapsed.
    assert result["root_mean_square_bond_length"] == pytest.approx(rms_bond)


def test_keys_contract():
    """The function exposes the three expected keys."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    result = calculate_physics_metrics(coords)
    assert set(result.keys()) == {
        "end_to_end",
        "radius_of_gyration",
        "root_mean_square_bond_length",
    }


# ---------------------------------------------------------------------------
# save_pdb (B5: unified signature; sequence=1-letter string, resnames
# optional 3-letter mapping)
# ---------------------------------------------------------------------------


def test_save_pdb_creates_file():
    coords = np.array([[0.0, 0.0, 0.0], [1.46, 0.0, 0.0]])
    labels = [(0, "N", "N"), (0, "CA", "C")]
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
        fname = f.name
    try:
        save_pdb(coords, labels, filename=fname, energy=-42.0, sequence="A")
        assert os.path.exists(fname)
    finally:
        os.unlink(fname)


def test_save_pdb_remark_contains_energy():
    coords = np.array([[0.0, 0.0, 0.0], [1.46, 0.0, 0.0]])
    labels = [(0, "N", "N"), (0, "CA", "C")]
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as f:
        fname = f.name
    try:
        save_pdb(coords, labels, filename=fname, energy=-42.0, sequence="A")
        with open(fname) as fh:
            content = fh.read()
        assert "REMARK" in content
        assert "-42.000" in content
    finally:
        os.unlink(fname)


def test_save_pdb_atom_lines():
    coords = np.array([[0.0, 0.0, 0.0], [1.46, 0.0, 0.0], [2.0, 1.0, 0.0]])
    labels = [(0, "N", "N"), (0, "CA", "C"), (0, "C", "C")]
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
        fname = f.name
    try:
        save_pdb(coords, labels, filename=fname, sequence="G")
        with open(fname) as fh:
            atom_lines = [line for line in fh if line.startswith("ATOM")]
        assert len(atom_lines) == 3
    finally:
        os.unlink(fname)


def test_save_pdb_atom_names_in_output():
    coords = np.array([[0.0, 0.0, 0.0], [1.46, 0.0, 0.0]])
    labels = [(0, "N", "N"), (0, "CA", "C")]
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
        fname = f.name
    try:
        save_pdb(coords, labels, filename=fname, sequence="A")
        with open(fname) as fh:
            content = fh.read()
        assert "N" in content
        assert "CA" in content
    finally:
        os.unlink(fname)


def test_save_pdb_default_energy_zero():
    coords = np.array([[0.0, 0.0, 0.0]])
    labels = [(0, "CA", "C")]
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
        fname = f.name
    try:
        save_pdb(coords, labels, filename=fname, sequence="G")
        with open(fname) as fh:
            content = fh.read()
        assert "0.000" in content
    finally:
        os.unlink(fname)


def test_save_pdb_end_record():
    """A well-formed PDB must terminate with an `END` record (the
    folder-method behavior that has now been unified in the free
    function)."""
    coords = np.array([[0.0, 0.0, 0.0]])
    labels = [(0, "CA", "C")]
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
        fname = f.name
    try:
        save_pdb(coords, labels, filename=fname, sequence="G")
        with open(fname) as fh:
            lines = [line.rstrip("\n") for line in fh if line.strip()]
        assert lines[-1] == "END"
    finally:
        os.unlink(fname)


def test_save_pdb_include_hydrogens_false_filters():
    """`include_hydrogens=False` must drop H atoms."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    labels = [(0, "CA", "C"), (0, "H", "H")]
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
        fname = f.name
    try:
        save_pdb(coords, labels, filename=fname, sequence="G", include_hydrogens=False)
        with open(fname) as fh:
            atom_lines = [line for line in fh if line.startswith("ATOM")]
        assert len(atom_lines) == 1
    finally:
        os.unlink(fname)


def test_save_pdb_resnames_override_sequence():
    """When the caller supplies `resnames`, that mapping wins over
    the `sequence` fallback."""
    coords = np.array([[0.0, 0.0, 0.0]])
    labels = [(0, "CA", "C")]
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
        fname = f.name
    try:
        # Sequence says A (ALA) but resnames says MSE (selenomethionine)
        save_pdb(
            coords, labels, filename=fname,
            sequence="A", resnames={0: "MSE"},
        )
        with open(fname) as fh:
            content = fh.read()
        assert "MSE" in content
        assert " ALA " not in content
    finally:
        os.unlink(fname)


# ---------------------------------------------------------------------------
# get_ground_truth_backbone (B6: User-Agent, timeout, retries, HTTPS-only)
# ---------------------------------------------------------------------------


def _write_minimal_pdb(path: str, n_ca: int = 5) -> None:
    """Write a minimal PDB file with ``n_ca`` CA atoms and a few
    header lines, suitable for being parsed by
    ``get_ground_truth_backbone`` without hitting the network."""
    lines = [
        "HEADER    MINIMAL TEST PDB                             01-JAN-00   TEST",
        "TITLE     A SHORT TEST PDB FOR GET_GROUND_TRUTH_BACKBONE",
        # Pad the file to > _MIN_VALID_PDB_BYTES so the size validator
        # in _download_pdb does not reject it.  200 bytes is the floor.
        "REMARK   1 THIS IS A FAKE PDB USED ONLY BY THE QTF UNIT TESTS.",
        "REMARK   2 IT IS NOT BIOLOGICALLY MEANINGFUL.",
    ]
    for i in range(n_ca):
        serial = i + 1
        x, y, z = float(i), 0.0, 0.0
        lines.append(
            f"ATOM  {serial:5d}  CA  ALA A{serial:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
    lines.append("END\n")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def test_get_ground_truth_backbone_uses_cache(tmp_path, monkeypatch):
    """If a cached PDB file already exists, the network must NOT be
    touched. B6 also ensures the cache file is loaded even if it
    was downloaded by a previous version of QTF."""
    cache_file = tmp_path / "1CRN.pdb"
    _write_minimal_pdb(str(cache_file), n_ca=3)

    def _explode(*_a, **_k):
        raise AssertionError("urlopen must not be called when the cache is warm")

    monkeypatch.setattr("qtf.utils.pdb.urllib.request.urlopen", _explode)

    coords = get_ground_truth_backbone("1CRN", cache_dir=str(tmp_path))
    assert coords.shape == (3, 3)


def test_get_ground_truth_backbone_user_agent_and_timeout(tmp_path, monkeypatch):
    """The download request must:
      * include a `User-Agent` header (B6: RCSB returns 403 without one),
      * pass a finite `timeout=` to `urlopen` (B6: default timeout is
        unbounded, hangs for minutes on rate-limited responses).
    """
    captured: dict = {}

    class _FakeResponse:
        def __init__(self, body: bytes):
            self._body = body

        def read(self) -> bytes:
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    def _fake_urlopen(req, timeout=None):
        captured["req"] = req
        captured["timeout"] = timeout
        # Build a body large enough to pass the size validator
        # (>=200 bytes) and that contains at least one CA atom line.
        body = (
            b"REMARK   1 FAKE PDB USED ONLY BY QTF UNIT TESTS, NOT BIOLOGICALLY MEANINGFUL.\n"
            b"REMARK   2 PAD TO PASS THE 200-BYTE SIZE VALIDATOR.  PAD.  PAD.  PAD.  PAD.  PAD.\n"
            b"ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        )
        return _FakeResponse(body)

    monkeypatch.setattr("qtf.utils.pdb.urllib.request.urlopen", _fake_urlopen)

    coords = get_ground_truth_backbone("1ABC", cache_dir=str(tmp_path))

    # The Request was built with a User-Agent
    assert "User-agent" in captured["req"].headers
    assert captured["req"].headers["User-agent"].startswith("QTF/")
    # The urlopen call received a finite timeout
    assert captured["timeout"] is not None
    assert captured["timeout"] > 0
    # The URL is HTTPS only
    assert captured["req"].full_url.startswith("https://")
    # And the parser saw the single CA atom we wrote
    assert coords.shape == (1, 3)


def test_get_ground_truth_backbone_retries_on_503(tmp_path, monkeypatch):
    """A 503 Service Unavailable response must be retried. After the
    retry budget is exhausted, the function raises a RuntimeError
    (not an urllib.error.HTTPError, so callers can catch a single
    exception type regardless of cause)."""
    call_count = {"n": 0}
    sleeps: list[float] = []

    good_body = (
        b"REMARK   1 FAKE PDB USED ONLY BY QTF UNIT TESTS, NOT BIOLOGICALLY MEANINGFUL.\n"
        b"REMARK   2 PAD TO PASS THE 200-BYTE SIZE VALIDATOR.  PAD.  PAD.  PAD.  PAD.  PAD.\n"
        b"ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
    )

    class _FakeResponse:
        def __init__(self, body: bytes):
            self._body = body

        def read(self) -> bytes:
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    def _fake_urlopen(req, timeout=None):
        call_count["n"] += 1
        if call_count["n"] < 3:
            # First two attempts: 503 Service Unavailable
            raise urllib.error.HTTPError(
                req.full_url, 503, "Service Unavailable", {}, None
            )
        # Third attempt: success
        return _FakeResponse(good_body)

    # Short-circuit the sleep so the test does not take 3+ seconds.
    monkeypatch.setattr("qtf.utils.pdb.time.sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr("qtf.utils.pdb.urllib.request.urlopen", _fake_urlopen)

    coords = get_ground_truth_backbone("2GB1", cache_dir=str(tmp_path))

    assert call_count["n"] == 3, f"expected 3 attempts, got {call_count['n']}"
    # The retry sleep was the documented exponential backoff
    assert sleeps[:2] == [0.5, 1.0]
    assert coords.shape == (1, 3)


def test_get_ground_truth_backbone_eventually_fails_after_503_exhausted(tmp_path, monkeypatch):
    """If the server keeps returning 503, the function must give up
    after the retry budget and raise a RuntimeError, not a raw
    urllib HTTPError."""
    monkeypatch.setattr("qtf.utils.pdb.time.sleep", lambda _s: None)

    def _fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 503, "Service Unavailable", {}, None
        )

    monkeypatch.setattr("qtf.utils.pdb.urllib.request.urlopen", _fake_urlopen)

    with pytest.raises(RuntimeError, match="503"):
        get_ground_truth_backbone("5AWL", cache_dir=str(tmp_path))


def test_get_ground_truth_backbone_rejects_http_url(monkeypatch):
    """B6: only HTTPS is allowed. A future refactor that flips the
    URL template to http:// must surface a hard error rather than
    silently leaking credentials or accepting MITM."""
    from qtf.utils import pdb as qtf_pdb
    monkeypatch.setattr(
        qtf_pdb, "_PDB_DOWNLOAD_URL_TEMPLATE", "http://files.rcsb.org/download/{pdb_id}.pdb"
    )

    # Force a download by ensuring the cache misses. We point cache_dir
    # at a fresh tmp dir.
    import tempfile as _tempfile
    with _tempfile.TemporaryDirectory() as cache_dir:
        with pytest.raises(ValueError, match="HTTPS"):
            get_ground_truth_backbone("5AWL", cache_dir=cache_dir)


def test_get_ground_truth_backbone_detects_truncation(tmp_path, monkeypatch):
    """If the server returns an empty or suspiciously small body
    (e.g. an HTML error page that snuck through a redirect), the
    function must raise a RuntimeError and NOT cache the bad file."""

    class _FakeResponse:
        def __init__(self, body: bytes):
            self._body = body

        def read(self) -> bytes:
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    def _fake_urlopen(req, timeout=None):
        # 50 bytes is below the 200-byte floor.
        return _FakeResponse(b"too short\n")

    monkeypatch.setattr("qtf.utils.pdb.urllib.request.urlopen", _fake_urlopen)
    monkeypatch.setattr("qtf.utils.pdb.time.sleep", lambda _s: None)

    target = tmp_path / "5AWL.pdb"
    with pytest.raises(RuntimeError, match="truncated"):
        get_ground_truth_backbone("5AWL", cache_dir=str(tmp_path))
    # The bad response must NOT have been cached.
    assert not target.exists()


def test_get_ground_truth_backbone_retries_on_urlerror(tmp_path, monkeypatch):
    """Transient URLErrors (e.g. DNS hiccup) must be retried with
    the same exponential backoff as HTTP 5xx."""

    good_body = (
        b"REMARK   1 FAKE PDB USED ONLY BY QTF UNIT TESTS, NOT BIOLOGICALLY MEANINGFUL.\n"
        b"REMARK   2 PAD TO PASS THE 200-BYTE SIZE VALIDATOR.  PAD.  PAD.  PAD.  PAD.  PAD.\n"
        b"ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
    )

    class _FakeResponse:
        def __init__(self, body: bytes):
            self._body = body

        def read(self) -> bytes:
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    call_count = {"n": 0}
    sleeps: list[float] = []

    def _fake_urlopen(req, timeout=None):
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise urllib.error.URLError("simulated DNS failure")
        return _FakeResponse(good_body)

    monkeypatch.setattr("qtf.utils.pdb.time.sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr("qtf.utils.pdb.urllib.request.urlopen", _fake_urlopen)

    coords = get_ground_truth_backbone("6LYZ", cache_dir=str(tmp_path))

    assert call_count["n"] == 2
    assert sleeps[:1] == [0.5]
    assert coords.shape == (1, 3)


def test_get_ground_truth_backbone_creates_cache_dir(tmp_path, monkeypatch):
    """`cache_dir` may not exist yet; the function must create it
    rather than crash with FileNotFoundError."""
    fresh = tmp_path / "deep" / "nested" / "cache"
    assert not fresh.exists()

    good_body = (
        b"REMARK   1 FAKE PDB USED ONLY BY QTF UNIT TESTS, NOT BIOLOGICALLY MEANINGFUL.\n"
        b"REMARK   2 PAD TO PASS THE 200-BYTE SIZE VALIDATOR.  PAD.  PAD.  PAD.  PAD.  PAD.\n"
        b"ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
    )

    class _FakeResponse:
        def __init__(self, body: bytes):
            self._body = body

        def read(self) -> bytes:
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    monkeypatch.setattr(
        "qtf.utils.pdb.urllib.request.urlopen",
        lambda req, timeout=None: _FakeResponse(good_body),
    )
    coords = get_ground_truth_backbone("7XYZ", cache_dir=str(fresh))
    assert coords.shape == (1, 3)
    assert fresh.is_dir()
    assert (fresh / "7XYZ.pdb").exists()


# ---------------------------------------------------------------------------
# pdb_id_from_path
# ---------------------------------------------------------------------------


class TestPdbIdFromPath:
    """Verify that pdb_id_from_path extracts valid 4-char PDB IDs."""

    def test_valid_pdb_id_from_path(self):
        from qtf.utils.workflow import pdb_id_from_path
        assert pdb_id_from_path("/data/pdbs/5awl.pdb") == "5AWL"

    def test_valid_pdb_id_uppercased(self):
        from qtf.utils.workflow import pdb_id_from_path
        assert pdb_id_from_path("/data/pdbs/5awL.pdb") == "5AWL"

    def test_valid_pdb_id_no_ext(self):
        from qtf.utils.workflow import pdb_id_from_path
        assert pdb_id_from_path("1XYZ") == "1XYZ"

    def test_none_returns_empty(self):
        from qtf.utils.workflow import pdb_id_from_path
        assert pdb_id_from_path(None) == ""

    def test_invalid_stem_returns_empty(self):
        from qtf.utils.workflow import pdb_id_from_path
        assert pdb_id_from_path("/data/pdbs/my_protein.pdb") == ""

    def test_pdb_id_starts_with_zero_returns_empty(self):
        from qtf.utils.workflow import pdb_id_from_path
        assert pdb_id_from_path("0ABC.pdb") == ""

    def test_short_stem_returns_empty(self):
        from qtf.utils.workflow import pdb_id_from_path
        assert pdb_id_from_path("5A.pdb") == ""

    def test_long_stem_returns_empty(self):
        from qtf.utils.workflow import pdb_id_from_path
        assert pdb_id_from_path("5ABCD.pdb") == ""

    def test_local_pdb_path_with_subdirs(self):
        from qtf.utils.workflow import pdb_id_from_path
        assert pdb_id_from_path(
            "/home/user/experimental_structures/pdb_files/6LYT.pdb"
        ) == "6LYT"
