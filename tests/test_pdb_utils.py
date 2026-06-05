"""Tests for PDB utility functions (save_pdb, calculate_physics_metrics)."""

import os
import tempfile

import numpy as np
import pytest

from qtf.utils.pdb import calculate_physics_metrics, save_pdb


# ---------------------------------------------------------------------------
# calculate_physics_metrics
# ---------------------------------------------------------------------------


def test_e2e_collinear():
    """Three atoms in a line: end-to-end = 2.0."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    e2e, rg = calculate_physics_metrics(coords)
    assert e2e == pytest.approx(2.0)


def test_rg_collinear():
    """Three atoms in a line: Rg = sqrt(2/3)."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    e2e, rg = calculate_physics_metrics(coords)
    expected_rg = np.sqrt(np.mean([1.0, 0.0, 1.0]))
    assert rg == pytest.approx(expected_rg)


def test_single_atom():
    """A single atom: both metrics should be zero."""
    coords = np.array([[5.0, 3.0, 1.0]])
    e2e, rg = calculate_physics_metrics(coords)
    assert e2e == pytest.approx(0.0)
    assert rg == pytest.approx(0.0)


def test_e2e_returns_euclidean():
    coords = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    e2e, _ = calculate_physics_metrics(coords)
    assert e2e == pytest.approx(5.0)


def test_rg_symmetric_square():
    """4 atoms at corners of a square: Rg = distance from corner to centroid."""
    coords = np.array([[1.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0]])
    _, rg = calculate_physics_metrics(coords)
    # centroid = (0,0,0), each distance = sqrt(2)
    assert rg == pytest.approx(np.sqrt(2.0))


def test_returns_floats():
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    e2e, rg = calculate_physics_metrics(coords)
    assert isinstance(e2e, float)
    assert isinstance(rg, float)


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
