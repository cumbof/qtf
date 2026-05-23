"""Tests for QuantumBiophysicsFolder."""

import numpy as np
import pytest

from qtf.core.folder import QuantumBiophysicsFolder


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestInit:
    def test_sequence_uppercased(self):
        f = QuantumBiophysicsFolder("ga")
        assert f.sequence == "GA"

    def test_n_residues(self, folder_ga):
        assert folder_ga.n_residues == 2

    def test_default_force_field(self, folder_ga):
        assert folder_ga.force_field == "charmm"

    def test_custom_force_field_amber(self):
        f = QuantumBiophysicsFolder("A", force_field="amber")
        assert f.force_field == "amber"

    def test_custom_force_field_opls(self):
        f = QuantumBiophysicsFolder("A", force_field="opls")
        assert f.force_field == "opls"

    def test_n_qubits_at_least_two(self, folder_ga):
        assert folder_ga.n_qubits >= 2

    def test_n_params_positive(self, folder_ga):
        assert folder_ga.n_params > 0

    def test_total_angles_positive(self, folder_ga):
        assert folder_ga.total_angles > 0

    def test_dof_map_has_phi_and_psi(self, folder_ga):
        types = {d["type"] for d in folder_ga.dof_map}
        assert "phi" in types
        assert "psi" in types

    def test_dof_map_length_matches_total_angles(self, folder_ga):
        assert len(folder_ga.dof_map) == folder_ga.total_angles

    def test_cache_initialized(self, folder_ga):
        assert folder_ga._cache_initialized is True

    def test_ansatz_parameters_match_n_params(self, folder_ga):
        assert folder_ga.ansatz.num_parameters == folder_ga.n_params

    def test_single_residue_folder(self):
        f = QuantumBiophysicsFolder("G")
        assert f.n_residues == 1
        assert f.total_angles >= 2  # at least phi and psi


# ---------------------------------------------------------------------------
# _build_charges
# ---------------------------------------------------------------------------


class TestBuildCharges:
    def test_charmm_backbone_N(self):
        charges = QuantumBiophysicsFolder._build_charges("charmm")
        assert charges["N"] == pytest.approx(-0.47)

    def test_amber_backbone_N(self):
        charges = QuantumBiophysicsFolder._build_charges("amber")
        assert charges["N"] == pytest.approx(-0.42)

    def test_opls_backbone_N(self):
        charges = QuantumBiophysicsFolder._build_charges("opls")
        assert charges["N"] == pytest.approx(-0.50)

    def test_unknown_ff_falls_back_to_charmm(self):
        charges_unknown = QuantumBiophysicsFolder._build_charges("unknown_ff")
        charges_charmm = QuantumBiophysicsFolder._build_charges("charmm")
        assert charges_unknown == charges_charmm

    def test_common_charge_oxt(self):
        charges = QuantumBiophysicsFolder._build_charges("charmm")
        assert charges["OXT"] == pytest.approx(-1.0)

    def test_common_charge_nz(self):
        charges = QuantumBiophysicsFolder._build_charges("amber")
        assert charges["NZ"] == pytest.approx(1.0)

    def test_all_ffs_contain_backbone_atoms(self):
        for ff in ("charmm", "amber", "opls"):
            charges = QuantumBiophysicsFolder._build_charges(ff)
            for atom in ("N", "CA", "C", "O"):
                assert atom in charges, f"{atom} missing in {ff}"


# ---------------------------------------------------------------------------
# _nerf_step
# ---------------------------------------------------------------------------


class TestNerfStep:
    # Use non-collinear points to avoid a degenerate cross product in NERF
    _A = np.array([0.0, 0.0, 0.0])
    _B = np.array([1.0, 0.0, 0.0])
    _C = np.array([1.5, 1.0, 0.0])

    def test_output_shape(self):
        result = QuantumBiophysicsFolder._nerf_step(self._A, self._B, self._C, 1.5, np.pi / 2, 0.0)
        assert result.shape == (3,)

    def test_bond_length_respected(self):
        """The placed atom should be ~bond_len away from c."""
        bond_len = 1.5
        result = QuantumBiophysicsFolder._nerf_step(self._A, self._B, self._C, bond_len, np.pi / 2, 0.0)
        dist = np.linalg.norm(result - self._C)
        assert dist == pytest.approx(bond_len, rel=1e-3)

    def test_different_torsions_give_different_positions(self):
        r1 = QuantumBiophysicsFolder._nerf_step(self._A, self._B, self._C, 1.5, 2.0, 0.0)
        r2 = QuantumBiophysicsFolder._nerf_step(self._A, self._B, self._C, 1.5, 2.0, np.pi / 2)
        assert not np.allclose(r1, r2)

    def test_returns_ndarray(self):
        result = QuantumBiophysicsFolder._nerf_step(self._A, self._B, self._C, 1.46, 1.9, 0.0)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# build_full_structure
# ---------------------------------------------------------------------------


class TestBuildFullStructure:
    def test_returns_three_values(self, folder_ga):
        out = folder_ga.build_full_structure(np.zeros(folder_ga.total_angles))
        assert len(out) == 3

    def test_coords_shape(self, folder_ga):
        coords, labels, bonds = folder_ga.build_full_structure(np.zeros(folder_ga.total_angles))
        assert coords.ndim == 2
        assert coords.shape[1] == 3
        assert len(coords) == len(labels)

    def test_labels_contain_ca(self, folder_ga):
        _, labels, _ = folder_ga.build_full_structure(np.zeros(folder_ga.total_angles))
        atom_names = [lbl[1] for lbl in labels]
        assert "CA" in atom_names

    def test_labels_contain_backbone(self, folder_ga):
        _, labels, _ = folder_ga.build_full_structure(np.zeros(folder_ga.total_angles))
        atom_names = {lbl[1] for lbl in labels}
        assert {"N", "CA", "C"}.issubset(atom_names)

    def test_bonds_are_pairs(self, folder_ga):
        _, _, bonds = folder_ga.build_full_structure(np.zeros(folder_ga.total_angles))
        for bond in bonds:
            assert len(bond) == 2

    def test_n_ca_atoms_equals_n_residues(self, folder_ga):
        _, labels, _ = folder_ga.build_full_structure(np.zeros(folder_ga.total_angles))
        ca_count = sum(1 for lbl in labels if lbl[1] == "CA")
        assert ca_count == folder_ga.n_residues

    def test_different_angles_give_different_structures(self, folder_ga):
        coords0, _, _ = folder_ga.build_full_structure(np.zeros(folder_ga.total_angles))
        coords1, _, _ = folder_ga.build_full_structure(np.ones(folder_ga.total_angles))
        # At least some coordinates should differ
        assert not np.allclose(coords0, coords1)


# ---------------------------------------------------------------------------
# energy_function
# ---------------------------------------------------------------------------


class TestEnergyFunction:
    def test_returns_finite_float(self, folder_ga):
        params = np.zeros(folder_ga.n_params)
        result = folder_ga.energy_function(params)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_different_params_finite(self, folder_ga):
        params = np.ones(folder_ga.n_params) * 0.5
        result = folder_ga.energy_function(params)
        assert np.isfinite(result)

    def test_tracker_logs_when_set(self, folder_ga):
        from qtf.core.tracker import LandscapeTracker

        tracker = LandscapeTracker()
        folder_ga.tracker = tracker
        folder_ga.energy_function(np.zeros(folder_ga.n_params))
        assert len(tracker.history) >= 1
        folder_ga.tracker = None  # reset


# ---------------------------------------------------------------------------
# get_smart_initialization
# ---------------------------------------------------------------------------


class TestGetSmartInitialization:
    def test_returns_correct_shape(self, folder_ga):
        params = folder_ga.get_smart_initialization(n_attempts=3, seed=42)
        assert params.shape == (folder_ga.n_params,)

    def test_reproducible_with_same_seed(self, folder_ga):
        p1 = folder_ga.get_smart_initialization(n_attempts=3, seed=42)
        p2 = folder_ga.get_smart_initialization(n_attempts=3, seed=42)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds_differ(self, folder_ga):
        p1 = folder_ga.get_smart_initialization(n_attempts=3, seed=1)
        p2 = folder_ga.get_smart_initialization(n_attempts=3, seed=999)
        assert not np.array_equal(p1, p2)

    def test_returns_ndarray(self, folder_ga):
        params = folder_ga.get_smart_initialization(n_attempts=2, seed=0)
        assert isinstance(params, np.ndarray)


# ---------------------------------------------------------------------------
# Residue-name conventions (regression tests for the 3-letter vs 1-letter bug)
# ---------------------------------------------------------------------------


class TestResidueNameConvention:
    """The folder stores ``self.sequence`` as single-letter codes; several
    energy terms used to compare it against three-letter codes (``"VAL"``,
    ``"PRO"``, ``"PHE"`` …) which silently returned 0 for every input.

    These tests pin the convention and guard against regression.
    """

    def test_rotamer_energy_nonzero_for_VIT(self):
        f = QuantumBiophysicsFolder("V")
        # chi1 in the gauche+ basin (~-1.047 rad) should give a strong negative
        angle_dict = {"0_chi1": -1.047}
        e = f._calculate_rotamer_energy(angle_dict)
        assert e < -1.0, f"expected strong negative rotamer energy, got {e}"

    def test_rotamer_energy_nonzero_for_aromatic(self):
        f = QuantumBiophysicsFolder("F")
        angle_dict = {"0_chi1": -1.047}
        e = f._calculate_rotamer_energy(angle_dict)
        assert e < 0.0

    def test_rotamer_energy_nonzero_for_proline(self):
        f = QuantumBiophysicsFolder("P")
        # PRO uses a quadratic penalty around ±0.5 rad
        angle_dict = {"0_chi1": 2.0}
        e = f._calculate_rotamer_energy(angle_dict)
        assert e > 0.0

    def test_rotamer_default_branch_for_generic_residue(self):
        # Glycine has no chi1 (no entry in angle_dict) → energy stays 0
        f = QuantumBiophysicsFolder("A")
        angle_dict = {"0_chi1": 0.0}
        e = f._calculate_rotamer_energy(angle_dict)
        # The generic branch is 1.0 * (1 + cos(3*chi)) = 2.0 at chi=0
        assert e == pytest.approx(2.0)

    def test_hydrophobic_mask_nonempty_for_hydrophobic_sequence(self):
        f = QuantumBiophysicsFolder("VVV")
        # With the bug, mask_hydrophobic was all-False for any sequence
        assert f.mask_hydrophobic.sum() > 0

    def test_hydrophobic_mask_empty_for_polar_sequence(self):
        f = QuantumBiophysicsFolder("DDD")
        # D (Asp) is not in the hydrophobic set
        assert f.mask_hydrophobic.sum() == 0
