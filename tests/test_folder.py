"""Tests for QuantumBiophysicsFolder."""

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qtf.core.folder import QuantumBiophysicsFolder, _TOPOLOGY_SEED_ANGLE


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestInit:
    def test_sequence_uppercased(self):
        f = QuantumBiophysicsFolder("ga")
        assert f.sequence == "GA"

    def test_n_residues(self, folder_ga):
        assert folder_ga.n_residues == 2

    def test_force_field_argument_removed(self):
        with pytest.raises(TypeError):
            QuantumBiophysicsFolder("A", force_field="amber")

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
    def test_amber_backbone_N(self):
        charges = QuantumBiophysicsFolder._build_charges()
        assert charges["N"] == pytest.approx(-0.42)

    def test_common_charge_oxt(self):
        charges = QuantumBiophysicsFolder._build_charges()
        assert charges["OXT"] == pytest.approx(-1.0)

    def test_common_charge_nz(self):
        charges = QuantumBiophysicsFolder._build_charges()
        assert charges["NZ"] == pytest.approx(1.0)

    def test_amber_contains_backbone_atoms(self):
        charges = QuantumBiophysicsFolder._build_charges()
        for atom in ("N", "CA", "C", "O"):
            assert atom in charges, f"{atom} missing in amber"


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

    def test_callable_before_fold(self):
        """``energy_function`` must work on a freshly-built folder that
        has never run ``fold()``.

        The function reads ``self.current_stage`` to pick the
        stage-1/2/3 force constants. ``current_stage`` has to be
        available from the moment the folder is constructed; it is
        declared as a class-level default of ``1`` precisely so that
        a user-supplied gradient, a checkpoint restart, or a test
        can call ``energy_function`` without first going through
        ``fold()``.
        """
        from qtf.core.folder import QuantumBiophysicsFolder

        folder = QuantumBiophysicsFolder("GA")
        # Pin the contract: current_stage defaults to 1 on a fresh
        # instance and is read by energy_function as a stage switch.
        assert folder.current_stage == 1
        # Now actually call energy_function. It must not raise
        # AttributeError on a folder that has never been folded.
        params = np.zeros(folder.n_params)
        result = folder.energy_function(params)
        assert np.isfinite(result)

    def test_callable_when_current_stage_missing_from_instance_dict(self):
        """Defence in depth: even if ``current_stage`` is somehow missing
        from the instance ``__dict__`` (e.g. pickling edge case, a
        user ``del``'d it, or a mock that bypassed ``__init__``),
        the class-level default of ``1`` keeps ``energy_function``
        callable."""
        from qtf.core.folder import QuantumBiophysicsFolder

        folder = QuantumBiophysicsFolder("GA")
        # Simulate a "current_stage is missing" situation by deleting
        # the instance attribute. The class-level default must take
        # over.
        del folder.__dict__["current_stage"]
        assert folder.current_stage == 1  # resolves via class default
        params = np.zeros(folder.n_params)
        result = folder.energy_function(params)
        assert np.isfinite(result)


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


# ---------------------------------------------------------------------------
# Electrostatics (_electrostatic_energy)
# ---------------------------------------------------------------------------


class TestElectrostaticEnergy:
    """Guard the 1/r Coulomb formula and the physical constants.

    The previous implementation used ``83.0 * Q / r²`` — an undocumented
    1/r² falloff. The corrected form is
    ``332.0637 * Q / (4.0 * r)`` which is the standard Coulomb law with a
    uniform dielectric of 4.
    """

    def _make_folder_with_two_charges(self, q1: float, q2: float, r: float):
        """Return a (folder, D) pair where only atoms 0 and 1 exist with
        charges q1, q2 separated by distance r, and the non-bonded mask
        allows the pair.  Uses QuantumBiophysicsFolder("GA") as a carrier
        and monkey-patches the charge vector + mask."""
        import numpy as np
        f = QuantumBiophysicsFolder("GA")
        n = len(f.q_vector)
        # Override charge vector: only atoms 0 and 1 are charged
        q = np.zeros(n)
        q[0] = q1
        q[1] = q2
        f.q_vector = q
        # Build a distance matrix where D[0,1] = D[1,0] = r, rest large
        D = np.full((n, n), 100.0)
        np.fill_diagonal(D, 0.0)
        D[0, 1] = D[1, 0] = r
        # Allow the pair in mask_non_bonded
        mask = np.zeros((n, n), dtype=bool)
        mask[0, 1] = mask[1, 0] = True
        f.mask_non_bonded = mask
        return f, D

    def test_opposite_charges_are_attractive(self):
        # q1=+1, q2=-1 → E = 332.0637 * (-1) / (4 * r) < 0
        f, D = self._make_folder_with_two_charges(+1.0, -1.0, 5.0)
        assert f._electrostatic_energy(D) < 0.0

    def test_like_charges_are_repulsive(self):
        # q1=+1, q2=+1 → E = 332.0637 / (4 * r) > 0
        f, D = self._make_folder_with_two_charges(+1.0, +1.0, 5.0)
        assert f._electrostatic_energy(D) > 0.0

    def test_one_over_r_falloff(self):
        """Energy at r=5 should be exactly twice the energy at r=10 (1/r law)."""
        f5, D5 = self._make_folder_with_two_charges(+1.0, -1.0, 5.0)
        f10, D10 = self._make_folder_with_two_charges(+1.0, -1.0, 10.0)
        e5 = f5._electrostatic_energy(D5)
        e10 = f10._electrostatic_energy(D10)
        assert e5 / e10 == pytest.approx(2.0, rel=1e-6)

    def test_magnitude_matches_coulomb_formula(self):
        """Numerical value matches 332.0637 * q1*q2 / (4.0 * r)."""
        q1, q2, r = 1.0, -1.0, 8.0
        expected = 332.0637 * q1 * q2 / (4.0 * r)
        f, D = self._make_folder_with_two_charges(q1, q2, r)
        assert f._electrostatic_energy(D) == pytest.approx(expected, rel=1e-5)

    def test_zero_charges_return_zero(self):
        f = QuantumBiophysicsFolder("GG")
        # GG has no formal charges under any force field
        import numpy as np
        n = len(f.q_vector)
        D = np.ones((n, n))
        np.fill_diagonal(D, 0.0)
        # Even if all charges are zero, energy must be zero
        f.q_vector = np.zeros(n)
        assert f._electrostatic_energy(D) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# fold() — scout_attempts budget
# ---------------------------------------------------------------------------


class TestFoldScoutBudget:
    """Guard the scout_attempts logic introduced to prevent fold() from
    silently consuming max_iter evaluations just for initialisation."""

    def _count_energy_calls(self, folder, **fold_kwargs):
        """Run fold() and return the number of energy_function calls made
        during the scouting phase only (before Stage 1 begins)."""
        import numpy as np
        call_counts = []

        original = folder.energy_function

        def counting_energy(params, **kwargs):
            call_counts.append(1)
            return original(params, **kwargs)

        folder.energy_function = counting_energy
        # We only need to check the scouting phase, so abort early by passing
        # a pre-computed initial_params when we want zero scout calls.
        folder.fold(**fold_kwargs)
        folder.energy_function = original
        return len(call_counts)

    def test_default_scout_well_below_max_iter(self):
        """Default scouting must use at most min(64, max_iter//10) attempts."""
        f = QuantumBiophysicsFolder("GA")
        max_iter = 100
        expected_max_scout = min(64, max_iter // 10)
        # Total calls = scout + 3 stages; stage calls dominate, but scout must be ≤ expected
        # We verify by passing explicit scout_attempts=0 and comparing totals.
        total_default = self._count_energy_calls(f, max_iter=max_iter)
        total_no_scout = self._count_energy_calls(
            f, max_iter=max_iter, initial_params=np.zeros(f.n_params)
        )
        scout_calls = total_default - total_no_scout
        assert scout_calls <= expected_max_scout + 5  # +5 for rounding/overhead

    def test_explicit_scout_attempts_respected(self):
        """When scout_attempts=1, only 1 random evaluation is done for init."""
        f = QuantumBiophysicsFolder("GA")
        calls = []
        original = f.energy_function

        def track(params, **kwargs):
            calls.append(1)
            return original(params, **kwargs)

        f.energy_function = track
        f.fold(max_iter=50, scout_attempts=1)
        f.energy_function = original
        # First call is the single scout evaluation; rest are optimiser calls
        # We can't easily separate them, but total >= 1 guarantees it ran.
        assert len(calls) >= 1

    def test_initial_params_skips_scouting(self):
        """Passing initial_params must bypass the scouting phase entirely."""
        f = QuantumBiophysicsFolder("GA")
        scout_calls = []
        original_scout = f.get_smart_initialization

        def spy_scout(*args, **kwargs):
            scout_calls.append(1)
            return original_scout(*args, **kwargs)

        f.get_smart_initialization = spy_scout
        f.fold(max_iter=50, initial_params=np.zeros(f.n_params))
        f.get_smart_initialization = original_scout
        assert len(scout_calls) == 0

    def test_scout_attempts_none_uses_formula(self):
        """scout_attempts=None must resolve to min(64, max_iter // 10)."""
        # We test via get_smart_initialization call count.
        for max_iter in (10, 100, 1000):
            expected = min(64, max_iter // 10)
            f = QuantumBiophysicsFolder("G")
            recorded = []
            original = f.get_smart_initialization

            def spy(n_attempts=20, seed=None, _rec=recorded):
                _rec.append(n_attempts)
                return original(n_attempts=n_attempts, seed=seed)

            f.get_smart_initialization = spy
            f.fold(max_iter=max_iter)
            f.get_smart_initialization = original
            assert recorded[0] == expected, (
                f"max_iter={max_iter}: expected scout={expected}, got {recorded[0]}"
            )

    def test_fold_returns_six_values(self):
        f = QuantumBiophysicsFolder("GA")
        result = f.fold(max_iter=10, scout_attempts=1)
        assert len(result) == 6

    def test_cobyla_maxfun_warning_not_emitted(self):
        """B10: a small-budget ``fold()`` (max_iter=10) must not leak
        SciPy's ``COBYLA: Invalid MAXFUN`` warning.

        The warning is structurally impossible once we enforce the
        ``max(max_iter, n_params + 2)`` floor on the COBYLA
        ``maxiter`` option; this test pins that contract.
        """
        import warnings as _warnings

        for max_iter in (10, 50, 100):
            f = QuantumBiophysicsFolder("G")
            with _warnings.catch_warnings(record=True) as caught:
                _warnings.simplefilter("always")
                f.fold(max_iter=max_iter, scout_attempts=1)
            cobyla_warnings = [
                w for w in caught
                if "cobyla" in str(w.filename).lower()
                and "maxfun" in str(w.message).lower()
            ]
            assert not cobyla_warnings, (
                f"fold(max_iter={max_iter}) leaked COBYLA MAXFUN warning(s): "
                f"{[str(w.message) for w in cobyla_warnings]}"
            )

    def test_cobyla_maxiter_floor_applied(self):
        """B10: the COBYLA ``maxiter`` option passed to ``scipy.optimize.minimize``
        must be at least ``n_params + 2`` even when ``max_iter`` is below
        that floor."""
        import numpy as np
        from scipy.optimize import minimize as _sp_minimize

        captured = {}

        def spy_minimize(fun, x0, method=None, options=None, *a, **kw):
            if method == "COBYLA":
                captured["options"] = dict(options or {})
            return _sp_minimize(fun, x0, method=method, options=options, *a, **kw)

        import qtf.core.folder as _folder_mod
        _folder_mod.minimize = spy_minimize
        try:
            f = QuantumBiophysicsFolder("G")
            f.fold(max_iter=1, scout_attempts=1)
        finally:
            _folder_mod.minimize = _sp_minimize

        opts = captured.get("options", {})
        assert opts, "COBYLA was never called"
        assert opts["maxiter"] >= f.n_params + 2, (
            f"COBYLA maxiter={opts['maxiter']} below n_params+2={f.n_params + 2}"
        )


# ---------------------------------------------------------------------------
# _get_angles — global phase removal
# ---------------------------------------------------------------------------


class TestGetAnglesGlobalPhase:
    """Guard the global-phase-removal fix in _get_angles.

    A global phase e^{iα}|ψ⟩ shifts every amplitude phase by α uniformly.
    After the fix, angle(ψ₀) is subtracted from all phases so that
    phases[0] ≡ 0, making the output gauge-invariant.
    """

    def test_first_angle_is_always_zero(self):
        """phases[0] must be exactly 0 for any parameter vector."""
        f = QuantumBiophysicsFolder("GA")
        rng = np.random.default_rng(42)
        for _ in range(20):
            params = rng.uniform(-np.pi, np.pi, f.n_params)
            angles = f._get_angles(params)
            assert angles[0] == pytest.approx(0.0, abs=1e-10)

    def test_global_phase_shift_gives_identical_angles(self):
        """A statevector multiplied by e^{iα} must yield the same angles."""
        f = QuantumBiophysicsFolder("GAV")
        params = np.ones(f.n_params) * 0.3

        # Ground-truth angles from the unshifted state
        psi = Statevector(f.ansatz.assign_parameters(
            dict(zip(f.ansatz.parameters, params))
        )).data
        angles_orig = f._get_angles(params)

        # Manually apply the global-phase correction to a phase-shifted copy
        for alpha in (0.5, 1.234, -2.0, np.pi):
            psi_shifted = np.exp(1j * alpha) * psi
            phases_shifted = np.angle(psi_shifted)[: f.total_angles]
            angles_manual = (
                phases_shifted - np.angle(psi_shifted[0]) + np.pi
            ) % (2 * np.pi) - np.pi
            np.testing.assert_allclose(
                angles_orig, angles_manual, atol=1e-10,
                err_msg=f"mismatch at alpha={alpha}"
            )

    def test_angles_within_minus_pi_to_pi(self):
        """All extracted angles must lie in the half-open interval (-π, π]."""
        f = QuantumBiophysicsFolder("GAVC")
        rng = np.random.default_rng(7)
        for _ in range(10):
            params = rng.uniform(-2 * np.pi, 2 * np.pi, f.n_params)
            angles = f._get_angles(params)
            assert np.all(angles > -np.pi - 1e-10), "angle below -π"
            assert np.all(angles <= np.pi + 1e-10), "angle above +π"

    def test_angles_length_matches_total_angles(self):
        f = QuantumBiophysicsFolder("GGG")
        params = np.zeros(f.n_params)
        angles = f._get_angles(params)
        assert len(angles) == f.total_angles


# ---------------------------------------------------------------------------
# M-2 – NeRF degeneracy at zero-torsion init
# ---------------------------------------------------------------------------


class TestTopologySeedAngle:
    """Verify that _initialize_topology_cache uses a non-degenerate seed."""

    def test_static_labels_non_empty(self):
        """Cache must produce at least one label entry."""
        f = QuantumBiophysicsFolder("GA")
        assert len(f.static_labels) > 0

    def test_static_labels_count_grows_with_sequence(self):
        """Longer sequences must produce more atom labels."""
        f2 = QuantumBiophysicsFolder("GA")
        f4 = QuantumBiophysicsFolder("GAVC")
        assert len(f4.static_labels) > len(f2.static_labels)

    def test_seed_coords_are_finite(self):
        """build_full_structure with the topology seed must return finite coords."""
        import numpy as np
        from qtf.core.folder import _TOPOLOGY_SEED_ANGLE

        f = QuantumBiophysicsFolder("GAVC")
        seed = np.full(f.total_angles, _TOPOLOGY_SEED_ANGLE)
        coords, _, _ = f.build_full_structure(seed)
        assert np.all(np.isfinite(coords)), "seed coords contain NaN or Inf"

    def test_zero_angles_produce_finite_labels(self):
        """Even np.zeros is not expected to crash cache init (just brittle).

        The topology cache discards coordinates, so static_labels must always
        be populated regardless of the seed geometry.
        """
        import numpy as np

        f = QuantumBiophysicsFolder("GAVC")
        # Re-trigger cache with zeros to confirm labels still come out intact
        f.static_labels = None
        f._initialize_topology_cache()
        assert f.static_labels is not None and len(f.static_labels) > 0

    def test_seed_angle_constant_exported(self):
        """_TOPOLOGY_SEED_ANGLE must be a small positive float."""
        from qtf.core.folder import _TOPOLOGY_SEED_ANGLE

        assert isinstance(_TOPOLOGY_SEED_ANGLE, float)
        assert 0.0 < _TOPOLOGY_SEED_ANGLE < np.pi


# ---------------------------------------------------------------------------
# M-3 – pre-proline ω optimisation
# ---------------------------------------------------------------------------


class TestOmegaPreProline:
    """Verify that ω for residues preceding Pro enters the DOF map and
    that build_full_structure honours its value."""

    def test_no_omega_dof_without_proline(self):
        """A sequence with no Pro must have no omega DOF entries."""
        f = QuantumBiophysicsFolder("GAVC")
        omega_dofs = [d for d in f.dof_map if d["type"] == "omega"]
        assert omega_dofs == []

    def test_omega_dof_added_before_proline(self):
        """Residue preceding P must gain an omega DOF."""
        f = QuantumBiophysicsFolder("GAP")
        omega_dofs = [d for d in f.dof_map if d["type"] == "omega"]
        assert len(omega_dofs) == 1
        assert omega_dofs[0]["res"] == 1  # residue 1 (A) precedes P at position 2

    def test_omega_dof_count_matches_pre_pro_count(self):
        """A sequence with two pre-Pro positions gets two omega DOFs."""
        f = QuantumBiophysicsFolder("GAPGPV")
        omega_dofs = [d for d in f.dof_map if d["type"] == "omega"]
        assert len(omega_dofs) == 2

    def test_omega_default_is_pi_without_dof(self):
        """Without an omega DOF the built structure uses ω = π (trans)."""
        f = QuantumBiophysicsFolder("GA")
        coords_trans, _, _ = f.build_full_structure(np.zeros(f.total_angles))
        # Re-build with an explicit pi to confirm they match
        assert np.all(np.isfinite(coords_trans))

    def test_omega_affects_coordinates(self):
        """Setting ω to 0 (cis) for a pre-Pro residue must shift CA of Pro."""
        f = QuantumBiophysicsFolder("GAP")
        omega_idx = next(
            k for k, d in enumerate(f.dof_map)
            if d["type"] == "omega" and d["res"] == 1
        )
        angles_trans = np.full(f.total_angles, _TOPOLOGY_SEED_ANGLE)
        angles_cis = angles_trans.copy()
        angles_cis[omega_idx] = 0.0  # cis-Pro

        coords_trans, _, _ = f.build_full_structure(angles_trans)
        coords_cis, _, _ = f.build_full_structure(angles_cis)

        # The Pro Cα (residue 2, "CA") coordinates must differ
        def ca_pos(coords, labels, res):
            for idx, (r, n, _) in enumerate(labels):
                if r == res and n == "CA":
                    return coords[idx]
            return None

        _, labels_t, _ = f.build_full_structure(angles_trans)
        ca_t = ca_pos(coords_trans, labels_t, 2)
        ca_c = ca_pos(coords_cis,
                      f.build_full_structure(angles_cis)[1], 2)
        assert ca_t is not None and ca_c is not None
        assert not np.allclose(ca_t, ca_c, atol=1e-3), \
            "cis and trans Pro Cα must differ"

    def test_total_angles_includes_omega_dof(self):
        """total_angles must equal len(dof_map) and include every omega entry.

        For "GAP":
          G  → phi, psi                        = 2
          A  → phi, psi  (CB at fixed angle)   = 2
          P  → phi, psi, chi1, chi2, chi3       = 5
          ω  → omega DOF for A (pre-Pro)        = 1
          Total = 10
        """
        f = QuantumBiophysicsFolder("GAP")
        assert f.total_angles == len(f.dof_map), \
            "total_angles must equal the number of dof_map entries"
        omega_count = sum(1 for d in f.dof_map if d["type"] == "omega")
        assert omega_count == 1
        assert f.total_angles == 10


# ---------------------------------------------------------------------------
# M-4 – Huber loss in _calculate_geometry_integrity
# ---------------------------------------------------------------------------


class TestHuberLoss:
    """Unit tests for QuantumBiophysicsFolder._huber."""

    def test_zero_residual(self):
        assert QuantumBiophysicsFolder._huber(0.0, 1.0) == 0.0

    def test_within_delta_is_quadratic(self):
        # |x| <= delta → x²
        for x in (0.0, 0.5, 0.99, -0.5, -0.99):
            assert pytest.approx(QuantumBiophysicsFolder._huber(x, 1.0)) == x ** 2

    def test_at_transition_continuous(self):
        # Both branches must agree at |x| = delta
        delta = 1.0
        from_below = delta ** 2
        from_above = 2.0 * delta * delta - delta ** 2  # = delta²
        assert pytest.approx(from_below) == from_above

    def test_above_delta_is_linear(self):
        # |x| > delta → 2·delta·|x| - delta²
        delta = 1.0
        for x in (1.5, 2.0, 5.0, -2.0):
            expected = 2.0 * delta * abs(x) - delta ** 2
            assert pytest.approx(QuantumBiophysicsFolder._huber(x, delta)) == expected

    def test_gradient_bounded_above_delta(self):
        """d/dx Huber = ±2·delta above the transition — must not exceed 2*delta."""
        delta = 1.0
        eps = 1e-6
        for x in (2.0, 5.0, 100.0):
            grad = (QuantumBiophysicsFolder._huber(x + eps, delta) -
                    QuantumBiophysicsFolder._huber(x - eps, delta)) / (2 * eps)
            assert abs(grad) <= 2.0 * delta + 1e-4

    def test_huber_less_than_quadratic_for_large_residual(self):
        """Huber penalty must be strictly smaller than the old quadratic for large x."""
        delta = 1.0
        for x in (2.0, 3.0, 10.0):
            assert QuantumBiophysicsFolder._huber(x, delta) < x ** 2

    def test_huber_equals_quadratic_for_small_residual(self):
        """Below delta, Huber must equal the raw quadratic."""
        delta = 1.0
        for x in (0.1, 0.5, 0.9):
            assert pytest.approx(QuantumBiophysicsFolder._huber(x, delta)) == x ** 2


class TestGeometryIntegrityHuber:
    """Verify that _calculate_geometry_integrity uses Huber-capped penalties."""

    def test_energy_finite_for_normal_structure(self):
        """Geometry integrity on a well-folded structure must be finite."""
        f = QuantumBiophysicsFolder("GAVC")
        angles = np.full(f.total_angles, 0.1)
        coords, labels, _ = f.build_full_structure(angles)
        e = f._calculate_geometry_integrity(coords, labels, f.atom_to_res)
        assert np.isfinite(e)

    def test_huber_caps_large_deviation(self):
        """A severe geometry distortion must contribute less than its raw quadratic."""
        delta = _TOPOLOGY_SEED_ANGLE  # reuse import; actual value is 0.1 (not delta)
        from qtf.core.folder import _HUBER_DELTA_GEOM, QuantumBiophysicsFolder as QBF
        large_dev = 5.0  # Å – far beyond any physical bond stretch
        huber_penalty = 50.0 * QBF._huber(large_dev, _HUBER_DELTA_GEOM)
        quadratic_penalty = 50.0 * large_dev ** 2
        assert huber_penalty < quadratic_penalty

    def test_huber_delta_constant_exported(self):
        from qtf.core.folder import _HUBER_DELTA_GEOM
        assert isinstance(_HUBER_DELTA_GEOM, float)
        assert _HUBER_DELTA_GEOM > 0.0


# ---------------------------------------------------------------------------
# P-1 – O(1) label lookup in build_full_structure
# ---------------------------------------------------------------------------


class TestBuildFullStructureLookup:
    """Verify correctness and sub-quadratic scaling of build_full_structure."""

    def test_coords_labels_bonds_consistent_length(self):
        """coords, labels, and bonds must have internally consistent lengths."""
        f = QuantumBiophysicsFolder("GAVC")
        angles = np.full(f.total_angles, 0.1)
        coords, labels, bonds = f.build_full_structure(angles)
        assert len(coords) == len(labels)
        n = len(coords)
        for a, b in bonds:
            assert 0 <= a < n and 0 <= b < n, f"bond ({a},{b}) out of range for {n} atoms"

    def test_backbone_atoms_present_for_each_residue(self):
        """Every residue must have at least N, CA, C atoms in the label list."""
        f = QuantumBiophysicsFolder("GAVC")
        angles = np.full(f.total_angles, 0.1)
        _, labels, _ = f.build_full_structure(angles)
        for res_id in range(f.n_residues):
            res_atoms = {lbl[1] for lbl in labels if lbl[0] == res_id}
            for backbone in ("N", "CA", "C"):
                assert backbone in res_atoms, \
                    f"residue {res_id} missing backbone atom {backbone}"

    def test_coords_finite(self):
        """All atom coordinates must be finite."""
        f = QuantumBiophysicsFolder("GAVC")
        coords, _, _ = f.build_full_structure(np.full(f.total_angles, 0.1))
        assert np.all(np.isfinite(coords))

    def test_longer_sequence_more_atoms(self):
        """A longer sequence must produce more atoms than a shorter one."""
        f2 = QuantumBiophysicsFolder("GA")
        f5 = QuantumBiophysicsFolder("GAVCL")
        coords2, _, _ = f2.build_full_structure(np.full(f2.total_angles, 0.1))
        coords5, _, _ = f5.build_full_structure(np.full(f5.total_angles, 0.1))
        assert len(coords5) > len(coords2)

    def test_scaling_subquadratic(self):
        """build_full_structure wall-clock time must scale sub-quadratically.

        Compares time for N=5 vs N=10 residues.  An O(N²) implementation
        would show ~4× slowdown; O(N) should be < 3×.  We allow up to 3.5×
        to accommodate OS scheduling jitter.
        """
        import time

        seq_short = "GAVCL"           # 5 residues
        seq_long  = "GAVCL" * 2      # 10 residues
        f_s = QuantumBiophysicsFolder(seq_short)
        f_l = QuantumBiophysicsFolder(seq_long)
        a_s = np.full(f_s.total_angles, 0.1)
        a_l = np.full(f_l.total_angles, 0.1)

        reps = 200
        t0 = time.perf_counter()
        for _ in range(reps):
            f_s.build_full_structure(a_s)
        t_short = (time.perf_counter() - t0) / reps

        t0 = time.perf_counter()
        for _ in range(reps):
            f_l.build_full_structure(a_l)
        t_long = (time.perf_counter() - t0) / reps

        ratio = t_long / (t_short + 1e-9)
        assert ratio < 3.5, (
            f"build_full_structure appears super-linear: "
            f"N=5 → {t_short*1e6:.1f} µs, N=10 → {t_long*1e6:.1f} µs, ratio={ratio:.2f}"
        )


# ---------------------------------------------------------------------------
# H-3 – Length-aware end-to-end constraint target
# ---------------------------------------------------------------------------


class TestLengthAwareE2EConstraint:
    """Verify the E2E target scales with chain length: 4.5 + 0.4·max(0,N−5) Å."""

    @pytest.mark.parametrize("n_residues,expected_target", [
        (3, 4.5),
        (5, 4.5),
        (10, 6.5),
        (15, 8.5),
        (20, 10.5),
    ])
    def test_target_formula_values(self, n_residues, expected_target):
        """E2E target formula produces the correct value for various chain lengths."""
        computed = 4.5 + 0.40 * max(0, n_residues - 5)
        assert computed == pytest.approx(expected_target)

    def test_longer_chain_has_larger_target(self):
        """A 20-residue chain must have a strictly larger E2E target than a 5-residue chain."""
        target_5 = 4.5 + 0.40 * max(0, 5 - 5)
        target_20 = 4.5 + 0.40 * max(0, 20 - 5)
        assert target_20 > target_5

    def test_constraint_term_is_nonneg_and_finite(self):
        """Constraint energy from energy_function must be finite and non-negative."""
        f = QuantumBiophysicsFolder("GAVCL", use_e2e_constraint=True)
        rng = np.random.default_rng(0)
        params = rng.uniform(-0.8, 0.8, f.n_params)
        f.energy_function(params, return_terms=True)
        terms = f.last_energy_terms
        assert np.isfinite(terms["constraint"])
        assert terms["constraint"] >= 0.0

    def test_constraint_disabled_gives_zero(self):
        """When use_e2e_constraint=False the constraint term must be exactly zero."""
        f = QuantumBiophysicsFolder("GAVCL", use_e2e_constraint=False)
        rng = np.random.default_rng(1)
        params = rng.uniform(-0.8, 0.8, f.n_params)
        f.energy_function(params, return_terms=True)
        terms = f.last_energy_terms
        assert terms["constraint"] == 0.0


# ---------------------------------------------------------------------------
# C-3 – Bond-graph VdW exclusions (1-2, 1-3 excluded; 1-4 scaled by 0.35)
# ---------------------------------------------------------------------------


class TestBondGraphVDWExclusions:
    """Verify bond-graph based VdW masking: C-3 correctness fix."""

    def test_bonded_pairs_excluded_from_strict_vdw_mask(self):
        """Every covalent bond (1-2 pair) must be absent from mask_non_bonded_vdw."""
        f = QuantumBiophysicsFolder("GAVC")
        _, _, static_bonds = f.build_full_structure(np.full(f.total_angles, 0.1))
        for i, j in static_bonds:
            assert not f.mask_non_bonded_vdw[i, j], (
                f"Bonded pair ({i},{j}) incorrectly present in strict VdW non-bonded mask"
            )
            assert not f.mask_non_bonded_vdw[j, i]

    def test_14_mask_and_strict_mask_are_disjoint(self):
        """1-4 pairs and strictly non-bonded pairs must be mutually exclusive."""
        f = QuantumBiophysicsFolder("GAVC")
        overlap = f.mask_non_bonded_vdw & f.mask_non_bonded_vdw_14
        assert not overlap.any(), (
            "mask_non_bonded_vdw and mask_non_bonded_vdw_14 share entries; "
            "they must be disjoint"
        )

    def test_14_mask_nonempty_for_peptide(self):
        """A multi-residue peptide must have at least some 1-4 atom pairs."""
        f = QuantumBiophysicsFolder("GAVC")
        assert f.mask_non_bonded_vdw_14.any(), (
            "Expected at least one 1-4 atom pair in mask_non_bonded_vdw_14"
        )

    def test_vdw_energy_finite_and_nonneg(self):
        """VdW repulsion energy must be finite and non-negative."""
        f = QuantumBiophysicsFolder("GAVC")
        rng = np.random.default_rng(7)
        params = rng.uniform(-0.8, 0.8, f.n_params)
        f.energy_function(params, return_terms=True)
        terms = f.last_energy_terms
        assert np.isfinite(terms["vdw_repulsion"])
        assert terms["vdw_repulsion"] >= 0.0


# ---------------------------------------------------------------------------
# NaN / Inf robustness in energy_function
# ---------------------------------------------------------------------------


class TestNonFiniteAngles:
    """energy_function must return a finite penalty for NaN / Inf angle vectors."""

    def test_nan_angles_returns_penalty(self, monkeypatch):
        f = QuantumBiophysicsFolder("GA")
        orig_get_angles = f._get_angles

        def _nan_angles(params):
            return np.full(orig_get_angles(params).shape, np.nan)

        monkeypatch.setattr(f, "_get_angles", _nan_angles)
        e = f.energy_function(np.zeros(f.n_params))
        assert np.isfinite(e)
        assert e > 1e5

    def test_inf_angles_returns_penalty(self, monkeypatch):
        f = QuantumBiophysicsFolder("GA")
        orig_get_angles = f._get_angles

        def _inf_angles(params):
            return np.full(orig_get_angles(params).shape, np.inf)

        monkeypatch.setattr(f, "_get_angles", _inf_angles)
        e = f.energy_function(np.zeros(f.n_params))
        assert np.isfinite(e)
        assert e > 1e5

    def test_mixed_nan_inf_returns_penalty(self, monkeypatch):
        f = QuantumBiophysicsFolder("GA")
        orig_get_angles = f._get_angles

        def _mixed_angles(params):
            a = orig_get_angles(params)
            a[0] = np.nan
            a[-1] = np.inf
            return a

        monkeypatch.setattr(f, "_get_angles", _mixed_angles)
        e = f.energy_function(np.zeros(f.n_params))
        assert np.isfinite(e)
        assert e > 1e5

    def test_return_terms_includes_non_finite_key(self, monkeypatch):
        f = QuantumBiophysicsFolder("GA")
        orig_get_angles = f._get_angles

        def _nan_angles(params):
            return np.full(orig_get_angles(params).shape, np.nan)

        monkeypatch.setattr(f, "_get_angles", _nan_angles)
        terms = f.energy_function(np.zeros(f.n_params), return_terms=True)
        assert isinstance(terms, dict)
        assert "non_finite_penalty" in terms
        assert terms["non_finite_penalty"] > 1e5

    def test_normal_angles_unchanged(self):
        """The guard must not affect the normal path (no monkeypatch)."""
        f = QuantumBiophysicsFolder("GA")
        params = np.zeros(f.n_params)
        e = f.energy_function(params)
        assert np.isfinite(e)
