"""QuantumBiophysicsFolder — hybrid quantum-classical protein structure predictor.

Architecture
------------
1. **Quantum Actor**: a parameterised Qiskit circuit whose statevector or
   sampled readout encodes backbone/side-chain torsion angles.
2. **Classical Critic**: a physics-based energy function (hydrophobicity, H-bonds,
   electrostatics, sterics, Ramachandran bias, geometry integrity).
3. **Optimisation Loop**: COBYLA + SLSQP in three progressive stages (collapse →
   refine → relax).

References
----------
* Kyte & Doolittle (1982) hydrophobicity scale.
* PHEAT coarse atom-name charge profiles for native scoring; exact force fields are external scorers.
* Bondi (1964) van der Waals radii.
* Engh & Huber (1991) bond/angle parameters.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import numpy as np
from pheat import Atom, HeavyAtomStructure, ResidueGeometry, ResidueGeometryStructure, write_pdb
from pheat.models import RESIDUE_GEOMETRY_BACKBONE_LENGTHS
from pheat.residue_geometry import (
    ANGLE_CA_C_N,
    ANGLE_N_CA_C,
    ANGLE_UNITS,
    CA_C,
    CA_CB,
    C_N,
    C_O,
    N_CA,
    PCA_N_CD,
    PRO_N_CD,
    PYL_CA2_CG2,
    residue_angle_specs,
    structure_from_residue_geometry,
)
from pheat.residues import SIDECHAIN_STEPS, one_to_three
from pheat.roundtrip import normalize_max_chi, normalize_stored_angles
from qiskit import transpile
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

from qtf.core.circuits import build_circuit
from qtf.core.tracker import LandscapeTracker
from qtf.scoring import canonical_score_model, is_qtf_score_model, score_classic_folder, score_pheat_structure

logger = logging.getLogger(__name__)

# Coulomb's law prefactor: 332.0637 kcal mol⁻¹ Å e⁻²
# (charges in elementary charges, distances in Ångströms, energy in kcal/mol)
_COULOMB_PREFACTOR: float = 332.0637
# Uniform implicit-solvent dielectric constant (ε ≈ 4 is the standard choice
# for buried/intermediate environments in coarse force fields; see e.g.
# Warshel & Russell, Q Rev Biophys 1984).
_DIELECTRIC: float = 4.0


# Seed angle (rad) used when calling build_full_structure inside
# _initialize_topology_cache.  All-zero torsions place every backbone atom
# along the same line (collinear), zeroing the cross products in _nerf_step
# and making the reference frame degenerate.  A small non-zero value avoids
# this without affecting the cache output, which discards coordinates and
# only retains atom labels.
_TOPOLOGY_SEED_ANGLE: float = 0.1
_HUBER_DELTA_GEOM: float = 1.0

PHEAT_FAILURE_PENALTY = 1.0e12
LENGTH_ENCODING_SCOPES = ("shared-by-type", "per-residue")
TRANSPILE_OPTIMIZATION_LEVELS = (0, 1, 2, 3)
DEFAULT_BACKBONE_LENGTH_SPAN_A = 0.08
DEFAULT_SIDECHAIN_LENGTH_SPAN_A = 0.12
MIN_ENCODED_BOND_LENGTH_A = 0.5
_UNSET = object()


def _backend_display_name(backend) -> str:
    if backend is None:
        return "statevector"
    try:
        name = getattr(backend, "name", None)
        return name() if callable(name) else str(name or backend)
    except Exception:
        return str(backend)


def _normalize_length_encoding_scope(value: str) -> str:
    normalized = str(value or "shared-by-type").strip().lower()
    if normalized not in LENGTH_ENCODING_SCOPES:
        raise ValueError(
            "length_encoding_scope must be one of "
            + ", ".join(LENGTH_ENCODING_SCOPES)
        )
    return normalized


def _normalize_transpile_optimization_level(value, context: str = "transpile_optimization_level") -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "default", "auto"}:
        return None
    try:
        level = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be one of none, 0, 1, 2, or 3.") from exc
    if level not in TRANSPILE_OPTIMIZATION_LEVELS:
        raise ValueError(f"{context} must be one of none, 0, 1, 2, or 3.")
    return level


def _normalize_transpile_seed(value, context: str = "transpile_seed") -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "default", "auto"}:
        return None
    try:
        seed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be a non-negative integer or none.") from exc
    if seed < 0:
        raise ValueError(f"{context} must be a non-negative integer or none.")
    return seed


def _wrap_radians(value: float) -> float:
    return (float(value) + math.pi) % (2 * math.pi) - math.pi


class QuantumBiophysicsFolder:
    """Hybrid quantum-classical protein folder."""

    current_stage: int = 1

    # ------------------------------------------------------------------
    # Force-field tables
    # ------------------------------------------------------------------
    _HYDROPHOBICITY: dict[str, float] = {
        "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5,
        "M": 1.9, "A": 1.8, "G": -0.4, "T": -0.7, "S": -0.8,
        "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2, "E": -3.5,
        "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
    }

    _VDW_RADII: dict[str, float] = {
        "H": 0.6, "C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8,
    }

    def __init__(
        self,
        sequence: str,
        force_field: str = "protein-coarse-charge-v1",
        *,
        selective_chi_map: Optional[dict[str, list[str]]] = None,
        angle_units: str = "radians",
        stored_angles=None,
        stored_lengths=None,
        max_chi=None,
        include_terminal_oxt: bool = False,
        geometry_mode: Optional[str] = None,
        geometry_table: Optional[Any] = None,
        geometry_profile: Optional[str] = None,
        score_model: str = "pheat-generic",
        bond_angle_encoding: str = "centered",
        tau_center_deg: float = ANGLE_N_CA_C,
        tau_span_deg: float = 25.0,
        theta_center_deg: float = ANGLE_CA_C_N,
        theta_span_deg: float = 25.0,
        length_encoding_scope: str = "shared-by-type",
        backbone_length_span: float = DEFAULT_BACKBONE_LENGTH_SPAN_A,
        sidechain_length_span: float = DEFAULT_SIDECHAIN_LENGTH_SPAN_A,
        optimizer_angle_mode: str = "statevector",
        optimizer_backend=None,
        optimizer_shots: int = 4096,
        basis_circuit_batching: str = "auto",
        transpile_optimization_level: Optional[int] = None,
        transpile_seed: Optional[int] = None,
        reference_residue_geometry: Optional[ResidueGeometryStructure] = None,
        base_residue_geometry: Optional[ResidueGeometryStructure] = None,
        circuit_template: Optional[dict[str, Any]] = None,
        circuit: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Parameters
        ----------
        sequence:
            Single-letter amino acid sequence (e.g. ``"MAGTWY"``).
        force_field:
            Coarse native scoring profile selector used only by QTF-native metadata paths;
            exact force-field engines are configured through external PHEAT scorers.
        """
        self.sequence = sequence.upper()
        self.n_residues = len(self.sequence)
        self.force_field = force_field.lower()
        self.selective_chi_map = selective_chi_map or {}
        self.angle_units = str(angle_units).lower()
        self.stored_angles = normalize_stored_angles(stored_angles or ())
        self.stored_lengths = ResidueGeometryStructure(
            residues=[],
            stored_lengths=stored_lengths,
        ).stored_lengths
        self.max_chi = normalize_max_chi(max_chi)
        self.include_terminal_oxt = bool(include_terminal_oxt)
        self.geometry_mode = None if geometry_mode in (None, "") else str(geometry_mode)
        self.geometry_table = None if geometry_table in (None, "") else geometry_table
        self.geometry_profile = None if geometry_profile in (None, "") else str(geometry_profile)
        self.reference_residue_geometry = reference_residue_geometry
        self.score_model = canonical_score_model(score_model)
        self.active_score_model = self.score_model
        self.optimizer_angle_mode = str(optimizer_angle_mode).lower()
        self.optimizer_backend = optimizer_backend
        self.optimizer_shots = int(optimizer_shots)
        self.basis_circuit_batching = str(basis_circuit_batching).strip().lower()
        self.transpile_optimization_level = _normalize_transpile_optimization_level(
            transpile_optimization_level
        )
        self.transpile_seed = _normalize_transpile_seed(transpile_seed)
        self.basis_circuit_batching_stats = {
            "requested": self.basis_circuit_batching,
            "calls": 0,
            "batched_calls": 0,
            "serial_calls": 0,
            "local_statevector_calls": 0,
            "fallback_calls": 0,
            "basis_circuits": 0,
            "backend_jobs": 0,
            "last_effective": None,
            "fallback_reasons": [],
            "last_transpile_optimization_level": None,
            "last_transpile_seed": None,
        }
        self.bond_angle_encoding = str(bond_angle_encoding).lower()
        self.tau_center_deg = float(tau_center_deg)
        self.tau_span_deg = float(tau_span_deg)
        self.theta_center_deg = float(theta_center_deg)
        self.theta_span_deg = float(theta_span_deg)
        self.length_encoding_scope = _normalize_length_encoding_scope(length_encoding_scope)
        self.backbone_length_span = float(backbone_length_span)
        self.sidechain_length_span = float(sidechain_length_span)
        self.circuit_template = circuit_template
        self.circuit = circuit
        self.circuit_metadata = {}
        self.last_score = None
        self.last_score_error = None
        self.last_structure = None
        self.last_residue_geometry = None
        self.base_residue_geometry = base_residue_geometry
        self.pheat_chi_dofs_by_residue: dict[int, list[str]] = {}

        if self.optimizer_angle_mode not in {"statevector", "sampler"}:
            raise ValueError("optimizer_angle_mode must be 'statevector' or 'sampler'.")
        if self.angle_units not in ANGLE_UNITS:
            raise ValueError(f"angle_units must be one of {', '.join(ANGLE_UNITS)}")
        if self.bond_angle_encoding not in {"centered", "raw"}:
            raise ValueError("bond_angle_encoding must be 'centered' or 'raw'.")
        if self.backbone_length_span <= 0:
            raise ValueError("backbone_length_span must be positive.")
        if self.sidechain_length_span <= 0:
            raise ValueError("sidechain_length_span must be positive.")
        if self.optimizer_shots <= 0:
            raise ValueError("optimizer_shots must be positive.")
        if self.basis_circuit_batching not in {"auto", "on", "off"}:
            raise ValueError("basis_circuit_batching must be one of auto, on, or off.")

        logger.info("Initialising QuantumBiophysicsFolder | FF=%s | seq=%s", self.force_field.upper(), self.sequence)

        self.HYDROPHOBICITY = self._HYDROPHOBICITY
        self.VDW_RADII = self._VDW_RADII

        self.CHARGES = self._build_charges(self.force_field)

        # ------------------------------------------------------------------
        # Degrees of freedom
        # ------------------------------------------------------------------
        # PHEAT owns the residue-level angle definitions. QTF keeps a compact
        # dof_map view for circuit encoding and optimizer bookkeeping.
        self._rebuild_dof_map()

        self._rebuild_quantum_register()

        self.current_stage = 1
        self.active_score_options = None
        self.tracker: LandscapeTracker | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_quantum_register(self) -> None:
        self.total_dofs = len(self.dof_map)
        self.total_angles = self.total_dofs
        build = build_circuit(
            total_angles=self.total_dofs,
            circuit_template=self.circuit_template,
            circuit=self.circuit,
        )
        self.ansatz = build.circuit
        self.n_qubits = build.n_qubits
        self.reps = build.reps
        self.n_params = build.n_params
        self.circuit_metadata = build.metadata()
        self.circuit_metadata.update(
            {
                "total_dofs": self.total_dofs,
                "total_angle_dofs": self.total_angle_dofs,
                "total_length_dofs": self.total_length_dofs,
                "length_encoding_scope": self.length_encoding_scope,
            }
        )
        self._cache_initialized = False
        self._initialize_topology_cache()

    @staticmethod
    def _build_charges(force_field: str = "protein-coarse-charge-v1") -> dict[str, float]:
        common: dict[str, float] = {
            "OXT": -1.0,
            "NZ": 1.0, "NH1": 0.5, "NH2": 0.5,
            "OD1": -0.5, "OD2": -0.5, "OE1": -0.5, "OE2": -0.5,
            "ND2": 0.5, "NE2": 0.5,
            "SG": -0.1, "SD": -0.1,
            "HE2": 0.4, "ND1": -0.4,
        }
        protein_coarse: dict[str, float] = {
            "N": -0.42, "H": 0.27, "CA": 0.00, "C": 0.60, "O": -0.57,
            "OG": -0.6, "HG": 0.4, "OG1": -0.6, "HG1": 0.4, "OH": -0.5, "HH": 0.4,
            "NE1": -0.4, "HE1": 0.3,
        }
        charges = common.copy()
        if force_field != "protein-coarse-charge-v1":
            logger.warning(
                "Unknown native scoring profile '%s'. Defaulting to protein-coarse-charge-v1.",
                force_field,
            )
        charges.update(protein_coarse)
        return charges


    def _rebuild_dof_map(self) -> None:
        self.pheat_angle_specs = residue_angle_specs(
            self.sequence,
            selective_chi_map=self.selective_chi_map or None,
            max_chi=self.max_chi,
            stored_angles=self.stored_angles,
            angle_units=self.angle_units,
        )
        self.pheat_chi_dofs_by_residue = {}
        self.dof_map: list[dict] = []
        self.dof_specs: list[dict] = []
        for spec in self.pheat_angle_specs:
            res_idx = int(spec["residue_index"])
            angle_name = str(spec["angle_name"])
            self.dof_map.append({"res": res_idx, "type": angle_name})
            self.dof_specs.append({"kind": "angle", "res": res_idx, "type": angle_name})
            if angle_name.startswith("chi"):
                self.pheat_chi_dofs_by_residue.setdefault(res_idx, []).append(angle_name)

        self.length_targets_by_residue = self._build_length_targets_by_residue()
        self.length_dof_specs = self._build_length_dof_specs()
        for spec in self.length_dof_specs:
            self.dof_specs.append(spec)
            self.dof_map.append({"res": spec.get("residue_index"), "type": f"length:{spec['key']}"})

        self.total_angle_dofs = len(self.pheat_angle_specs)
        self.total_length_dofs = len(self.length_dof_specs)
        self.total_dofs = len(self.dof_specs)

    def _build_length_targets_by_residue(self) -> dict[int, list[dict[str, Any]]]:
        targets: dict[int, list[dict[str, Any]]] = {}
        if not self.stored_lengths:
            return targets
        for res_idx, aa in enumerate(self.sequence):
            resname = one_to_three(aa)
            residue_targets = [
                {"key": "N-CA", "default": N_CA, "class": "backbone"},
                {"key": "CA-C", "default": CA_C, "class": "backbone"},
                {"key": "C-O", "default": C_O, "class": "backbone"},
            ]
            if res_idx < self.n_residues - 1:
                residue_targets.append({"key": "C-N", "default": C_N, "class": "backbone"})
            if resname != "GLY":
                residue_targets.append({"key": "CA-CB", "default": CA_CB, "class": "sidechain"})
            for step in SIDECHAIN_STEPS.get(resname, []):
                residue_targets.append(
                    {
                        "key": f"{step.parent}-{step.atom}",
                        "default": float(step.length),
                        "class": "sidechain",
                    }
                )
            for ring_target in self._ring_closure_length_targets(resname):
                residue_targets.append(ring_target)
            selected = [
                target
                for target in residue_targets
                if self._length_requested(str(target["key"]))
            ]
            if selected:
                targets[res_idx] = selected
        return targets

    def _build_length_dof_specs(self) -> list[dict[str, Any]]:
        if not self.length_targets_by_residue:
            return []
        if self.length_encoding_scope == "per-residue":
            return [
                {
                    "kind": "length",
                    "scope": "per-residue",
                    "residue_index": res_idx,
                    "res": res_idx,
                    "key": target["key"],
                    "class": target["class"],
                }
                for res_idx, targets in self.length_targets_by_residue.items()
                for target in targets
            ]

        specs_by_key: dict[str, dict[str, Any]] = {}
        for targets in self.length_targets_by_residue.values():
            for target in targets:
                key = str(target["key"])
                specs_by_key.setdefault(
                    key,
                    {
                        "kind": "length",
                        "scope": "shared-by-type",
                        "residue_index": None,
                        "res": None,
                        "key": key,
                        "class": target["class"],
                    },
                )
        return list(specs_by_key.values())

    @staticmethod
    def _ring_closure_length_targets(resname: str) -> list[dict[str, Any]]:
        if resname in {"PRO", "HYP"}:
            return [{"key": "N-CD", "default": PRO_N_CD, "class": "sidechain"}]
        if resname == "PCA":
            return [{"key": "N-CD", "default": PCA_N_CD, "class": "sidechain"}]
        if resname == "PYL":
            return [{"key": "CA2-CG2", "default": PYL_CA2_CG2, "class": "sidechain"}]
        return []

    def _length_requested(self, key: str) -> bool:
        if "all" in self.stored_lengths:
            return True
        if key in self.stored_lengths:
            return True
        if key in RESIDUE_GEOMETRY_BACKBONE_LENGTHS:
            return "backbone" in self.stored_lengths
        return "sidechain" in self.stored_lengths

    def _get_angles(
        self,
        params: np.ndarray,
        mode: str = "statevector",
        backend=None,
        shots: int = 4096,
        transpile_optimization_level=_UNSET,
        transpile_seed=_UNSET,
    ) -> np.ndarray:
        """Map circuit parameters to torsion angles via statevector phases or sampled bases.

        The 2ⁿ complex amplitudes of the statevector each carry a phase in
        ``(-π, π]``.  The first ``total_angles`` phases are used as torsion
        angles after removing the **global phase**.

        A global rotation ``e^{iα}|ψ⟩`` shifts every amplitude phase by α
        uniformly, but is physically unobservable — it would add a spurious
        common offset to every dihedral and waste one degree of freedom.
        The gauge is fixed by subtracting ``angle(ψ₀)`` (the phase of the
        ``|0…0⟩`` basis-state amplitude) from all extracted phases, so that
        ``phases[0]`` is always 0.  The result is wrapped back into
        ``(-π, π]``.

        **Consequence**: the zeroth entry of ``dof_map`` — φ of residue 0,
        the N-terminal φ dihedral — is held at 0 and cannot be optimised.
        This is acceptable: the N-terminal φ has no backbone predecessor and
        is conventionally undefined (or set to a reference value).  The
        optimiser therefore controls ``K − 1`` independent phase degrees of
        freedom for K total torsion angles.
        """
        if mode == "sampler":
            return self._get_sampler_angles(
                params,
                backend=backend,
                shots=shots,
                transpile_optimization_level=transpile_optimization_level,
                transpile_seed=transpile_seed,
            )
        param_dict = dict(zip(self.ansatz.parameters, params))
        bound_circuit = self.ansatz.assign_parameters(param_dict)
        psi = Statevector(bound_circuit).data
        phases = np.angle(psi)[: self.total_angles]
        # Remove global phase: pin phases[0] to 0 and wrap into (-π, π].
        phases = (phases - np.angle(psi[0]) + np.pi) % (2 * np.pi) - np.pi
        return phases

    def _get_sampler_angles(
        self,
        params: np.ndarray,
        *,
        backend,
        shots: int,
        transpile_optimization_level=_UNSET,
        transpile_seed=_UNSET,
    ) -> np.ndarray:
        if shots <= 0:
            raise ValueError("shots must be positive for sampler angle extraction.")
        if backend is None:
            try:
                from qiskit_aer import AerSimulator
            except ImportError as exc:
                raise RuntimeError("Sampler angle extraction requires a backend or qiskit-aer.") from exc
            backend = AerSimulator()

        param_dict = dict(zip(self.ansatz.parameters, params))
        bound_circuit = self.ansatz.assign_parameters(param_dict)
        n_states = 2 ** self.n_qubits
        is_statevector_shots = _backend_display_name(backend) == "statevector-shots"
        optimization_level = (
            self.transpile_optimization_level
            if transpile_optimization_level is _UNSET
            else _normalize_transpile_optimization_level(transpile_optimization_level)
        )
        seed_transpiler = (
            self.transpile_seed
            if transpile_seed is _UNSET
            else _normalize_transpile_seed(transpile_seed)
        )

        def _transpile_kwargs():
            kwargs = {}
            if optimization_level is not None:
                kwargs["optimization_level"] = int(optimization_level)
            if seed_transpiler is not None:
                kwargs["seed_transpiler"] = int(seed_transpiler)
            return kwargs

        def _basis_circuit(qc, basis: str):
            circuit = qc.copy()
            if basis == "X":
                for qubit in range(self.n_qubits):
                    circuit.h(qubit)
            elif basis == "Y":
                for qubit in range(self.n_qubits):
                    circuit.sdg(qubit)
                    circuit.h(qubit)
            if not is_statevector_shots:
                circuit.measure_all()
            return circuit

        circuits = [
            _basis_circuit(bound_circuit, "Z"),
            _basis_circuit(bound_circuit, "X"),
            _basis_circuit(bound_circuit, "Y"),
        ]

        def _statevector_counts(qc, basis_offset: int):
            statevector = Statevector(qc)
            seed = getattr(backend, "seed", None)
            if seed is not None:
                statevector.seed(int(seed) + basis_offset)
            return statevector.sample_counts(shots)

        def _counts_to_pvec(counts):
            pvec = np.zeros(n_states, dtype=float)
            total = sum(counts.values())
            if total <= 0:
                raise ValueError("shot sampling produced no measurement counts.")
            for bitstring, count in counts.items():
                idx = int(str(bitstring).replace(" ", "")[::-1], 2)
                pvec[idx] += count / total
            return pvec

        def _serial_counts():
            counts = []
            for circuit in circuits:
                tqc = transpile(circuit, backend, **_transpile_kwargs())
                counts.append(backend.run(tqc, shots=shots).result().get_counts())
            return counts

        def _batched_counts():
            tqcs = transpile(circuits, backend, **_transpile_kwargs())
            result = backend.run(tqcs, shots=shots).result()
            return [result.get_counts(index) for index in range(len(circuits))]

        requested = getattr(self, "basis_circuit_batching", "auto")
        if is_statevector_shots:
            counts = [_statevector_counts(circuit, offset) for offset, circuit in enumerate(circuits)]
            effective = "local_statevector_serial" if requested == "off" else "local_statevector"
            self._record_basis_circuit_batching(
                effective=effective,
                backend=backend,
                transpile_optimization_level=optimization_level,
                transpile_seed=seed_transpiler,
            )
            return self._angles_from_probability_vectors(*[_counts_to_pvec(item) for item in counts])

        if requested == "off":
            counts = _serial_counts()
            self._record_basis_circuit_batching(
                effective="serial",
                backend=backend,
                transpile_optimization_level=optimization_level,
                transpile_seed=seed_transpiler,
            )
            return self._angles_from_probability_vectors(*[_counts_to_pvec(item) for item in counts])

        try:
            counts = _batched_counts()
            self._record_basis_circuit_batching(
                effective="batched",
                backend=backend,
                transpile_optimization_level=optimization_level,
                transpile_seed=seed_transpiler,
            )
        except Exception as exc:
            if requested == "on":
                raise RuntimeError(
                    "Basis-circuit batching was requested but the backend did not accept the batched Z/X/Y job: "
                    f"{exc}"
                ) from exc
            counts = _serial_counts()
            self._record_basis_circuit_batching(
                effective="fallback_serial",
                backend=backend,
                transpile_optimization_level=optimization_level,
                transpile_seed=seed_transpiler,
                reason=f"{_backend_display_name(backend)}: {exc}",
            )
        return self._angles_from_probability_vectors(*[_counts_to_pvec(item) for item in counts])

    def _record_basis_circuit_batching(
        self,
        *,
        effective: str,
        backend,
        reason: Optional[str] = None,
        transpile_optimization_level: Optional[int] = None,
        transpile_seed: Optional[int] = None,
    ) -> None:
        stats = getattr(self, "basis_circuit_batching_stats", None)
        if not isinstance(stats, dict):
            return
        stats["calls"] = int(stats.get("calls") or 0) + 1
        stats["basis_circuits"] = int(stats.get("basis_circuits") or 0) + 3
        stats["last_effective"] = effective
        stats["last_backend"] = _backend_display_name(backend) if backend is not None else "statevector"
        stats["last_transpile_optimization_level"] = transpile_optimization_level
        stats["last_transpile_seed"] = transpile_seed
        if effective == "batched":
            stats["batched_calls"] = int(stats.get("batched_calls") or 0) + 1
            stats["backend_jobs"] = int(stats.get("backend_jobs") or 0) + 1
        elif effective.startswith("local_statevector"):
            stats["local_statevector_calls"] = int(stats.get("local_statevector_calls") or 0) + 1
        else:
            stats["serial_calls"] = int(stats.get("serial_calls") or 0) + 1
            stats["backend_jobs"] = int(stats.get("backend_jobs") or 0) + 3
            if effective == "fallback_serial":
                stats["fallback_calls"] = int(stats.get("fallback_calls") or 0) + 1
        if reason:
            reasons = list(stats.get("fallback_reasons") or [])
            if reason not in reasons:
                reasons.append(reason)
            stats["fallback_reasons"] = reasons[-10:]

    def _angles_from_probability_vectors(self, pZ, pX, pY):
        n_states = 2 ** self.n_qubits
        state_angles = 2.0 * np.pi * np.arange(n_states) / n_states

        def _marginal_angles(pvec):
            angles = []
            for qubit in range(self.n_qubits):
                mask = np.array([(index >> qubit) & 1 for index in range(n_states)], dtype=float)
                angles.append(2.0 * np.pi * np.dot(pvec, mask) - np.pi)
            return np.array(angles)

        def _circular_mean(pvec):
            return np.arctan2(np.sum(pvec * np.sin(state_angles)), np.sum(pvec * np.cos(state_angles)))

        cdf_angles = 2.0 * np.pi * np.cumsum(pZ) - np.pi
        eps = 1e-12
        kl_zx = float(np.sum(pZ * np.log((pZ + eps) / (pX + eps))))
        kl_zy = float(np.sum(pZ * np.log((pZ + eps) / (pY + eps))))
        base = np.concatenate(
            [
                _marginal_angles(pZ),
                _marginal_angles(pX),
                _marginal_angles(pY),
                [_circular_mean(pZ), _circular_mean(pX), _circular_mean(pY)],
                cdf_angles,
                [np.arctan(kl_zx) * 2.0 - np.pi / 2.0, np.arctan(kl_zy) * 2.0 - np.pi / 2.0],
            ]
        )
        base = np.clip(base, -np.pi, np.pi)
        if len(base) >= self.total_angles:
            return base[: self.total_angles]
        out = np.zeros(self.total_angles, dtype=float)
        out[: len(base)] = base
        for idx in range(len(base), self.total_angles):
            i = idx % len(base)
            j = (idx * 3 + 1) % len(base)
            m = (idx * 7 + 2) % len(base)
            value = 0.60 * base[i] + 0.30 * np.sin(base[j]) + 0.10 * np.cos(base[m])
            out[idx] = (value + np.pi) % (2 * np.pi) - np.pi
        return out

    @staticmethod
    def _nerf_step(
        a: np.ndarray, b: np.ndarray, c: np.ndarray,
        bond_len: float, bond_angle: float, torsion: float,
    ) -> np.ndarray:
        """Natural Extension Reference Frame (NERF) atom placement."""
        bc = c - b
        bc_u = bc / (np.linalg.norm(bc) + 1e-9)
        ab = b - a
        n = np.cross(ab, bc_u)
        n_u = n / (np.linalg.norm(n) + 1e-9)
        bx_n = np.cross(n_u, bc_u)
        M = np.column_stack((bc_u, bx_n, n_u))
        theta_supp = np.pi - bond_angle
        d = np.array([
            bond_len * np.cos(theta_supp),
            bond_len * np.cos(torsion) * np.sin(theta_supp),
            bond_len * np.sin(torsion) * np.sin(theta_supp),
        ])
        return c + (M @ d)

    def _raw_dof_maps(self, dof_vector: np.ndarray) -> tuple[dict[str, float], dict[tuple[Any, str], float]]:
        angle_values: dict[str, float] = {}
        length_values: dict[tuple[Any, str], float] = {}
        for spec, value in zip(self.dof_specs, dof_vector):
            if spec["kind"] == "angle":
                angle_values[f"{spec['res']}_{spec['type']}"] = float(value)
            elif spec["kind"] == "length":
                key = str(spec["key"])
                if spec.get("scope") == "shared-by-type":
                    length_values[(None, key)] = float(value)
                else:
                    length_values[(int(spec["residue_index"]), key)] = float(value)
        return angle_values, length_values

    def _angle_dict(self, angle_vector: np.ndarray) -> dict[str, float]:
        return self._raw_dof_maps(angle_vector)[0]

    def _to_configured_units(self, angle_radians: Optional[float]) -> Optional[float]:
        if angle_radians is None:
            return None
        if self.angle_units == "degrees":
            return float(math.degrees(angle_radians))
        return float(angle_radians)

    def _angle_from_configured_units(self, angle_value: Optional[float]) -> Optional[float]:
        if angle_value is None:
            return None
        if self.angle_units == "degrees":
            return float(math.radians(float(angle_value)))
        return float(angle_value)

    def _residue_angle_radians(self, residue: Optional[ResidueGeometry], name: str) -> Optional[float]:
        if residue is None:
            return None
        return self._angle_from_configured_units(getattr(residue, name, None))

    def _residue_chi_radians(self, residue: Optional[ResidueGeometry], chi_name: str) -> Optional[float]:
        if residue is None or not chi_name.startswith("chi"):
            return None
        try:
            chi_index = int(chi_name[3:]) - 1
        except ValueError:
            return None
        if not (0 <= chi_index < len(residue.chi)):
            return None
        return self._angle_from_configured_units(residue.chi[chi_index])

    def _encoded_torsion_value(
        self,
        raw_radians: Optional[float],
        base_radians: Optional[float],
    ) -> Optional[float]:
        if raw_radians is None:
            return self._to_configured_units(base_radians)
        if base_radians is None:
            return self._to_configured_units(float(raw_radians))
        return self._to_configured_units(_wrap_radians(float(base_radians) + float(raw_radians)))

    def _bond_angle_value(
        self,
        raw_radians: Optional[float],
        *,
        center_deg: float,
        span_deg: float,
        base_radians: Optional[float] = None,
    ) -> Optional[float]:
        if raw_radians is None:
            return self._to_configured_units(base_radians)
        if self.bond_angle_encoding == "raw":
            if base_radians is not None:
                return self._to_configured_units(float(base_radians) + float(raw_radians))
            return self._to_configured_units(float(raw_radians))
        base_deg = math.degrees(base_radians) if base_radians is not None else center_deg
        angle_deg = base_deg + span_deg * math.sin(float(raw_radians))
        return self._to_configured_units(math.radians(angle_deg))

    def _template_residue(self, res_idx: int) -> Optional[ResidueGeometry]:
        template = self.reference_residue_geometry
        if template is None or not (0 <= res_idx < len(template.residues)):
            return None
        return template.residues[res_idx]

    def _base_residue(self, res_idx: int) -> Optional[ResidueGeometry]:
        base = self.base_residue_geometry
        if base is None or not (0 <= res_idx < len(base.residues)):
            return None
        return base.residues[res_idx]

    def _template_disulfide_bonds(self):
        template = self.reference_residue_geometry
        if template is None:
            return []
        return list(template.disulfide_bonds)

    @staticmethod
    def _residue_bond_length(residue: Optional[ResidueGeometry], key: str) -> Optional[float]:
        if residue is None:
            return None
        value = residue.bond_lengths.get(key)
        return None if value is None else float(value)

    def _length_span_for_target(self, target: dict[str, Any]) -> float:
        if target.get("class") == "backbone":
            return self.backbone_length_span
        return self.sidechain_length_span

    def _encoded_bond_lengths(
        self,
        res_idx: int,
        *,
        base_residue: Optional[ResidueGeometry],
        template_residue: Optional[ResidueGeometry],
        length_values: dict[tuple[Any, str], float],
    ) -> dict[str, float]:
        bond_lengths = dict(template_residue.bond_lengths) if template_residue is not None else {}
        for target in self.length_targets_by_residue.get(res_idx, []):
            key = str(target["key"])
            raw_value = length_values.get((res_idx, key), length_values.get((None, key), 0.0))
            base_length = (
                self._residue_bond_length(base_residue, key)
                or self._residue_bond_length(template_residue, key)
                or float(target["default"])
            )
            encoded = base_length + self._length_span_for_target(target) * math.sin(float(raw_value))
            bond_lengths[key] = max(MIN_ENCODED_BOND_LENGTH_A, float(encoded))
        return bond_lengths

    def angle_vector_to_residue_geometry(self, angle_vector: np.ndarray) -> ResidueGeometryStructure:
        """Convert a QTF raw DOF vector to PHEAT residue geometry."""

        angle_dict, length_values = self._raw_dof_maps(angle_vector)
        residues = []
        for res_idx, aa in enumerate(self.sequence):
            template_residue = self._template_residue(res_idx)
            base_residue = self._base_residue(res_idx)
            metadata_residue = template_residue or base_residue
            chain_id = metadata_residue.chain_id if metadata_residue is not None else "A"
            resseq = metadata_residue.resseq if metadata_residue is not None else res_idx + 1
            icode = metadata_residue.icode if metadata_residue is not None else ""
            chi_values = []
            for chi_name in self.pheat_chi_dofs_by_residue.get(res_idx, []):
                key = f"{res_idx}_{chi_name}"
                if key in angle_dict:
                    chi_values.append(
                        self._encoded_torsion_value(
                            angle_dict[key],
                            self._residue_chi_radians(base_residue, chi_name),
                        )
                    )

            omega = None
            theta = None
            if res_idx < self.n_residues - 1:
                if "omega" in self.stored_angles:
                    omega = self._encoded_torsion_value(
                        angle_dict.get(f"{res_idx}_omega"),
                        self._residue_angle_radians(base_residue, "omega"),
                    )
                if "theta" in self.stored_angles:
                    theta = self._bond_angle_value(
                        angle_dict.get(f"{res_idx}_theta"),
                        center_deg=self.theta_center_deg,
                        span_deg=self.theta_span_deg,
                        base_radians=self._residue_angle_radians(base_residue, "theta"),
                    )

            tau = None
            if "tau" in self.stored_angles:
                tau = self._bond_angle_value(
                    angle_dict.get(f"{res_idx}_tau"),
                    center_deg=self.tau_center_deg,
                    span_deg=self.tau_span_deg,
                    base_radians=self._residue_angle_radians(base_residue, "tau"),
                )

            residues.append(
                ResidueGeometry(
                    name=one_to_three(aa),
                    phi=self._encoded_torsion_value(
                        angle_dict.get(f"{res_idx}_phi"),
                        self._residue_angle_radians(base_residue, "phi"),
                    ),
                    psi=self._encoded_torsion_value(
                        angle_dict.get(f"{res_idx}_psi"),
                        self._residue_angle_radians(base_residue, "psi"),
                    ),
                    omega=omega,
                    tau=tau,
                    theta=theta,
                    chi=chi_values,
                    bond_lengths=self._encoded_bond_lengths(
                        res_idx,
                        base_residue=base_residue,
                        template_residue=template_residue,
                        length_values=length_values,
                    ),
                    chain_id=chain_id,
                    resseq=resseq,
                    icode=icode,
                )
            )

        residue_geometry = ResidueGeometryStructure(
            residues=residues,
            name=f"qtf:{self.sequence}",
            angle_units=self.angle_units,
            metadata={
                "source": "qtf",
                "sequence": self.sequence,
                "angle_source": "pheat.residue_angle_specs",
                "selective_chi_map": {
                    str(key): list(value)
                    for key, value in self.selective_chi_map.items()
                },
                "stored_angles": list(self.stored_angles),
                "stored_lengths": list(self.stored_lengths),
                "max_chi": self.max_chi,
                "bond_angle_encoding": self.bond_angle_encoding,
                "tau_center_deg": self.tau_center_deg,
                "tau_span_deg": self.tau_span_deg,
                "theta_center_deg": self.theta_center_deg,
                "theta_span_deg": self.theta_span_deg,
                "length_encoding_scope": self.length_encoding_scope,
                "backbone_length_span": self.backbone_length_span,
                "sidechain_length_span": self.sidechain_length_span,
                "total_angle_dofs": self.total_angle_dofs,
                "total_length_dofs": self.total_length_dofs,
            },
            stored_angles=self.stored_angles,
            stored_lengths=self.stored_lengths,
            disulfide_bonds=self._template_disulfide_bonds(),
        )
        self.last_residue_geometry = residue_geometry
        return residue_geometry

    def structure_from_angle_vector(self, angle_vector: np.ndarray):
        """Build a PHEAT heavy-atom structure from a QTF angle vector."""

        structure = structure_from_residue_geometry(
            self.angle_vector_to_residue_geometry(angle_vector),
            include_terminal_oxt=self.include_terminal_oxt,
            geometry_mode=self.geometry_mode,
            geometry_table=self.geometry_table,
            geometry_profile=self.geometry_profile,
        )
        self.last_structure = structure
        return structure

    def set_base_residue_geometry(self, residue_geometry: Optional[ResidueGeometryStructure]) -> None:
        """Set the geometry used as the zero-control baseline for active DOFs."""

        if residue_geometry is not None and len(residue_geometry.residues) != self.n_residues:
            raise ValueError(
                "base_residue_geometry residue count must match the folder sequence "
                f"({len(residue_geometry.residues)} != {self.n_residues})"
            )
        self.base_residue_geometry = residue_geometry

    def update_geometry_encoding(
        self,
        *,
        stored_angles=_UNSET,
        stored_lengths=_UNSET,
        max_chi=_UNSET,
        selective_chi_map=_UNSET,
        length_encoding_scope=_UNSET,
        backbone_length_span=_UNSET,
        sidechain_length_span=_UNSET,
    ) -> bool:
        """Update active geometry DOFs and rebuild the circuit when needed."""

        old_signature = self.geometry_encoding_signature()
        if stored_angles is not _UNSET:
            self.stored_angles = normalize_stored_angles(stored_angles or ())
        if stored_lengths is not _UNSET:
            self.stored_lengths = ResidueGeometryStructure(
                residues=[],
                stored_lengths=stored_lengths,
            ).stored_lengths
        if max_chi is not _UNSET:
            self.max_chi = normalize_max_chi(max_chi)
        if selective_chi_map is not _UNSET:
            self.selective_chi_map = dict(selective_chi_map or {})
        if length_encoding_scope is not _UNSET:
            self.length_encoding_scope = _normalize_length_encoding_scope(length_encoding_scope)
        if backbone_length_span is not _UNSET:
            self.backbone_length_span = float(backbone_length_span)
        if sidechain_length_span is not _UNSET:
            self.sidechain_length_span = float(sidechain_length_span)
        if self.backbone_length_span <= 0:
            raise ValueError("backbone_length_span must be positive.")
        if self.sidechain_length_span <= 0:
            raise ValueError("sidechain_length_span must be positive.")

        self._rebuild_dof_map()
        changed = self.geometry_encoding_signature() != old_signature
        if changed:
            self._rebuild_quantum_register()
        return changed

    def geometry_encoding_signature(self) -> tuple:
        selective = tuple(
            (str(key), tuple(str(item) for item in value))
            for key, value in sorted((self.selective_chi_map or {}).items())
        )
        return (
            tuple(self.stored_angles),
            tuple(self.stored_lengths),
            self.max_chi,
            selective,
            self.length_encoding_scope,
            float(self.backbone_length_span),
            float(self.sidechain_length_span),
            tuple((spec["kind"], spec.get("residue_index"), spec.get("type"), spec.get("key")) for spec in self.dof_specs),
        )

    def structure_from_coords_labels(self, coords, labels) -> HeavyAtomStructure:
        atoms = []
        for serial, (pos, label) in enumerate(zip(coords, labels), start=1):
            if len(label) >= 8:
                rid, atom_name, element, chain_id, resseq, icode, resname, record_name = label[:8]
            else:
                rid, atom_name, element = label[:3]
                template_residue = self._template_residue(int(rid))
                chain_id = template_residue.chain_id if template_residue is not None else "A"
                resseq = template_residue.resseq if template_residue is not None else int(rid) + 1
                icode = template_residue.icode if template_residue is not None else ""
                resname = one_to_three(self.sequence[int(rid)])
                record_name = "ATOM"
            rid = int(rid)
            atom_name = str(atom_name).strip()
            atoms.append(
                Atom(
                    name=atom_name,
                    element=str(element).strip().upper() or atom_name[0].upper(),
                    x=float(pos[0]),
                    y=float(pos[1]),
                    z=float(pos[2]),
                    resname=str(resname).strip().upper() or one_to_three(self.sequence[rid]),
                    chain_id=str(chain_id or ""),
                    resseq=int(resseq),
                    icode=str(icode or ""),
                    record_name=str(record_name or "ATOM"),
                    serial=serial,
                    occupancy=1.0,
                    bfactor=0.0,
                )
            )
        return HeavyAtomStructure(
            atoms=atoms,
            name=f"qtf:{self.sequence}",
            metadata={"source": "qtf"},
            disulfide_bonds=self._template_disulfide_bonds(),
            atom_scope="heavy",
        )

    @staticmethod
    def _structure_to_arrays(structure) -> tuple[np.ndarray, list, list]:
        coords = []
        labels = []
        residue_index = {key: index for index, key in enumerate(structure.residue_keys())}
        for atom in structure.atoms:
            rid = residue_index.get(atom.residue_key, int(atom.resseq) - 1)
            coords.append([atom.x, atom.y, atom.z])
            labels.append((rid, atom.name.strip().upper(), atom.element.strip().upper()))
        return np.asarray(coords, dtype=float), labels, []

    def build_full_structure(
        self, angle_vector: np.ndarray
    ) -> tuple[np.ndarray, list, list]:
        """Build PHEAT-derived 3-D Cartesian coordinates from a torsion-angle vector.

        The angle vector is indexed by ``dof_map`` and converted to a PHEAT
        residue-geometry structure before PHEAT reconstructs heavy atoms.

        Returns
        -------
        coords : ndarray, shape (N_atoms, 3)
        labels : list of (res_id, atom_name, element)
        bonds  : list of (atom_idx_a, atom_idx_b)
        """
        return self._structure_to_arrays(self.structure_from_angle_vector(angle_vector))

    def _initialize_topology_cache(self) -> None:
        """Pre-compute static atom properties for vectorised energy evaluation.

        A single call to ``build_full_structure`` is made here to harvest the
        atom label list (``static_labels``).  The resulting coordinates are
        **discarded** — only the labels, element symbols, and residue-index
        mapping are retained for use by the energy function.

        The seed torsion vector is ``_TOPOLOGY_SEED_ANGLE`` (0.1 rad) rather
        than all-zeros.  All-zero torsions place every successive backbone
        atom exactly along the same direction, making every cross product in
        ``_nerf_step`` identically zero.  Although the 1e-9 floor in the
        normalisation prevents a division-by-zero crash, the resulting
        reference frames are degenerate and the coordinates are meaningless.
        Using a small non-zero seed is costless (coordinates are unused) and
        removes this brittleness.
        """
        seed = np.full(self.total_angles, _TOPOLOGY_SEED_ANGLE)
        dummy_coords, self.static_labels, _ = self.build_full_structure(seed)
        n_atoms = len(dummy_coords)

        self.atom_to_res = np.array([x[0] for x in self.static_labels], dtype=int)
        self.atom_names = np.array([x[1] for x in self.static_labels])
        self.atom_elems = np.array([x[2] for x in self.static_labels])

        self.q_vector = np.zeros(n_atoms)
        for k, (rid, name, _) in enumerate(self.static_labels):
            q = self.CHARGES.get(name, 0.0)
            res_name = self.sequence[rid]
            if res_name == "H":
                if name == "NE2":
                    q = -0.4
                if name == "ND1":
                    q = -0.4
            if rid == 0 or rid == self.n_residues - 1:
                if name in {"N", "CA", "C", "O", "OXT", "H1", "H2", "H3", "H"}:
                    q = 0.0
            self.q_vector[k] = q

        self.vdw_radii_vector = np.array([self.VDW_RADII.get(x[2], 1.7) for x in self.static_labels])
        self.mask_heavy = np.array([not x.startswith("H") for x in self.atom_names], dtype=bool)

        hydro_res_set = {"A", "V", "L", "I", "M", "F", "W", "P", "C"}
        self.mask_hydrophobic = np.zeros(n_atoms, dtype=bool)
        for k, (rid, name, _) in enumerate(self.static_labels):
            if self.sequence[rid] in hydro_res_set and name.startswith("C"):
                self.mask_hydrophobic[k] = True

        res_diff_matrix = np.abs(self.atom_to_res[:, None] - self.atom_to_res[None, :])
        self.mask_non_bonded = res_diff_matrix >= 2
        adjacency = [set() for _ in range(n_atoms)]
        _, _, static_bonds = self.build_full_structure(seed)
        for i, j in static_bonds:
            if 0 <= i < n_atoms and 0 <= j < n_atoms:
                adjacency[i].add(j)
                adjacency[j].add(i)
        graph_dist = np.full((n_atoms, n_atoms), 99, dtype=int)
        np.fill_diagonal(graph_dist, 0)
        for i in range(n_atoms):
            frontier = {i}
            visited = {i}
            depth = 0
            while frontier and depth < 3:
                depth += 1
                next_frontier = set()
                for node in frontier:
                    for nbr in adjacency[node]:
                        if nbr in visited:
                            continue
                        visited.add(nbr)
                        graph_dist[i, nbr] = min(graph_dist[i, nbr], depth)
                        graph_dist[nbr, i] = min(graph_dist[nbr, i], depth)
                        next_frontier.add(nbr)
                frontier = next_frontier
        offdiag = ~np.eye(n_atoms, dtype=bool)
        self.mask_non_bonded_vdw = offdiag & (graph_dist > 3)
        self.mask_non_bonded_vdw_14 = offdiag & (graph_dist == 3)
        self.atom_lookup = {
            (int(rid), str(name).upper()): idx
            for idx, (rid, name, _elem) in enumerate(self.static_labels)
        }

        self.idx_N_atoms = np.where(self.atom_names == "N")[0]
        self.idx_O_atoms = np.where(self.atom_names == "O")[0]
        self.idx_SG_atoms = np.where(self.atom_names == "SG")[0]
        self._cache_initialized = True

    # ------------------------------------------------------------------
    # Energy function
    # ------------------------------------------------------------------

    def _angle_vector_from_params(
        self,
        params,
        *,
        angle_mode: str,
        backend=None,
        shots: int = 4096,
        transpile_optimization_level=_UNSET,
        transpile_seed=_UNSET,
    ):
        if angle_mode == "statevector":
            return self._get_angles(params, mode="statevector")
        if angle_mode == "sampler":
            if backend is None:
                raise ValueError("sampler angle mode requires a backend.")
            return self._get_angles(
                params,
                mode="sampler",
                backend=backend,
                shots=shots,
                transpile_optimization_level=transpile_optimization_level,
                transpile_seed=transpile_seed,
            )
        raise ValueError(f"Unknown angle mode: {angle_mode}")

    @staticmethod
    def _angle_mode_for_backend(backend) -> str:
        return "statevector" if backend is None else "sampler"

    def score_model_for_params(
        self,
        params,
        model: str,
        *,
        angle_mode: str,
        backend=None,
        shots: int = 4096,
        options: Optional[dict[str, Any]] = None,
        transpile_optimization_level=_UNSET,
        transpile_seed=_UNSET,
    ) -> tuple[dict, float]:
        active_model = canonical_score_model(model)
        angle_vec = self._angle_vector_from_params(
            params,
            angle_mode=angle_mode,
            backend=backend,
            shots=shots,
            transpile_optimization_level=transpile_optimization_level,
            transpile_seed=transpile_seed,
        )
        if is_qtf_score_model(active_model):
            score = score_classic_folder(
                self,
                params,
                model=active_model,
                angle_vector=angle_vec,
                options=options,
                return_terms=True,
            )
            payload = score.to_dict()
            payload["status"] = "ok"
            return payload, float(score.total)

        structure = self.structure_from_angle_vector(angle_vec)
        score = score_pheat_structure(structure, model=active_model, **dict(options or {}))
        payload = score.to_dict()
        payload["status"] = "ok"
        return payload, float(score.total)

    def energy_function(self, params: np.ndarray, return_terms: bool = False) -> float:
        """Evaluate the configured QTF or PHEAT score for the encoded structure."""
        active_model = canonical_score_model(getattr(self, "active_score_model", self.score_model))
        try:
            angle_probe = self._get_angles(params, mode=self.optimizer_angle_mode, backend=self.optimizer_backend, shots=self.optimizer_shots)
            if not np.isfinite(angle_probe).all():
                n_bad = int(np.count_nonzero(~np.isfinite(angle_probe)))
                penalty = 1e6 + 1e3 * n_bad
                self.last_energy_terms = {"non_finite_penalty": penalty, "total": penalty}
                if return_terms:
                    return dict(self.last_energy_terms)
                return float(penalty)
            score_payload, objective = self.score_model_for_params(
                params,
                active_model,
                angle_mode=self.optimizer_angle_mode,
                backend=self.optimizer_backend,
                shots=self.optimizer_shots,
                options=getattr(self, "active_score_options", None),
                transpile_optimization_level=self.transpile_optimization_level,
                transpile_seed=self.transpile_seed,
            )
            self.last_score_error = None
        except Exception as exc:
            score_payload = {
                "model": active_model,
                "status": "unavailable",
                "error": str(exc),
                "total": None,
                "units": None,
                "terms": {},
                "warnings": [],
                "citations": [],
                "metadata": {},
            }
            objective = PHEAT_FAILURE_PENALTY
            self.last_score_error = str(exc)

        self.last_score = score_payload
        raw_terms = dict(score_payload.get("terms", {}) or {})
        self.last_energy_terms = {
            "score_model": active_model,
            "score_total": float(objective),
            "total": float(objective),
            **{f"{active_model}_{key}": float(value) for key, value in raw_terms.items()},
        }

        candidate_records = getattr(self, "candidate_records", None)
        if candidate_records is not None:
            iteration = int(
                getattr(self.tracker, "current_iter", len(candidate_records))
                if self.tracker is not None
                else len(candidate_records)
            )
            candidate_records.append(
                {
                    "candidate_id": len(candidate_records),
                    "iteration": iteration,
                    "phase_index": getattr(self, "current_stage", None),
                    "phase_name": getattr(self, "current_phase_name", None),
                    "phase_label": getattr(self, "current_phase_label", None),
                    "score_model": active_model,
                    "objective": float(objective),
                    "status": score_payload.get("status"),
                    "params": np.asarray(params, dtype=float).copy(),
                }
            )

        if self.tracker is not None:
            self.tracker.log(objective)

        if return_terms:
            return objective, dict(self.last_energy_terms)
        return objective

    def _electrostatic_energy(self, D: np.ndarray) -> float:
        """Coulomb electrostatic energy in kcal/mol.

        Uses the standard prefactor 332.0637 kcal mol⁻¹ Å e⁻² and a uniform
        implicit-solvent dielectric ``_DIELECTRIC`` (default 4.0), giving::

            E = 332.0637 · qᵢqⱼ / (4.0 · rᵢⱼ)

        A hard lower bound of 1.0 Å is applied to ``rᵢⱼ`` to prevent
        singularities at unphysically short distances.
        """
        Q_mat = np.outer(self.q_vector, self.q_vector)
        elec_mask = np.triu(self.mask_non_bonded, k=1) & (np.abs(Q_mat) > 0.0001)
        if not np.any(elec_mask):
            return 0.0
        r_elec = np.maximum(D[elec_mask], 1.0)
        return float(np.sum(_COULOMB_PREFACTOR * Q_mat[elec_mask] / (_DIELECTRIC * r_elec)))

    def _calculate_rotamer_energy(self, angle_dict: dict) -> float:
        energy = 0.0
        for i in range(self.n_residues):
            res_name = self.sequence[i]
            key = f"{i}_chi1"
            if key in angle_dict:
                chi = angle_dict[key]
                if res_name in ("V", "I", "T", "VAL", "ILE", "THR"):
                    energy += -3.0 * (np.exp(-(chi - np.pi) ** 2 / 0.5) + np.exp(-(chi - (-1.047)) ** 2 / 0.5))
                elif res_name in ("P", "PRO"):
                    energy += 10.0 * min((chi - (-0.5)) ** 2, (chi - 0.5) ** 2)
                elif res_name in ("W", "F", "Y", "H", "TRP", "PHE", "TYR", "HIS"):
                    energy += -2.0 * (np.exp(-(chi - np.pi) ** 2 / 0.5) + np.exp(-(chi - (-1.047)) ** 2 / 0.5))
                else:
                    energy += 1.0 * (1.0 + np.cos(3.0 * chi))
        return energy

    def _calculate_aromatic_quadrupole(
        self, coords: np.ndarray, labels: list, atom_to_res_idx: np.ndarray
    ) -> float:
        aromatics: list[tuple] = []
        for r_idx in np.unique(atom_to_res_idx):
            if self.sequence[r_idx] in ("F", "Y", "W", "H", "PHE", "TYR", "TRP", "HIS"):
                mask = atom_to_res_idx == r_idx
                r_coords = coords[mask]
                r_names = self.atom_names[mask]
                ring_mask = np.isin(r_names, ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"])
                ring_atoms = r_coords[ring_mask]
                if len(ring_atoms) > 2:
                    centroid = np.mean(ring_atoms, axis=0)
                    v1 = ring_atoms[1] - ring_atoms[0]
                    v2 = ring_atoms[2] - ring_atoms[0]
                    normal = np.cross(v1, v2)
                    normal /= np.linalg.norm(normal) + 1e-9
                    aromatics.append((centroid, normal))
        energy_pi = 0.0
        for i in range(len(aromatics)):
            for j in range(i + 1, len(aromatics)):
                c1, n1 = aromatics[i]
                c2, n2 = aromatics[j]
                dist = np.linalg.norm(c1 - c2)
                if dist > 7.0:
                    continue
                alignment = abs(np.dot(n1, n2))
                if alignment < 0.3 and 4.5 < dist < 6.0:
                    energy_pi -= 4.0 * np.exp(-(dist - 5.0) ** 2)
                elif alignment > 0.8 and 3.4 < dist < 4.5:
                    energy_pi -= 5.0 * np.exp(-(dist - 3.8) ** 2)
        return energy_pi

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def get_smart_initialization(self, n_attempts: int = 20, seed: int | None = None) -> np.ndarray:
        """Sample random parameter sets and return the one with the lowest energy.

        Parameters
        ----------
        n_attempts:
            Number of random samples to evaluate.
        seed:
            Random seed. When ``None``, NumPy uses non-deterministic entropy.
        """
        rng = np.random.default_rng(seed)
        logger.info(
            "Scouting %d starting points (seed=%s, score_model=%s)",
            n_attempts,
            "unseeded" if seed is None else seed,
            getattr(self, "active_score_model", self.score_model),
        )

        best_params: np.ndarray | None = None
        best_energy = float("inf")
        for _ in range(n_attempts):
            trial_params = rng.uniform(-np.pi, np.pi, self.n_params)
            e = self.energy_function(trial_params)
            if e < best_energy:
                best_energy = e
                best_params = trial_params

        logger.info("Best start found: energy=%.2f", best_energy)
        if best_params is None:
            raise ValueError("get_smart_initialization: n_attempts must be at least 1.")
        return best_params

    # ------------------------------------------------------------------
    # Folding
    # ------------------------------------------------------------------

    def fold(
        self,
        max_iter: int = 2000,
        initial_params: np.ndarray | None = None,
        scout_attempts: int | None = None,
        phase_schedule: list[dict] | None = None,
    ) -> tuple[np.ndarray, list, list, LandscapeTracker, np.ndarray, float]:
        """Run the configurable optimisation phase curriculum.

        Parameters
        ----------
        max_iter:
            Maximum number of energy evaluations allowed for **each** of the
            three optimisation stages (COBYLA collapse, SLSQP refine, SLSQP
            relax).  The default of 2000 is a generous budget suitable for
            peptides up to ~20 residues.
        initial_params:
            Pre-computed circuit parameters to use as the starting point for
            Stage 1.  When *None* (the default) a scouting phase randomly
            samples ``scout_attempts`` parameter vectors and picks the one
            with the lowest energy.
        scout_attempts:
            Number of random parameter vectors evaluated during the scouting
            phase (only used when ``initial_params`` is *None*).  A larger
            value improves the quality of the starting point at a linear cost
            in energy evaluations.

            If *None* (the default) the value is computed as
            ``min(64, max_iter // 10)``, which keeps the scouting budget at
            most 10 % of one optimisation stage while still sampling at least
            a few dozen points.  Pass an explicit integer to override —
            e.g. ``scout_attempts=1`` for the fastest possible run during
            testing, or ``scout_attempts=200`` for a thorough global search.
        phase_schedule:
            Optional configurable phase list. When omitted, the default
            collapse/refine/relax curriculum is used.

        Returns
        -------
        coords, labels, bonds, tracker, final_params, final_energy
        """
        logger.info("Starting quantum folding (max_iter=%d)", max_iter)
        self.tracker = LandscapeTracker()

        if initial_params is None:
            n_scout = min(64, max_iter // 10) if scout_attempts is None else scout_attempts
            logger.info("Scouting %d starting points…", n_scout)
            init_params = self.get_smart_initialization(n_attempts=n_scout)
        else:
            init_params = initial_params

        phases = phase_schedule or [
            {
                "name": "collapse",
                "label": "Mechanical collapse",
                "optimizer": "COBYLA",
                "objective": {
                    "options": {
                        "hydrophobic_gamma": 15.0,
                        "end_to_end_weight": 50.0,
                        "end_to_end_target": 5.5,
                    }
                },
                "optimizer_options": {"rhobeg": 1.0},
            },
            {
                "name": "refine",
                "label": "Physics refinement",
                "optimizer": "SLSQP",
                "objective": {
                    "options": {
                        "hydrophobic_gamma": 15.0,
                        "end_to_end_weight": 50.0,
                        "end_to_end_target": 5.5,
                    }
                },
                "tol": 1e-6,
                "optimizer_options": {"disp": False},
            },
            {
                "name": "relax",
                "label": "Natural relaxation",
                "optimizer": "SLSQP",
                "objective": {
                    "options": {
                        "hydrophobic_gamma": 5.0,
                        "end_to_end_weight": 5.0,
                        "end_to_end_target": 5.5,
                    }
                },
                "tol": 1e-6,
                "optimizer_options": {"disp": False},
            },
        ]

        params = init_params
        self.phase_results = []
        self.phase_structures = []
        result = None
        for index, phase in enumerate(phases, start=1):
            name = str(phase.get("name") or f"phase-{index}")
            label = str(phase.get("label") or name)
            optimizer = str(phase.get("optimizer") or "SLSQP")
            options = {"maxiter": int(phase.get("maxiter") or max_iter)}
            options.update(dict(phase.get("optimizer_options") or phase.get("options") or {}))
            tol = phase.get("tol")
            objective = phase.get("objective") or {}
            self.active_score_options = dict(objective.get("options") or {})
            if phase.get("score_model"):
                self.active_score_model = canonical_score_model(phase["score_model"])
            self.current_stage = index
            self.current_phase_name = name
            self.current_phase_label = label
            self.tracker.mark_stage(f"Phase{index}:{name}")
            logger.info("Phase %d: %s (%s)…", index, label, optimizer)
            kwargs = {"method": optimizer, "options": options}
            if tol is not None:
                kwargs["tol"] = float(tol)
            result = minimize(self.energy_function, params, **kwargs)
            params = result.x
            coords_i, labels_i, bonds_i = self.build_full_structure(self._get_angles(params))
            self.phase_structures.append(
                {
                    "index": index,
                    "name": name,
                    "label": label,
                    "coords": coords_i,
                    "labels": labels_i,
                    "bonds": bonds_i,
                    "params": params,
                    "energy": float(result.fun),
                    "success": bool(getattr(result, "success", False)),
                    "status": getattr(result, "status", None),
                    "message": str(getattr(result, "message", "")),
                }
            )
            self.phase_results.append(
                {
                    "index": index,
                    "name": name,
                    "label": label,
                    "optimizer": optimizer,
                    "energy": float(result.fun),
                    "success": bool(getattr(result, "success", False)),
                    "status": getattr(result, "status", None),
                    "message": str(getattr(result, "message", "")),
                    "objective": objective,
                }
            )
            logger.info("  %s energy: %.2f", label, result.fun)

        self.active_score_options = None
        if result is None:
            raise ValueError("fold() phase_schedule must contain at least one phase.")
        coords, labels, bonds = self.build_full_structure(self._get_angles(result.x))
        logger.info("  Final energy: %.2f", result.fun)
        return coords, labels, bonds, self.tracker, result.x, float(result.fun)

    def save_pdb(self, coords, labels, filename="structure.pdb", energy=0.0):
        write_pdb(
            self.structure_from_coords_labels(coords, labels),
            filename,
            remarks=[f"ENERGY: {float(energy):.3f}"],
        )

    def save_reduced_pdb(self, ca_coords, filename="reduced.pdb", sidechain_centroids=None, energy=0.0):
        atoms = []
        for serial, pos in enumerate(ca_coords, start=1):
            rid = serial - 1
            template_residue = self._template_residue(rid)
            chain_id = template_residue.chain_id if template_residue is not None else "A"
            resseq = template_residue.resseq if template_residue is not None else rid + 1
            icode = template_residue.icode if template_residue is not None else ""
            atoms.append(
                Atom(
                    name="CA",
                    element="C",
                    x=float(pos[0]),
                    y=float(pos[1]),
                    z=float(pos[2]),
                    resname=one_to_three(self.sequence[rid]),
                    chain_id=chain_id,
                    resseq=resseq,
                    icode=icode,
                    record_name="ATOM",
                    serial=serial,
                    occupancy=1.0,
                    bfactor=0.0,
                )
            )
        write_pdb(
            HeavyAtomStructure(
                atoms=atoms,
                name=f"qtf:{self.sequence}:ca",
                metadata={"source": "qtf", "representation": "ca"},
                disulfide_bonds=self._template_disulfide_bonds(),
            ),
            filename,
            remarks=[f"ENERGY: {float(energy):.3f}"],
        )
