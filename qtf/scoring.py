"""Unified scoring helpers for QTF and PHEAT-backed recipes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np


QTF_SCORE_MODELS: tuple[str, ...] = ()
_COULOMB_PREFACTOR: float = 332.0637
_DIELECTRIC: float = 4.0


@dataclass
class ScoreResult:
    """PHEAT-compatible score payload used by QTF-native scoring."""

    model: str
    total: float
    units: str = "arbitrary"
    terms: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "total": float(self.total),
            "units": self.units,
            "terms": {key: float(value) for key, value in self.terms.items()},
            "warnings": list(self.warnings),
            "citations": list(self.citations),
            "metadata": dict(self.metadata),
        }


def available_pheat_score_models() -> list[str]:
    """Return QTF-canonical PHEAT score names runnable in this environment."""

    return [
        item["public_model"]
        for item in pheat_score_model_capabilities()
        if item.get("available")
    ]


def supported_pheat_score_models() -> list[str]:
    """Return every QTF-canonical PHEAT score name supported by PHEAT."""

    return [item["public_model"] for item in pheat_score_model_capabilities()]


def pheat_score_model_capabilities() -> list[dict[str, Any]]:
    """Return PHEAT score capability metadata using QTF public names."""

    capabilities = _pheat_model_capabilities_from_package()
    public_capabilities = []
    for item in capabilities:
        raw_model = _normalize_model_name(item.get("model"))
        if not raw_model:
            continue
        public_model = _public_pheat_score_name(raw_model)
        payload = dict(item)
        payload["model"] = public_model
        payload["public_model"] = public_model
        payload["pheat_model"] = raw_model
        payload["raw_model"] = raw_model
        payload["available"] = bool(item.get("available"))
        payload["supported"] = bool(item.get("supported", True))
        payload.setdefault("reason", None)
        payload.setdefault("requires", [])
        payload.setdefault("optional_requires", [])
        payload.setdefault("units", "arbitrary")
        public_capabilities.append(payload)
    return public_capabilities


def available_score_models() -> list[str]:
    """Return all user-facing score model names."""

    return [*QTF_SCORE_MODELS, *available_pheat_score_models()]


def pheat_model_name(model: str) -> str:
    """Map a QTF-canonical PHEAT model name to the PHEAT package model name."""

    normalized = _normalize_model_name(model)
    capabilities = {item["public_model"]: item for item in pheat_score_model_capabilities()}
    if normalized not in capabilities:
        raise ValueError(
            f"PHEAT score model '{normalized}' is not supported. "
            f"Supported models: {', '.join(supported_pheat_score_models()) or 'none'}."
        )
    capability = capabilities[normalized]
    if not capability.get("available"):
        reason = capability.get("reason") or "not available in the current Python environment"
        raise ValueError(f"PHEAT score model '{normalized}' is unavailable: {reason}.")
    return str(capability["pheat_model"])


def canonical_score_model(model: str, *, engine: Optional[str] = None) -> str:
    """Validate and normalize a score model name for QTF recipes."""

    normalized = _normalize_model_name(model)
    if engine == "classic":
        if normalized not in QTF_SCORE_MODELS:
            raise ValueError(f"Classic score model must be one of {', '.join(QTF_SCORE_MODELS)}.")
        return normalized
    if engine == "pheat":
        pheat_model_name(normalized)
        return normalized
    if normalized in QTF_SCORE_MODELS or normalized in available_pheat_score_models():
        return normalized
    raise ValueError(f"Score model must be one of {', '.join(available_score_models())}.")


def is_qtf_score_model(model: str) -> bool:
    return _normalize_model_name(model) in QTF_SCORE_MODELS


def is_pheat_score_model(model: str) -> bool:
    return _normalize_model_name(model).startswith("pheat-")


def _normalize_model_name(model) -> str:
    return str(model or "").strip().lower().replace("_", "-")


def _public_pheat_score_name(raw_model: str) -> str:
    normalized = _normalize_model_name(raw_model)
    return normalized if normalized.startswith("pheat-") else f"pheat-{normalized}"


def _pheat_model_capabilities_from_package() -> list[dict[str, Any]]:
    """Load PHEAT score capabilities from the installed PHEAT package."""

    try:
        from pheat.scoring import model_capabilities
    except Exception as exc:
        raise RuntimeError(
            "QTF requires pheat.scoring.model_capabilities() for PHEAT score discovery."
        ) from exc

    capabilities = model_capabilities()
    if not isinstance(capabilities, list):
        raise RuntimeError("pheat.scoring.model_capabilities() must return a list.")
    return capabilities


def score_pheat_structure(structure, model: str, **kwargs):
    """Score a PHEAT ``HeavyAtomStructure`` using canonical QTF score names."""

    from pheat.scoring import score_structure

    canonical = canonical_score_model(model, engine="pheat")
    result = score_structure(structure, model=pheat_model_name(canonical), **kwargs)
    result.model = canonical
    return result


def score_classic_folder(
    folder,
    params: np.ndarray,
    *,
    model: Optional[str] = None,
    angle_vector: Optional[np.ndarray] = None,
    options: Optional[Mapping[str, Any]] = None,
    return_terms: bool = True,
) -> ScoreResult:
    """Evaluate the QTF-native classic objective for a folder and parameter vector."""

    force_field = canonical_score_model(model or folder.force_field, engine="classic")
    if getattr(folder, "force_field", force_field) != force_field:
        folder.force_field = force_field
        folder.CHARGES = folder._build_charges(force_field)
        folder._cache_initialized = False

    if not folder._cache_initialized:
        folder._initialize_topology_cache()

    score_options = dict(options or {})
    if "hydrophobic_gamma" in score_options:
        gamma = float(score_options["hydrophobic_gamma"])
    else:
        gamma = 5.0 if folder.current_stage == 3 else 15.0
    if "end_to_end_weight" in score_options:
        constraint_strength = float(score_options["end_to_end_weight"])
    else:
        constraint_strength = 5.0 if folder.current_stage == 3 else 50.0
    end_to_end_target = float(score_options.get("end_to_end_target", 5.5))

    angle_vec = np.asarray(angle_vector, dtype=float) if angle_vector is not None else folder._get_angles(params)
    coords, _, _ = folder.build_full_structure(angle_vec)
    diffs = coords[:, None, :] - coords[None, :, :]
    distances = np.sqrt(np.sum(diffs**2, axis=-1)) + 1e-9

    terms: dict[str, float] = {}

    ca_indices = [i for i, lbl in enumerate(folder.static_labels) if lbl[1] == "CA"]
    if len(ca_indices) >= 2:
        dist_ends = np.linalg.norm(coords[ca_indices[0]] - coords[ca_indices[-1]])
        terms["end_to_end"] = constraint_strength * (dist_ends - end_to_end_target) ** 2
    else:
        terms["end_to_end"] = 0.0

    hydro_dists = distances[folder.mask_hydrophobic, :]
    weights = 1.0 / (1.0 + np.exp(1.0 * (hydro_dists - 6.0)))
    neighbor_counts = np.sum(weights, axis=1) - 1.0
    burial_fractions = np.clip(neighbor_counts / 15.0, 0.0, 1.0)
    terms["hydrophobic_burial"] = float(np.sum(gamma * 30.0 * (1.0 - burial_fractions)))

    terms["hbond"] = _classic_hbond_term(folder, coords)
    terms["electrostatic"] = _classic_electrostatic_term(folder, distances)
    terms["disulfide"] = _classic_disulfide_term(folder, distances)
    terms["steric"] = _classic_steric_term(folder, distances)

    angle_dict = {f"{x['res']}_{x['type']}": val for x, val in zip(folder.dof_map, angle_vec)}
    terms["rotamer"] = float(folder._calculate_rotamer_energy(angle_dict))
    terms["aromatic"] = float(
        folder._calculate_aromatic_quadrupole(coords, folder.static_labels, folder.atom_to_res)
    )
    terms["ramachandran"] = _classic_ramachandran_term(folder, angle_dict)
    structure = getattr(folder, "last_structure", None)
    if structure is None:
        structure = folder.structure_from_coords_labels(coords, folder.static_labels)
    geometry_score = score_pheat_structure(structure, "pheat-geometry-integrity")
    terms["geometry_integrity"] = float(geometry_score.total)
    geometry_subterms = {
        f"geometry_integrity.{key}": float(value)
        for key, value in getattr(geometry_score, "terms", {}).items()
    }

    total = float(sum(terms.values()))
    if return_terms:
        terms.update(geometry_subterms)
    else:
        terms = {}
    return ScoreResult(
        model=force_field,
        total=total,
        terms=terms,
        citations=[
            "kyte-doolittle-1982",
            "bondi-1964-vdw-radii",
            "qtf-classic-native-score",
        ],
        metadata={
            "engine": "qtf",
            "geometry_source": "pheat",
            "force_field": force_field,
            "stage": int(folder.current_stage),
            "options": {
                "hydrophobic_gamma": gamma,
                "end_to_end_weight": constraint_strength,
                "end_to_end_target": end_to_end_target,
            },
        },
    )


def _classic_hbond_term(folder, coords: np.ndarray) -> float:
    term = 0.0
    atom_lookup = getattr(folder, "atom_lookup", None)
    if atom_lookup is None:
        atom_lookup = {
            (int(rid), str(name).upper()): idx
            for idx, (rid, name, _elem) in enumerate(folder.static_labels)
        }
    for i_n in folder.idx_N_atoms:
        res_d = folder.atom_to_res[i_n]
        idx_ca = atom_lookup.get((int(res_d), "CA"))
        idx_prev_c = atom_lookup.get((int(res_d) - 1, "C"))
        if idx_ca is None or idx_prev_c is None:
            pos_h = coords[i_n] + np.array([0, 0, 1.0])
            pos_n = coords[i_n]
        else:
            p_c = coords[idx_prev_c]
            p_n = coords[i_n]
            p_ca = coords[idx_ca]
            v_nc = p_c - p_n
            v_nc /= np.linalg.norm(v_nc)
            v_nca = p_ca - p_n
            v_nca /= np.linalg.norm(v_nca)
            v_h = -(v_nc + v_nca)
            v_h /= np.linalg.norm(v_h)
            pos_h = p_n + v_h * 1.01
            pos_n = p_n
        o_coords = coords[folder.idx_O_atoms]
        o_res = folder.atom_to_res[folder.idx_O_atoms]
        valid_mask = np.abs(o_res - res_d) >= 2
        if not np.any(valid_mask):
            continue
        valid_o_coords = o_coords[valid_mask]
        d_ho = np.linalg.norm(valid_o_coords - pos_h, axis=1)
        close_mask = d_ho < 3.5
        if not np.any(close_mask):
            continue
        final_d_ho = d_ho[close_mask]
        final_o_coords = valid_o_coords[close_mask]
        v_hn = pos_n - pos_h
        v_hn /= np.linalg.norm(v_hn)
        v_ho = final_o_coords - pos_h
        v_ho /= np.linalg.norm(v_ho, axis=1)[:, None]
        angle_cos = np.dot(v_ho, v_hn)
        ang_mask = angle_cos < -0.4
        radial_term = np.exp(-(final_d_ho - 2.0) ** 2 / 0.5)
        angular_term = (np.abs(angle_cos) - 0.4) * 2.0
        term += float(np.sum(-25.0 * radial_term * angular_term * ang_mask))
    return term


def _classic_electrostatic_term(folder, distances: np.ndarray) -> float:
    q_mat = np.outer(folder.q_vector, folder.q_vector)
    elec_mask = np.triu(folder.mask_non_bonded, k=1) & (np.abs(q_mat) > 0.0001)
    if not np.any(elec_mask):
        return 0.0
    r_elec = np.maximum(distances[elec_mask], 1.0)
    return float(np.sum(_COULOMB_PREFACTOR * q_mat[elec_mask] / (_DIELECTRIC * r_elec)))


def _classic_disulfide_term(folder, distances: np.ndarray) -> float:
    if len(folder.idx_SG_atoms) <= 1:
        return 0.0
    sg_dists = distances[np.ix_(folder.idx_SG_atoms, folder.idx_SG_atoms)]
    sg_mask = np.triu(np.ones_like(sg_dists, dtype=bool), k=1)
    valid_dists = sg_dists[sg_mask]
    bond_strengths = np.exp(-(valid_dists - 2.05) ** 2 / 0.5)
    active_bonds = valid_dists < 3.0
    term = -float(np.sum(25.0 * bond_strengths * active_bonds))
    full_strengths = np.exp(-(sg_dists - 2.05) ** 2 / 0.5) * (sg_dists < 3.0)
    np.fill_diagonal(full_strengths, 0.0)
    saturation = np.sum(full_strengths, axis=1)
    overload = saturation - 1.0
    penalty_mask = overload > 0.1
    if np.any(penalty_mask):
        term += float(np.sum(40.0 * overload[penalty_mask] ** 2))
    return term


def _classic_steric_term(folder, distances: np.ndarray) -> float:
    sigma_mat = folder.vdw_radii_vector[:, None] + folder.vdw_radii_vector[None, :]
    heavy_mat = folder.mask_heavy[:, None] & folder.mask_heavy[None, :]
    vdw_mask = np.triu(folder.mask_non_bonded & heavy_mat, k=1)
    if not np.any(vdw_mask):
        return 0.0
    r_vdw = distances[vdw_mask]
    s_vdw = sigma_mat[vdw_mask]
    collision_mask = r_vdw < s_vdw
    if not np.any(collision_mask):
        return 0.0
    r_col = r_vdw[collision_mask]
    s_col = s_vdw[collision_mask]
    term = (s_col / (r_col + 0.1)) ** 12
    high_e = term > 50.0
    if np.any(high_e):
        term[high_e] = 50.0 + np.log(term[high_e] - 49.0)
    return float(np.sum(0.1 * term))


def _classic_ramachandran_term(folder, angle_dict: Mapping[str, float]) -> float:
    term = 0.0
    for i in range(folder.n_residues):
        if f"{i}_phi" in angle_dict and f"{i}_psi" in angle_dict:
            phi = angle_dict[f"{i}_phi"]
            psi = angle_dict[f"{i}_psi"]
            aa = folder.sequence[i]
            d_helix = (phi - (-1.0)) ** 2 + (psi - (-0.8)) ** 2
            d_sheet = (phi - (-2.3)) ** 2 + (psi - 2.4) ** 2
            if aa == "G":
                d_helix_l = (phi - 1.0) ** 2 + (psi - 0.8) ** 2
                d_sheet_l = (phi - 2.3) ** 2 + (psi - (-2.4)) ** 2
                term += -3.0 * np.exp(-min(d_helix, d_sheet, d_helix_l, d_sheet_l) / 0.6)
            else:
                d_forbidden = (phi - (-2.0)) ** 2 + (psi - 1.0) ** 2
                term += (
                    -3.0 * np.exp(-d_helix / 0.6)
                    - 3.0 * np.exp(-d_sheet / 0.6)
                    + 5.0 * np.exp(-d_forbidden / 1.0)
                )
    return float(term)
