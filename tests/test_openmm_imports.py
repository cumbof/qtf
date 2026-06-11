"""Tests for external molecular-scoring ownership.

QTF does not import or manage OpenMM directly; external molecular mechanics
scorers are discovered and validated through PHEAT.
"""


def test_qtf_discovers_pheat_openmm_scorer_capability():
    from qtf.scoring import supported_pheat_score_models

    assert "pheat-openmm-prepared" in supported_pheat_score_models()


def test_pheat_openmm_options_validate_without_qtf_openmm_helpers():
    from pheat.scoring import validate_scoring_options

    result = validate_scoring_options("pheat-openmm-prepared", {})
    assert result["model"] == "pheat-openmm-prepared"
    assert "errors" in result
    assert "warnings" in result


def test_external_executable_validation_is_reported_by_pheat():
    from pheat.scoring import validate_scoring_options

    result = validate_scoring_options("pheat-gromacs-mdrun", {"gromacs_forcefield": "amber99sb-ildn"})
    assert result["model"] == "pheat-gromacs-mdrun"
    assert "ok" in result
    assert isinstance(result["errors"], list)
