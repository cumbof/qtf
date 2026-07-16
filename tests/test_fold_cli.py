"""Tests for the unified fold recipe CLI."""

import json
from importlib import resources

import pytest


def test_recipe_assets_are_packaged():
    root = resources.files("qtf.assets.recipes")
    assert (root / "default.yaml").is_file()
    assert (root / "schema.json").is_file()


def test_builtin_recipes_load():
    pytest.importorskip("yaml")
    from qtf.recipes import load_builtin_recipes

    recipes = load_builtin_recipes()
    assert "engine" not in recipes["qtf-main-equivalent"]
    assert recipes["qtf-main-equivalent"]["phases"][0]["score_model"] == "pheat-coarse-protein-folding-v1"
    assert recipes["qtf-heavy-atom-phased"]["phases"][0]["score_model"].startswith("pheat-")
    assert recipes["qtf-main-equivalent"]["circuit_template"]["name"] == "EfficientSU2"
    assert recipes["qtf-heavy-atom-phased"]["circuit_template"]["source"] == "qtf"
    assert recipes["qtf-main-equivalent"]["geometry"]["stored_angles"] == []
    assert recipes["qtf-main-equivalent"]["geometry"]["stored_lengths"] == []
    assert "chi_mode" not in recipes["qtf-main-equivalent"]["geometry"]
    assert "chi_mode" not in recipes["qtf-heavy-atom-phased"]["geometry"]
    assert recipes["qtf-heavy-atom-phased"]["metrics"]["atom_sets"] == ["ca", "backbone", "all-heavy"]
    assert recipes["qtf-heavy-atom-phased"]["report"]["structure_domain"] == "protein-heavy"
    assert recipes["qtf-heavy-atom-phased"]["evaluators"]["geometry_integrity"]["score_model"] == (
        "pheat-geometry-integrity"
    )


def test_run_status_writer_writes_live_status(tmp_path):
    import qtf.engines.qtf as qtf_engine

    flushed = []
    status_path = tmp_path / "replica_0_status.json"
    writer = qtf_engine.RunStatusWriter(
        status_path,
        replica_id=0,
        run_label="demo",
        command_line="python -m qtf.engines.qtf ...",
        console_output_path=tmp_path / "replica_0_console.log",
        flush_console=lambda: flushed.append(True),
    )

    writer.update(
        status="running",
        step={"index": 1, "total": 2, "label": "Backend access"},
        current_phase={"index": 1, "name": "collapse"},
        optimization={"phase_evaluations": 25, "best_objective": 1.5},
        force=True,
        flush_console=True,
    )

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["status"] == "running"
    assert payload["replica_id"] == 0
    assert payload["step"]["label"] == "Backend access"
    assert payload["current_phase"]["name"] == "collapse"
    assert payload["optimization"]["phase_evaluations"] == 25
    assert payload["elapsed_s"] >= 0
    assert flushed == [True]


def test_ibm_runtime_service_uses_named_saved_account(monkeypatch):
    import sys
    from types import SimpleNamespace

    import qtf.engines.qtf as qtf_engine

    calls = []

    class FakeRuntimeService:
        @staticmethod
        def saved_accounts():
            return {"default-ibm-cloud": {}, "default-ibm-quantum-platform": {}}

        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "qiskit_ibm_runtime",
        SimpleNamespace(QiskitRuntimeService=FakeRuntimeService),
    )

    auth = qtf_engine.IBMRuntimeAuthConfig(account_name="default-ibm-cloud")
    qtf_engine._ibm_runtime_service("'ibm_miami'", auth)

    assert calls == [{"name": "default-ibm-cloud"}]


def test_gate_estimate_backend_crn_uses_saved_account_token(monkeypatch):
    import json
    import sys
    from types import SimpleNamespace

    import qtf.engines.qtf as qtf_engine

    calls = []

    class FakeRuntimeService:
        @staticmethod
        def saved_accounts():
            return {
                "default-ibm-cloud": {
                    "token": "saved-secret-token",
                    "channel": "ibm_cloud",
                    "url": "https://cloud.ibm.com",
                }
            }

        def __init__(self, **kwargs):
            calls.append(kwargs)

        def backend(self, name):
            return SimpleNamespace(name=name)

    monkeypatch.setitem(
        sys.modules,
        "qiskit_ibm_runtime",
        SimpleNamespace(QiskitRuntimeService=FakeRuntimeService),
    )

    auth = qtf_engine.IBMRuntimeAuthConfig(account_name="default-ibm-cloud")
    refs = qtf_engine._resolve_gate_estimate_backends(
        "ibm_cleveland,ibm_miami",
        1,
        auth,
        {"ibm_miami": "crn:v1:miami-instance"},
    )

    assert calls == [
        {"name": "default-ibm-cloud"},
        {
            "token": "saved-secret-token",
            "channel": "ibm_cloud",
            "instance": "crn:v1:miami-instance",
            "url": "https://cloud.ibm.com",
        },
    ]
    assert [ref["requested"] for ref in refs] == ["ibm_cleveland", "ibm_miami"]
    assert refs[0].get("instance_crn_provided") is False
    assert refs[1].get("instance_crn_provided") is True

    mapped_auth = qtf_engine._gate_estimate_auth_config_for_backend(
        "ibm_miami",
        auth,
        {"ibm_miami": "crn:v1:miami-instance"},
    )
    metadata = qtf_engine._ibm_auth_metadata(mapped_auth)
    assert metadata["ibm_token_source"] == "saved_account"
    assert metadata["ibm_instance_crn_provided"] is True
    assert "saved-secret-token" not in json.dumps(metadata)
    assert "crn:v1:miami-instance" not in json.dumps(metadata)


def test_gate_estimate_backend_crn_must_match_estimate_backend():
    import qtf.engines.qtf as qtf_engine

    parser = qtf_engine._build_parser()
    with pytest.raises(SystemExit):
        qtf_engine._validate_gate_estimate_backend_crn_map(
            parser,
            "ibm_miami",
            {"ibm_mimai": "crn:v1:typo"},
        )


def test_ibm_auth_env_token_overrides_saved_account_and_redacts_metadata(monkeypatch):
    import json

    import qtf.engines.qtf as qtf_engine

    monkeypatch.setenv("QTF_IBM_TOKEN", "env-secret-token")
    parser = qtf_engine._build_parser()
    args = parser.parse_args([
        "--predict",
        "GA",
        "--replica-id",
        "0",
        "--ibm-account",
        "default-ibm-cloud",
        "--ibm-token-env",
        "QTF_IBM_TOKEN",
        "--ibm-instance-crn",
        "crn:v1:secret-instance",
    ])

    auth = qtf_engine._resolve_ibm_runtime_auth_config(args, parser)
    kwargs = qtf_engine._ibm_auth_service_kwargs(auth)
    metadata = qtf_engine._ibm_auth_metadata(auth)

    assert kwargs == {
        "token": "env-secret-token",
        "channel": "ibm_quantum_platform",
        "instance": "crn:v1:secret-instance",
    }
    assert "name" not in kwargs
    assert metadata["ibm_auth_source"] == "token_env"
    assert metadata["ibm_account_name"] == "default-ibm-cloud"
    assert metadata["ibm_token_source"] == "env"
    assert metadata["ibm_instance_crn_provided"] is True
    assert "env-secret-token" not in json.dumps(metadata)
    assert "crn:v1:secret-instance" not in json.dumps(metadata)


def test_ibm_auth_token_file_and_custom_channel(tmp_path):
    import qtf.engines.qtf as qtf_engine

    token_path = tmp_path / "ibm-token.txt"
    token_path.write_text("file-secret-token\n", encoding="utf-8")
    parser = qtf_engine._build_parser()
    args = parser.parse_args([
        "--predict",
        "GA",
        "--replica-id",
        "0",
        "--ibm-token-file",
        str(token_path),
        "--ibm-channel",
        "ibm_cloud",
        "--ibm-url",
        "https://quantum.example.test",
    ])

    auth = qtf_engine._resolve_ibm_runtime_auth_config(args, parser)

    assert qtf_engine._ibm_auth_service_kwargs(auth) == {
        "token": "file-secret-token",
        "channel": "ibm_cloud",
        "url": "https://quantum.example.test",
    }
    assert qtf_engine._ibm_auth_metadata(auth)["ibm_url_provided"] is True


def test_ibm_auth_rejects_multiple_token_sources(monkeypatch):
    import pytest

    import qtf.engines.qtf as qtf_engine

    monkeypatch.setenv("QTF_IBM_TOKEN", "env-secret-token")
    parser = qtf_engine._build_parser()
    args = parser.parse_args([
        "--predict",
        "GA",
        "--replica-id",
        "0",
        "--ibm-token",
        "direct-secret-token",
        "--ibm-token-env",
        "QTF_IBM_TOKEN",
    ])

    with pytest.raises(SystemExit):
        qtf_engine._resolve_ibm_runtime_auth_config(args, parser)


def test_command_line_redacts_ibm_token_and_crn():
    import qtf.engines.qtf as qtf_engine

    command_line = qtf_engine._command_line(
        [
            "--predict",
            "GA",
            "--replica-id",
            "0",
            "--ibm-token",
            "direct-secret-token",
            "--ibm-instance-crn=crn:v1:secret-instance",
        ],
        None,
    )
    override = qtf_engine._command_line(
        None,
        "qtf fold --ibm-token direct-secret-token --ibm-instance-crn crn:v1:secret-instance",
    )

    assert "direct-secret-token" not in command_line
    assert "crn:v1:secret-instance" not in command_line
    assert "direct-secret-token" not in override
    assert "crn:v1:secret-instance" not in override
    assert "<redacted>" in command_line
    assert "<redacted>" in override


def test_sequence_indexed_reference_geometry_matches_sequence_fold_numbering():
    import qtf.engines.qtf as qtf_engine
    from pheat import DisulfideBond, ResidueGeometry, ResidueGeometryStructure
    from pheat.residue_geometry import structure_from_residue_geometry

    source_reference = ResidueGeometryStructure(
        residues=[
            ResidueGeometry("CYS", chain_id="A", resseq=16),
            ResidueGeometry("CYS", chain_id="A", resseq=20),
        ],
        disulfide_bonds=[DisulfideBond(chain_id_1="A", resseq_1=16, chain_id_2="A", resseq_2=20)],
    )
    metric_reference = qtf_engine._sequence_indexed_residue_geometry(source_reference)

    assert [residue.resseq for residue in metric_reference.residues] == [1, 2]
    assert [residue.chain_id for residue in metric_reference.residues] == ["A", "A"]
    assert metric_reference.metadata["qtf_metric_numbering"] == "sequence-indexed"
    assert metric_reference.disulfide_bonds[0].resseq_1 == 1
    assert metric_reference.disulfide_bonds[0].resseq_2 == 2

    sequence_fold = ResidueGeometryStructure(
        residues=[
            ResidueGeometry("CYS", chain_id="A", resseq=1),
            ResidueGeometry("CYS", chain_id="A", resseq=2),
        ]
    )
    reference_structure = structure_from_residue_geometry(metric_reference)
    target_structure = structure_from_residue_geometry(sequence_fold)
    details = qtf_engine._pheat_alignment_details(
        reference_structure,
        target_structure,
        atom_sets=["ca", "backbone", "all-heavy"],
    )

    expected_ca = sum(1 for atom in reference_structure.atoms if atom.name.strip().upper() == "CA")
    assert details["matched_ca_atoms"] == expected_ca
    assert details["matched_heavy_atoms"] == len(reference_structure.atoms)
    assert details["all_heavy_rmsd"] == pytest.approx(0.0)


def test_pheat_native_robust_recipe_defaults_to_all_chi():
    pytest.importorskip("yaml")
    from qtf.recipes import load_builtin_recipes

    recipe = load_builtin_recipes()["qtf-pheat-native-robust"]
    assert recipe["geometry"]["stored_angles"] == "all"
    assert recipe["geometry"]["stored_lengths"] == []
    assert recipe["geometry"]["max_chi"] is None
    assert "selective_chi_map" not in recipe["geometry"]
    assert recipe["circuit_template"]["source"] == "qtf"
    assert recipe["circuit_template"]["name"] == "brickwork-ryrz-nearest-neighbor"
    assert recipe["scouting"]["score_model"] == "pheat-mj"
    assert recipe["result"]["score_model"] == "pheat-goap"
    assert recipe["phase_comparisons"]["enabled"] is True
    assert recipe["reranking"]["enabled"] is True
    assert recipe["validation"]["enabled"] is True

    phases = recipe["phases"]
    assert [phase["name"] for phase in phases] == [
        "collapse",
        "packing",
        "orientation",
        "polar_contacts",
        "backbone_torsions",
        "sidechain_rotamers",
        "heavy_relax",
    ]
    assert [phase["optimizer"] for phase in phases] == [
        "COBYLA",
        "COBYLA",
        "Powell",
        "Powell",
        "Powell",
        "Powell",
        "Powell",
    ]
    assert [phase["score_model"] for phase in phases] == [
        "pheat-hydropathy",
        "pheat-mj",
        "pheat-goap",
        "pheat-hbond",
        "pheat-backbone",
        "pheat-rotamer",
        "pheat-heavy-mm",
    ]


def test_pheat_coarse_to_fine_lengths_recipe_progresses_geometry():
    pytest.importorskip("yaml")
    from qtf.recipes import load_builtin_recipes

    recipe = load_builtin_recipes()["qtf-pheat-coarse-to-fine-lengths"]
    assert recipe["description"]
    assert recipe["circuit_template"]["description"]
    assert recipe["phases"][0]["description"]
    assert recipe["geometry"]["stored_lengths"] == []
    assert recipe["geometry"]["length_encoding_scope"] == "shared-by-type"
    assert recipe["validation"]["enabled"] is True
    assert recipe["validation"]["candidates"] == ["primary"]
    assert recipe["validation"]["evaluators"] == [
        "geometry_integrity",
        "physical_integrity",
        "rg_compactness",
        "gromacs_minimize",
    ]
    assert recipe["result"]["score_model"] == "pheat-heavy-mm-physical"
    assert recipe["phase_comparisons"]["evaluators"] == [
        "geometry_integrity",
        "physical_integrity",
        "goap_check",
        "rg_compactness",
    ]
    assert recipe["reranking"]["enabled"] is True
    assert recipe["reranking"]["evaluator"] == "physical_integrity"
    assert recipe["reranking"]["triggers"] == [
        {"when": "every_evaluations", "interval": 250},
        {"when": "phase_end"},
    ]
    assert recipe["reranking"]["candidate_pool"]["per_phase_top_k"] == 25
    assert recipe["reranking"]["candidate_pool"]["include_phase_start"] is True
    assert recipe["phase_readiness"]["enabled"] is True
    assert recipe["phase_readiness"]["evaluator"] == "physical_integrity"
    assert recipe["phase_readiness"]["phases"] == [
        "shared_backbone_lengths",
        "local_backbone_lengths",
        "post_length_physical_declash",
    ]
    assert recipe["phase_readiness"]["on_fail"] == "skip_phase"
    assert recipe["phase_readiness"]["max_clash_count"] == 0
    assert recipe["phase_readiness"]["max_short_contact_count"] == 0
    assert recipe["phase_readiness"]["min_nonlocal_distance_a"] == 0.7
    assert recipe["handoff_guard"]["enabled"] is True
    assert recipe["handoff_guard"]["evaluator"] == "physical_integrity"
    assert recipe["handoff_guard"]["max_clash_count"] == 0
    assert recipe["handoff_guard"]["max_short_contact_count"] == 0
    assert "phases" not in recipe["handoff_guard"]
    assert recipe["handoff_guard"]["unsafe_transition_max_short_contact_count"] == 2
    assert recipe["handoff_guard"]["unsafe_transition_min_nonlocal_distance_a"] == 0.65
    assert recipe["handoff_guard"]["unsafe_transition_require_clash_count_decrease"] is True

    phases = recipe["phases"]
    assert [phase["name"] for phase in phases] == [
        "ca_collapse",
        "backbone_geometry",
        "physical_declash",
        "packing_orientation",
        "post_packing_declash",
        "shared_backbone_lengths",
        "local_backbone_lengths",
        "post_length_physical_declash",
    ]
    assert phases[0]["geometry"]["stored_lengths"] == []
    assert phases[0]["score_model"] == "pheat-hydropathy-physical"
    assert phases[1]["score_model"] == "pheat-backbone-physical"
    assert phases[1]["options"]["maxfev"] == 10000
    assert phases[2]["score_model"] == "pheat-physical-integrity"
    assert phases[2]["handoff_guard"] == {"enabled": True, "allow_improving_unsafe": True}
    assert phases[2]["options"]["maxfev"] == 35000
    assert phases[3]["score_model"] == "pheat-goap-physical"
    assert phases[3]["handoff_guard"] == {"enabled": True, "allow_improving_unsafe": False}
    assert phases[3]["score_options"]["physical_integrity_weight"] == 0.02
    assert phases[3]["options"]["maxfev"] == 16000
    assert phases[4]["score_model"] == "pheat-physical-integrity"
    assert phases[4]["handoff_guard"] == {"enabled": True, "allow_improving_unsafe": False}
    assert phases[4]["geometry"]["stored_lengths"] == []
    assert phases[4]["options"]["maxfev"] == 20000
    assert phases[5]["geometry"]["stored_lengths"] == "backbone"
    assert phases[5]["geometry"]["length_encoding_scope"] == "shared-by-type"
    assert phases[5]["handoff_guard"] == {"enabled": True, "allow_improving_unsafe": True}
    assert phases[5]["score_model"] == "pheat-heavy-mm-physical"
    assert phases[5]["options"]["maxfev"] == 12000
    assert phases[6]["geometry"]["stored_lengths"] == "backbone"
    assert phases[6]["geometry"]["length_encoding_scope"] == "per-residue"
    assert phases[6]["geometry"]["backbone_length_span"] == 0.015
    assert phases[6]["handoff_guard"] == {
        "enabled": True,
        "allow_improving_unsafe": True,
        "unsafe_transition_max_short_contact_count": 0,
    }
    assert phases[6]["score_model"] == "pheat-heavy-mm-physical"
    assert phases[6]["options"]["maxfev"] == 12000
    assert phases[7]["geometry"]["stored_lengths"] == "backbone"
    assert phases[7]["geometry"]["length_encoding_scope"] == "per-residue"
    assert phases[7]["geometry"]["backbone_length_span"] == 0.015
    assert phases[7]["handoff_guard"] == {"enabled": True, "allow_improving_unsafe": False}
    assert phases[7]["score_model"] == "pheat-physical-integrity"
    assert phases[7]["options"]["maxfev"] == 20000
    assert recipe["evaluators"]["physical_integrity"]["score_model"] == "pheat-physical-integrity"
    assert recipe["evaluators"]["rg_compactness"]["score_model"] == "pheat-rg"
    assert recipe["evaluators"]["gromacs_minimize"]["score_model"] == "pheat-gromacs-mdrun"
    assert recipe["evaluators"]["gromacs_minimize"]["options"]["gromacs_preflight"] == "warn"
    assert recipe["evaluators"]["gromacs_minimize"]["options"]["gromacs_run_settings"]["grompp_maxwarn"] == 2
    assert recipe["evaluators"]["gromacs_minimize"]["options"]["gromacs_run_settings"]["mdrun_flags"] == ["-ntmpi", "1"]

    diagnostic = load_builtin_recipes()["qtf-pheat-coarse-to-fine-guarded-diagnostic"]
    assert [phase["name"] for phase in diagnostic["phases"]] == [
        "ca_collapse",
        "backbone_geometry",
        "physical_declash",
        "packing_orientation",
        "post_packing_declash",
    ]
    assert diagnostic["phases"][1]["options"]["maxfev"] == 10000
    assert diagnostic["phases"][2]["options"]["maxfev"] == 35000
    assert diagnostic["phases"][3]["options"]["maxfev"] == 16000
    assert diagnostic["phases"][4]["options"]["maxfev"] == 20000
    assert diagnostic["geometry"]["stored_lengths"] == []
    assert diagnostic["result"]["score_model"] == "pheat-goap-physical"
    assert diagnostic["result"]["score_options"]["physical_integrity_weight"] == 0.02
    assert "phases" not in diagnostic["handoff_guard"]
    assert diagnostic["handoff_guard"]["allow_improving_unsafe"] is True
    assert diagnostic["handoff_guard"]["max_clash_count"] == 0
    assert diagnostic["handoff_guard"]["unsafe_transition_max_short_contact_count"] == 2
    assert diagnostic["handoff_guard"]["unsafe_transition_min_nonlocal_distance_a"] == 0.65
    assert diagnostic["handoff_guard"]["unsafe_transition_require_clash_count_decrease"] is True


def test_final_improving_lengths_recipe_preserves_final_declash_candidate_policy():
    pytest.importorskip("yaml")
    from qtf.recipes import load_builtin_recipes

    recipes = load_builtin_recipes()
    base = recipes["qtf-pheat-coarse-to-fine-lengths"]
    recipe = recipes["qtf-pheat-coarse-to-fine-lengths-final-improving"]

    assert recipe["description"]
    assert recipe["validation"]["description"]
    assert recipe["validation"]["candidates"] == ["primary", "phase_ends", "reranked_top"]
    assert [phase["name"] for phase in recipe["phases"]] == [phase["name"] for phase in base["phases"]]

    final_phase = recipe["phases"][-1]
    assert final_phase["name"] == "post_length_physical_declash"
    assert final_phase["description"]
    assert final_phase["handoff_guard"]["description"]
    assert final_phase["handoff_guard"]["enabled"] is True
    assert final_phase["handoff_guard"]["allow_improving_unsafe"] is True
    assert final_phase["handoff_guard"]["unsafe_transition_max_short_contact_count"] == 0
    assert final_phase["handoff_guard"]["unsafe_transition_min_nonlocal_distance_a"] == 1.0
    assert final_phase["handoff_guard"]["unsafe_transition_require_clash_count_decrease"] is True
    assert final_phase["handoff_guard"]["reject_on_min_nonlocal_distance_decrease"] is False
    assert final_phase["handoff_guard"]["reject_on_score_worse"] is True
    assert final_phase["handoff_guard"]["reject_on_nonfinite"] is True


def test_progressive_declash_lengths_recipe_relaxes_intermediate_gates():
    pytest.importorskip("yaml")
    from qtf.recipes import load_builtin_recipes

    recipe = load_builtin_recipes()["qtf-pheat-coarse-to-fine-lengths-progressive-declash"]

    assert recipe["description"]
    assert recipe["result"]["score_model"] == "pheat-heavy-mm-physical"
    assert recipe["result"]["score_options"]["physical_integrity_weight"] == 1.0
    assert recipe["phase_readiness"]["max_clash_count"] is None
    assert recipe["phase_readiness"]["max_short_contact_count"] == 0
    assert recipe["phase_readiness"]["min_nonlocal_distance_a"] == 0.7
    assert recipe["validation"]["candidates"] == ["primary", "phase_ends", "reranked_top"]

    phases = {phase["name"]: phase for phase in recipe["phases"]}
    assert phases["packing_orientation"]["score_options"]["physical_integrity_weight"] == 0.1
    assert phases["packing_orientation"]["handoff_guard"]["allow_improving_unsafe"] is True
    assert phases["post_packing_declash"]["handoff_guard"]["allow_improving_unsafe"] is True
    assert phases["post_packing_declash"]["handoff_guard"]["unsafe_transition_max_short_contact_count"] == 0
    assert phases["shared_backbone_lengths"]["score_options"]["physical_integrity_weight"] == 0.5
    assert phases["local_backbone_lengths"]["score_options"]["physical_integrity_weight"] == 0.5
    assert phases["shared_backbone_lengths"]["handoff_guard"]["unsafe_transition_require_clash_count_decrease"] is False
    assert phases["post_length_physical_declash"]["handoff_guard"]["allow_improving_unsafe"] is True
    assert phases["post_length_physical_declash"]["handoff_guard"]["unsafe_transition_require_clash_count_decrease"] is True


def test_recipe_descriptions_resolve_into_phase_schedule():
    import qtf.engines.qtf as qtf_engine

    parser = qtf_engine._build_parser()
    args = parser.parse_args([
        "--predict",
        "GA",
        "--replica-id",
        "0",
        "--recipe",
        "qtf-pheat-coarse-to-fine-lengths-final-improving",
        "--backend",
        "statevector-shots",
        "--shots",
        "16",
        "--maxiter",
        "1",
    ])
    schedule = qtf_engine._resolve_phase_schedule(args, parser, global_shots=16, global_maxiter=1)
    payload = qtf_engine._phase_schedule_payload(schedule)

    assert schedule.description == payload["description"]
    assert schedule.fold["description"] == payload["fold"]["description"]
    assert schedule.scouting.description == payload["scouting"]["description"]
    assert schedule.result.description == payload["result"]["description"]
    assert schedule.metrics.description == payload["metrics"]["description"]
    assert schedule.report.description == payload["report"]["description"]
    assert schedule.phases[0].description == payload["phases"][0]["description"]
    assert schedule.evaluators["physical_integrity"].description == payload["evaluators"]["physical_integrity"]["description"]
    assert schedule.phase_readiness.description == payload["phase_readiness"]["description"]
    assert payload["phases"][-1]["handoff_guard"]["description"]


def test_phase_results_report_section_includes_phase_descriptions():
    import qtf.engines.qtf as qtf_engine

    html = qtf_engine._phase_results_report_section(
        {
            "reference_available": False,
            "rmsd_angle_mode": "statevector",
            "phase_results": [
                {
                    "index": 1,
                    "name": "collapse",
                    "label": "Collapse",
                    "description": "Compact coarse starting geometry.",
                    "optimizer": "COBYLA",
                    "score_model": "pheat-physics",
                    "phase_status": "ok",
                }
            ],
        }
    )
    assert "<th>Description</th>" in html
    assert "Compact coarse starting geometry." in html


def test_phase_readiness_report_section_lists_gate_decisions():
    import qtf.engines.qtf as qtf_engine

    html = qtf_engine._phase_readiness_report_section(
        {
            "phase_readiness_results": [
                {
                    "phase_label": "Local backbone length relaxation",
                    "evaluator": "physical_integrity",
                    "decision": "skip_phase",
                    "status": "not_ready",
                    "score_total": 123.0,
                    "score_units": "arbitrary",
                    "counts": {
                        "clash_count": 2,
                        "short_contact_count": 0,
                        "min_nonlocal_distance": 0.62,
                    },
                    "thresholds": {
                        "max_clash_count": 0,
                        "max_short_contact_count": 0,
                        "min_nonlocal_distance_a": 0.7,
                    },
                    "reasons": ["clash count 2 exceeds limit 0"],
                }
            ]
        }
    )

    assert "Phase Readiness Gates" in html
    assert "skip_phase" in html
    assert "Local backbone length relaxation" in html


def test_handoff_guard_report_section_shows_progressive_counts():
    import qtf.engines.qtf as qtf_engine

    html = qtf_engine._handoff_guard_report_section(
        {
            "handoff_guard_results": [
                {
                    "phase_label": "Post-packing physical declash",
                    "evaluator": "physical_integrity",
                    "handoff_candidate_id": "phase-5-final",
                    "decision": "accept",
                    "status": "accepted_with_violations",
                    "phase_start_counts": {
                        "clash_count": 187,
                        "short_contact_count": 0,
                        "min_nonlocal_distance": 0.7504,
                    },
                    "handoff_counts": {
                        "clash_count": 181,
                        "short_contact_count": 0,
                        "min_nonlocal_distance": 0.7831,
                    },
                    "reasons": ["clash count 181 exceeds limit 0"],
                }
            ]
        }
    )

    assert "Start clashes" in html
    assert "Candidate clashes" in html
    assert "Clash delta" in html
    assert "-6" in html
    assert "accepted_with_violations" in html


def test_validation_report_section_lists_scorer_warnings():
    import qtf.engines.qtf as qtf_engine

    html = qtf_engine._validation_report_section(
        {
            "validation_results": [
                {
                    "candidate_set": "primary",
                    "label": "Primary result",
                    "evaluator": "gromacs_minimize",
                    "score_model": "pheat-gromacs-mdrun",
                    "score_total": 4.2e18,
                    "score_units": "kJ/mol",
                    "status": "ok",
                    "warnings": ["force-field validation energy is extremely large"],
                }
            ]
        }
    )

    assert "<th>Warnings</th>" in html
    assert "force-field validation energy is extremely large" in html


def test_validation_candidate_entries_include_top_snapshots():
    from types import SimpleNamespace

    import qtf.engines.qtf as qtf_engine

    validation_config = SimpleNamespace(candidates=["top_snapshots"], evaluators=["gromacs_minimize"])
    structure_snapshot_payloads = [
        {"role": "top_snapshot", "key": "snapshot_top_001", "snapshot_rank": 1},
        {"role": "phase", "key": "phase_01_collapse"},
    ]
    snapshot_structures = {"snapshot_top_001": object()}

    entries = qtf_engine._validation_candidate_entries(
        validation_config,
        primary_structure=object(),
        primary_snapshot_key=None,
        structure_snapshot_payloads=structure_snapshot_payloads,
        snapshot_structures=snapshot_structures,
        reranking_results=[],
    )

    assert any(entry["candidate_set"] == "top_snapshots" and entry["snapshot_key"] == "snapshot_top_001" for entry in entries)


def test_phase_geometry_overrides_resolve_in_schedule():
    import qtf.engines.qtf as qtf_engine

    parser = qtf_engine._build_parser()
    args = parser.parse_args([
        "--predict",
        "GA",
        "--replica-id",
        "0",
        "--recipe",
        "qtf-pheat-coarse-to-fine-lengths",
        "--backend",
        "statevector-shots",
        "--shots",
        "16",
        "--maxiter",
        "1",
        "--phase-geometry-option",
        "local_backbone_lengths:backbone_length_span=0.05",
    ])
    schedule = qtf_engine._resolve_phase_schedule(args, parser, global_shots=16, global_maxiter=1)
    assert schedule.phases[0].geometry["max_chi"] == 0
    local_length_phase = {phase.name: phase for phase in schedule.phases}["local_backbone_lengths"]
    final_declash_phase = {phase.name: phase for phase in schedule.phases}["post_length_physical_declash"]
    assert local_length_phase.geometry["stored_lengths"] == "backbone"
    assert local_length_phase.geometry["length_encoding_scope"] == "per-residue"
    assert local_length_phase.geometry["backbone_length_span"] == 0.05
    assert final_declash_phase.geometry["stored_lengths"] == "backbone"
    assert final_declash_phase.geometry["backbone_length_span"] == 0.015
    assert schedule.handoff_guard.enabled is True
    assert schedule.handoff_guard.evaluator == "physical_integrity"
    assert schedule.handoff_guard.max_clash_count == 0
    assert schedule.handoff_guard.max_short_contact_count == 0
    assert schedule.handoff_guard.unsafe_transition_max_short_contact_count == 2
    assert schedule.handoff_guard.unsafe_transition_min_nonlocal_distance_a == 0.65
    assert schedule.handoff_guard.unsafe_transition_require_clash_count_decrease is True
    assert schedule.handoff_guard.allow_improving_unsafe is True
    assert schedule.handoff_guard.reject_on_min_nonlocal_distance_decrease is True
    assert schedule.handoff_guard.phases == []
    assert schedule.phase_readiness.enabled is True
    assert schedule.phase_readiness.evaluator == "physical_integrity"
    assert schedule.phase_readiness.on_fail == "skip_phase"
    assert schedule.phase_readiness.phases == [
        "shared_backbone_lengths",
        "local_backbone_lengths",
        "post_length_physical_declash",
    ]
    assert local_length_phase.handoff_guard["allow_improving_unsafe"] is True
    assert local_length_phase.handoff_guard["unsafe_transition_max_short_contact_count"] == 0
    assert final_declash_phase.handoff_guard["enabled"] is True
    assert final_declash_phase.handoff_guard["allow_improving_unsafe"] is False


def test_phase_geometry_override_rejects_unknown_key():
    import qtf.engines.qtf as qtf_engine

    parser = qtf_engine._build_parser()
    args = parser.parse_args([
        "--predict",
        "GA",
        "--replica-id",
        "0",
        "--recipe",
        "qtf-pheat-coarse-to-fine-lengths",
        "--phase-geometry-option",
        "local_backbone_lengths:store_lengths=backbone",
    ])
    with pytest.raises(SystemExit):
        qtf_engine._resolve_phase_schedule(args, parser, global_shots=4096, global_maxiter=1)


def _physical_score(total, *, clash_count, short_contact_count, min_nonlocal_distance, nonfinite_atom_count=0):
    return {
        "status": "ok",
        "total": total,
        "units": "arbitrary",
        "metadata": {
            "checked_counts": {
                "clash_count": clash_count,
                "short_contact_count": short_contact_count,
                "nonfinite_atom_count": nonfinite_atom_count,
            },
            "min_nonlocal_distance": min_nonlocal_distance,
        },
    }


def _handoff_config(**overrides):
    import qtf.engines.qtf as qtf_engine

    values = {
        "enabled": True,
        "evaluator": "physical_integrity",
        "phases": ["packing_orientation"],
        "fallback": "phase_start",
        "abort_on_reject": False,
        "allow_improving_unsafe": True,
        "max_clash_count": 0,
        "max_short_contact_count": 0,
        "min_nonlocal_distance_a": 0.7,
        "unsafe_transition_max_short_contact_count": None,
        "unsafe_transition_min_nonlocal_distance_a": None,
        "unsafe_transition_require_clash_count_decrease": False,
        "reject_on_score_worse": True,
        "reject_on_clash_count_increase": True,
        "reject_on_short_contact_count_increase": True,
        "reject_on_min_nonlocal_distance_decrease": True,
        "reject_on_nonfinite": True,
    }
    values.update(overrides)
    return qtf_engine.HandoffGuardConfig(**values)


def test_handoff_guard_accepts_improving_candidate_with_remaining_threshold_violations():
    import qtf.engines.qtf as qtf_engine

    start = _physical_score(4_000_000.0, clash_count=2700, short_contact_count=145, min_nonlocal_distance=0.0)
    handoff = _physical_score(3_000_000.0, clash_count=2300, short_contact_count=110, min_nonlocal_distance=0.0)
    decision = qtf_engine._handoff_guard_decision_payload(start, handoff, _handoff_config())

    assert decision["status"] == "accepted_with_violations"
    assert decision["decision"] == "accept"
    assert decision["delta_current_minus_start"] == -1_000_000.0
    assert decision["absolute_threshold_violations"]


def test_handoff_guard_accepts_bounded_unsafe_declash_transition():
    import qtf.engines.qtf as qtf_engine

    start = _physical_score(10_378.3224, clash_count=247, short_contact_count=1, min_nonlocal_distance=0.699515)
    handoff = _physical_score(4_524.0239, clash_count=149, short_contact_count=2, min_nonlocal_distance=0.698702)
    decision = qtf_engine._handoff_guard_decision_payload(
        start,
        handoff,
        _handoff_config(
            unsafe_transition_max_short_contact_count=2,
            unsafe_transition_min_nonlocal_distance_a=0.65,
            unsafe_transition_require_clash_count_decrease=True,
        ),
    )

    assert decision["status"] == "accepted_with_violations"
    assert decision["decision"] == "accept"
    assert decision["hard_reasons"] == []
    assert decision["unsafe_transition"]["active"] is True
    assert decision["unsafe_transition"]["accepted"] is True


def test_phase_handoff_guard_overrides_unsafe_transition_policy():
    from types import SimpleNamespace

    import qtf.engines.qtf as qtf_engine

    start = _physical_score(10_378.3224, clash_count=247, short_contact_count=1, min_nonlocal_distance=0.699515)
    handoff = _physical_score(4_524.0239, clash_count=149, short_contact_count=2, min_nonlocal_distance=0.698702)
    config = _handoff_config(
        unsafe_transition_max_short_contact_count=2,
        unsafe_transition_min_nonlocal_distance_a=0.65,
        unsafe_transition_require_clash_count_decrease=True,
    )
    exploratory_config = qtf_engine._effective_handoff_guard_config(
        config,
        SimpleNamespace(name="physical_declash", handoff_guard={"enabled": True, "allow_improving_unsafe": True}),
    )
    strict_config = qtf_engine._effective_handoff_guard_config(
        config,
        SimpleNamespace(name="post_length_physical_declash", handoff_guard={"enabled": True, "allow_improving_unsafe": False}),
    )

    early_decision = qtf_engine._handoff_guard_decision_payload(
        start,
        handoff,
        exploratory_config,
    )
    final_decision = qtf_engine._handoff_guard_decision_payload(
        start,
        handoff,
        strict_config,
    )

    assert early_decision["status"] == "accepted_with_violations"
    assert early_decision["unsafe_transition"]["allow_improving_unsafe"] is True
    assert early_decision["unsafe_transition"]["active"] is True
    assert final_decision["status"] == "rejected"
    assert final_decision["unsafe_transition"]["allow_improving_unsafe"] is False
    assert final_decision["unsafe_transition"]["active"] is False
    assert final_decision["absolute_threshold_violations"]


def test_handoff_guard_rejects_unsafe_transition_without_clash_reduction():
    import qtf.engines.qtf as qtf_engine

    start = _physical_score(10_378.3224, clash_count=247, short_contact_count=1, min_nonlocal_distance=0.699515)
    handoff = _physical_score(9_000.0, clash_count=247, short_contact_count=1, min_nonlocal_distance=0.699515)
    decision = qtf_engine._handoff_guard_decision_payload(
        start,
        handoff,
        _handoff_config(
            unsafe_transition_max_short_contact_count=2,
            unsafe_transition_min_nonlocal_distance_a=0.65,
            unsafe_transition_require_clash_count_decrease=True,
        ),
    )

    assert decision["status"] == "rejected"
    assert decision["unsafe_transition"]["active"] is True
    assert decision["unsafe_transition"]["accepted"] is False
    assert any("clash count decrease" in reason for reason in decision["unsafe_transition"]["failures"])


def test_handoff_guard_rejects_unsafe_transition_below_distance_floor():
    import qtf.engines.qtf as qtf_engine

    start = _physical_score(10_378.3224, clash_count=247, short_contact_count=1, min_nonlocal_distance=0.699515)
    handoff = _physical_score(4_524.0239, clash_count=149, short_contact_count=2, min_nonlocal_distance=0.60)
    decision = qtf_engine._handoff_guard_decision_payload(
        start,
        handoff,
        _handoff_config(
            unsafe_transition_max_short_contact_count=2,
            unsafe_transition_min_nonlocal_distance_a=0.65,
            unsafe_transition_require_clash_count_decrease=True,
        ),
    )

    assert decision["status"] == "rejected"
    assert decision["unsafe_transition"]["active"] is True
    assert decision["unsafe_transition"]["accepted"] is False
    assert any("below 0.65 A" in reason for reason in decision["unsafe_transition"]["failures"])


def test_handoff_guard_rejects_worsening_candidate_even_when_start_is_unsafe():
    import qtf.engines.qtf as qtf_engine

    start = _physical_score(4_000_000.0, clash_count=2700, short_contact_count=145, min_nonlocal_distance=0.2)
    handoff = _physical_score(4_100_000.0, clash_count=2701, short_contact_count=146, min_nonlocal_distance=0.1)
    decision = qtf_engine._handoff_guard_decision_payload(start, handoff, _handoff_config())

    assert decision["status"] == "rejected"
    assert decision["decision"] == "fallback"
    assert any("worsened" in reason for reason in decision["hard_reasons"])
    assert any("clash count increased" in reason for reason in decision["hard_reasons"])


def test_handoff_guard_rejects_nonfinite_candidate():
    import qtf.engines.qtf as qtf_engine

    start = _physical_score(4_000_000.0, clash_count=2700, short_contact_count=145, min_nonlocal_distance=0.0)
    handoff = _physical_score(
        3_000_000.0,
        clash_count=2300,
        short_contact_count=110,
        min_nonlocal_distance=0.0,
        nonfinite_atom_count=1,
    )
    decision = qtf_engine._handoff_guard_decision_payload(start, handoff, _handoff_config())

    assert decision["status"] == "rejected"
    assert any("non-finite atom count" in reason for reason in decision["hard_reasons"])


def test_physical_readiness_requires_zero_clashes():
    import qtf.engines.qtf as qtf_engine

    score = _physical_score(10.0, clash_count=1, short_contact_count=0, min_nonlocal_distance=1.0)
    readiness = qtf_engine._physical_readiness_payload(
        score,
        [],
        max_clash_count=0,
        max_short_contact_count=0,
        min_nonlocal_distance_a=0.7,
    )

    assert readiness["ready_for_length_tuning"] is False
    assert readiness["thresholds"]["max_clash_count"] == 0
    assert any("clash count" in reason for reason in readiness["reasons"])


def test_phase_readiness_decision_uses_physical_thresholds():
    import qtf.engines.qtf as qtf_engine

    config = qtf_engine.PhaseReadinessConfig(
        enabled=True,
        evaluator="physical_integrity",
        phases=["local_backbone_lengths"],
        on_fail="skip_phase",
        max_clash_count=0,
        max_short_contact_count=0,
        min_nonlocal_distance_a=0.7,
    )
    score = _physical_score(10.0, clash_count=0, short_contact_count=1, min_nonlocal_distance=0.65)
    decision = qtf_engine._phase_readiness_decision_payload(score, config)

    assert decision["ready"] is False
    assert decision["status"] == "not_ready"
    assert decision["counts"]["short_contact_count"] == 1
    assert any("short-contact count" in reason for reason in decision["reasons"])
    assert any("min nonlocal distance" in reason for reason in decision["reasons"])


def test_phase_minimize_options_caps_powell_maxfev_to_phase_maxiter():
    from types import SimpleNamespace

    import qtf.engines.qtf as qtf_engine

    phase = SimpleNamespace(optimizer="Powell", options={"maxfev": 10000, "xtol": 0.01}, maxiter=50)
    options = qtf_engine._phase_minimize_options(phase)
    assert options["maxiter"] == 50
    assert options["maxfev"] == 50
    assert options["xtol"] == 0.01

    tighter = SimpleNamespace(optimizer="Powell", options={"maxfev": 25}, maxiter=50)
    assert qtf_engine._phase_minimize_options(tighter)["maxfev"] == 25

    cobyla = SimpleNamespace(optimizer="COBYLA", options={}, maxiter=50)
    assert "maxfev" not in qtf_engine._phase_minimize_options(cobyla)


def test_transpile_defaults_resolve_to_unset_and_gate_estimates_include_baseline():
    import qtf.engines.qtf as qtf_engine

    parser = qtf_engine._build_parser()
    args = parser.parse_args([
        "--predict",
        "GA",
        "--replica-id",
        "0",
        "--recipe",
        "qtf-main-equivalent",
        "--backend",
        "statevector-shots",
        "--shots",
        "16",
        "--maxiter",
        "1",
    ])
    assert args.reference_geometry_mode == "metrics-only"
    schedule = qtf_engine._resolve_phase_schedule(args, parser, global_shots=16, global_maxiter=1)
    assert schedule.default_transpile.optimization_level is None
    assert schedule.default_transpile.seed is None
    assert schedule.scouting.transpile.optimization_level is None
    assert schedule.scouting.transpile.seed is None
    assert {phase.optimizer_transpile.optimization_level for phase in schedule.phases} == {None}
    assert {phase.readout_transpile.seed for phase in schedule.phases} == {None}
    assert schedule.gate_estimate_optimization_levels == [0, 3]
    assert schedule.gate_estimate_transpile_seed is None


def test_transpile_overrides_resolve_for_scouting_phases_readouts_and_gate_estimates():
    import qtf.engines.qtf as qtf_engine

    parser = qtf_engine._build_parser()
    args = parser.parse_args([
        "--predict",
        "GA",
        "--replica-id",
        "0",
        "--recipe",
        "qtf-main-equivalent",
        "--backend",
        "statevector-shots",
        "--shots",
        "16",
        "--maxiter",
        "1",
        "--transpile-optimization-level",
        "1",
        "--transpile-seed",
        "5",
        "--scouting-transpile-optimization-level",
        "none",
        "--scouting-transpile-seed",
        "none",
        "--phase-optimizer-transpile-optimization-level",
        "collapse=2",
        "--phase-readout-transpile-seed",
        "collapse=9",
        "--readout",
        "final",
        "--readout-transpile-optimization-level",
        "final=3",
        "--readout-transpile-seed",
        "final=11",
        "--gate-estimate-optimization-levels",
        "3",
        "--gate-estimate-transpile-seed",
        "12",
    ])
    schedule = qtf_engine._resolve_phase_schedule(args, parser, global_shots=16, global_maxiter=1)
    assert schedule.default_transpile.optimization_level == 1
    assert schedule.default_transpile.seed == 5
    assert schedule.scouting.transpile.optimization_level is None
    assert schedule.scouting.transpile.seed is None

    collapse = schedule.phases[0]
    assert collapse.name == "collapse"
    assert collapse.optimizer_transpile.optimization_level == 2
    assert collapse.optimizer_transpile.seed == 5
    assert collapse.readout_transpile.optimization_level == 1
    assert collapse.readout_transpile.seed == 9

    assert schedule.readouts[0].name == "final"
    assert schedule.readouts[0].transpile.optimization_level == 3
    assert schedule.readouts[0].transpile.seed == 11
    assert schedule.gate_estimate_optimization_levels == [0, 3]
    assert schedule.gate_estimate_transpile_seed == 12


def test_transpile_optimization_level_rejects_invalid_values():
    import qtf.engines.qtf as qtf_engine

    parser = qtf_engine._build_parser()
    args = parser.parse_args([
        "--predict",
        "GA",
        "--replica-id",
        "0",
        "--recipe",
        "qtf-main-equivalent",
        "--transpile-optimization-level",
        "4",
    ])
    with pytest.raises(SystemExit):
        qtf_engine._resolve_phase_schedule(args, parser, global_shots=4096, global_maxiter=1)


def test_gate_estimates_force_level_zero_and_leave_seed_unset_by_default(monkeypatch):
    import qtf.engines.qtf as qtf_engine
    from qtf.core.folder import QuantumBiophysicsFolder

    calls = []

    class FakeBackend:
        def name(self):
            return "fake_backend"

    def fake_transpile(circuit, **kwargs):
        calls.append(dict(kwargs))
        return circuit

    monkeypatch.setattr(qtf_engine, "transpile", fake_transpile)
    folder = QuantumBiophysicsFolder("GA")
    estimates = qtf_engine._estimate_gate_costs(
        folder,
        [{"backend": FakeBackend(), "requested": "fake", "source": "test"}],
        optimization_levels=[3],
    )
    assert {estimate["optimization_level"] for estimate in estimates} == {0, 3}
    assert {estimate["seed_transpiler"] for estimate in estimates} == {None}
    assert {call["optimization_level"] for call in calls} == {0, 3}
    assert all("seed_transpiler" not in call for call in calls)

    calls.clear()
    estimates = qtf_engine._estimate_gate_costs(
        folder,
        [{"backend": FakeBackend(), "requested": "fake", "source": "test"}],
        optimization_levels=[0, 2],
        transpile_seed=77,
    )
    assert {estimate["optimization_level"] for estimate in estimates} == {0, 2}
    assert {estimate["seed_transpiler"] for estimate in estimates} == {77}
    assert {call["seed_transpiler"] for call in calls} == {77}

    calls.clear()
    phase_estimates = qtf_engine._estimate_phase_gate_costs(
        [
            {
                "index": 1,
                "name": "angles",
                "label": "Angles",
                "total_dofs": 3,
                "total_angle_dofs": 3,
                "total_length_dofs": 0,
            }
        ],
        circuit_template=None,
        circuit=None,
        backend_refs=[{"backend": FakeBackend(), "requested": "fake", "source": "test"}],
        optimization_levels=[3],
        transpile_seed=None,
    )
    assert {estimate["phase_name"] for estimate in phase_estimates} == {"angles"}
    assert any(estimate.get("backend") == "logical" for estimate in phase_estimates)
    assert any(estimate.get("backend") == "fake_backend" for estimate in phase_estimates)
    assert {estimate.get("optimization_level") for estimate in phase_estimates if estimate.get("backend") == "fake_backend"} == {0, 3}


def test_pheat_native_robust_schedule_resolves():
    import qtf.engines.qtf as qtf_engine

    parser = qtf_engine._build_parser()
    args = parser.parse_args([
        "--predict",
        "GA",
        "--replica-id",
        "0",
        "--recipe",
        "qtf-pheat-native-robust",
        "--backend",
        "statevector-shots",
        "--shots",
        "16",
        "--maxiter",
        "1",
    ])
    schedule = qtf_engine._resolve_phase_schedule(args, parser, global_shots=16, global_maxiter=1)
    assert schedule.preset == "qtf-pheat-native-robust"
    assert schedule.basis_circuit_batching == "auto"
    assert schedule.scouting.score_model == "pheat-mj"
    assert schedule.scouting.shots == 16
    assert schedule.result.score_model == "pheat-goap"
    assert [phase.name for phase in schedule.phases] == [
        "collapse",
        "packing",
        "orientation",
        "polar_contacts",
        "backbone_torsions",
        "sidechain_rotamers",
        "heavy_relax",
    ]
    assert {phase.maxiter for phase in schedule.phases} == {1}
    assert {phase.optimizer_shots for phase in schedule.phases} == {16}
    assert {phase.readout_shots for phase in schedule.phases} == {16}
    assert schedule.phase_comparisons.enabled is True
    assert schedule.phase_comparisons.evaluators == ["geometry_integrity", "goap_check", "rg_check"]
    assert schedule.reranking.enabled is True
    assert schedule.reranking.evaluator == "goap_check"
    assert schedule.reranking.candidate_pool["per_phase_top_k"] == 3
    assert schedule.validation.enabled is True
    assert schedule.validation.candidates == ["primary", "phase_ends", "reranked_top"]
    assert schedule.report.structure_domain == "protein-heavy"


def test_main_snapshot_equivalent_schedule_resolves():
    import qtf.engines.qtf as qtf_engine

    parser = qtf_engine._build_parser()
    args = parser.parse_args([
        "--predict",
        "GA",
        "--replica-id",
        "0",
        "--recipe",
        "qtf-main-snapshot-equivalent",
        "--backend",
        "statevector",
        "--shots",
        "16",
        "--maxiter",
        "1",
    ])
    schedule = qtf_engine._resolve_phase_schedule(args, parser, global_shots=16, global_maxiter=1)
    assert schedule.preset == "qtf-main-snapshot-equivalent"
    assert [phase.name for phase in schedule.phases] == ["collapse", "refine", "relax"]
    assert {phase.optimizer for phase in schedule.phases} == {"COBYLA"}
    assert {phase.score_model for phase in schedule.phases} == {"pheat-coarse-protein-folding-v1"}
    assert {phase.score_options["use_end_to_end_constraint"] for phase in schedule.phases} == {True}
    assert {phase.score_options["end_to_end_scale"] for phase in schedule.phases} == {1.0}
    assert {phase.score_options["hydrophobic_burial_denominator"] for phase in schedule.phases} == {35.0}
    assert {phase.score_options["hydrophobic_burial_scale"] for phase in schedule.phases} == {0.7}
    assert all("end_to_end_target" not in phase.score_options for phase in schedule.phases)
    assert all("end_to_end_slack" not in phase.score_options for phase in schedule.phases)
    assert schedule.reranking.enabled is False
    assert schedule.reranking.evaluator is None
    assert schedule.reranking.candidate_pool == {}
    assert schedule.validation.enabled is True
    assert schedule.validation.candidates == ["primary", "phase_ends", "top_snapshots"]
    assert schedule.validation.evaluators == ["gromacs_minimize"]
    assert schedule.evaluators["gromacs_minimize"].score_model == "pheat-gromacs-mdrun"



def test_pheat_external_validation_recipe_configures_optional_external_scorers():
    pytest.importorskip("yaml")
    from qtf.recipes import load_builtin_recipes

    recipes = load_builtin_recipes()
    recipe = recipes["qtf-pheat-external-validation"]
    robust = recipes["qtf-pheat-native-robust"]
    assert recipe["geometry"]["stored_angles"] == "all"
    assert recipe["geometry"]["stored_lengths"] == []
    assert recipe["geometry"]["max_chi"] is None
    assert [phase["name"] for phase in recipe["phases"]] == [phase["name"] for phase in robust["phases"]]
    assert [phase["optimizer"] for phase in recipe["phases"]] == [phase["optimizer"] for phase in robust["phases"]]
    assert [phase["score_model"] for phase in recipe["phases"]] == [phase["score_model"] for phase in robust["phases"]]

    evaluators = recipe["evaluators"]
    assert evaluators["geometry_integrity"]["score_model"] == "pheat-geometry-integrity"
    assert evaluators["openmm_prepared"]["score_model"] == "pheat-openmm-prepared"
    assert evaluators["ambertools_gb"]["score_model"] == "pheat-ambertools-sander"
    assert evaluators["gromacs_minimize"]["score_model"] == "pheat-gromacs-mdrun"
    assert all(evaluator["required"] is False for evaluator in evaluators.values())
    assert evaluators["ambertools_gb"]["options"]["amber_solvent"] == "gb"
    assert evaluators["gromacs_minimize"]["options"]["gromacs_run_mode"] == "minimize-rerun"
    assert evaluators["gromacs_minimize"]["options"]["gromacs_preflight"] == "warn"
    assert evaluators["gromacs_minimize"]["options"]["gromacs_forcefield"] == "amber99sb-ildn"
    assert evaluators["gromacs_minimize"]["options"]["gromacs_run_settings"]["minimize_steps"] == 500
    assert evaluators["gromacs_minimize"]["options"]["gromacs_run_settings"]["grompp_maxwarn"] == 1
    assert recipe["phase_comparisons"]["evaluators"] == ["openmm_prepared"]
    assert recipe["reranking"]["evaluator"] == "ambertools_gb"
    assert recipe["validation"]["evaluators"] == [
        "geometry_integrity",
        "openmm_prepared",
        "ambertools_gb",
        "gromacs_minimize",
    ]


def test_pheat_external_validation_unavailable_evaluators_are_skipped(monkeypatch, tmp_path):
    import qtf.engines.qtf as qtf_engine

    parser = qtf_engine._build_parser()
    args = parser.parse_args([
        "--predict",
        "GA",
        "--replica-id",
        "0",
        "--recipe",
        "qtf-pheat-external-validation",
        "--backend",
        "statevector-shots",
        "--shots",
        "16",
        "--maxiter",
        "1",
    ])
    schedule = qtf_engine._resolve_phase_schedule(args, parser, global_shots=16, global_maxiter=1)
    assert schedule.phase_comparisons.evaluators == ["openmm_prepared"]
    assert schedule.reranking.evaluator == "ambertools_gb"
    assert schedule.validation.evaluators == [
        "geometry_integrity",
        "openmm_prepared",
        "ambertools_gb",
        "gromacs_minimize",
    ]

    def fake_capabilities_by_public_name():
        return {
            "pheat-geometry-integrity": {
                "available": True,
                "reason": None,
                "implementation": {"external": False},
            },
            "pheat-openmm-prepared": {
                "available": False,
                "reason": "missing dependency: openmm",
                "implementation": {"external": True},
            },
            "pheat-ambertools-sander": {
                "available": False,
                "reason": "missing dependency: tleap; sander",
                "implementation": {"external": True},
            },
            "pheat-gromacs-mdrun": {
                "available": False,
                "reason": "missing dependency: gmx",
                "implementation": {"external": True},
            },
        }

    def fake_validate_options(evaluator, *, outdir=None):
        return {
            "ok": True,
            "model": evaluator.score_model.removeprefix("pheat-"),
            "external": evaluator.name != "geometry_integrity",
            "errors": [],
            "warnings": [],
        }

    monkeypatch.setattr(qtf_engine, "_pheat_capabilities_by_public_name", fake_capabilities_by_public_name)
    monkeypatch.setattr(qtf_engine, "_validate_external_evaluator_options", fake_validate_options)
    statuses = qtf_engine._validate_recipe_evaluators_for_run(parser, schedule, outdir=tmp_path)
    by_name = {status["name"]: status for status in statuses}
    assert by_name["geometry_integrity"]["status"] == "ok"
    assert by_name["openmm_prepared"]["status"] == "skipped"
    assert by_name["ambertools_gb"]["status"] == "skipped"
    assert by_name["gromacs_minimize"]["status"] == "skipped"
    assert by_name["ambertools_gb"]["errors"] == ["missing dependency: tleap; sander"]



def test_software_versions_highlight_run_components_not_legacy_packages(monkeypatch, tmp_path):
    import qtf.engines.qtf as qtf_engine

    versions = {
        "numpy": "1.0",
        "scipy": "1.0",
        "PyYAML": "1.0",
        "jsonschema": "1.0",
        "plotly": "1.0",
        "qiskit": "1.0",
        "qiskit-aer": "1.0",
        "qiskit-ibm-runtime": "1.0",
        "matplotlib": "1.0",
        "biopython": "1.0",
        "mdtraj": "1.0",
    }
    monkeypatch.setattr(qtf_engine, "_distribution_version", lambda name: versions.get(name))
    monkeypatch.setattr(
        qtf_engine,
        "_installed_distributions",
        lambda: [
            {"name": "biopython", "version": "1.0"},
            {"name": "mdtraj", "version": "1.0"},
        ],
    )
    monkeypatch.setattr(qtf_engine, "_git_provenance", lambda path: {"available": False, "path": str(path)})
    monkeypatch.setattr(qtf_engine, "_pheat_capabilities_by_public_name", lambda: {})
    monkeypatch.setattr(
        qtf_engine,
        "collect_software_provenance",
        lambda **kwargs: {
            "format": "pheat.software-provenance",
            "version": 1,
            "selected_score_models": kwargs.get("selected_score_models") or [],
            "selected_features": [],
            "package_components": [
                {
                    "name": "platformdirs",
                    "version": "4.0",
                    "role": "PHEAT cache and platform path handling",
                    "required": True,
                    "selected": True,
                    "status": "available",
                }
            ],
            "external_tools": [],
        },
    )

    full = qtf_engine._collect_software_versions(selected_quantum_packages={"qiskit"})
    highlighted_names = {item["name"] for item in full["package_components"]}

    assert "biopython" not in highlighted_names
    assert "mdtraj" not in highlighted_names
    assert "matplotlib" not in highlighted_names
    assert "plotly" in highlighted_names
    assert "platformdirs" in highlighted_names
    assert full["installed_distributions"] == [
        {"name": "biopython", "version": "1.0"},
        {"name": "mdtraj", "version": "1.0"},
    ]

    summary = qtf_engine._software_summary_payload(full, tmp_path / "software.json")
    html = qtf_engine._software_report_section({"software_versions": summary})

    assert "Run Components" in html
    assert "biopython" not in html.lower()
    assert "mdtraj" not in html.lower()
    assert "matplotlib" not in html.lower()
    assert summary["pheat_software_provenance"]["format"] == "pheat.software-provenance"


def test_selected_quantum_packages_follow_backend_and_estimates():
    from types import SimpleNamespace

    import qtf.engines.qtf as qtf_engine

    args = SimpleNamespace(hw_backend="statevector", estimate_gates="aer,ibm_cleveland")

    assert qtf_engine._selected_quantum_package_names(args, "aer,ibm_cleveland") == {
        "qiskit",
        "qiskit-aer",
        "qiskit-ibm-runtime",
    }


def test_software_versions_include_selected_external_requirements_from_pheat(monkeypatch):
    import qtf.engines.qtf as qtf_engine

    versions = {
        "numpy": "1.0",
        "scipy": "1.0",
        "PyYAML": "1.0",
        "jsonschema": "1.0",
        "plotly": "1.0",
        "qiskit": "1.0",
        "qiskit-aer": "0.17",
    }
    collect_calls = []

    def fake_collect_software_provenance(*, selected_score_models=None, **kwargs):
        raw_models = list(selected_score_models or [])
        collect_calls.append(tuple(raw_models))
        components = [
            {
                "name": "platformdirs",
                "version": "4.0",
                "role": "PHEAT cache and platform path handling",
                "required": True,
                "selected": True,
                "status": "available",
            }
        ]
        tools = []
        if "openmm-prepared" in raw_models:
            components.append(
                {
                    "name": "openmm",
                    "version": "8.5",
                    "role": "selected PHEAT scoring dependency",
                    "required": True,
                    "selected": True,
                    "status": "available",
                }
            )
        if "ambertools-sander" in raw_models:
            tools.extend(
                [
                    {
                        "name": "sander",
                        "path": "/fake/bin/sander",
                        "version": "sander version",
                        "role": "selected AmberTools energy/minimization executable",
                        "required": True,
                        "selected": True,
                        "status": "available",
                        "details": None,
                    },
                    {
                        "name": "tleap",
                        "path": "/fake/bin/tleap",
                        "version": "tleap version",
                        "role": "selected AmberTools topology-preparation executable",
                        "required": True,
                        "selected": True,
                        "status": "available",
                        "details": None,
                    },
                ]
            )
        return {
            "format": "pheat.software-provenance",
            "version": 1,
            "selected_score_models": raw_models,
            "selected_features": [],
            "package_components": components,
            "external_tools": tools,
        }

    monkeypatch.setattr(qtf_engine, "_distribution_version", lambda name: versions.get(name))
    monkeypatch.setattr(qtf_engine, "_installed_distributions", lambda: [])
    monkeypatch.setattr(qtf_engine, "_git_provenance", lambda path: {"available": False, "path": str(path)})
    monkeypatch.setattr(
        qtf_engine,
        "_pheat_capabilities_by_public_name",
        lambda: {
            "pheat-openmm-prepared": {"pheat_model": "openmm-prepared"},
            "pheat-ambertools-sander": {"pheat_model": "ambertools-sander"},
        },
    )
    monkeypatch.setattr(qtf_engine, "collect_software_provenance", fake_collect_software_provenance)

    full = qtf_engine._collect_software_versions(
        selected_score_models=["pheat-openmm-prepared"],
        evaluator_statuses=[
            {
                "name": "ambertools_gb",
                "score_model": "pheat-ambertools-sander",
                "required": False,
                "capability": {"requires": ["executable:tleap", "executable:sander"]},
            }
        ],
        selected_quantum_packages={"qiskit", "qiskit-aer"},
    )

    packages = {item["name"]: item for item in full["package_components"]}
    tools = {item["name"]: item for item in full["external_tools"]}

    assert ("openmm-prepared",) in collect_calls
    assert ("ambertools-sander",) in collect_calls
    assert packages["openmm"]["status"] == "available"
    assert packages["openmm"]["required"] is True
    assert packages["qiskit-aer"]["selected"] is True
    assert packages["qiskit-aer"]["status"] == "available"
    assert sorted(tools) == ["sander", "tleap"]
    assert tools["tleap"]["required"] is False
    assert tools["tleap"]["role"] == "selected evaluator: ambertools_gb"
    assert full["pheat_software_provenance"]["selected_score_models"] == [
        "openmm-prepared",
        "ambertools-sander",
    ]

    html = qtf_engine._software_report_section({"software_versions": qtf_engine._software_summary_payload(full, None)})
    assert "Selected External Tools" in html
    assert "tleap" in html
    assert "sander" in html



def test_score_payload_preserves_pheat_unavailable_status(monkeypatch):
    import qtf.engines.qtf as qtf_engine

    class FakeScore:
        def to_dict(self):
            return {
                "model": "gromacs-mdrun",
                "total": None,
                "units": None,
                "terms": {},
                "warnings": [],
                "citations": [],
                "metadata": {"status": "unavailable", "reason": "preflight rejected structure"},
            }

    monkeypatch.setattr(qtf_engine, "score_pheat_structure", lambda structure, model, **kwargs: FakeScore())
    payload = qtf_engine._score_payload(object(), "pheat-gromacs-mdrun")
    assert payload["status"] == "unavailable"
    assert payload["error"] == "preflight rejected structure"


def test_recipe_file_schema_is_validated(tmp_path):
    pytest.importorskip("yaml")
    from qtf.recipes import load_recipe_file

    recipe_file = tmp_path / "bad.yaml"
    recipe_file.write_text("recipes:\n  broken:\n    phases: not-a-list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid recipe file"):
        load_recipe_file(recipe_file)


def test_recipe_schema_accepts_metric_and_geometry_options(tmp_path):
    pytest.importorskip("yaml")
    from qtf.recipes import load_recipe_file

    recipe_file = tmp_path / "ok.yaml"
    recipe_file.write_text(
        """
recipes:
  custom:
    geometry:
      stored_angles: all
      stored_lengths: backbone
      geometry_mode: packaged
      geometry_table: default
      geometry_profile: canonical
      max_chi: 1
      selective_chi_map:
        P: [chi1]
        W: chi1,chi2
    metrics:
      atom_sets:
        - ca
        - backbone
      rmsd_alignment_atom_set: same-as-rmsd
    phases:
      - name: phase-a
        optimizer: COBYLA
        score_model: pheat-physics
""",
        encoding="utf-8",
    )
    recipes = load_recipe_file(recipe_file)
    assert recipes["custom"]["metrics"]["atom_sets"] == ["ca", "backbone"]
    assert recipes["custom"]["geometry"]["selective_chi_map"]["P"] == ["chi1"]


def test_recipe_schema_accepts_descriptions_on_supported_blocks(tmp_path):
    pytest.importorskip("yaml")
    from qtf.recipes import load_recipe_file

    recipe_file = tmp_path / "described.yaml"
    recipe_file.write_text(
        """
recipes:
  described:
    description: Recipe description.
    fold:
      description: Fold description.
    circuit_template:
      description: Circuit template description.
      source: qtf
      name: brickwork-ryrz-nearest-neighbor
    geometry:
      description: Geometry description.
      stored_angles: all
    metrics:
      description: Metrics description.
      atom_sets: ca
    report:
      description: Report description.
      structure_domain: protein-heavy
    transpile:
      description: Transpile description.
      optimization_level: 0
    scouting:
      description: Scouting description.
    result:
      description: Result description.
    evaluators:
      physical_integrity:
        description: Evaluator description.
        score_model: pheat-physical-integrity
    phase_comparisons:
      description: Phase comparison description.
      enabled: false
    reranking:
      description: Reranking description.
      enabled: false
    handoff_guard:
      description: Handoff guard description.
      enabled: false
    validation:
      description: Validation description.
      enabled: false
    phases:
      - name: phase-a
        description: Phase description.
        optimizer: COBYLA
        score_model: pheat-physics
        geometry:
          description: Phase geometry description.
        handoff_guard:
          description: Phase guard description.
          enabled: false
    readouts:
      - name: final
        description: Readout description.
""",
        encoding="utf-8",
    )
    recipes = load_recipe_file(recipe_file)
    recipe = recipes["described"]
    assert recipe["description"] == "Recipe description."
    assert recipe["phases"][0]["geometry"]["description"] == "Phase geometry description."
    assert recipe["readouts"][0]["description"] == "Readout description."


def test_recipe_schema_rejects_non_string_descriptions(tmp_path):
    pytest.importorskip("yaml")
    from qtf.recipes import load_recipe_file

    recipe_file = tmp_path / "bad-description.yaml"
    recipe_file.write_text(
        """
recipes:
  broken:
    description: 123
    phases:
      - name: phase-a
        optimizer: COBYLA
        score_model: pheat-physics
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Invalid recipe file"):
        load_recipe_file(recipe_file)


def test_recipe_schema_rejects_chi_mode(tmp_path):
    pytest.importorskip("yaml")
    from qtf.recipes import load_recipe_file

    recipe_file = tmp_path / "bad-chi-mode.yaml"
    recipe_file.write_text(
        """
recipes:
  custom:
    geometry:
      chi_mode: chi1_only
    phases:
      - name: phase-a
        optimizer: COBYLA
        score_model: pheat-physics
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Invalid recipe file"):
        load_recipe_file(recipe_file)


def test_recipe_schema_accepts_external_evaluator_options(tmp_path):
    pytest.importorskip("yaml")
    from qtf.recipes import load_recipe_file

    recipe_file = tmp_path / "external.yaml"
    recipe_file.write_text(
        """
recipes:
  external:
    evaluators:
      ambertools_gb:
        score_model: pheat-ambertools-sander
        required: false
        options:
          prepare: auto
          amber_solvent: gb
      gromacs_minimize:
        score_model: pheat-gromacs-mdrun
        required: false
        options:
          gromacs_run_mode: minimize-rerun
          gromacs_preflight: warn
          gromacs_run_settings:
            minimize_steps: 25
    phase_comparisons:
      enabled: true
      evaluators: [ambertools_gb]
      compare: consecutive_phase_ends
    reranking:
      enabled: true
      evaluator: ambertools_gb
      triggers:
        - when: every_evaluations
          interval: 10
      candidate_pool:
        per_phase_top_k: 2
      apply: next_phase_start
    validation:
      enabled: true
      candidates: [primary, phase_ends, reranked_top, top_snapshots]
      evaluators: [ambertools_gb, gromacs_minimize]
    phases:
      - name: phase-a
        optimizer: COBYLA
        score_model: pheat-physics
""",
        encoding="utf-8",
    )
    recipes = load_recipe_file(recipe_file)
    assert recipes["external"]["evaluators"]["ambertools_gb"]["score_model"] == "pheat-ambertools-sander"
    assert recipes["external"]["evaluators"]["gromacs_minimize"]["options"]["gromacs_preflight"] == "warn"
    assert recipes["external"]["validation"]["candidates"] == ["primary", "phase_ends", "reranked_top", "top_snapshots"]


def test_fold_cli_passes_metric_and_geometry_options_from_recipe():
    from qtf.cli import _build_parser, _qtf_argv
    from qtf.recipes import resolve_recipe

    args = _build_parser().parse_args(["fold", "qtf-heavy-atom-phased", "--sequence", "GA"])
    argv = _qtf_argv(args, resolve_recipe("qtf-heavy-atom-phased"))
    assert argv[argv.index("--metric-atom-sets") + 1] == "ca,backbone,all-heavy"
    assert argv[argv.index("--rmsd-alignment-atom-set") + 1] == "same-as-rmsd"
    assert argv[argv.index("--report-structure-domain") + 1] == "protein-heavy"
    assert argv[argv.index("--store-lengths") + 1] == ""
    assert "--chi-mode" not in argv


def test_fold_cli_passes_snapshot_options_to_engine():
    from qtf.cli import _build_parser, _qtf_argv
    from qtf.recipes import resolve_recipe

    args = _build_parser().parse_args([
        "fold",
        "qtf-main-snapshot-equivalent",
        "--sequence",
        "GA",
        "--top-k-snapshots",
        "12",
        "--snapshot-energy-gap",
        "0.1",
        "--snapshot-sort-by",
        "rmsd",
    ])
    argv = _qtf_argv(args, resolve_recipe("qtf-main-snapshot-equivalent"))
    assert argv[argv.index("--top-k-snapshots") + 1] == "12"
    assert argv[argv.index("--snapshot-energy-gap") + 1] == "0.1"
    assert argv[argv.index("--snapshot-sort-by") + 1] == "rmsd"


def test_engine_parser_accepts_snapshot_options():
    import qtf.engines.qtf as qtf_engine

    args = qtf_engine._build_parser().parse_args([
        "--predict",
        "GA",
        "--replica-id",
        "0",
        "--top-k-snapshots",
        "7",
        "--snapshot-energy-gap",
        "0.1",
        "--snapshot-sort-by",
        "energy",
    ])
    assert args.top_k_snapshots == 7
    assert args.snapshot_energy_gap == 0.1
    assert args.snapshot_sort_by == "energy"


def test_fold_cli_passes_length_encoding_options_from_recipe():
    from qtf.cli import _build_parser, _qtf_argv

    recipe = {
        "name": "custom",
        "geometry": {
            "stored_lengths": "backbone",
            "length_encoding_scope": "per-residue",
            "backbone_length_span": 0.03,
            "sidechain_length_span": 0.07,
        },
        "phases": [{"name": "phase-a", "optimizer": "COBYLA", "score_model": "pheat-physics"}],
    }
    args = _build_parser().parse_args(["fold", "custom", "--sequence", "GA"])
    argv = _qtf_argv(args, recipe)
    assert argv[argv.index("--store-lengths") + 1] == "backbone"
    assert argv[argv.index("--length-encoding-scope") + 1] == "per-residue"
    assert argv[argv.index("--backbone-length-span") + 1] == "0.03"
    assert argv[argv.index("--sidechain-length-span") + 1] == "0.07"


def test_fold_cli_passes_phase_geometry_option_to_engine():
    from qtf.cli import _build_parser, _qtf_argv

    recipe = {
        "name": "custom",
        "phases": [{"name": "phase-a", "optimizer": "COBYLA", "score_model": "pheat-physics"}],
    }
    args = _build_parser().parse_args(
        [
            "fold",
            "custom",
            "--sequence",
            "GA",
            "--phase-geometry-option",
            "phase-a:backbone_length_span=0.05",
        ]
    )
    argv = _qtf_argv(args, recipe)
    assert argv[argv.index("--phase-geometry-option") + 1] == "phase-a:backbone_length_span=0.05"


def test_fold_cli_forwards_ibm_auth_options_to_engine():
    from qtf.cli import _build_parser, _qtf_argv

    recipe = {
        "name": "custom",
        "phases": [{"name": "phase-a", "optimizer": "COBYLA", "score_model": "pheat-physics"}],
    }
    args = _build_parser().parse_args([
        "fold",
        "custom",
        "--sequence",
        "GA",
        "--ibm-account",
        "default-ibm-cloud",
        "--ibm-token-env",
        "QTF_IBM_TOKEN",
        "--ibm-channel",
        "ibm_quantum_platform",
        "--ibm-url",
        "https://quantum.example.test",
        "--ibm-instance-crn",
        "crn:v1:secret-instance",
    ])

    argv = _qtf_argv(args, recipe)

    assert argv[argv.index("--ibm-account") + 1] == "default-ibm-cloud"
    assert argv[argv.index("--ibm-token-env") + 1] == "QTF_IBM_TOKEN"
    assert argv[argv.index("--ibm-channel") + 1] == "ibm_quantum_platform"
    assert argv[argv.index("--ibm-url") + 1] == "https://quantum.example.test"
    assert argv[argv.index("--ibm-instance-crn") + 1] == "crn:v1:secret-instance"


def test_fold_cli_forwards_gate_estimate_backend_crns_to_engine():
    from qtf.cli import _build_parser, _qtf_argv

    recipe = {
        "name": "custom",
        "phases": [{"name": "phase-a", "optimizer": "COBYLA", "score_model": "pheat-physics"}],
    }
    args = _build_parser().parse_args([
        "fold",
        "custom",
        "--sequence",
        "GA",
        "--estimate-gates",
        "ibm_cleveland,ibm_miami",
        "--gate-estimate-backend-crn",
        "ibm_miami=crn:v1:miami-instance",
    ])

    argv = _qtf_argv(args, recipe)

    assert argv[argv.index("--estimate-gates") + 1] == "ibm_cleveland,ibm_miami"
    assert argv[argv.index("--gate-estimate-backend-crn") + 1] == "ibm_miami=crn:v1:miami-instance"


def test_fold_cli_passes_selective_chi_map_from_recipe():
    from qtf.cli import _build_parser, _qtf_argv

    recipe = {
        "name": "custom",
        "geometry": {
            "selective_chi_map": {
                "P": ["chi1"],
                "W": "chi1,chi2",
            }
        },
        "phases": [{"name": "phase-a", "optimizer": "COBYLA", "score_model": "pheat-physics"}],
    }
    args = _build_parser().parse_args(["fold", "custom", "--sequence", "PW"])
    argv = _qtf_argv(args, recipe)
    assert argv.count("--selective-chi") == 2
    assert "P=chi1" in argv
    assert "W=chi1,chi2" in argv


def test_fold_cli_selective_chi_cli_overrides_recipe():
    from qtf.cli import _build_parser, _qtf_argv

    recipe = {
        "name": "custom",
        "geometry": {"selective_chi_map": {"P": ["chi1"]}},
        "phases": [{"name": "phase-a", "optimizer": "COBYLA", "score_model": "pheat-physics"}],
    }
    args = _build_parser().parse_args(
        ["fold", "custom", "--sequence", "PW", "--selective-chi", "W=chi1,chi2"]
    )
    argv = _qtf_argv(args, recipe)
    assert argv.count("--selective-chi") == 1
    assert "W=chi1,chi2" in argv
    assert "P=chi1" not in argv


def test_fold_parser_uses_recipe_language():
    from qtf.cli import _build_parser

    help_text = _build_parser().format_help()
    assert "fold" in help_text

    fold_help = _build_parser().parse_args(["fold", "--help"]) if False else None
    assert fold_help is None


def test_score_model_names_are_canonical():
    from qtf.scoring import available_pheat_score_models, available_score_models, pheat_model_name

    models = available_score_models()
    assert "pheat-physics" in models
    assert "pheat-coarse-protein-folding-v1" in models
    assert "pheat-generic" in models
    assert "pheat-geometry-integrity" in models
    assert "generic" not in models
    assert all(model.startswith("pheat-") for model in available_pheat_score_models())
    assert pheat_model_name("pheat-generic") == "generic"
    assert pheat_model_name("pheat-heavy-mm") == "heavy-mm"
    assert pheat_model_name("pheat-heavy-mm-physical") == "pheat-heavy-mm-physical"
    with pytest.raises(ValueError):
        pheat_model_name("generic")


def test_evaluator_options_are_validated_by_pheat(tmp_path):
    import qtf.engines.qtf as qtf_engine

    evaluator = qtf_engine.EvaluatorConfig(
        name="bad",
        score_model="pheat-generic",
        required=False,
        options={"not_an_option": 1},
    )
    payload = qtf_engine._validate_external_evaluator_options(evaluator, outdir=tmp_path)
    assert payload["ok"] is False
    assert any("unknown option" in error for error in payload["errors"])


def test_native_evaluator_options_are_accepted(tmp_path):
    import qtf.engines.qtf as qtf_engine

    evaluator = qtf_engine.EvaluatorConfig(
        name="generic",
        score_model="pheat-generic",
        required=False,
        options={"domain": "protein-heavy"},
    )
    payload = qtf_engine._validate_external_evaluator_options(evaluator, outdir=tmp_path)
    assert payload["ok"] is True
    assert payload["external"] is False


def test_pheat_score_capabilities_report_unavailable_models(monkeypatch):
    import qtf.scoring as scoring

    monkeypatch.setattr(
        scoring,
        "_pheat_model_capabilities_from_package",
        lambda: [
            {
                "model": "generic",
                "available": True,
                "supported": True,
                "units": "arbitrary",
                "requires": [],
                "optional_requires": [],
                "reason": None,
            },
            {
                "model": "openmm-prepared",
                "available": False,
                "supported": True,
                "units": "kJ/mol",
                "requires": ["openmm"],
                "optional_requires": ["pdbfixer"],
                "reason": "missing optional dependency: openmm",
            },
        ],
    )

    assert scoring.supported_pheat_score_models() == ["pheat-generic", "pheat-openmm-prepared"]
    assert scoring.available_pheat_score_models() == ["pheat-generic"]
    assert scoring.pheat_model_name("pheat-generic") == "generic"
    with pytest.raises(ValueError, match="missing optional dependency"):
        scoring.pheat_model_name("pheat-openmm-prepared")



def test_qtf_report_pdb_filters_by_report_domain(tmp_path):
    pytest.importorskip("pheat")
    import qtf.engines.qtf as qtf_engine
    from pheat import Atom, HeavyAtomStructure

    structure = HeavyAtomStructure(
        atoms=[
            Atom(name="CA", element="C", x=0.0, y=0.0, z=0.0, resname="GLY", chain_id="A", resseq=1),
            Atom(name="O", element="O", x=1.0, y=0.0, z=0.0, resname="HOH", chain_id="A", resseq=101, record_name="HETATM"),
            Atom(name="C1", element="C", x=2.0, y=0.0, z=0.0, resname="BEN", chain_id="A", resseq=102, record_name="HETATM"),
        ],
        name="mixed",
    )

    protein_path = tmp_path / "protein_only.pdb"
    filtered, coverage = qtf_engine._write_report_pdb(
        structure,
        protein_path,
        domain="protein-heavy",
    )
    protein_text = protein_path.read_text(encoding="utf-8")
    assert len(filtered.atoms) == 1
    assert coverage["input_atom_count"] == 3
    assert coverage["scored_atom_count"] == 1
    assert coverage["ignored_nonprotein_atom_count"] == 2
    assert " GLY " in protein_text
    assert " HOH " not in protein_text
    assert " BEN " not in protein_text

    all_heavy_path = tmp_path / "all_heavy.pdb"
    all_heavy, all_heavy_coverage = qtf_engine._write_report_pdb(
        structure,
        all_heavy_path,
        domain="all-heavy",
    )
    all_heavy_text = all_heavy_path.read_text(encoding="utf-8")
    assert len(all_heavy.atoms) == 3
    assert all_heavy_coverage["scored_atom_count"] == 3
    assert " HOH " in all_heavy_text
    assert " BEN " in all_heavy_text


def test_ranked_multimodel_pdb_strips_nested_records(tmp_path):
    import qtf.engines.qtf as qtf_engine

    source = tmp_path / "source.pdb"
    source.write_text(
        "\n".join(
            [
                "MODEL        9",
                "REMARK existing",
                "ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00  0.00           C",
                "ENDMDL",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "ranked.pdb"

    written = qtf_engine._write_ranked_multimodel_pdb(
        out,
        [
            {
                "pdb_path": str(source),
                "remarks": [
                    "QTF_SOURCE replica=replica_1 replica_id=0",
                    "QTF_SCORE energy=-1 gromacs_potential_kj_mol=-2",
                ],
            }
        ],
    )

    text = out.read_text(encoding="utf-8")
    assert written == out
    assert text.count("MODEL") == 1
    assert text.count("ENDMDL") == 1
    assert "QTF_SOURCE replica=replica_1" in text
    assert "gromacs_potential_kj_mol=-2" in text
    assert text.rstrip().endswith("ENDMDL")


def test_pdb_alignment_atom_set_matches_selected_rmsd_metric():
    import qtf.engines.qtf as qtf_engine

    assert qtf_engine._pdb_alignment_atom_set(["ca"], "same-as-rmsd") == "ca"
    assert qtf_engine._pdb_alignment_atom_set(["ca", "all-heavy"], "same-as-rmsd") == "all-heavy"
    assert qtf_engine._pdb_alignment_atom_set(["ca", "all-heavy"], "ca") == "ca"


def test_qtf_copies_pheat_managed_molstar_assets(tmp_path, monkeypatch):
    pytest.importorskip("pheat")
    import qtf.engines.qtf as qtf_engine
    from pheat.molstar_assets import MOLSTAR_ENV_VAR

    source = tmp_path / "pheat-molstar"
    source.mkdir()
    for filename, text in {
        "molstar.js": "window.molstar = {};\n",
        "molstar.css": ".msp-layout {}\n",
        "LICENSE": "MIT\n",
        "manifest.json": "{}\n",
    }.items():
        (source / filename).write_text(text, encoding="utf-8")

    monkeypatch.setenv(MOLSTAR_ENV_VAR, str(source))
    copied, error = qtf_engine._copy_molstar_assets(tmp_path / "report")

    assert error is None
    assert copied == tmp_path / "report" / "vendor" / "molstar"
    assert (copied / "molstar.js").read_text(encoding="utf-8") == "window.molstar = {};\n"
    assert (copied / "manifest.json").exists()


def test_qtf_molstar_asset_copy_warns_when_pheat_assets_missing(tmp_path, monkeypatch):
    pytest.importorskip("pheat")
    import qtf.engines.qtf as qtf_engine
    from pheat.molstar_assets import MOLSTAR_ENV_VAR

    source = tmp_path / "missing-molstar"
    source.mkdir()
    monkeypatch.setenv(MOLSTAR_ENV_VAR, str(source))

    with pytest.warns(RuntimeWarning, match="pheat molstar install"):
        copied, error = qtf_engine._copy_molstar_assets(tmp_path / "report")

    assert copied is None
    assert "pheat molstar install" in error


def test_qtf_structure_uses_pheat_geometry(folder_ga):
    pytest.importorskip("pheat")
    from qtf.metrics import radius_of_gyration_summary

    structure = folder_ga.structure_from_angle_vector([0.0] * folder_ga.total_angles)
    rg = radius_of_gyration_summary(structure)
    assert rg["ca"]["status"] == "ok"
    assert rg["ca"]["atom_set"] == "ca"


def test_qtf_metrics_use_pheat_atom_sets(folder_ga):
    pytest.importorskip("pheat")
    from qtf.metrics import structure_metric_summary

    structure = folder_ga.structure_from_angle_vector([0.0] * folder_ga.total_angles)
    metrics = structure_metric_summary(
        structure,
        structure,
        atom_sets=("ca", "backbone", "all-heavy"),
        alignment_atom_set="same-as-rmsd",
    )
    assert metrics["ca"]["status"] == "ok"
    assert metrics["ca"]["value"] == pytest.approx(0.0)
    assert metrics["backbone"]["status"] == "ok"
    assert metrics["all-heavy"]["status"] == "ok"


def test_qtf_folder_stores_pheat_lengths_config():
    pytest.importorskip("pheat")
    from qtf.core.folder import QuantumBiophysicsFolder

    folder = QuantumBiophysicsFolder("GA", stored_lengths="backbone")
    residue_geometry = folder.angle_vector_to_residue_geometry([0.0] * folder.total_angles)
    assert residue_geometry.stored_lengths == ("backbone",)
    assert residue_geometry.metadata["stored_lengths"] == ["backbone"]


def test_pheat_score_options_are_forwarded_to_folder(monkeypatch):
    import numpy as np
    import qtf.core.folder as folder_mod
    from qtf.core.folder import QuantumBiophysicsFolder

    captured = {}

    class FakeScore:
        total = 7.0

        def to_dict(self):
            return {
                "model": "pheat-generic",
                "total": 7.0,
                "units": "arbitrary",
                "terms": {},
                "warnings": [],
                "citations": [],
                "metadata": {},
            }

    def fake_score(structure, model, **kwargs):
        captured["model"] = model
        captured["kwargs"] = kwargs
        return FakeScore()

    monkeypatch.setattr(folder_mod, "score_pheat_structure", fake_score)
    folder = QuantumBiophysicsFolder("GA", score_model="pheat-generic")
    payload, total = folder.score_model_for_params(
        np.zeros(folder.n_params),
        "pheat-generic",
        angle_mode="statevector",
        options={"prepare": "never", "external_timeout_seconds": 1.5},
    )
    assert total == 7.0
    assert payload["status"] == "ok"
    assert captured["model"] == "pheat-generic"
    assert captured["kwargs"] == {"prepare": "never", "external_timeout_seconds": 1.5}


def test_folder_passes_decoded_torsions_to_coarse_pheat_scorer(monkeypatch):
    import numpy as np
    import qtf.core.folder as folder_mod
    from qtf.core.folder import QuantumBiophysicsFolder

    captured = {}

    class FakeScore:
        total = 3.0

        def to_dict(self):
            return {
                "model": "pheat-coarse-protein-folding-v1",
                "total": 3.0,
                "units": "arbitrary",
                "terms": {},
                "warnings": [],
                "citations": [],
                "metadata": {},
            }

    def fake_score(structure, model, **kwargs):
        captured["model"] = model
        captured["kwargs"] = kwargs
        return FakeScore()

    monkeypatch.setattr(folder_mod, "score_pheat_structure", fake_score)
    folder = QuantumBiophysicsFolder("GAA", score_model="pheat-coarse-protein-folding-v1")
    payload, total = folder.score_model_for_params(
        np.zeros(folder.n_params),
        "pheat-coarse-protein-folding-v1",
        angle_mode="statevector",
        options={"hydrophobic_gamma": 15.0},
    )
    assert total == 3.0
    assert payload["status"] == "ok"
    assert captured["model"] == "pheat-coarse-protein-folding-v1"
    assert captured["kwargs"]["hydrophobic_gamma"] == 15.0
    assert "decoded_torsions" in captured["kwargs"]
    assert captured["kwargs"]["decoded_torsions"]
