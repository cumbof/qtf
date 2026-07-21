import os

import numpy as np
import pandas as pd

from qtf.cli import fold as fold_cli
from qtf.core.folder import QuantumBiophysicsFolder


class _FakeFolder:
    static_labels = [(0, "CA", "C"), (0, "CB", "C")]
    chi_mode = "all"
    n_params = 2
    n_qubits = 1
    ansatz = object()

    def compute_sidechain_centroids(self, coords, labels):
        return np.asarray([[0.5, 0.0, 0.0]], dtype=float)

    def save_reduced_pdb(self, coords, filename, sidechain_centroids=None, energy=None):
        with open(filename, "w", encoding="utf-8") as handle:
            handle.write("MODEL\n")

    def save_pdb(self, coords, labels, filename, energy=None, remarks=None, include_hydrogens=False):
        with open(filename, "w", encoding="utf-8") as handle:
            handle.write("MODEL\n")


class _FakeManager:
    def __init__(self, folder):
        self.folder = folder
        self.top_k_snapshots = None
        self.snapshot_energy_gap = None
        coords = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
        labels = [(0, "CA", "C"), (0, "CB", "C")]
        self.results = [
            {
                "id": 0,
                "seed": 123,
                "type": "fake",
                "energy": -1.0,
                "coords": coords,
                "labels": labels,
                "bonds": [],
                "params": np.asarray([0.1, 0.2], dtype=float),
                "best_snapshots": [
                    {"energy": -2.0, "coords": coords, "labels": labels, "bonds": []},
                    {"energy": -1.5, "coords": coords, "labels": labels, "bonds": []},
                ],
            }
        ]
        self.initial_params_list = None

    def run_ensemble(
        self,
        n_runs,
        max_iter,
        top_k_snapshots=0,
        snapshot_energy_gap=0.0,
        initial_params_list=None,
    ):
        self.top_k_snapshots = top_k_snapshots
        self.snapshot_energy_gap = snapshot_energy_gap
        self.initial_params_list = initial_params_list

    def get_results(self, ranked=True):
        return self.results

    def select_top(self, top_k=1, top_frac=None):
        return self.results[:top_k]


class _RmsdFakeManager(_FakeManager):
    def __init__(self, folder):
        super().__init__(folder)
        coords = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
        labels = [(0, "CA", "C"), (0, "CB", "C")]
        self.results[0]["coords"] = coords
        self.results[0]["labels"] = labels
        self.results[0]["best_snapshots"] = [
            {"energy": -2.0, "coords": np.asarray([[10.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float), "labels": labels, "bonds": []},
            {"energy": -1.5, "coords": np.asarray([[20.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float), "labels": labels, "bonds": []},
        ]


def _patch_fast_fold(monkeypatch):
    monkeypatch.setattr(fold_cli.utils, "make_folder", lambda **kwargs: _FakeFolder())
    monkeypatch.setattr(fold_cli, "EnsembleFoldingManager", _FakeManager)
    monkeypatch.setattr(fold_cli.utils, "calculate_physics_metrics", lambda coords: {"end_to_end": 1.0, "radius_of_gyration": 0.5})
    monkeypatch.setattr(
        fold_cli.utils,
        "rmsd_selection_metadata",
        lambda *args, **kwargs: {
            "rmsd_atom_selection": "ca",
            "rmsd_excludes_terminal_residues": False,
            "rmsd_start_residue_1indexed": 1,
            "rmsd_end_residue_1indexed": 1,
            "rmsd_n_selected_residues": 1,
            "rmsd_n_selected_atoms": 1,
            "rmsd_n_aligned": 1,
            "rmsd_n_matched": 1,
            "rmsd_n_missing": 0,
        },
    )
    monkeypatch.setattr(fold_cli, "nonlocal_heavy_clash_metrics", lambda coords, labels: {})
    monkeypatch.setattr(fold_cli, "adjacent_heavy_clash_metrics", lambda coords, labels: {})
    monkeypatch.setattr(fold_cli.qtf_gromacs, "ring_penetration_metrics", lambda coords, labels: {})


def test_fold_cli_defaults_to_window_omega_mode(monkeypatch, tmp_path):
    _patch_fast_fold(monkeypatch)
    make_folder_kwargs = {}

    def fake_make_folder(**kwargs):
        make_folder_kwargs.update(kwargs)
        return _FakeFolder()

    def fake_gromacs_postprocess_structure(enabled, full_pdb_path, gromacs_dir, coords, labels, ca_coords, sidechain_centroid_fn, **kwargs):
        return {
            "coords": coords,
            "labels": labels,
            "ca_coords": ca_coords,
            "sidechain_centroids": sidechain_centroid_fn(coords, labels),
            "nonlocal_clash_metrics": {},
            "local_clash_metrics": {},
            "ring_penetration_metrics": {},
            "gromacs_info": {"gromacs_status": "not_run"},
        }

    monkeypatch.setattr(fold_cli.utils, "make_folder", fake_make_folder)
    monkeypatch.setattr(fold_cli.utils, "gromacs_postprocess_structure", fake_gromacs_postprocess_structure)

    fold_cli.main(
        [
            "--predict",
            "GA",
            "--mode",
            "predict_only",
            "--ensemble_size",
            "1",
            "--maxiter",
            "1",
            "--top_k",
            "1",
            "--top_k_snapshots",
            "0",
            "--gromacs_minimize",
            "0",
            "--output_root",
            str(tmp_path),
        ]
    )

    assert make_folder_kwargs["omega_mode"] == "window"


def test_best_snapshot_tracker_enforces_energy_gap():
    folder = QuantumBiophysicsFolder.__new__(QuantumBiophysicsFolder)
    energies = iter([-10.0, -9.9, -10.1, -9.74, -9.73, -9.4])

    def fake_energy_function(params, **kwargs):
        return next(energies)

    folder.energy_function = fake_energy_function
    tracker = folder._build_best_k_tracker(3, min_energy_gap=0.25)

    for idx in range(6):
        tracker(np.asarray([idx], dtype=float))

    kept = sorted(-entry[0] for entry in tracker._heap)
    assert kept == [-10.1, -9.74, -9.4]


def test_best_snapshot_tracker_rejects_bridging_energy_gap_candidate():
    folder = QuantumBiophysicsFolder.__new__(QuantumBiophysicsFolder)
    energies = iter([0.0, 0.18, 0.09])

    def fake_energy_function(params, **kwargs):
        return next(energies)

    folder.energy_function = fake_energy_function
    tracker = folder._build_best_k_tracker(3, min_energy_gap=0.1)

    for idx in range(3):
        tracker(np.asarray([idx], dtype=float))

    kept = sorted(-entry[0] for entry in tracker._heap)
    assert kept == [0.0, 0.18]


def test_reference_alignment_transform_applies_to_full_structure():
    reference = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    labels = [(0, "N", "N"), (0, "CA", "C"), (0, "C", "C"), (1, "CA", "C")]
    rotation = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    translation = np.asarray([4.0, -2.0, 7.0], dtype=float)
    model = reference @ rotation + translation

    aligned, rmsd, _meta, transform = fold_cli.utils.align_structure_to_reference(
        model,
        labels,
        reference,
        labels,
        "heavy",
        "all",
    )

    assert np.allclose(aligned, reference)
    assert np.isclose(rmsd, 0.0)

    centroid = fold_cli.utils.apply_alignment_transform({0: model[0]}, transform)
    assert np.allclose(centroid[0], reference[0])


def test_fold_snapshot_gromacs_follows_gromacs_minimize(monkeypatch, tmp_path):
    _patch_fast_fold(monkeypatch)
    minimized_snapshot_inputs = []
    minimized_snapshot_workdirs = []
    final_postprocess_enabled = []

    def fake_minimize_pdb_with_gromacs(input_pdb, workdir, **kwargs):
        minimized_snapshot_inputs.append(input_pdb)
        minimized_snapshot_workdirs.append(workdir)
        return {
            "gromacs_status": "ok",
            "gromacs_potential_kj_mol": -10.0,
            "gromacs_potential_kcal_mol": -2.39,
            "gromacs_final_max_force": 50.0,
            "gromacs_converged_fmax_lt_100": True,
            "gromacs_minimized_full_pdb_path": "",
        }

    def fake_gromacs_postprocess_structure(enabled, full_pdb_path, gromacs_dir, coords, labels, ca_coords, sidechain_centroid_fn, **kwargs):
        final_postprocess_enabled.append(enabled)
        return {
            "coords": coords,
            "labels": labels,
            "ca_coords": ca_coords,
            "sidechain_centroids": sidechain_centroid_fn(coords, labels),
            "nonlocal_clash_metrics": {},
            "local_clash_metrics": {},
            "ring_penetration_metrics": {},
            "gromacs_info": {
                "gromacs_status": "ok" if enabled else "not_run",
                "gromacs_potential_kj_mol": -1.0 if enabled else np.nan,
            },
        }

    monkeypatch.setattr(fold_cli.qtf_gromacs, "minimize_pdb_with_gromacs", fake_minimize_pdb_with_gromacs)
    monkeypatch.setattr(fold_cli.utils, "gromacs_postprocess_structure", fake_gromacs_postprocess_structure)

    fold_cli.main(
        [
            "--predict",
            "GA",
            "--mode",
            "predict_only",
            "--ensemble_size",
            "1",
            "--maxiter",
            "1",
            "--top_k",
            "1",
            "--top_k_snapshots",
            "2",
            "--gromacs_minimize",
            "1",
            "--output_root",
            str(tmp_path),
        ]
    )

    assert final_postprocess_enabled == [True]
    assert len(minimized_snapshot_inputs) == 2
    run_dir = next(path for path in tmp_path.iterdir() if path.is_dir())
    assert minimized_snapshot_workdirs == [
        str(run_dir / "gromacs_minimized_models" / "snapshots" / "replica_1" / "snapshot_1"),
        str(run_dir / "gromacs_minimized_models" / "snapshots" / "replica_1" / "snapshot_2"),
    ]
    assert all("raw_models/snapshots/replica_1" in path for path in minimized_snapshot_inputs)
    assert all(not os.path.exists(path) for path in minimized_snapshot_inputs)
    snapshot_csv = next(tmp_path.glob("*/snapshot_ranked.csv"))
    snapshots = pd.read_csv(snapshot_csv)
    assert snapshots["snapshot_gromacs_enabled"].tolist() == [True, True]
    assert snapshots["snapshot_raw_pdb_retained"].tolist() == [False, False]
    assert snapshots["snapshot_pdb_path"].isna().tolist() == [True, True]


def test_fold_snapshot_gromacs_disabled_with_gromacs_minimize_zero(monkeypatch, tmp_path):
    _patch_fast_fold(monkeypatch)
    minimized_snapshot_inputs = []
    final_postprocess_enabled = []

    monkeypatch.setattr(
        fold_cli.qtf_gromacs,
        "minimize_pdb_with_gromacs",
        lambda input_pdb, workdir, **kwargs: minimized_snapshot_inputs.append(input_pdb),
    )

    def fake_gromacs_postprocess_structure(enabled, full_pdb_path, gromacs_dir, coords, labels, ca_coords, sidechain_centroid_fn, **kwargs):
        final_postprocess_enabled.append(enabled)
        return {
            "coords": coords,
            "labels": labels,
            "ca_coords": ca_coords,
            "sidechain_centroids": sidechain_centroid_fn(coords, labels),
            "nonlocal_clash_metrics": {},
            "local_clash_metrics": {},
            "ring_penetration_metrics": {},
            "gromacs_info": {"gromacs_status": "not_run"},
        }

    monkeypatch.setattr(fold_cli.utils, "gromacs_postprocess_structure", fake_gromacs_postprocess_structure)

    fold_cli.main(
        [
            "--predict",
            "GA",
            "--mode",
            "predict_only",
            "--ensemble_size",
            "1",
            "--maxiter",
            "1",
            "--top_k",
            "1",
            "--top_k_snapshots",
            "2",
            "--gromacs_minimize",
            "0",
            "--output_root",
            str(tmp_path),
        ]
    )

    assert final_postprocess_enabled == [False]
    assert minimized_snapshot_inputs == []
    snapshot_csv = next(tmp_path.glob("*/snapshot_ranked.csv"))
    snapshots = pd.read_csv(snapshot_csv)
    assert snapshots["snapshot_gromacs_enabled"].tolist() == [False, False]
    assert snapshots["snapshot_raw_pdb_retained"].tolist() == [True, True]
    assert all(os.path.exists(path) for path in snapshots["snapshot_pdb_path"])
    run_dir = snapshot_csv.parent
    snapshot_pdb = run_dir / "snapshot_ranked.pdb"
    ensemble_pdb = run_dir / "ensemble_ranked.pdb"
    assert snapshot_pdb.exists()
    assert ensemble_pdb.exists()
    snapshot_text = snapshot_pdb.read_text()
    ensemble_text = ensemble_pdb.read_text()
    assert sum(1 for line in snapshot_text.splitlines() if line.startswith("MODEL")) == 2
    assert sum(1 for line in ensemble_text.splitlines() if line.startswith("MODEL")) == 1
    assert "QTF_SOURCE replica=replica_1" in snapshot_text
    assert "QTF_RANK file_rank=1 energy_rank=1 rmsd_rank=" in snapshot_text
    assert "gromacs_potential_kj_mol=NA" in snapshot_text
    assert "QTF_SOURCE replica=replica_1" in ensemble_text
    assert "gromacs_potential_kj_mol=NA" in ensemble_text


def test_fold_keeps_raw_snapshot_when_gromacs_fails(monkeypatch, tmp_path):
    _patch_fast_fold(monkeypatch)
    minimized_snapshot_inputs = []
    final_postprocess_enabled = []

    def fake_minimize_pdb_with_gromacs(input_pdb, workdir, **kwargs):
        minimized_snapshot_inputs.append(input_pdb)
        return {
            "gromacs_status": "failed",
            "gromacs_message": "command failed: gmx mdrun",
            "gromacs_potential_kj_mol": np.nan,
            "gromacs_potential_kcal_mol": np.nan,
            "gromacs_final_max_force": np.nan,
            "gromacs_converged_fmax_lt_100": False,
            "gromacs_minimized_full_pdb_path": "",
        }

    def fake_gromacs_postprocess_structure(enabled, full_pdb_path, gromacs_dir, coords, labels, ca_coords, sidechain_centroid_fn, **kwargs):
        final_postprocess_enabled.append(enabled)
        return {
            "coords": coords,
            "labels": labels,
            "ca_coords": ca_coords,
            "sidechain_centroids": sidechain_centroid_fn(coords, labels),
            "nonlocal_clash_metrics": {},
            "local_clash_metrics": {},
            "ring_penetration_metrics": {},
            "gromacs_info": {"gromacs_status": "ok"},
        }

    monkeypatch.setattr(fold_cli.qtf_gromacs, "minimize_pdb_with_gromacs", fake_minimize_pdb_with_gromacs)
    monkeypatch.setattr(fold_cli.utils, "gromacs_postprocess_structure", fake_gromacs_postprocess_structure)

    fold_cli.main(
        [
            "--predict",
            "GA",
            "--mode",
            "predict_only",
            "--ensemble_size",
            "1",
            "--maxiter",
            "1",
            "--top_k",
            "1",
            "--top_k_snapshots",
            "2",
            "--gromacs_minimize",
            "1",
            "--output_root",
            str(tmp_path),
        ]
    )

    assert final_postprocess_enabled == [True]
    assert all(os.path.exists(path) for path in minimized_snapshot_inputs)
    snapshot_csv = next(tmp_path.glob("*/snapshot_ranked.csv"))
    snapshots = pd.read_csv(snapshot_csv)
    assert snapshots["snapshot_gromacs_status"].tolist() == ["failed", "failed"]
    assert snapshots["snapshot_raw_pdb_retained"].tolist() == [True, True]
    assert all(os.path.exists(path) for path in snapshots["snapshot_pdb_path"])


def test_snapshot_rmsd_sort_uses_effective_gromacs_rmsd_when_available(monkeypatch, tmp_path):
    _patch_fast_fold(monkeypatch)
    monkeypatch.setattr(fold_cli, "EnsembleFoldingManager", _RmsdFakeManager)
    monkeypatch.setattr(
        fold_cli.utils,
        "load_reference_rmsd_coords",
        lambda *args, **kwargs: (np.asarray([[0.0, 0.0, 0.0]], dtype=float), [(0, "CA", "C")], {}),
    )

    def fake_rmsd_between_structures(coords, labels, *args, **kwargs):
        marker = float(np.asarray(coords)[0, 0])
        rmsd_by_marker = {
            0.0: 9.0,
            10.0: 1.0,
            20.0: 4.0,
            110.0: 5.0,
            120.0: 0.5,
        }
        return rmsd_by_marker.get(marker, 9.0), {}

    def fake_align_structure_to_reference(coords, labels, *args, **kwargs):
        rmsd, meta = fake_rmsd_between_structures(coords, labels, *args, **kwargs)
        return np.asarray(coords, dtype=float), rmsd, meta, None

    def fake_minimize_pdb_with_gromacs(input_pdb, workdir, **kwargs):
        os.makedirs(workdir, exist_ok=True)
        minimized_pdb = os.path.join(workdir, "minimized.pdb")
        with open(minimized_pdb, "w", encoding="utf-8") as handle:
            handle.write("MODEL\n")
        return {
            "gromacs_status": "ok",
            "gromacs_potential_kj_mol": -10.0,
            "gromacs_potential_kcal_mol": -2.39,
            "gromacs_final_max_force": 50.0,
            "gromacs_converged_fmax_lt_100": True,
            "gromacs_minimized_full_pdb_path": minimized_pdb,
        }

    def fake_parse_pdb_atoms(path):
        marker = 110.0 if "snapshot_1" in path else 120.0
        return np.asarray([[marker, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float), [(0, "CA", "C"), (0, "CB", "C")]

    def fake_gromacs_postprocess_structure(enabled, full_pdb_path, gromacs_dir, coords, labels, ca_coords, sidechain_centroid_fn, **kwargs):
        return {
            "coords": coords,
            "labels": labels,
            "ca_coords": ca_coords,
            "sidechain_centroids": sidechain_centroid_fn(coords, labels),
            "nonlocal_clash_metrics": {},
            "local_clash_metrics": {},
            "ring_penetration_metrics": {},
            "gromacs_info": {"gromacs_status": "ok"},
        }

    monkeypatch.setattr(fold_cli.utils, "align_structure_to_reference", fake_align_structure_to_reference)
    monkeypatch.setattr(fold_cli.utils, "rmsd_between_structures", fake_rmsd_between_structures)
    monkeypatch.setattr(fold_cli.qtf_gromacs, "minimize_pdb_with_gromacs", fake_minimize_pdb_with_gromacs)
    monkeypatch.setattr(fold_cli.qtf_gromacs, "parse_pdb_atoms", fake_parse_pdb_atoms)
    monkeypatch.setattr(fold_cli.utils, "gromacs_postprocess_structure", fake_gromacs_postprocess_structure)

    fold_cli.main(
        [
            "--predict",
            "GA",
            "--reference_pdb",
            "fake.pdb",
            "--mode",
            "predict_and_compare",
            "--ensemble_size",
            "1",
            "--maxiter",
            "1",
            "--top_k",
            "1",
            "--top_k_snapshots",
            "2",
            "--snapshot_sort_by",
            "rmsd",
            "--gromacs_minimize",
            "1",
            "--output_root",
            str(tmp_path),
        ]
    )

    snapshot_csv = next(tmp_path.glob("*/snapshot_ranked.csv"))
    snapshots = pd.read_csv(snapshot_csv)
    assert snapshots["snapshot_rank_within_replica"].tolist() == [2, 1]
    assert snapshots["snapshot_rmsd_to_reference_A"].tolist() == [4.0, 1.0]
    assert snapshots["snapshot_gromacs_rmsd_to_reference_A"].tolist() == [0.5, 5.0]
    assert snapshots["snapshot_effective_rmsd_to_reference_A"].tolist() == [0.5, 5.0]
    snapshot_pdb = snapshot_csv.parent / "snapshot_ranked.pdb"
    assert snapshot_pdb.exists()
    snapshot_text = snapshot_pdb.read_text()
    assert "snapshot_rank_within_replica=2" in snapshot_text.split("ENDMDL")[0]
    assert "QTF_RANK file_rank=1 energy_rank=2 rmsd_rank=1" in snapshot_text
    assert "gromacs_potential_kj_mol=-10" in snapshot_text


def test_fold_writes_circuit_parameter_artifacts(monkeypatch, tmp_path):
    _patch_fast_fold(monkeypatch)

    def fake_gromacs_postprocess_structure(enabled, full_pdb_path, gromacs_dir, coords, labels, ca_coords, sidechain_centroid_fn, **kwargs):
        return {
            "coords": coords,
            "labels": labels,
            "ca_coords": ca_coords,
            "sidechain_centroids": sidechain_centroid_fn(coords, labels),
            "nonlocal_clash_metrics": {},
            "local_clash_metrics": {},
            "ring_penetration_metrics": {},
            "gromacs_info": {"gromacs_status": "not_run"},
        }

    monkeypatch.setattr(fold_cli.utils, "gromacs_postprocess_structure", fake_gromacs_postprocess_structure)

    fold_cli.main(
        [
            "--predict",
            "GA",
            "--mode",
            "predict_only",
            "--ensemble_size",
            "1",
            "--maxiter",
            "1",
            "--top_k",
            "1",
            "--top_k_snapshots",
            "0",
            "--gromacs_minimize",
            "0",
            "--output_root",
            str(tmp_path),
        ]
    )

    run_dir = next(path for path in tmp_path.iterdir() if path.is_dir())
    params_dir = run_dir / "circuit_parameters"
    manifest_path = params_dir / "circuit_parameters.json"
    json_path = params_dir / "replica_1_params.json"
    npz_path = params_dir / "replica_1_params.npz"
    assert manifest_path.exists()
    assert json_path.exists()
    assert npz_path.exists()

    import json as _json
    manifest = _json.loads(manifest_path.read_text())
    payload = _json.loads(json_path.read_text())
    assert manifest["format"] == "qtf.circuit_parameters.v1"
    assert manifest["replicas"][0]["json_path"] == "replica_1_params.json"
    assert payload["params"] == [0.1, 0.2]

    ensemble = pd.read_csv(run_dir / "ensemble_ranked.csv")
    assert os.path.exists(ensemble.loc[0, "circuit_params_json_path"])
    assert os.path.exists(ensemble.loc[0, "circuit_params_npz_path"])


def test_fold_accepts_saved_circuit_parameter_manifest_as_warm_start(monkeypatch, tmp_path):
    _patch_fast_fold(monkeypatch)
    captured_managers = []

    class CapturingManager(_FakeManager):
        def __init__(self, folder):
            super().__init__(folder)
            captured_managers.append(self)

    def fake_gromacs_postprocess_structure(enabled, full_pdb_path, gromacs_dir, coords, labels, ca_coords, sidechain_centroid_fn, **kwargs):
        return {
            "coords": coords,
            "labels": labels,
            "ca_coords": ca_coords,
            "sidechain_centroids": sidechain_centroid_fn(coords, labels),
            "nonlocal_clash_metrics": {},
            "local_clash_metrics": {},
            "ring_penetration_metrics": {},
            "gromacs_info": {"gromacs_status": "not_run"},
        }

    monkeypatch.setattr(fold_cli, "EnsembleFoldingManager", CapturingManager)
    monkeypatch.setattr(fold_cli.utils, "gromacs_postprocess_structure", fake_gromacs_postprocess_structure)

    params_dir = tmp_path / "previous" / "circuit_parameters"
    params_dir.mkdir(parents=True)
    params_json = params_dir / "replica_1_params.json"
    params_json.write_text(
        (
            "{"
            '"format": "qtf.circuit_parameters.v1", '
            '"n_params": 2, '
            '"params": [0.3, 0.4]'
            "}"
        ),
        encoding="utf-8",
    )
    (params_dir / "circuit_parameters.json").write_text(
        (
            "{"
            '"format": "qtf.circuit_parameters.v1", '
            '"replicas": [{"ensemble_id": 0, "json_path": "replica_1_params.json"}]'
            "}"
        ),
        encoding="utf-8",
    )

    fold_cli.main(
        [
            "--predict",
            "GA",
            "--mode",
            "predict_only",
            "--ensemble_size",
            "1",
            "--maxiter",
            "1",
            "--top_k",
            "1",
            "--top_k_snapshots",
            "0",
            "--gromacs_minimize",
            "0",
            "--initial_params",
            str(params_dir / "circuit_parameters.json"),
            "--output_root",
            str(tmp_path / "new"),
        ]
    )

    assert len(captured_managers) == 1
    assert captured_managers[0].initial_params_list is not None
    assert np.allclose(captured_managers[0].initial_params_list[0], [0.3, 0.4])


def test_fold_initial_params_select_best_energy_uses_lowest_raw_energy(monkeypatch, tmp_path):
    _patch_fast_fold(monkeypatch)
    captured_managers = []

    class CapturingManager(_FakeManager):
        def __init__(self, folder):
            super().__init__(folder)
            captured_managers.append(self)

    monkeypatch.setattr(fold_cli, "EnsembleFoldingManager", CapturingManager)
    monkeypatch.setattr(
        fold_cli.utils,
        "gromacs_postprocess_structure",
        lambda enabled, full_pdb_path, gromacs_dir, coords, labels, ca_coords, sidechain_centroid_fn, **kwargs: {
            "coords": coords,
            "labels": labels,
            "ca_coords": ca_coords,
            "sidechain_centroids": sidechain_centroid_fn(coords, labels),
            "nonlocal_clash_metrics": {},
            "local_clash_metrics": {},
            "ring_penetration_metrics": {},
            "gromacs_info": {"gromacs_status": "not_run"},
        },
    )

    previous = tmp_path / "previous"
    params_dir = previous / "circuit_parameters"
    params_dir.mkdir(parents=True)
    np.savez_compressed(params_dir / "replica_1_params.npz", params=np.asarray([1.0, 1.1]))
    np.savez_compressed(params_dir / "replica_2_params.npz", params=np.asarray([2.0, 2.2]))
    np.savez_compressed(params_dir / "replica_3_params.npz", params=np.asarray([3.0, 3.3]))
    pd.DataFrame(
        [
            {
                "ensemble_id": 0,
                "energy": -1.0,
                "rmsd_to_reference_A": 1.0,
                "gromacs_potential_kcal_mol": -999.0,
                "circuit_params_npz_path": str(params_dir / "replica_1_params.npz"),
            },
            {
                "ensemble_id": 1,
                "energy": -5.0,
                "rmsd_to_reference_A": 5.0,
                "gromacs_potential_kcal_mol": -10.0,
                "circuit_params_npz_path": str(params_dir / "replica_2_params.npz"),
            },
            {
                "ensemble_id": 2,
                "energy": -3.0,
                "rmsd_to_reference_A": 0.5,
                "gromacs_potential_kcal_mol": -20.0,
                "circuit_params_npz_path": str(params_dir / "replica_3_params.npz"),
            },
        ]
    ).to_csv(previous / "ensemble_ranked.csv", index=False)

    fold_cli.main(
        [
            "--predict",
            "GA",
            "--mode",
            "predict_only",
            "--ensemble_size",
            "1",
            "--maxiter",
            "1",
            "--top_k",
            "1",
            "--top_k_snapshots",
            "0",
            "--gromacs_minimize",
            "0",
            "--initial_params",
            str(previous),
            "--initial_params_select",
            "best_energy",
            "--output_root",
            str(tmp_path / "new"),
        ]
    )

    assert np.allclose(captured_managers[0].initial_params_list[0], [2.0, 2.2])


def test_fold_initial_params_select_best_rmsd_uses_lowest_rmsd(monkeypatch, tmp_path):
    _patch_fast_fold(monkeypatch)
    captured_managers = []

    class CapturingManager(_FakeManager):
        def __init__(self, folder):
            super().__init__(folder)
            captured_managers.append(self)

    monkeypatch.setattr(fold_cli, "EnsembleFoldingManager", CapturingManager)
    monkeypatch.setattr(
        fold_cli.utils,
        "gromacs_postprocess_structure",
        lambda enabled, full_pdb_path, gromacs_dir, coords, labels, ca_coords, sidechain_centroid_fn, **kwargs: {
            "coords": coords,
            "labels": labels,
            "ca_coords": ca_coords,
            "sidechain_centroids": sidechain_centroid_fn(coords, labels),
            "nonlocal_clash_metrics": {},
            "local_clash_metrics": {},
            "ring_penetration_metrics": {},
            "gromacs_info": {"gromacs_status": "not_run"},
        },
    )

    previous = tmp_path / "previous"
    params_dir = previous / "circuit_parameters"
    params_dir.mkdir(parents=True)
    np.savez_compressed(params_dir / "replica_1_params.npz", params=np.asarray([1.0, 1.1]))
    np.savez_compressed(params_dir / "replica_2_params.npz", params=np.asarray([2.0, 2.2]))
    pd.DataFrame(
        [
            {
                "ensemble_id": 0,
                "energy": -100.0,
                "rmsd_to_reference_A": 9.0,
                "circuit_params_npz_path": str(params_dir / "replica_1_params.npz"),
            },
            {
                "ensemble_id": 1,
                "energy": -1.0,
                "rmsd_to_reference_A": 1.2,
                "circuit_params_npz_path": str(params_dir / "replica_2_params.npz"),
            },
        ]
    ).to_csv(previous / "ensemble_ranked.csv", index=False)

    fold_cli.main(
        [
            "--predict",
            "GA",
            "--mode",
            "predict_only",
            "--ensemble_size",
            "1",
            "--maxiter",
            "1",
            "--top_k",
            "1",
            "--top_k_snapshots",
            "0",
            "--gromacs_minimize",
            "0",
            "--initial_params",
            str(previous),
            "--initial_params_select",
            "best_rmsd",
            "--output_root",
            str(tmp_path / "new"),
        ]
    )

    assert np.allclose(captured_managers[0].initial_params_list[0], [2.0, 2.2])
