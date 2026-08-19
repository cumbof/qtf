from pathlib import Path
from subprocess import CompletedProcess

from qtf.utils import gromacs


def test_minimize_pdb_with_gromacs_compacts_successful_workdir(tmp_path, monkeypatch):
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C\n"
        "END\n"
    )
    workdir = tmp_path / "gmx"

    monkeypatch.setattr(gromacs, "find_gmx", lambda: "gmx")

    def fake_run(cmd, cwd, log_path, input_text=None):
        command = cmd[1]
        if command == "pdb2gmx":
            Path(cmd[5]).write_text("processed\n")
            Path(cmd[7]).write_text("topology\n")
            Path(cmd[9]).write_text("posre\n")
        elif command == "editconf" and "em.gro" not in cmd:
            Path(cmd[5]).write_text("boxed\n")
        elif command == "grompp":
            (cwd / "em.tpr").write_text("tpr\n")
        elif command == "mdrun":
            (cwd / "em.gro").write_text("gro\n")
            (cwd / "em.edr").write_text("edr\n")
            (cwd / "em.log").write_text("Maximum force     =  9.0e+01\nconverged to Fmax < 100\n")
        elif command == "energy":
            Path(cmd[5]).write_text("@ title\n0 -123.4\n")
        elif command == "editconf":
            Path(cmd[5]).write_text("minimized\n")
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write("$ " + " ".join(cmd) + "\n")
        return CompletedProcess(cmd, 0, stdout="")

    monkeypatch.setattr(gromacs, "_run", fake_run)

    result = gromacs.minimize_pdb_with_gromacs(str(input_pdb), str(workdir))

    assert result["gromacs_status"] == "ok"
    assert result["gromacs_potential_kj_mol"] == -123.4
    assert result["gromacs_converged_fmax_lt_100"] is True
    assert sorted(path.name for path in workdir.iterdir()) == ["gromacs_minimize.log", "minimized.pdb"]


def test_minimize_pdb_with_gromacs_keeps_failed_workdir_for_debugging(tmp_path, monkeypatch):
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "END\n"
    )
    workdir = tmp_path / "gmx"

    monkeypatch.setattr(gromacs, "find_gmx", lambda: "gmx")

    def fake_run(cmd, cwd, log_path, input_text=None):
        (cwd / "processed.gro").write_text("partial\n")
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write("failure\n")
        return CompletedProcess(cmd, 1, stdout="failed")

    monkeypatch.setattr(gromacs, "_run", fake_run)

    result = gromacs.minimize_pdb_with_gromacs(str(input_pdb), str(workdir))

    assert result["gromacs_status"] == "failed"
    assert (workdir / "gromacs_minimize.log").exists()
    assert (workdir / "processed.gro").exists()
    assert (workdir / "prepared_input.pdb").exists()


def test_minimize_pdb_with_gromacs_rejects_nonconverged_output(tmp_path, monkeypatch):
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\nEND\n"
    )
    workdir = tmp_path / "gmx"
    monkeypatch.setattr(gromacs, "find_gmx", lambda: "gmx")

    def fake_run(cmd, cwd, log_path, input_text=None):
        command = cmd[1]
        if command == "pdb2gmx":
            Path(cmd[5]).write_text("processed\n")
            Path(cmd[7]).write_text("topology\n")
            Path(cmd[9]).write_text("posre\n")
        elif command == "editconf" and "em.gro" not in cmd:
            Path(cmd[5]).write_text("boxed\n")
        elif command == "grompp":
            (cwd / "em.tpr").write_text("tpr\n")
        elif command == "mdrun":
            (cwd / "em.gro").write_text("gro\n")
            (cwd / "em.edr").write_text("edr\n")
            (cwd / "em.log").write_text("Maximum force     =  1.0e+20\n")
        elif command == "energy":
            Path(cmd[5]).write_text("@ title\n0 1.0e+23\n")
        elif command == "editconf":
            Path(cmd[5]).write_text("invalid minimized structure\n")
        return CompletedProcess(cmd, 0, stdout="")

    monkeypatch.setattr(gromacs, "_run", fake_run)
    result = gromacs.minimize_pdb_with_gromacs(str(input_pdb), str(workdir))

    assert result["gromacs_status"] == "failed"
    assert result["gromacs_converged"] is False
    assert result["gromacs_minimized_full_pdb_path"] == ""
    assert "invalid potential energy" in result["gromacs_message"]


def test_refine_pdb_with_gromacs_retains_named_artifacts(tmp_path, monkeypatch):
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text("MODEL\nENDMDL\n", encoding="utf-8")
    output_pdb = tmp_path / "results" / "replica_0_final_gromacs_refined.pdb"
    output_log = tmp_path / "results" / "replica_0_final_gromacs_refined.log"

    def fake_minimize(pdb_path, workdir, **kwargs):
        generated_pdb = Path(workdir) / "minimized.pdb"
        generated_log = Path(workdir) / "gromacs_minimize.log"
        generated_pdb.write_text("MINIMIZED\n", encoding="utf-8")
        generated_log.write_text("CONVERGED\n", encoding="utf-8")
        return {
            "gromacs_status": "ok",
            "gromacs_workdir": str(workdir),
            "gromacs_minimized_full_pdb_path": str(generated_pdb),
            "gromacs_log_path": str(generated_log),
        }

    monkeypatch.setattr(gromacs, "minimize_pdb_with_gromacs", fake_minimize)
    result = gromacs.refine_pdb_with_gromacs(
        str(input_pdb), output_pdb, log_path=output_log
    )

    assert output_pdb.read_text(encoding="utf-8") == "MINIMIZED\n"
    assert output_log.read_text(encoding="utf-8") == "CONVERGED\n"
    assert result["gromacs_minimized_full_pdb_path"] == str(output_pdb.resolve())
    assert result["gromacs_log_path"] == str(output_log.resolve())
    assert result["gromacs_workdir"] is None


def test_refine_pdb_with_gromacs_retains_failure_log_only(tmp_path, monkeypatch):
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text("MODEL\nENDMDL\n", encoding="utf-8")
    output_pdb = tmp_path / "minimized.pdb"

    def fake_minimize(pdb_path, workdir, **kwargs):
        generated_log = Path(workdir) / "gromacs_minimize.log"
        generated_log.write_text("FAILED\n", encoding="utf-8")
        return {
            "gromacs_status": "failed",
            "gromacs_workdir": str(workdir),
            "gromacs_minimized_full_pdb_path": "",
            "gromacs_log_path": str(generated_log),
        }

    monkeypatch.setattr(gromacs, "minimize_pdb_with_gromacs", fake_minimize)
    result = gromacs.refine_pdb_with_gromacs(str(input_pdb), output_pdb)

    assert not output_pdb.exists()
    assert output_pdb.with_suffix(".log").read_text(encoding="utf-8") == "FAILED\n"
    assert result["gromacs_workdir"] is None
