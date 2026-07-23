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
