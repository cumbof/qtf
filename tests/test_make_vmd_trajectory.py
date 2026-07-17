from qtf.cli.make_vmd_trajectory import write_vmd_trajectory


def test_write_vmd_trajectory_keeps_common_atoms(tmp_path):
    source = tmp_path / "snapshot_ranked.pdb"
    source.write_text(
        "MODEL        1\n"
        "REMARK first\n"
        "ATOM      1    N GLY A   1       0.000   0.000   0.000  1.00  0.00            N\n"
        "ATOM      2   H1 GLY A   1       0.100   0.000   0.000  1.00  0.00            H\n"
        "ATOM      3   CA GLY A   1       1.000   0.000   0.000  1.00  0.00            C\n"
        "ENDMDL\n"
        "MODEL        2\n"
        "REMARK second\n"
        "ATOM      1    N GLY A   1       0.000   1.000   0.000  1.00  0.00            N\n"
        "ATOM      2    H GLY A   1       0.100   1.000   0.000  1.00  0.00            H\n"
        "ATOM      3   CA GLY A   1       1.000   1.000   0.000  1.00  0.00            C\n"
        "ENDMDL\n",
        encoding="utf-8",
    )
    output = tmp_path / "snapshot_ranked_vmd.pdb"

    summary = write_vmd_trajectory(source, output)
    text = output.read_text(encoding="utf-8")
    model_blocks = [block for block in text.split("ENDMDL") if "MODEL" in block]
    atom_counts = [
        sum(1 for line in block.splitlines() if line.startswith("ATOM"))
        for block in model_blocks
    ]

    assert summary["models"] == 2
    assert summary["atoms_per_model"] == 2
    assert summary["source_atom_count_groups"] == {3: 2}
    assert atom_counts == [2, 2]
    assert "QTF_VMD_COMMON_TOPOLOGY atoms_written=2" in text
    assert " H1 " not in text
    assert " H GLY " not in text
