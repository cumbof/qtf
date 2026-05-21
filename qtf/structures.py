"""Structure conversion helpers shared by QTF engines and reports."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def qtf_structure_to_pheat(
    coords: np.ndarray,
    labels: Sequence[tuple],
    sequence: str,
    *,
    name: str = "qtf-structure",
):
    """Convert QTF ``coords``/``labels`` arrays to a PHEAT heavy-atom structure."""

    from pheat import Atom, HeavyAtomStructure
    from pheat.residues import one_to_three

    atoms = []
    sequence = sequence.upper()
    for serial, (coord, label) in enumerate(zip(coords, labels), start=1):
        res_id, atom_name, element = label[:3]
        res_index = int(res_id)
        resname = one_to_three(sequence[res_index])
        atoms.append(
            Atom(
                str(atom_name),
                str(element),
                float(coord[0]),
                float(coord[1]),
                float(coord[2]),
                resname,
                chain_id="A",
                resseq=res_index + 1,
                record_name="ATOM",
                serial=serial,
            )
        )
    return HeavyAtomStructure(
        atoms=atoms,
        name=name,
        metadata={
            "source": "qtf",
            "sequence": sequence,
            "label_format": "(res_id, atom_name, element)",
        },
        atom_scope="heavy",
    )
