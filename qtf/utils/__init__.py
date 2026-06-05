"""Public surface of :mod:`qtf.utils`.

B5 unified several previously-duplicated helpers under this namespace.
The PDB I/O and biophysics helpers in :mod:`qtf.utils.pdb` are
eagerly imported because they have no top-level dependency on the
rest of QTF. The workflow helpers in :mod:`qtf.utils.workflow`
import :class:`qtf.core.folder.QuantumBiophysicsFolder` at the top
of their module, so importing them eagerly would create a circular
import (``qtf.core.folder`` -> ``qtf.utils`` -> ``qtf.utils.workflow``
-> ``qtf.core.folder``). They are therefore exposed through a lazy
``__getattr__`` so that ``from qtf.utils import adjacent_heavy_clash_metrics``
still works without breaking the import order.
"""

from .pdb import (
    save_pdb,
    get_ground_truth_backbone,
    calculate_physics_metrics,
    calculate_physics_metrics_rich,
)

__all__ = [
    # PDB I/O and metrics (eagerly imported above)
    "save_pdb",
    "get_ground_truth_backbone",
    "calculate_physics_metrics",
    "calculate_physics_metrics_rich",
    # Workflow helpers (lazily resolved via __getattr__)
    "AA3_TO_1",
    "adjacent_heavy_clash_metrics",
    "nonlocal_heavy_clash_metrics",
    "pdb_id_from_path",
]

# Lazy re-exports for the workflow helpers. Module-level __getattr__
# is invoked on `import qtf.utils; qtf.utils.<name>` and on
# `from qtf.utils import <name>`, so callers see a clean public surface
# without paying the import cost (or paying the circular-import cost)
# up front.
_LAZY_NAMES = {
    "AA3_TO_1": "qtf.utils.workflow",
    "adjacent_heavy_clash_metrics": "qtf.utils.workflow",
    "nonlocal_heavy_clash_metrics": "qtf.utils.workflow",
    "pdb_id_from_path": "qtf.utils.workflow",
}


def __getattr__(name):
    if name in _LAZY_NAMES:
        import importlib
        mod = importlib.import_module(_LAZY_NAMES[name])
        value = getattr(mod, name)
        globals()[name] = value  # cache for subsequent lookups
        return value
    raise AttributeError(f"module 'qtf.utils' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_NAMES.keys()))


# qtf.utils.gromacs and qtf.utils.workflow also export a wider set of
# helpers (kabsch callers, RMSD between structures, panel loaders,
# GROMACS plumbing). Import them directly:
#     from qtf.utils import gromacs, workflow
# Those modules require the optional `[workflows]` extras (mdtraj,
# biopython, openmm) and are *not* part of the default install.
