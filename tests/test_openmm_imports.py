"""Tests for the openmm/pyrosetta import handling in :mod:`qtf.core.folder`.

B9: a botched ``openmm`` install that fails to load its compiled C++
extensions used to be reported as "OpenMM not installed" because the
catch on the import block swallowed *any* ``Exception``. The user
would then chase a phantom missing-package issue when the real fix
was to reinstall the broken openmm package. These tests pin the
narrower exception handling.
"""

import builtins
import logging

import pytest


def test_load_openmm_swallows_module_not_found(monkeypatch):
    """A genuine ``ModuleNotFoundError`` is swallowed; symbols are None."""
    from qtf.core import folder

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "openmm" or name.startswith("openmm."):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    symbols, available = folder._load_openmm()
    assert available is False
    assert symbols == (None, None, None, None, None, None, None)


def test_load_openmm_reraises_runtime_error(monkeypatch, caplog):
    """A non-``ModuleNotFoundError`` failure is logged and re-raised."""
    from qtf.core import folder

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "openmm" or name.startswith("openmm."):
            raise RuntimeError("CUDA/dll mismatch on a real install")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with caplog.at_level(logging.ERROR, logger="qtf.core.folder"):
        with pytest.raises(RuntimeError, match="CUDA/dll mismatch"):
            folder._load_openmm()

    # The original error must be logged at ERROR level so it shows up
    # in production logs (the user is otherwise left with no breadcrumb).
    error_messages = [record.getMessage() for record in caplog.records]
    assert any("OpenMM import failed" in m for m in error_messages), (
        f"expected an 'OpenMM import failed' log line, got {error_messages}"
    )
    assert any("CUDA/dll mismatch" in m for m in error_messages)


def test_load_openmm_reraises_os_error(monkeypatch):
    """An ``OSError`` (e.g. missing shared library) is also re-raised."""
    from qtf.core import folder

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "openmm" or name.startswith("openmm."):
            raise OSError("libOpenMM.so.8: cannot open shared object file")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(OSError, match="libOpenMM.so.8"):
        folder._load_openmm()


def test_ensure_openmm_raises_import_error_with_conda_hint(monkeypatch):
    """When openmm is unavailable, ``_ensure_openmm`` raises ``ImportError``
    pointing the user at ``conda install -c conda-forge openmm``."""
    from qtf.core import folder

    # Simulate "openmm is missing" without re-running the module-level
    # import. We replace the module-level ``_OPENMM_AVAILABLE`` flag
    # and force ``_openmm_ready`` to be False on the folder instance.
    monkeypatch.setattr(folder, "_OPENMM_AVAILABLE", False, raising=False)
    # ``_openmm_ready`` is an instance attribute initialised in
    # ``__init__``; a freshly-built folder has it False by default.
    folder_instance = folder.QuantumBiophysicsFolder("GA")
    assert folder_instance._openmm_ready is False

    with pytest.raises(ImportError, match=r"conda install -c conda-forge openmm"):
        folder_instance._ensure_openmm()
