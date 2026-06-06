"""Tests for the optional-dependency contract on QTF public surfaces.

B11: a user who installs ``pip install qtf`` (no extras) and then
runs ``import qtf`` (or ``from qtf.analysis import ...`` /
``from qtf.visualization import ...``) should not have to pay the
import cost of pandas / plotly / matplotlib, and a missing
package should surface as a clear ``ImportError`` with an
actionable hint, not a bare ``ModuleNotFoundError`` raised from a
module-level ``import`` statement.

The tests below verify the contract at three levels:

1. **Static**: the module-level source of the affected files must
   not contain ``import pandas``, ``import matplotlib``, or
   ``import plotly`` (each is a heavy optional dep). A grep is
   enough — if a future contributor adds such an import back at
   module top, the test fails.
2. **Runtime proxy**: the lazy-proxy classes resolve the real
   module on first attribute access and cache it in ``globals()``.
3. **Clear error**: calling the lazy loader without the dep
   raises ``ImportError`` with the install-command hint.
"""

import importlib
import inspect
import re
import sys

import pytest


# ---------------------------------------------------------------------------
# Static checks: no heavy optional dep at module top
# ---------------------------------------------------------------------------


def _read(path: str) -> str:
    with open(path) as fh:
        return fh.read()


def test_qtf_init_does_not_import_optional_deps():
    """``qtf/__init__.py`` must not import pandas/plotly/matplotlib."""
    src = _read("qtf/__init__.py")
    for needle in ("import pandas", "from pandas",
                    "import plotly", "from plotly",
                    "import matplotlib", "from matplotlib"):
        assert needle not in src, (
            f"qtf/__init__.py contains a module-level {needle!r}; "
            f"this is a heavy optional dep and should be lazy."
        )


def _module_level_imports(src: str) -> list[str]:
    """Return the set of import statements that sit at the module's
    top indentation level (column 0). Imports nested inside a
    function or a class body are ignored."""
    out = []
    for line in src.splitlines():
        if not line.strip():
            continue
        # Top-level lines start at column 0. Lines starting with
        # whitespace are inside a function/class body.
        if line[0] in (" ", "\t"):
            continue
        stripped = line.strip()
        if (stripped.startswith("import ") or stripped.startswith("from ")):
            out.append(stripped)
    return out


def test_visualization_plots_does_not_import_plotly_at_top():
    """``qtf/visualization/plots.py`` must lazy-load plotly."""
    src = _read("qtf/visualization/plots.py")
    top = _module_level_imports(src)
    for stmt in top:
        assert "plotly" not in stmt, (
            f"qtf/visualization/plots.py has {stmt!r} at module top; "
            f"plotly must be lazy-loaded via _require_plotly()."
        )


def test_analysis_panel_does_not_import_pandas_or_matplotlib_at_top():
    """``qtf/analysis/panel.py`` must lazy-load pandas and matplotlib."""
    src = _read("qtf/analysis/panel.py")
    top = _module_level_imports(src)
    for stmt in top:
        assert "pandas" not in stmt, (
            f"qtf/analysis/panel.py has {stmt!r} at module top; "
            f"pandas must be lazy-loaded via the _LazyPandas proxy."
        )
        assert "matplotlib" not in stmt, (
            f"qtf/analysis/panel.py has {stmt!r} at module top; "
            f"matplotlib must be lazy-loaded via the _LazyPyplot proxy."
        )


def test_analysis_ranking_does_not_import_pandas_at_top():
    """``qtf/analysis/ranking.py`` must lazy-load pandas."""
    src = _read("qtf/analysis/ranking.py")
    top = _module_level_imports(src)
    for stmt in top:
        assert "pandas" not in stmt, (
            f"qtf/analysis/ranking.py has {stmt!r} at module top; "
            f"pandas must be lazy-loaded via the _LazyPandas proxy."
        )


# ---------------------------------------------------------------------------
# Runtime: the lazy proxies resolve the real module on first use
# ---------------------------------------------------------------------------


def test_panel_lazy_pandas_proxy_resolves_to_real_pandas():
    from qtf.analysis import panel
    # First attribute access triggers the import.
    df = panel.pd.DataFrame({"a": [1, 2, 3]})
    assert df.shape == (3, 1)
    # After first access, the proxy is replaced by the real module.
    assert panel.pd is not panel._LazyPandas  # type: ignore[attr-defined]
    assert panel.pd.DataFrame is __import__("pandas").DataFrame


def test_panel_lazy_pyplot_proxy_resolves_to_real_pyplot():
    pytest.importorskip("matplotlib")
    from qtf.analysis import panel
    # First attribute access triggers the import.
    fig = panel.plt.figure()
    panel.plt.close(fig)
    # After first access, the proxy is replaced by the real module.
    assert panel.plt is not panel._LazyPyplot  # type: ignore[attr-defined]


def test_ranking_lazy_pandas_proxy_resolves_to_real_pandas():
    from qtf.analysis import ranking
    df = ranking.pd.DataFrame({"a": [1]})
    assert df.shape == (1, 1)
    assert ranking.pd is not ranking._LazyPandas  # type: ignore[attr-defined]


def test_visualization_require_plotly_returns_plotly_symbols():
    from qtf.visualization import plots
    go, make_subplots = plots._require_plotly()
    # go is the real plotly.graph_objects module
    import plotly.graph_objects as real_go
    assert go is real_go


# ---------------------------------------------------------------------------
# Runtime: the clear error message on missing optional dep
# ---------------------------------------------------------------------------


def test_panel_lazy_pandas_raises_clear_error_when_pandas_missing(monkeypatch):
    """If pandas cannot be imported, the ``_LazyPandas.__getattr__``
    proxy must raise ``ImportError`` with the install hint, not a bare
    ``ModuleNotFoundError``.

    We instantiate a *fresh* ``_LazyPandas`` proxy here (not the one
    bound to ``panel.pd``) so the test is independent of any cached
    state from earlier tests in the same module.
    """
    import builtins
    from qtf.analysis.panel import _LazyPandas

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pandas" or name.startswith("pandas."):
            raise ModuleNotFoundError("No module named 'pandas'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    proxy = _LazyPandas()
    with pytest.raises(ImportError, match=r"pandas is required"):
        _ = proxy.DataFrame


def test_panel_lazy_pyplot_raises_clear_error_when_matplotlib_missing(monkeypatch):
    """If matplotlib cannot be imported, the ``_LazyPyplot.__getattr__``
    proxy must raise ``ImportError`` with the workflows-install hint."""
    import builtins
    from qtf.analysis.panel import _LazyPyplot

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ModuleNotFoundError("No module named 'matplotlib'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    proxy = _LazyPyplot()
    with pytest.raises(ImportError, match=r"qtf\[workflows\]"):
        _ = proxy.figure


def test_visualization_require_plotly_raises_clear_error_when_plotly_missing(monkeypatch):
    """If plotly cannot be imported, ``_require_plotly()`` must raise
    ``ImportError`` with the install hint."""
    import builtins
    from qtf.visualization import plots

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "plotly" or name.startswith("plotly."):
            raise ModuleNotFoundError("No module named 'plotly'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match=r"plotly is required"):
        plots._require_plotly()
