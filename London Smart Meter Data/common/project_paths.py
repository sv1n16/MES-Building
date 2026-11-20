"""Project folders, experiment routing and figure metadata, shared by every script.

Every script starts with

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
    import project_paths  # noqa: E402

    project_paths.register(__file__)

register() does three things:

1. puts common/ and every experiment folder on sys.path, so scripts import each
   other by module name wherever they live (e.g. the real-time scripts import
   admm_bilateral_p2p_valley from common/);
2. records the experiment of the script that was run (the first register() call
   wins, so a library imported from another experiment does not change it).
   results_dir() is then Results/<experiment>/[<SHOWCASE_PLOTS_SUBDIR>], and
   MES_EXPERIMENT overrides the experiment (run_scaling.py uses it so the
   day-ahead scripts it launches write into 04_day_ahead_scaling/);
3. tags every figure saved with matplotlib (PNG, PDF, SVG) or plotly (HTML) with
   metadata: title, description, experiment, producing script and command line,
   dataset, git commit and creation time. Read it back with
   ``python common/tag_figures.py --show <figure>``.
"""

from __future__ import annotations

import datetime as _dt
import importlib.abc
import importlib.util
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMON_DIR = ROOT / "common"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "Results"
LCL_DIR = ROOT / "Small LCL Data"
PV_DIR = ROOT / "PV Data"

EXPERIMENTS = (
    "01_data_preparation",
    "02_day_ahead_sharing_methods",
    "03_day_ahead_sensitivity",
    "04_day_ahead_scaling",
    "05_realtime_methods",
    "06_realtime_scaling",
    "07_communication_failure",
    "08_minute_resolution",
    "09_ramp_rate_control",
)

_experiment: str | None = None
_entry_script: Path | None = None


def register(script, experiment: str | None = None) -> None:
    """Set up paths, the current experiment and figure tagging (see module docstring).

    experiment names the results folder when the script is not inside an
    experiment folder (the runnable modules in common/).
    """
    global _experiment, _entry_script
    for folder in (COMMON_DIR, *(ROOT / name for name in EXPERIMENTS)):
        if str(folder) not in sys.path:
            sys.path.append(str(folder))
    if _experiment is None:
        script = Path(script).resolve()
        folder = script.parent.name
        _experiment = os.environ.get("MES_EXPERIMENT", "").strip() or experiment or (
            folder if folder in EXPERIMENTS else "misc"
        )
        _entry_script = script
        _install_figure_hooks()


def experiment() -> str:
    return _experiment or "misc"


def experiment_dir(name: str | None = None) -> Path:
    """Results/<name> (default: the current experiment), created if missing."""
    path = RESULTS_DIR / (name or experiment())
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_dir() -> Path:
    """Where the current run writes: Results/<experiment>/[<SHOWCASE_PLOTS_SUBDIR>]."""
    path = experiment_dir()
    sub = os.environ.get("SHOWCASE_PLOTS_SUBDIR", "").strip()
    if sub:
        path = path / sub
        path.mkdir(parents=True, exist_ok=True)
    return path


def relative(path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


# ============================================================================
# FIGURE METADATA
# ============================================================================
_GIT = None


def _git_commit() -> str:
    global _GIT
    if _GIT is None:
        try:
            run = lambda *a: subprocess.run(  # noqa: E731
                ["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=10
            ).stdout.strip()
            commit = run("rev-parse", "--short", "HEAD")
            dirty = run("status", "--porcelain", "--", ".")
            _GIT = f"{commit}{' (uncommitted changes)' if dirty else ''}" if commit else "unknown"
        except Exception:  # noqa: BLE001 - metadata only
            _GIT = "unknown"
    return _GIT


def _dataset() -> str:
    central = sys.modules.get("central_optimisation_showcase")
    if central is not None and hasattr(central, "n_buildings"):
        stem = getattr(central, "_DATASET_STEM", "")
        return f"{stem} (N={central.n_buildings} buildings, T={central.time_horizon} h, day {central.SHOWCASE_DAY})"
    return os.environ.get("SHOWCASE_DATASET", "").strip()  # empty (left out) when no dataset is loaded


def _figure_title(figure) -> str:
    try:
        if getattr(figure, "_suptitle", None) is not None and figure._suptitle.get_text():
            return " ".join(figure._suptitle.get_text().split())
        for axis in figure.axes:
            if axis.get_title():
                return " ".join(axis.get_title().split())
    except Exception:  # noqa: BLE001
        pass
    return ""


def figure_metadata(path, title: str = "") -> dict:
    """Metadata for a figure written by this run (generic keys; see _for_format)."""
    import figure_catalogue

    name = Path(str(path)).name
    entry = figure_catalogue.lookup(name)
    return {
        "Title": entry.get("title") or title or Path(name).stem,
        "Description": entry.get("description", ""),
        "Experiment": experiment(),
        "Source": relative(_entry_script) if _entry_script else "",
        "Command": " ".join([Path(sys.executable).name, *(relative(a) if Path(a).exists() else a for a in sys.argv)]),
        "Dataset": _dataset(),
        "Plot subfolder": os.environ.get("SHOWCASE_PLOTS_SUBDIR", "").strip(),
        "Git commit": _git_commit(),
        "Creation Time": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "Software": f"Python {platform.python_version()}",
    }


def _for_format(meta: dict, fmt: str) -> dict | None:
    """Map the generic metadata to what each matplotlib backend accepts."""
    meta = {k: v for k, v in meta.items() if v}
    if fmt == "png":
        return meta
    details = "; ".join(f"{k}: {v}" for k, v in meta.items() if k not in ("Title", "Description"))
    if fmt == "pdf":
        return {"Title": meta.get("Title", ""), "Subject": meta.get("Description", ""),
                "Keywords": details, "Creator": meta.get("Source", "")}
    if fmt == "svg":
        return {"Title": meta.get("Title", ""), "Description": f"{meta.get('Description', '')} {details}".strip(),
                "Creator": meta.get("Source", ""), "Date": meta.get("Creation Time", "")}
    return None


def _patch_matplotlib(module) -> None:
    figure_class = module.Figure
    if getattr(figure_class.savefig, "_mes_tagged", False):
        return
    original = figure_class.savefig

    def savefig(self, fname, *args, **kwargs):
        try:
            fmt = kwargs.get("format") or (Path(os.fspath(fname)).suffix.lstrip(".").lower()
                                           if isinstance(fname, (str, os.PathLike)) else "")
            if not fmt:
                import matplotlib

                fmt = matplotlib.rcParams["savefig.format"]
            extra = _for_format(figure_metadata(fname, _figure_title(self)), fmt)
            if extra is not None:
                kwargs["metadata"] = {**extra, **(kwargs.get("metadata") or {})}
        except Exception as error:  # noqa: BLE001 - never block a save over metadata
            print(f"[project_paths] figure metadata skipped: {error!r}")
        return original(self, fname, *args, **kwargs)

    savefig._mes_tagged = True
    figure_class.savefig = savefig


def _patch_plotly(module) -> None:
    figure_class = module.BaseFigure
    if getattr(figure_class.write_html, "_mes_tagged", False):
        return
    original = figure_class.write_html

    def write_html(self, file, *args, **kwargs):
        result = original(self, file, *args, **kwargs)
        if isinstance(file, (str, os.PathLike)):
            try:
                import tag_figures

                title = getattr(getattr(self.layout, "title", None), "text", "") or ""
                tag_figures.tag_html(Path(file), figure_metadata(file, title))
            except Exception as error:  # noqa: BLE001
                print(f"[project_paths] figure metadata skipped: {error!r}")
        return result

    write_html._mes_tagged = True
    figure_class.write_html = write_html


_PATCHES = {"matplotlib.figure": _patch_matplotlib, "plotly.basedatatypes": _patch_plotly}


class _PatchOnImport(importlib.abc.MetaPathFinder):
    """Patch matplotlib / plotly when (and only if) a script imports them."""

    def find_spec(self, name, path, target=None):
        if name not in _PATCHES or name in sys.modules:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        loader, patch = spec.loader, _PATCHES[name]
        original_exec = loader.exec_module

        def exec_module(module):
            original_exec(module)
            patch(module)

        loader.exec_module = exec_module
        return spec


def _install_figure_hooks() -> None:
    for name, patch in _PATCHES.items():
        if name in sys.modules:
            patch(sys.modules[name])
    if not any(isinstance(finder, _PatchOnImport) for finder in sys.meta_path):
        sys.meta_path.insert(0, _PatchOnImport())
