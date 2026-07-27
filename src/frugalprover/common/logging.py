"""Central logging + console output for FrugalProver.

Rich-backed logging that stays import-light: ``rich`` is pure-Python with no
heavy transitive deps, so this module (and everything that logs through it)
remains importable on the CPU-only base install -- no torch required (see
CLAUDE.md, "Setup & commands").

Two output channels, kept deliberately separate:

* **logs** (status, progress, warnings, errors) -> a Rich handler on *stderr*.
* **reports** (the ``runs`` table, ``info`` version dump) -> ``console``, a Rich
  console on *stdout*. Keeping reports on stdout means ``frugalprover runs >
  table.txt`` captures the table and none of the log chatter.

Usage: modules take a logger at import (cheap, no side effects)::

    from frugalprover.common.logging import get_logger
    log = get_logger(__name__)

The CLI calls :func:`configure` once from ``main()``; standalone analysis
modules call it from their own ``main()``. :func:`configure` is idempotent and
re-runnable, so calling it again to raise the level or attach a per-run log file
just replaces the handlers.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Union

from rich.console import Console
from rich.logging import RichHandler

#: Rich console for *report* output (tables, version dumps) on stdout.
console = Console()

#: Rich console for *log* records on stderr. Shared with :func:`track` so a live
#: progress bar and a log line don't fight over the same stream.
_err_console = Console(stderr=True)

_ROOT = "frugalprover"
_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_configured = False


def _coerce(level: Union[str, int]) -> int:
    if isinstance(level, int):
        return level
    name = str(level).upper()
    if name not in _LEVELS:
        raise ValueError(f"unknown log level {level!r}; choose from {list(_LEVELS)}")
    return getattr(logging, name)


def configure(
    level: Union[str, int] = "INFO",
    *,
    log_file: Optional[Union[str, Path]] = None,
    quiet: bool = False,
) -> None:
    """Install (or reinstall) FrugalProver's log handlers on the package root.

    Idempotent -- our own handlers are cleared first, so raising the level or
    adding a file sink mid-run is just another call. ``quiet`` floors the console
    at WARNING while any file handler still records full DEBUG detail.
    """
    level = _coerce(level)
    console_level = max(level, logging.WARNING) if quiet else level

    root = logging.getLogger(_ROOT)
    root.setLevel(logging.DEBUG)   # handlers do the filtering, not the logger
    root.propagate = False         # don't double-log through the stdlib root
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()

    handler = RichHandler(
        console=_err_console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
        markup=False,
    )
    handler.setLevel(console_level)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
        )
        root.addHandler(fh)

    global _configured
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger under the ``frugalprover`` root.

    Configures a sensible default the first time it's called, so a standalone
    script (``python -m frugalprover.analysis.*``) still logs nicely even if it
    forgot to call :func:`configure`.
    """
    if not _configured:
        configure()
    return logging.getLogger(name)


def track(
    iterable: Iterable,
    description: str = "working",
    total: Optional[int] = None,
):
    """A Rich progress bar bound to the log (stderr) console.

    Replaces the old ``print(f"{i}/{n}", end="\\r")`` counters. Renders on the
    same stream as log records so the two don't collide, and vanishes on
    completion, leaving a clean transcript.
    """
    from rich.progress import track as _track

    return _track(iterable, description=description, total=total, console=_err_console)
