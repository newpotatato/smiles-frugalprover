"""Repo-anchored paths, so code resolves data/results the same way no matter
which directory it's launched from (repo root, a notebook, or Colab).

Anchoring on this file's location means `frugalprover sample` and an imported
`run_sample(cfg)` land in the same place without any `cd` or relative-path
guessing.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
CONFIG_DIR = REPO_ROOT / "configs"


def data_dir(run_name: str) -> Path:
    """Where a run's intermediate artifacts live (problems, budgets, states)."""
    return DATA_DIR / run_name


def run_dir(run_name: str) -> Path:
    """Where a run's final results live (model, metrics, plots)."""
    return RESULTS_DIR / run_name


def ensure_dir(path: Path) -> Path:
    """mkdir -p for a directory, or for a file's parent. Returns `path`."""
    target = path.parent if path.suffix else path
    target.mkdir(parents=True, exist_ok=True)
    return path
