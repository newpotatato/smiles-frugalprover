"""Repo-anchored paths, so scripts resolve data/results the same way no matter
which directory they're launched from.

Code lives under experiments/, pipeline/, extraction/; the shared data pool and
generated artifacts live at the repo root in data/ and results/. Anchoring on
this file's location (src/frugalprover/paths.py -> repo root is two parents up)
means `python experiments/A_oracle/oracle.py` and `python pipeline/data_prep.py`
both find data/ without any `cd` or relative-path guessing.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
