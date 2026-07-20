"""Reading and writing pipeline artifacts.

Two formats, chosen per stage:

- **JSONL** for problem and budget records. Human-readable, greppable, and
  append-friendly — which is what makes Stage 2 resumable when a Colab session
  dies four hours into a sweep.
- **Parquet** for hidden states. A few hundred rows of 1536-dim float32 vectors
  is tens of MB; JSON would be an order of magnitude larger and slower.

Every write also drops a `<name>.meta.json` beside the artifact recording what
produced it. **Nothing reads it as a precondition.** No stage refuses to run
because a meta is missing, stale, or from a different config — it exists so
that six weeks from now a bare parquet of float arrays can tell you which model
and which prompt produced it. If you bring in a file from elsewhere, just match
the columns; the meta is optional.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from frugalprover.common.paths import ensure_dir


# --------------------------------------------------------------------- JSONL

def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file. Blank lines are skipped (a truncated final line from
    an interrupted run is common enough to be worth tolerating)."""
    path = Path(path)
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Streaming read, for files too big to hold in memory."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]], meta: dict | None = None) -> Path:
    """Write records as JSONL, replacing any existing file."""
    path = ensure_dir(Path(path))
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    if meta is not None:
        write_meta(path, {**meta, "n_records": n})
    return path


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append one record and flush immediately.

    Flushing per record is deliberate: an expensive sweep that dies partway
    through should leave every completed problem on disk, so `existing_ids`
    can skip them on restart.
    """
    path = ensure_dir(Path(path))
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def existing_ids(path: str | Path, key: str = "id") -> set[str]:
    """Ids already present in a JSONL file — the resume mechanism. Empty set if
    the file doesn't exist yet."""
    path = Path(path)
    if not path.exists():
        return set()
    return {rec[key] for rec in iter_jsonl(path) if key in rec}


def sort_jsonl_by_id(path: str | Path) -> None:
    """Rewrite a JSONL file sorted by id. Called once at the end of a resumable
    stage so the on-disk order doesn't depend on how many times it crashed."""
    path = Path(path)
    if not path.exists():
        return
    records = read_jsonl(path)
    records.sort(key=lambda r: str(r.get("id", "")))
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------- Parquet

def read_table(path: str | Path):
    """Read a parquet artifact into a DataFrame."""
    import pandas as pd

    return pd.read_parquet(path)


def write_table(df, path: str | Path, meta: dict | None = None) -> Path:
    """Write a DataFrame to parquet.

    Warns (does not fail) when there's no `id` column: a hidden-states file
    without a join key can't be matched to problems, which is exactly the flaw
    in the pre-restructure parquet. A warning rather than an error because this
    helper is also useful for ad-hoc analysis frames.
    """
    import warnings

    path = ensure_dir(Path(path))
    if "id" not in df.columns:
        warnings.warn(
            f"{path.name} has no 'id' column — it won't join to problems.jsonl. "
            "See docs/ARTIFACTS.md (A3).",
            stacklevel=2,
        )
    df.to_parquet(path, index=False)
    if meta is not None:
        write_meta(path, {**meta, "n_records": len(df)})
    return path


# ---------------------------------------------------------------- model files

def save_model(model: Any, path: str | Path, meta: dict | None = None) -> Path:
    import joblib

    path = ensure_dir(Path(path))
    joblib.dump(model, path)
    if meta is not None:
        write_meta(path, meta)
    return path


def load_model(path: str | Path) -> Any:
    import joblib

    return joblib.load(path)


def write_json(path: str | Path, obj: Any) -> Path:
    path = ensure_dir(Path(path))
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ metadata

def meta_path(artifact: str | Path) -> Path:
    """`data/run/problems.jsonl` -> `data/run/problems.jsonl.meta.json`."""
    p = Path(artifact)
    return p.with_name(p.name + ".meta.json")


def write_meta(artifact: str | Path, meta: dict) -> Path:
    """Write the sidecar. Provenance fields are filled in automatically."""
    path = meta_path(artifact)
    full = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        **meta,
    }
    return write_json(path, full)


def read_meta(artifact: str | Path) -> dict | None:
    """Read the sidecar, or None if there isn't one. Callers must treat None as
    normal — externally-produced artifacts won't have one."""
    path = meta_path(artifact)
    if not path.exists():
        return None
    try:
        return read_json(path)
    except (json.JSONDecodeError, OSError):
        return None


def git_sha() -> str | None:
    """Short commit hash, with a `-dirty` suffix for uncommitted changes.
    None outside a git checkout."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if sha.returncode != 0:
            return None
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        dirty = "-dirty" if status.stdout.strip() else ""
        return sha.stdout.strip() + dirty
    except (OSError, subprocess.SubprocessError):
        return None
