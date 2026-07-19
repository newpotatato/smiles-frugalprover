"""`frugalprover runs` - compare every run under results/.

This is the experiment tracking. At this scale (dozens of runs, sklearn fits
that take seconds) a hosted tracker would cost an account and an network
dependency to replace a table you can print. The one thing a table has to do
that a plain listing doesn't is show *what differed* between runs -- otherwise
you're left diffing YAML by eye to work out why one scored better.

If the runs ever outgrow a terminal, mirroring `metrics.json` to W&B is a small
hook here; it isn't worth it before then.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from frugalprover.common.io import read_json


def collect_runs(results_dir: str | Path) -> list[dict]:
    """One summary dict per run directory containing a metrics.json."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []

    runs = []
    for run_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            metrics = read_json(metrics_path)
        except Exception:
            continue
        entry = {
            "run": run_dir.name,
            "mode": metrics.get("mode"),
            "layer": metrics.get("layer"),
            "metric": metrics.get("cv_metric"),
            "score": metrics.get("cv_score"),
            "n": metrics.get("n_problems"),
            "censored": metrics.get("n_censored"),
            "config": _flatten(_load_config(run_dir)),
        }
        runs.append(entry)
    return runs


def _load_config(run_dir: Path) -> dict:
    path = run_dir / "config.yaml"
    if not path.exists():
        return {}
    try:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    else:
        out[prefix] = d
    return out


def differing_keys(runs: list[dict], ignore: tuple[str, ...] = ("run_name",)) -> list[str]:
    """Config keys whose value isn't the same across all runs.

    Everything shared is noise in a comparison table; only what varies explains
    why the scores differ.
    """
    if len(runs) < 2:
        return []
    keys = set().union(*(set(r["config"]) for r in runs))
    varying = []
    for key in sorted(keys):
        if any(key.endswith(i) for i in ignore):
            continue
        values = {_hashable(r["config"].get(key)) for r in runs}
        if len(values) > 1:
            varying.append(key)
    return varying


def _hashable(v: Any) -> Any:
    """Comparable stand-in for a config value.

    Values can be arbitrarily nested (`extract.features` is a list of dicts),
    so serialize rather than trying to coerce to tuples.
    """
    import json

    if isinstance(v, (list, dict)):
        return json.dumps(v, sort_keys=True, default=str)
    return v


def print_runs_table(results_dir: str | Path) -> None:
    runs = collect_runs(results_dir)
    if not runs:
        print(f"no runs with a metrics.json under {results_dir}")
        print("run `frugalprover report` to produce one.")
        return

    varying = differing_keys(runs)
    headers = ["run", "mode", "layer", "score", "n", "censored"] + varying
    rows = []
    for r in runs:
        score = r["score"]
        rows.append([
            r["run"],
            r["mode"] or "-",
            r["layer"] or "-",
            f"{score:.3f}" if isinstance(score, (int, float)) else "-",
            str(r["n"] or "-"),
            str(r["censored"] if r["censored"] is not None else "-"),
        ] + [_fmt(r["config"].get(k)) for k in varying])

    widths = [max(len(str(h)), *(len(str(row[i])) for row in rows)) for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))

    if varying:
        print(f"\n(showing the {len(varying)} config key(s) that differ between runs)")
    else:
        print("\n(all runs share the same config)")

    scored = [r for r in runs if isinstance(r["score"], (int, float))]
    if len(scored) > 1:
        best = max(scored, key=lambda r: r["score"])
        print(f"best: {best['run']} - {best['metric']} {best['score']:.3f} at {best['layer']}")


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, list):
        return "[" + ",".join(str(x) for x in v) + "]"
    return str(v)
