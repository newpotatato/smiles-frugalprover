"""Adapter: pipeline artifacts -> the flat record dicts the analyses expect.

The analysis scripts in this package predate the staged pipeline. They were
written against the pilot's `pilot_results.jsonl`, where one line held a
problem, its surface features, its activations at every layer, and its budget
sweep all at once.

Rather than rewrite several hundred lines of verified statistics against the
new three-file layout -- and risk quietly changing a result while doing it --
this module rebuilds that flat shape from the new artifacts. The statistics
code is untouched; only where the data comes from has changed.

If you're writing a *new* analysis, don't use this. Use `OracleDataset`
directly.
"""
from __future__ import annotations

import re

from frugalprover.common.grading import SURFACE_KEYS, surface_features
from frugalprover.oracle.model.dataset import OracleDataset

_POOLED_RE = re.compile(r"^L(\d+)_")


def load_records(
    problems: str,
    hidden_states: str | None = None,
    budgets: str | None = None,
    pooling: str | None = None,
) -> list[dict]:
    """Load the artifacts and flatten them into old-style record dicts.

    `pooling` picks which pooled family to expose as `activations` when the
    parquet holds more than one (e.g. both `L14_mean` and `L14_last`). Defaults
    to whichever family covers the most layers, since that's the one a sweep
    wants.

    Note the key format: `activations` is keyed `layer_14`, not `L14_mean`. The
    analyses parse the index out with `key.split("_")[1]`, so this preserves
    that. The pooling used is reported once at load time so it isn't invisible.
    """
    ds = OracleDataset.load(problems, hidden_states, budgets, verbose=False)
    return records_from_dataset(ds, pooling=pooling)


def records_from_dataset(ds: OracleDataset, pooling: str | None = None) -> list[dict]:
    """Flatten an already-loaded :class:`OracleDataset`."""
    activations: dict[str, dict[str, list]] = {}
    if ds.hidden is not None:
        family = pooling or _default_pooling(ds)
        columns = [c for c in ds.pooled_columns() if c.endswith(f"_{family}")]
        if not columns:
            raise ValueError(
                f"no pooled columns with pooling {family!r}. Available: {ds.pooled_columns()}"
            )
        print(f"analysis: using pooling {family!r} across {len(columns)} layers")
        for col in columns:
            layer_idx = _POOLED_RE.match(col).group(1)
            vectors = ds.activations(col)
            activations[f"layer_{layer_idx}"] = [v.tolist() for v in vectors]

    records = []
    for i, p in enumerate(ds.problems):
        phi = surface_features(p.problem)
        rec = {
            "id": p.id,
            "type": p.type,
            "level_num": p.level,
            "problem": p.problem,
            "phi": {k: phi[k] for k in SURFACE_KEYS},
            "activations": {name: vals[i] for name, vals in activations.items()},
        }
        budget = ds.budgets.get(p.id)
        if budget is not None:
            # only the swept tier carries these; the analyses branch on
            # `"p_by_budget" in r`, so an unlabeled problem must simply not
            # have the key rather than have it set to None
            rec["p_by_budget"] = dict(budget.p)
            rec["b_star"] = budget.b_star
            if budget.sc:
                rec["sc_by_budget"] = dict(budget.sc)
        records.append(rec)
    return records


def _default_pooling(ds: OracleDataset) -> str:
    """The pooling family covering the most layers."""
    counts: dict[str, int] = {}
    for col in ds.pooled_columns():
        family = col.split("_", 1)[1]
        counts[family] = counts.get(family, 0) + 1
    if not counts:
        raise ValueError("hidden states contain no pooled columns")
    return max(counts, key=counts.get)


def add_paths_arg(parser) -> None:
    """Standard artifact arguments, shared by every analysis entry point."""
    parser.add_argument("--problems", default="data/pilot/problems.jsonl")
    parser.add_argument("--hidden-states", default="data/pilot/hidden_states.parquet")
    parser.add_argument("--budgets", default="data/pilot/budgets.jsonl")
    parser.add_argument("--pooling", default=None,
                        help="which pooled family to analyse (default: the one with most layers)")
    parser.add_argument("--out-dir", default="results/analysis",
                        help="where plots go (the old scripts wrote to the cwd)")
