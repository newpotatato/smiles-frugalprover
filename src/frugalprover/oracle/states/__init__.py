"""Stage 3 - hidden state extraction. Produces `hidden_states.parquet` (A3)."""
from __future__ import annotations

from frugalprover.common.config import ExtractConfig, PipelineConfig
from frugalprover.common.io import read_jsonl, write_table
from frugalprover.common.records import ProblemRecord
from frugalprover.oracle.states.base import HiddenStateExtractor
from frugalprover.oracle.states.pooling import geometry, pool, resolve_layer
from frugalprover.oracle.states.synthetic import SyntheticExtractor

__all__ = [
    "HiddenStateExtractor", "SyntheticExtractor", "TransformerHiddenStateExtractor",
    "EXTRACTORS", "build_extractor", "pool", "geometry", "resolve_layer", "run_extract",
]


def build_extractor(cfg: ExtractConfig) -> HiddenStateExtractor:
    """Construct the configured extractor.

    The transformer one is imported here rather than at module import, so that
    `frugalprover train` on a laptop never touches torch.
    """
    if cfg.extractor == "synthetic":
        return SyntheticExtractor(cfg)
    if cfg.extractor == "transformer":
        from frugalprover.oracle.states.hf_extractor import TransformerHiddenStateExtractor

        return TransformerHiddenStateExtractor(cfg)
    raise ValueError(
        f"unknown extract.extractor {cfg.extractor!r}. Available: ['transformer', 'synthetic']"
    )


EXTRACTORS = ["transformer", "synthetic"]


def __getattr__(name):
    if name == "TransformerHiddenStateExtractor":
        from frugalprover.oracle.states.hf_extractor import TransformerHiddenStateExtractor

        return TransformerHiddenStateExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def run_extract(cfg: PipelineConfig):
    """Extract hidden states for every problem and write A3.

    Not resumable, unlike Stage 2: a forward pass over a few hundred problems
    takes minutes, not hours, so restarting is cheaper than the bookkeeping.
    """
    import pandas as pd

    ec = cfg.extract
    problems_path = cfg.data_path(ec.problems)
    out = cfg.data_path(ec.output)

    if not problems_path.exists():
        raise FileNotFoundError(
            f"{problems_path} not found - run `frugalprover sample` first, or point "
            f"extract.problems at an existing problems file."
        )

    problems = [ProblemRecord.from_dict(d) for d in read_jsonl(problems_path)]
    extractor = build_extractor(ec)
    extractor.setup()

    rows = []
    try:
        for i in range(0, len(problems), ec.batch_size):
            batch = problems[i : i + ec.batch_size]
            rows.extend(extractor.extract_batch(batch))
            print(f"  {min(i + len(batch), len(problems))}/{len(problems)}", end="\r", flush=True)
        spec = extractor.spec
    finally:
        extractor.teardown()

    df = pd.DataFrame(rows)
    write_table(df, out, meta={
        "artifact": "hidden_states",
        "produced_by": f"frugalprover.oracle.states:{type(extractor).__name__}",
        "config": {k: v for k, v in ec.__dict__.items() if k not in ("features", "geometry")},
        **spec,
    })

    n_pooled = len(spec["pooled_columns"])
    n_geom = len(spec["geometry_columns"])
    print(f"\nwrote {len(df)} rows -> {out}")
    print(f"  {n_pooled} pooled columns (hidden_size={spec['hidden_size']}), "
          f"{n_geom} geometry columns")
    return df
