"""Stage 1 — dataset sampling. Produces `problems.jsonl` (A1)."""
from __future__ import annotations

from frugalprover.common.config import PipelineConfig
from frugalprover.common.io import write_jsonl
from frugalprover.common.logging import get_logger
from frugalprover.common.records import ProblemRecord
from frugalprover.oracle.sample.base import DatasetSampler
from frugalprover.oracle.sample.math_sampler import MathStratifiedSampler, describe, parse_level

log = get_logger(__name__)

__all__ = ["DatasetSampler", "MathStratifiedSampler", "describe", "parse_level", "run_sample"]


def run_sample(cfg: PipelineConfig) -> list[ProblemRecord]:
    """Draw the sample and write A1. Returns the records."""
    sampler = MathStratifiedSampler(cfg.sample)
    records = list(sampler.sample())
    out = cfg.data_path(cfg.sample.output)

    stats = describe(records)
    write_jsonl(
        out,
        (r.to_dict() for r in records),
        meta={
            "artifact": "problems",
            "produced_by": "frugalprover.oracle.sample:MathStratifiedSampler",
            "config": cfg.sample.__dict__,
            **stats,
        },
    )

    log.info("wrote %d problems -> %s", len(records), out)
    log.info("  levels: %s", stats["level_distribution"])
    log.info("  types:  %s", stats["type_distribution"])
    if stats["n_missing_level"]:
        log.info("  %d problems have no parseable level", stats["n_missing_level"])
    return records
