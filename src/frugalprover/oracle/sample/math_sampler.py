"""Stage 1 — a level/type-balanced sample of the MATH benchmark.

Balance matters because MATH's test split is not uniform across subjects or
difficulty levels, and an unbalanced sample would let the oracle learn "geometry
problems are long" instead of anything about solve effort. Sampling
`per_level_per_subject` from each (subject, level) cell and then capping keeps
all 35 cells represented.

Determinism: seeded shuffles plus source-derived ids mean two runs of this
stage on different machines produce the same problems with the same ids.
"""
from __future__ import annotations

import random
import re
from typing import Iterator

from frugalprover.common.config import SampleConfig
from frugalprover.common.grading import extract_boxed
from frugalprover.common.records import ProblemRecord

LEVEL_RE = re.compile(r"\d+")


def parse_level(level: str | int | None) -> int | None:
    """"Level 3" -> 3. MATH has a few rows with "Level ?"; those become None."""
    if level is None:
        return None
    if isinstance(level, int):
        return level
    m = LEVEL_RE.search(str(level))
    return int(m.group()) if m else None


class MathStratifiedSampler:
    """Balanced sample from the MATH benchmark on the HuggingFace Hub."""

    def __init__(self, cfg: SampleConfig):
        self.cfg = cfg

    def sample(self) -> Iterator[ProblemRecord]:
        from datasets import load_dataset

        cfg = self.cfg
        rng = random.Random(cfg.seed)
        pool: list[ProblemRecord] = []

        for subject in cfg.subjects:
            ds = load_dataset(cfg.dataset_name, subject, split=cfg.split)
            by_level: dict[int | None, list[ProblemRecord]] = {}

            for row_index, row in enumerate(ds):
                answer = extract_boxed(row["solution"])
                if answer is None and cfg.require_boxed_answer:
                    # no gold answer means no way to grade a solution attempt,
                    # so the problem is useless for budget labeling
                    continue
                level = parse_level(row.get("level"))
                by_level.setdefault(level, []).append(
                    ProblemRecord(
                        id=f"{subject}/{cfg.split}/{row_index}",
                        problem=row["problem"],
                        answer=answer or "",
                        type=row.get("type") or subject,
                        level=level,
                        solution=row["solution"],
                    )
                )

            for rows in by_level.values():
                rng.shuffle(rows)
                pool.extend(rows[: cfg.per_level_per_subject])

        rng.shuffle(pool)
        # sort by id so the file order is stable; the shuffle above already
        # decided *which* problems, and this decides only how they're written
        yield from sorted(pool[: cfg.n_problems], key=lambda r: r.id)


def describe(records: list[ProblemRecord]) -> dict:
    """Level and type distributions — printed after sampling and stored in the
    sidecar, so an unbalanced draw is visible immediately rather than showing up
    later as a confusing result."""
    levels = sorted({r.level for r in records if r.level is not None})
    types = sorted({r.type for r in records})
    return {
        "n_problems": len(records),
        "level_distribution": {str(lv): sum(1 for r in records if r.level == lv) for lv in levels},
        "type_distribution": {t: sum(1 for r in records if r.type == t) for t in types},
        "n_missing_level": sum(1 for r in records if r.level is None),
    }
