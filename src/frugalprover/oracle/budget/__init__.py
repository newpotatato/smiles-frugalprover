"""Stage 2 - budget labeling. Produces `budgets.jsonl` (A2).

The runner here handles everything an estimator shouldn't have to care about:
batching, resume, per-record flushing, ordering, and the sidecar. An estimator
implements one method.
"""
from __future__ import annotations

import random
import time

from frugalprover.common.config import AgentConfig, BudgetConfig, PipelineConfig
from frugalprover.common.io import (
    append_jsonl,
    existing_ids,
    read_jsonl,
    sort_jsonl_by_id,
    write_meta,
)
from frugalprover.common.logging import get_logger, track
from frugalprover.common.records import BudgetRecord, ProblemRecord
from frugalprover.oracle.budget.base import BudgetEstimator
from frugalprover.oracle.budget.mock import MockEstimator
from frugalprover.oracle.budget.sweep import TokenSweepEstimator

log = get_logger(__name__)

__all__ = [
    "BudgetEstimator", "MockEstimator", "TokenSweepEstimator",
    "ESTIMATORS", "build_estimator", "run_budget",
]

#: Register a new labeling strategy here and it becomes available as
#: `budget.estimator: <name>` with no other changes.
ESTIMATORS = {
    "sweep": TokenSweepEstimator,
    "mock": MockEstimator,
}


def build_estimator(
    cfg: BudgetConfig, agent_cfg: AgentConfig | None = None
) -> BudgetEstimator:
    try:
        cls = ESTIMATORS[cfg.estimator]
    except KeyError:
        raise ValueError(
            f"unknown budget.estimator {cfg.estimator!r}. Available: {sorted(ESTIMATORS)}"
        ) from None
    # The sweep estimator drives the pipeline's solving agent; the mock needs no
    # agent at all. Only pass the agent config to the estimator that uses it.
    if cls is TokenSweepEstimator:
        if agent_cfg is None:
            raise ValueError(
                "budget.estimator='sweep' needs the pipeline agent config; "
                "call build_estimator(cfg.budget, cfg.agent)."
            )
        return cls(cfg, agent_cfg)
    return cls(cfg)


def run_budget(cfg: PipelineConfig) -> list[BudgetRecord]:
    """Label problems with solve effort and write A2.

    Resumable: rerunning after a crash picks up only the problems missing from
    the output file. That matters because this is the stage that costs GPU
    hours -- losing four hours of sweeping to a dropped Colab session is the
    failure mode the whole append-and-flush design exists to prevent.
    """
    bc = cfg.budget
    # Started before anything loads: weight loading and a cold model download are
    # part of the wall clock the caller is budgeting.
    t0 = time.perf_counter()
    problems_path = cfg.data_path(bc.problems)
    out = cfg.data_path(bc.output)

    if not problems_path.exists():
        raise FileNotFoundError(
            f"{problems_path} not found - run `frugalprover sample` first, or point "
            f"budget.problems at an existing problems file."
        )

    problems = [ProblemRecord.from_dict(d) for d in read_jsonl(problems_path)]
    if bc.shuffle:
        # Before max_problems (so both truncation paths are covered) and before
        # the resume filter (so the permutation depends only on the file and the
        # seed -- shuffling `todo` instead would make the order depend on how
        # many times the run crashed).
        random.Random(bc.seed).shuffle(problems)
        log.info("shuffled %d problems with budget.seed=%d -- a truncated run stays "
                 "representative across subjects and levels", len(problems), bc.seed)
    if bc.max_problems is not None:
        problems = problems[: bc.max_problems]

    done = existing_ids(out)
    todo = [p for p in problems if p.id not in done]
    if done:
        log.info("resuming: %d already labeled, %d to go", len(done), len(todo))
    if not todo:
        log.info("nothing to do - all %d problems already in %s", len(problems), out)
        return [BudgetRecord.from_dict(d) for d in read_jsonl(out)]

    estimator = build_estimator(bc, cfg.agent)
    agent_desc = bc.agent if bc.estimator == "mock" else cfg.agent.prover.model
    log.info("labeling %d problems with estimator=%r agent=%r budgets=%s n_samples=%s",
             len(todo), bc.estimator, agent_desc, bc.budgets, bc.n_samples)

    estimator.setup()
    n_labeled = 0
    stopped_early = False
    batch_s: float | None = None  # EMA of per-batch wall time, for the deadline check
    try:
        starts = list(range(0, len(todo), bc.batch_size))
        for i in track(starts, description="labeling", total=len(starts)):
            if _should_stop(bc.time_budget_s, t0, batch_s):
                stopped_early = True
                break
            batch = todo[i : i + bc.batch_size]
            t_batch = time.perf_counter()
            try:
                for record in estimator.estimate_batch(batch):
                    append_jsonl(out, record.to_dict())
                    n_labeled += 1
            except Exception:
                if not bc.continue_on_error:
                    raise
                log.exception("batch at offset %d failed -- skipping %d problem(s). "
                              "Rerun to retry them.", i, len(batch))
                continue
            dt = time.perf_counter() - t_batch
            batch_s = dt if batch_s is None else 0.7 * batch_s + 0.3 * dt
    finally:
        estimator.teardown()

    if not out.exists():
        # Reachable when a time budget expires before the first batch finishes
        # (a deadline shorter than model loading, or a resumed run whose window
        # has already passed). Nothing to sort or summarize -- say so plainly
        # rather than dying on the missing file.
        log.warning("no batch completed before the run stopped -- nothing written to %s", out)
        return []

    sort_jsonl_by_id(out)
    records = [BudgetRecord.from_dict(d) for d in read_jsonl(out)]
    stats = describe(records)
    elapsed = time.perf_counter() - t0
    meta = {
        "artifact": "budgets",
        "produced_by": f"frugalprover.oracle.budget:{type(estimator).__name__}",
        "config": bc.__dict__,
        "n_records": len(records),
        "elapsed_s": round(elapsed, 1),
        "n_labeled_this_run": n_labeled,
        "n_remaining": max(0, len(todo) - n_labeled),
        "stopped_early": stopped_early,
        "shuffled": bc.shuffle,
        "order_seed": bc.seed if bc.shuffle else None,
        **stats,
    }
    # For an agent-driven sweep, record which agent actually produced the labels
    # (bc.agent is superseded by the full agent config under estimator='sweep').
    agent = getattr(estimator, "agent", None)
    if agent is not None:
        meta["agent_spec"] = agent.spec
    write_meta(out, meta)

    log.info("wrote %d budget labels -> %s", len(records), out)
    log.info("  solved: %d/%d  censored: %d",
             stats["n_solved"], len(records), stats["n_censored"])
    log.info("  b_star distribution: %s", stats["b_star_distribution"])
    if stats["single_pass"]:
        log.info("  single-budget run: usable for classification, not for regression")
    if stopped_early:
        log.warning("  stopped early: %d of %d problems still unlabeled -- rerun the same "
                    "command to continue where this left off",
                    len(todo) - n_labeled, len(todo))
    return records


def _should_stop(limit: float | None, t0: float, batch_s: float | None) -> bool:
    """Whether to stop before starting another batch.

    The predictive arm is the one that buys coverage: a batch started three
    minutes before the deadline finishes after it, wasting both those minutes and
    the batch. `batch_s` is an EMA, so it self-calibrates to the hardware.
    """
    if limit is None:
        return False
    elapsed = time.perf_counter() - t0
    if elapsed >= limit:
        log.warning("time budget %.0fs reached after %.0fs -- stopping between batches",
                    limit, elapsed)
        return True
    if batch_s is not None and elapsed + batch_s > limit:
        log.warning("time budget %.0fs: %.0fs elapsed and the next batch needs ~%.0fs -- "
                    "stopping now rather than starting one that would be cut off",
                    limit, elapsed, batch_s)
        return True
    return False


def describe(records: list[BudgetRecord]) -> dict:
    """Label distribution. Printed and stored in the sidecar so a degenerate
    sweep -- everything censored, or everything solved at the smallest budget --
    is visible immediately instead of surfacing later as a confusing CV score."""
    budgets = sorted({b for r in records for b in r.budgets})
    dist = {str(b): sum(1 for r in records if r.b_star == b) for b in budgets}
    dist["censored"] = sum(1 for r in records if r.censored)
    return {
        "n_solved": sum(1 for r in records if not r.censored),
        "n_censored": sum(1 for r in records if r.censored),
        "budgets": budgets,
        "b_star_distribution": dist,
        "single_pass": len(budgets) <= 1,
    }
