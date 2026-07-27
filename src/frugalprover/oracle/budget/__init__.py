"""Stage 2 - budget labeling. Produces `budgets.jsonl` (A2).

The runner here handles everything an estimator shouldn't have to care about:
batching, resume, per-record flushing, ordering, and the sidecar. An estimator
implements one method.
"""
from __future__ import annotations

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
    problems_path = cfg.data_path(bc.problems)
    out = cfg.data_path(bc.output)

    if not problems_path.exists():
        raise FileNotFoundError(
            f"{problems_path} not found - run `frugalprover sample` first, or point "
            f"budget.problems at an existing problems file."
        )

    problems = [ProblemRecord.from_dict(d) for d in read_jsonl(problems_path)]
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
    try:
        starts = range(0, len(todo), bc.batch_size)
        for i in track(starts, description="labeling", total=len(starts)):
            batch = todo[i : i + bc.batch_size]
            for record in estimator.estimate_batch(batch):
                append_jsonl(out, record.to_dict())
    finally:
        estimator.teardown()

    sort_jsonl_by_id(out)
    records = [BudgetRecord.from_dict(d) for d in read_jsonl(out)]
    stats = describe(records)
    meta = {
        "artifact": "budgets",
        "produced_by": f"frugalprover.oracle.budget:{type(estimator).__name__}",
        "config": bc.__dict__,
        "n_records": len(records),
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
    return records


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
