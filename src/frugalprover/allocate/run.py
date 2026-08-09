"""The allocation runner: drive one agent under several budget policies.

This is the H2 experiment (docs/RESEARCH_PLAN.md, Phase 2). Every arm solves the
*same* problems with the *same* agent under the *same* total token budget; the
only thing that differs is how that budget is split. Whatever solves more, wins.

Three details are what make the comparison hold up, and all three are here
rather than in the policies:

**Allocation is global, execution is chunked.** Caps for every arm are computed
once over the whole eval set, because a policy that could only see 25 problems
at a time would be solving a different (easier) problem than the one H2 asks
about. Execution then walks the set in chunks, running every arm on a chunk
before moving on -- so a run cut short by its time budget leaves all arms with
the same prefix, and the paired comparison survives.

**The agent is not told which arm it is in.** It receives a list of caps. That
is the whole integration surface: the oracle's opinion reaches the agent as
`max_new_tokens`, and nothing else changes.

**Resume is per (arm, problem).** The same append-and-flush design as Stage 2,
for the same reason -- this stage costs GPU hours on a rented clock.
"""
from __future__ import annotations

import random
import time

from frugalprover.allocate import metrics
from frugalprover.allocate.policies import (
    NEEDS_LABELS,
    NEEDS_ORACLE,
    Candidate,
    allocate,
    build_policy,
)
from frugalprover.common.config import AllocateConfig, PipelineConfig
from frugalprover.common.deadline import ema, should_stop
from frugalprover.common.grading import extract_answer, grade, surface_features
from frugalprover.common.io import (
    append_jsonl,
    existing_ids,
    read_jsonl,
    sort_jsonl_by_id,
    write_json,
    write_meta,
)
from frugalprover.common.logging import get_logger, track
from frugalprover.common.records import ProblemRecord

log = get_logger(__name__)

#: Row key for resume. One row per (arm, problem), so `id` alone would collide.
def _key(arm: str, problem_id: str) -> str:
    return f"{arm}::{problem_id}"


def _load_problems(cfg: PipelineConfig) -> list[ProblemRecord]:
    ac = cfg.allocate
    path = cfg.data_path(ac.problems)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run `frugalprover sample` first, or point "
            f"allocate.problems at an existing problems file."
        )
    problems = [ProblemRecord.from_dict(d) for d in read_jsonl(path)]

    if ac.exclude_labeled:
        # The oracle was fitted on the labeled problems, so scoring it on them
        # would measure memorisation. Excluding them here (rather than trusting
        # whoever prepared the file) is the difference between an out-of-sample
        # result and a meaningless one.
        labeled = existing_ids(cfg.data_path(ac.exclude_labeled))
        before = len(problems)
        problems = [p for p in problems if p.id not in labeled]
        log.info("excluded %d problem(s) already labeled in %s -- the eval set is "
                 "out-of-sample for the oracle", before - len(problems), ac.exclude_labeled)

    if ac.shuffle:
        random.Random(ac.seed).shuffle(problems)
        log.info("shuffled %d problems with allocate.seed=%d", len(problems), ac.seed)
    if ac.max_problems is not None:
        problems = problems[: ac.max_problems]
    return problems


def _candidates(cfg: PipelineConfig, problems: list[ProblemRecord]) -> list[Candidate]:
    """Problem records plus whatever the requested policies need to allocate."""
    ac = cfg.allocate
    cands = [
        Candidate(id=p.id, char_len=surface_features(p.problem)["char_len"])
        for p in problems
    ]
    if not (set(ac.policies) & NEEDS_ORACLE):
        return cands

    from frugalprover.common.io import load_model
    from frugalprover.oracle.model.dataset import OracleDataset

    model_path = cfg.result_path(ac.oracle)
    if not model_path.exists():
        model_path = cfg.data_path(ac.oracle)
    if not model_path.exists():
        raise FileNotFoundError(
            f"allocate.oracle={ac.oracle!r} not found -- policies "
            f"{sorted(set(ac.policies) & NEEDS_ORACLE)} need a fitted oracle. "
            f"Run `frugalprover train` or point allocate.oracle at an oracle.joblib."
        )
    model = load_model(model_path)
    ds = OracleDataset.load(
        problems=str(cfg.data_path(ac.problems)),
        hidden_states=str(cfg.data_path(ac.hidden_states)) if ac.hidden_states else None,
    )
    ds = ds.subset([p.id for p in problems])
    if len(ds) != len(problems):
        missing = len(problems) - len(ds)
        raise ValueError(
            f"{missing} of {len(problems)} eval problems have no hidden states, so the "
            f"oracle cannot score them. Run `frugalprover extract` over "
            f"{ac.problems} first (same extract settings the oracle was fitted with, "
            f"or its saved PCA does not apply)."
        )

    preds = {r["id"]: r for r in model.predict_rows(ds)}
    log.info("oracle %s: predicted budgets for %d problem(s) (layer=%s, cv %s=%.4f)",
             model_path.name, len(preds), getattr(model, "layer", None),
             getattr(model, "cv_metric", "?"), getattr(model, "cv_score", float("nan")))
    for c in cands:
        row = preds.get(c.id)
        if row is None:
            raise ValueError(f"oracle produced no prediction for {c.id!r}")
        c.b_hat = row["b_hat"]
        c.p_by_budget = {int(b): float(v) for b, v in row["p_by_budget_pred"].items()}
    return cands


def _plan(cfg: PipelineConfig, cands: list[Candidate]) -> dict[str, list[int]]:
    """Every arm's caps, computed once over the whole eval set."""
    ac = cfg.allocate
    total = ac.b_bar * len(cands)
    log.info("allocating B_tot = %d x %d = %s tokens across %d arm(s): %s",
             ac.b_bar, len(cands), f"{total:,}", len(ac.policies), ", ".join(ac.policies))
    plan = {}
    for name in ac.policies:
        if name in NEEDS_LABELS:
            raise ValueError(
                f"policy {name!r} needs measured B* labels, which the eval problems do "
                f"not have. It is the offline ceiling -- run it through "
                f"`python -m frugalprover.analysis.allocation_sim` instead."
            )
        build_policy(name)  # fail on a typo'd arm before any GPU work
        caps = allocate(name, cands, ac.grid, total, ac.triage_threshold)
        plan[name] = caps
        by_cap = {b: caps.count(b) for b in sorted(set(caps))}
        log.info("  %-14s sum=%s (%.0f%% of B_tot)  caps=%s",
                 name, f"{sum(caps):,}", 100 * sum(caps) / total, by_cap)
    return plan


def run_allocate(cfg: PipelineConfig) -> dict:
    """Run every arm over the eval set and write the per-problem rows + report."""
    ac = cfg.allocate
    t0 = time.perf_counter()
    out = cfg.data_path(ac.output)

    problems = _load_problems(cfg)
    if not problems:
        raise ValueError("no problems left to allocate over")
    cands = _candidates(cfg, problems)
    plan = _plan(cfg, cands)
    caps_by_arm = {arm: dict(zip([p.id for p in problems], caps)) for arm, caps in plan.items()}

    done = existing_ids(out, key="key")
    if done:
        log.info("resuming: %d (arm, problem) row(s) already on disk", len(done))

    from frugalprover.agent import build_agent

    agent = build_agent(cfg.agent)
    agent.setup()
    if cfg.agent.seed_mode == "none":
        log.warning("agent.seed_mode='none': arms will draw independent completions, so "
                    "the arm-vs-arm difference carries sampling noise on top of the "
                    "budget effect. Set agent.seed_mode=per_problem to pair them.")

    n_rows = 0
    stopped_early = False
    chunk_s: float | None = None
    try:
        starts = list(range(0, len(problems), ac.chunk_size))
        for i in track(starts, description="allocating", total=len(starts)):
            if should_stop(ac.time_budget_s, t0, chunk_s):
                stopped_early = True
                break
            chunk = problems[i : i + ac.chunk_size]
            t_chunk = time.perf_counter()
            try:
                for arm in ac.policies:
                    n_rows += _run_arm(agent, arm, chunk, caps_by_arm[arm], done, out, ac)
            except Exception:
                if not ac.continue_on_error:
                    raise
                log.exception("chunk at offset %d failed -- skipping %d problem(s). "
                              "Rerun to retry them.", i, len(chunk))
                continue
            chunk_s = ema(chunk_s, time.perf_counter() - t_chunk)
    finally:
        agent.teardown()

    if not out.exists():
        log.warning("no chunk completed before the run stopped -- nothing written to %s", out)
        return {}

    sort_jsonl_by_id(out)
    rows = read_jsonl(out)
    report = metrics.summarize(rows, list(ac.policies), baseline=ac.baseline)
    report["elapsed_s"] = round(time.perf_counter() - t0, 1)
    report["stopped_early"] = stopped_early
    report["b_bar"] = ac.b_bar
    report["grid"] = list(ac.grid)
    report["triage_threshold"] = ac.triage_threshold
    report["agent_spec"] = agent.spec

    write_meta(out, {
        "artifact": "allocation",
        "produced_by": "frugalprover.allocate:run_allocate",
        "config": ac.__dict__,
        "n_records": len(rows),
        "n_rows_this_run": n_rows,
        "stopped_early": stopped_early,
        "agent_spec": agent.spec,
        **{k: report[k] for k in ("n_scored", "n_dropped_partial", "elapsed_s")},
    })
    summary_path = cfg.result_path(ac.report)
    write_json(summary_path, report)

    log.info("allocation report (%d problem(s) scored in every arm):\n%s",
             report["n_scored"], metrics.format_table(report))
    log.info("wrote %d row(s) -> %s and the report -> %s", len(rows), out, summary_path)
    if stopped_early:
        log.warning("stopped early: rerun the same command to continue where this left off")
    return report


def _run_arm(
    agent,
    arm: str,
    chunk: list[ProblemRecord],
    caps: dict[str, int],
    done: set[str],
    out,
    ac: AllocateConfig,
) -> int:
    """One arm's pass over one chunk. Returns how many rows it wrote."""
    todo, todo_caps = [], []
    written = 0
    for p in chunk:
        if _key(arm, p.id) in done:
            continue
        cap = caps[p.id]
        if cap <= 0:
            # Abstention: the policy declined this problem. It still gets a row --
            # a skipped problem is an unsolved problem that cost nothing, and
            # leaving it out would quietly raise the arm's solve rate.
            append_jsonl(out, {
                "key": _key(arm, p.id), "id": p.id, "arm": arm, "cap": 0, "tokens": 0,
                "solved": False, "accepted": None, "status": "skipped", "rounds": 0,
                "answer": None, "gold": p.answer, "candidate": "",
            })
            written += 1
            continue
        todo.append(p)
        todo_caps.append(cap)

    if not todo:
        return written

    log.info("[%s] chunk of %d problem(s), caps %s", arm, len(todo),
             {b: todo_caps.count(b) for b in sorted(set(todo_caps))})
    samples = agent.solve_batch(todo, max_new_tokens=todo_caps, n_samples=ac.n_samples)
    traces = getattr(agent, "last_traces", [[] for _ in todo])

    for p, cap, row_samples, row_traces in zip(todo, todo_caps, samples, traces):
        tokens = sum(s.tokens for s in row_samples)
        # `solved` is any-of-n correct, matching how a caller would use the agent;
        # at the n_samples=1 this experiment runs at, it is just pass@1.
        solved = any(grade(s.text, p.answer) for s in row_samples)
        best = row_samples[0]
        trace = row_traces[0] if row_traces else {}
        append_jsonl(out, {
            "key": _key(arm, p.id), "id": p.id, "arm": arm, "cap": cap, "tokens": tokens,
            "solved": bool(solved),
            "accepted": trace.get("accepted"),
            "status": trace.get("status", "unknown"),
            "rounds": trace.get("rounds", 0),
            "answer": extract_answer(best.text),
            "gold": p.answer,
            "candidate": best.text,
        })
        written += 1
    return written
