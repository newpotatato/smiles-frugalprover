"""A fake budget estimator, so Stages 3-5 are runnable before Stage 2 exists.

The labels are invented. Nothing produced with this estimator is a research
result, and every artifact it writes says ``agent: "mock"`` so a run trained on
fake labels can't be mistaken for a real one.

Design of the fake: harder problems (higher `level`) need more budget, plus
seeded noise so it isn't perfectly predictable, plus some censored problems.
That shape matters -- a mock where every problem needs the same budget would
make the oracle look broken, and one with no noise would make it look perfect.
Neither would tell you whether the plumbing works.

Note the mock uses `level` to generate labels while the oracle is forbidden from
using it as a feature. That is the point: the oracle has to recover the signal
from activations, and on tiny-gpt2 with random-ish activations it mostly won't.
A low score on the smoke run is expected.
"""
from __future__ import annotations

import random

from frugalprover.common.config import BudgetConfig
from frugalprover.common.records import BudgetRecord, ProblemRecord


class MockEstimator:
    """Deterministic fake labels. No model, no GPU, instant."""

    def __init__(self, cfg: BudgetConfig, seed: int = 0):
        self.cfg = cfg
        self.seed = seed

    def setup(self) -> None:
        pass

    def teardown(self) -> None:
        pass

    def estimate_batch(self, problems: list[ProblemRecord]) -> list[BudgetRecord]:
        return [self._estimate_one(p) for p in problems]

    def _estimate_one(self, problem: ProblemRecord) -> BudgetRecord:
        cfg = self.cfg
        # seed off the problem id so a record is identical no matter which
        # batch it landed in or how many times the run was restarted
        rng = random.Random(f"{self.seed}:{problem.id}")

        level = problem.level if problem.level is not None else 3
        # difficulty in [0, 1]: level 1 -> 0.0, level 5 -> 1.0, plus noise
        difficulty = (level - 1) / 4 + rng.gauss(0, 0.18)
        difficulty = min(max(difficulty, 0.0), 1.15)  # >1 leaves room for censoring

        n_success = {}
        for i, budget in enumerate(cfg.budgets):
            # success probability rises with budget position, falls with difficulty
            ease = (i + 1) / len(cfg.budgets)
            p = min(max(ease - difficulty + 0.35, 0.0), 1.0)
            n_success[budget] = sum(rng.random() < p for _ in range(cfg.n_samples))

        # counts must be monotone in budget: more room to think can't lose you a
        # solve. Without this the sweep is noise and b_star is meaningless.
        running = 0
        for budget in cfg.budgets:
            running = max(running, n_success[budget])
            n_success[budget] = running

        return BudgetRecord.from_counts(
            problem_id=problem.id,
            agent="mock",
            budgets=list(cfg.budgets),
            n_samples=cfg.n_samples,
            n_success=n_success,
            success_threshold=cfg.success_threshold,
            tokens_spent=sum(cfg.budgets) * cfg.n_samples,
        )
