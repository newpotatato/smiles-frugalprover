"""The records that travel between pipeline stages.

These are plain dataclasses with `to_dict` / `from_dict`. They are a
*convenience*, not a gate: every stage reads and writes plain JSON dicts, so a
file produced by some other script — a colleague's export, a notebook, a
different dataset entirely — works as long as the documented columns are there.
Unknown fields survive a round-trip in `extra` rather than being dropped.

Full column documentation lives in docs/ARTIFACTS.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _split_known(d: dict[str, Any], known: set[str]) -> tuple[dict, dict]:
    core = {k: v for k, v in d.items() if k in known}
    extra = {k: v for k, v in d.items() if k not in known}
    return core, extra


@dataclass
class ProblemRecord:
    """One problem. Stage 1 output (A1), input to Stages 2 and 3.

    `id` is derived from the source dataset — "{subject}/{split}/{row_index}",
    e.g. "algebra/test/1423". It is the same string on any machine for anyone
    who samples the same source, which is what makes independently-produced
    files joinable without coordinating.

    `level` is the human difficulty annotation. It is kept for stratifying and
    for plots, and is deliberately NOT a model feature: a problem in the wild
    has no human difficulty label, so training on it would inflate scores and
    hide whether activations carry signal.
    """

    id: str
    problem: str
    answer: str
    type: str
    level: int | None = None
    solution: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    _KNOWN = {"id", "problem", "answer", "type", "level", "solution"}

    def to_dict(self) -> dict[str, Any]:
        d = {
            "id": self.id,
            "problem": self.problem,
            "answer": self.answer,
            "type": self.type,
            "level": self.level,
            "solution": self.solution,
        }
        d.update(self.extra)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProblemRecord:
        core, extra = _split_known(d, cls._KNOWN)
        return cls(**core, extra=extra)


@dataclass
class BudgetRecord:
    """Budget-sweep outcome for one problem. Stage 2 output (A2).

    A full sweep and a single fixed-budget pass share this shape — a single
    pass is just a sweep of length one:

        full sweep:  budgets=[128, 256, 512], p={"128": 0.0, ..., "512": 1.0}
        single pass: budgets=[256],           p={"256": 0.67}

    Dict keys are budget values as *strings* throughout. (The old code read
    `int(b)` in one place and `str(b)` in another off the same field.)

    `b_star` is the smallest budget whose success rate reaches
    `success_threshold`, or None when no swept budget did — "censored", meaning
    the true value is above the largest budget tried, not that it's missing.
    """

    id: str
    agent: str
    budgets: list[int]
    n_samples: int
    success_threshold: float
    n_success: dict[str, int]
    p: dict[str, float]
    p_ci: dict[str, list[float]] = field(default_factory=dict)
    b_star: int | None = None
    sc: dict[str, float] = field(default_factory=dict)
    tokens_spent: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    _KNOWN = {
        "id", "agent", "budgets", "n_samples", "success_threshold",
        "n_success", "p", "p_ci", "b_star", "sc", "tokens_spent",
    }

    @property
    def censored(self) -> bool:
        return self.b_star is None

    @property
    def single_pass(self) -> bool:
        """True when only one budget was tried, so `b_star` is a solved/unsolved
        label rather than a real effort estimate. Regression can't use these."""
        return len(self.budgets) == 1

    def solved_at(self, budget: int) -> bool:
        """Did this problem clear the threshold at `budget`?

        Monotone in budget: a problem solved at 128 counts as solved at 512 even
        if 512 wasn't swept, because more budget can't hurt.
        """
        return self.b_star is not None and budget >= self.b_star

    def to_dict(self) -> dict[str, Any]:
        d = {
            "id": self.id,
            "agent": self.agent,
            "budgets": list(self.budgets),
            "n_samples": self.n_samples,
            "success_threshold": self.success_threshold,
            "n_success": self.n_success,
            "p": self.p,
            "p_ci": self.p_ci,
            "b_star": self.b_star,
            "sc": self.sc,
            "tokens_spent": self.tokens_spent,
        }
        d.update(self.extra)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BudgetRecord:
        core, extra = _split_known(d, cls._KNOWN)
        return cls(**core, extra=extra)

    @classmethod
    def from_counts(
        cls,
        problem_id: str,
        agent: str,
        budgets: list[int],
        n_samples: int,
        n_success: dict[int, int],
        success_threshold: float,
        sc: dict[int, float] | None = None,
        tokens_spent: int | None = None,
    ) -> BudgetRecord:
        """Build a record from raw success counts — derives p, CIs and b_star.

        This is what an estimator implementation should call, so the b_star
        rule lives in exactly one place.
        """
        from frugalprover.common.grading import wilson_ci

        budgets = sorted(budgets)
        p = {str(b): n_success[b] / n_samples if n_samples else 0.0 for b in budgets}
        p_ci = {str(b): list(wilson_ci(n_success[b], n_samples)) for b in budgets}
        cleared = [b for b in budgets if p[str(b)] >= success_threshold]
        return cls(
            id=problem_id,
            agent=agent,
            budgets=budgets,
            n_samples=n_samples,
            success_threshold=success_threshold,
            n_success={str(b): n_success[b] for b in budgets},
            p=p,
            p_ci=p_ci,
            b_star=min(cleared) if cleared else None,
            sc={str(b): v for b, v in (sc or {}).items()},
            tokens_spent=tokens_spent,
        )
