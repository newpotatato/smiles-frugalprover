"""Allocation policies: a fixed total budget, split across problems.

This is the H2 machinery (docs/RESEARCH_PLAN.md, Phase 2). Given a total token
budget `B_tot` for a set of problems, a policy decides each problem's share:

    maximize  sum_i p_i(B_i)   subject to  sum_i B_i <= B_tot

**Everything here is pure.** A policy takes candidate records and returns a list
of integer caps -- no agent, no model, no I/O. That is deliberate: the offline
simulation (`analysis/allocation_sim.py`) and the live run (`allocate/run.py`)
must provably run *the same allocator*, or the simulation is not evidence about
the run.

Every cap lands on the budget grid the labels were measured at. Off-grid caps
would be cheap to produce and impossible to compare: the oracle's success curve
is only defined at the grid points, so a cap of 2600 would be scored against
`p(2048)` while costing 2600 tokens. A cap of 0 means *skip this problem* -- it
costs nothing and counts as unsolved, which is what abstention is.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from frugalprover.common.logging import get_logger

log = get_logger(__name__)


@dataclass
class Candidate:
    """One problem, as an allocator sees it.

    `p_by_budget` is a success curve over the grid -- *predicted* by the oracle
    for the `oracle_*` policies, *measured* from the labels for the `true_bstar`
    ceiling. The policies cannot tell the difference, which is the point: the
    ceiling is the same rule run with perfect information, so the gap between
    them is exactly the cost of prediction error.
    """

    id: str
    char_len: float = 0.0
    p_by_budget: dict[int, float] = field(default_factory=dict)
    #: Smallest grid budget the oracle expects to clear tau; None = none of them.
    b_hat: int | None = None

    def p(self, budget: int) -> float:
        """Success probability at `budget`; 0.0 for a skipped problem."""
        if budget <= 0:
            return 0.0
        return float(self.p_by_budget.get(budget, 0.0))


# --------------------------------------------------------------- grid helpers

def _next_step(cap: int, grid: list[int]) -> int | None:
    """The next grid value above `cap`, or None at the top."""
    for b in grid:
        if b > cap:
            return b
    return None


def _floor_to_grid(value: float, grid: list[int]) -> int:
    """Largest grid value <= `value`, or the smallest grid value."""
    below = [b for b in grid if b <= value]
    return below[-1] if below else grid[0]


def _upgrade_by(
    caps: list[int],
    grid: list[int],
    total: int,
    gain: Callable[[int, int, int], float],
) -> list[int]:
    """Spend what's left of `total` one grid step at a time, best gain first.

    `gain(i, current_cap, next_cap)` scores a candidate upgrade; the highest
    scoring affordable step is taken, repeatedly, until nothing fits. This is the
    knapsack/marginal-gain rule from RESEARCH_PLAN Phase 2, and it is shared by
    every policy that has any budget left to distribute -- the policies differ
    only in what they score.

    Greedy is optimal here up to one step: the steps are small relative to
    `total` and the measured curves are concave in practice, so the usual
    fractional-knapsack argument applies. Where they are not concave, greedy is
    still a policy -- it is just no longer provably the best one, which is worth
    knowing but does not affect a matched-compute comparison in which every arm
    is scored on what it actually spent.
    """
    spent = sum(caps)
    while True:
        best_i, best_next, best_gain = -1, 0, 0.0
        for i, cap in enumerate(caps):
            if cap <= 0:
                continue  # skipped by the policy; not a candidate for upgrades
            nxt = _next_step(cap, grid)
            if nxt is None or spent + (nxt - cap) > total:
                continue
            g = gain(i, cap, nxt)
            if g > best_gain:
                best_i, best_next, best_gain = i, nxt, g
        if best_i < 0:
            return caps
        spent += best_next - caps[best_i]
        caps[best_i] = best_next


def _seed_floor(priority: list[float], grid: list[int], total: int, skip: list[bool]) -> list[int]:
    """Give the smallest grid budget to as many problems as `total` affords.

    The grid has a floor: below `grid[0]` there is no cap a problem can be given
    except zero. So when `B_tot` cannot seat everyone at `grid[0]` -- which is
    what a total below `n * grid[0]` means -- somebody gets nothing, and the
    policy has to say who. Highest priority first, ties by index.

    Only binds at the very bottom of the compute sweep. At the operating points
    an experiment actually runs (`b_bar` on the grid, so `b_bar >= grid[0]`)
    every problem is seated and this is just `[grid[0]] * n`.
    """
    live = [i for i in range(len(priority)) if not skip[i]]
    seats = total // grid[0]
    caps = [0] * len(priority)
    if seats >= len(live):
        for i in live:
            caps[i] = grid[0]
        return caps
    log.warning("B_tot=%d seats only %d of %d problems at the smallest budget %d -- "
                "the rest get nothing", total, seats, len(live), grid[0])
    for i in sorted(live, key=lambda i: (-priority[i], i))[:seats]:
        caps[i] = grid[0]
    return caps


def _apportion(weights: list[float], grid: list[int], total: int, skip: list[bool]) -> list[int]:
    """Caps proportional to `weights`, snapped down to the grid, remainder
    redistributed largest-remainder first.

    Used by the policies that name a *target* per problem (a length, a B-hat)
    rather than a marginal gain. Proportional-then-floor always undershoots
    `total`; handing the slack to the largest fractional remainders is the
    standard apportionment fix and keeps the arm's spend comparable to uniform's.
    """
    live = [i for i in range(len(weights)) if not skip[i]]
    if not live:
        return [0] * len(weights)
    mass = sum(max(0.0, weights[i]) for i in live) or float(len(live))
    raw = {i: total * max(0.0, weights[i]) / mass for i in live}
    # Seat everyone at the floor first, then buy each problem up towards its
    # proportional share. Flooring `raw` directly would overspend whenever a
    # share fell below `grid[0]`, since there is no cap smaller than that to
    # round down to.
    caps = _seed_floor([raw.get(i, 0.0) for i in range(len(weights))], grid, total, skip)
    return _upgrade_by(caps, grid, total, lambda i, cap, nxt: raw[i] - cap)


# -------------------------------------------------------------------- policies

def uniform(cands: list[Candidate], grid: list[int], total: int, **_) -> list[int]:
    """Every problem gets the same cap. The floor every other arm must clear."""
    n = len(cands)
    if n == 0:
        return []
    base = _floor_to_grid(total / n, grid)
    caps = _seed_floor([0.0] * n, grid, total, [False] * n)
    if all(c > 0 for c in caps):
        caps = [base] * n
    # Rounding the even share down to the grid leaves a remainder, and an even
    # split cannot say who gets it. One step each to a *randomly* chosen subset,
    # rather than to the lowest indices: problems.jsonl is sorted by id, which
    # groups it by subject, so an index-ordered tiebreak would hand every spare
    # token to the alphabetically-early subjects -- and subject predicts
    # difficulty. That would hobble the baseline the oracle is measured against,
    # in a way that looks like an oracle result.
    #
    # Seeded, so an arm is reproducible. Exactly zero of this happens when
    # `b_bar` is on the grid, which is the configuration to prefer for a headline
    # comparison.
    import random

    jitter = random.Random(0)
    priority = [jitter.random() for _ in range(n)]
    return _upgrade_by(
        caps, grid, total, lambda i, cap, nxt: priority[i] if cap == base else 0.0
    )


def length(cands: list[Candidate], grid: list[int], total: int, **_) -> list[int]:
    """Budget proportional to problem length -- the deconfounder.

    Longer problems plausibly need more tokens for reasons that have nothing to
    do with reasoning difficulty. If the oracle cannot beat this, it has not
    shown anything (README, "Methodology notes").
    """
    return _apportion([c.char_len for c in cands], grid, total, [False] * len(cands))


def oracle_bhat(cands: list[Candidate], grid: list[int], total: int, **_) -> list[int]:
    """Budget proportional to the oracle's predicted B-hat.

    The most direct reading of "the oracle says this problem needs B": a problem
    the oracle expects to need 8192 gets eight times the share of one it expects
    to finish in 1024. Problems the oracle expects to clear at no grid budget ask
    for the top of the grid -- they are the hard ones, not the free ones. (That
    is the opposite of what `oracle_triage` does with them, which is why both
    arms are worth running.)
    """
    top = grid[-1]
    return _apportion(
        [float(c.b_hat if c.b_hat is not None else top) for c in cands],
        grid, total, [False] * len(cands),
    )


def oracle_greedy(cands: list[Candidate], grid: list[int], total: int, **_) -> list[int]:
    """Marginal-gain allocation: buy the steepest slice of success curve first.

    RESEARCH_PLAN Phase 2's stated rule. Unlike `oracle_bhat` this uses the whole
    predicted curve, so it will refuse to upgrade a problem whose predicted
    success barely moves and spend those tokens on one where it moves a lot.
    """
    top = grid[-1]
    caps = _seed_floor([c.p(top) for c in cands], grid, total, [False] * len(cands))
    return _upgrade_by(
        caps, grid, total,
        lambda i, cap, nxt: (cands[i].p(nxt) - cands[i].p(cap)) / (nxt - cap),
    )


def oracle_optimal(cands: list[Candidate], grid: list[int], total: int, **_) -> list[int]:
    """Exactly maximize `sum_i p_i(B_i)` subject to `sum_i B_i <= B_tot`.

    The README's objective, solved rather than approximated. `oracle_greedy`
    approximates it with the marginal-gain rule, which is optimal only while the
    success curves are concave -- and measured curves at n=3 are not: sampling
    noise makes `p(4096) < p(2048)` for some problems, and greedy then stops
    early on a zero or negative step while a better allocation exists further
    on. That matters most for the ceiling arm, where "the best any oracle could
    do" has to actually be the best.

    Every budget is a multiple of `grid[0]`, so this is a multiple-choice
    knapsack on a small integer capacity and an exact DP fits comfortably: at
    276 problems, five choices and ~2200 units of capacity it is a couple of
    million table updates. Skipping (`B_i = 0`, value 0) is one of the choices,
    so this triages on its own where that is genuinely worth it.
    """
    import numpy as np

    n = len(cands)
    if n == 0:
        return []
    unit = grid[0]
    capacity = total // unit
    choices = [0] + [b // unit for b in grid]          # in units, 0 = skip
    values = np.array(
        [[0.0] + [c.p(b) for b in grid] for c in cands], dtype=float
    )

    NEG = -np.inf
    best = np.full(capacity + 1, NEG)
    best[0] = 0.0
    # `taken[i, u]` is which choice problem i made when the table held u units.
    taken = np.zeros((n, capacity + 1), dtype=np.int8)
    for i in range(n):
        nxt = np.full(capacity + 1, NEG)
        pick = np.zeros(capacity + 1, dtype=np.int8)
        for ci, cost in enumerate(choices):
            if cost > capacity:
                continue
            shifted = np.full(capacity + 1, NEG)
            shifted[cost:] = best[: capacity + 1 - cost] + values[i, ci]
            better = shifted > nxt
            nxt = np.where(better, shifted, nxt)
            pick = np.where(better, ci, pick)
        best, taken[i] = nxt, pick

    caps = [0] * n
    u = int(np.argmax(best))
    for i in range(n - 1, -1, -1):
        ci = int(taken[i, u])
        caps[i] = choices[ci] * unit
        u -= choices[ci]
    return caps


def oracle_triage(
    cands: list[Candidate], grid: list[int], total: int, triage_threshold: float = 0.3, **_
) -> list[int]:
    """Abstain on the hopeless, spend their tokens on the rest.

    A problem whose predicted success at the *largest* grid budget is below
    `triage_threshold` gets nothing at all. The claim being tested is that those
    tokens buy more elsewhere -- and the honest cost of the claim is every
    skipped problem the uniform arm went on to solve, which `metrics.py` reports.
    """
    top = grid[-1]
    skip = [c.p(top) < triage_threshold for c in cands]
    caps = _seed_floor([c.p(top) for c in cands], grid, total, skip)
    if all(skip):
        log.warning("oracle_triage skipped all %d problems at threshold %.2f -- "
                    "the arm will solve nothing; lower allocate.triage_threshold",
                    len(cands), triage_threshold)
        return caps
    return _upgrade_by(
        caps, grid, total,
        lambda i, cap, nxt: (cands[i].p(nxt) - cands[i].p(cap)) / (nxt - cap),
    )


def true_bstar(cands: list[Candidate], grid: list[int], total: int, **_) -> list[int]:
    """The ceiling: the optimal allocation, computed on *measured* curves.

    Offline only -- it needs labels, which the live eval problems by construction
    do not have. The gap between this and the `oracle_*` arms is the cost of
    prediction error (RESEARCH_PLAN Phase 2: "the gap is the cost of prediction
    error"); the gap between this and `uniform` is the most any oracle could ever
    buy on this data, and therefore whether a better oracle is worth building.

    It is a true upper bound because `oracle_optimal` solves the allocation
    exactly -- with a greedy approximation here a learned arm could score above
    the "ceiling", which is how the noise in n=3 curves shows up if you let it.
    """
    return oracle_optimal(cands, grid, total)


#: Register a policy here and it becomes available as an arm in
#: `allocate.policies`, with no other change. Mirrors agent/__init__.py:AGENTS
#: and oracle/budget:ESTIMATORS.
POLICIES: dict[str, Callable[..., list[int]]] = {
    "uniform": uniform,
    "length": length,
    "oracle_bhat": oracle_bhat,
    "oracle_greedy": oracle_greedy,
    "oracle_optimal": oracle_optimal,
    "oracle_triage": oracle_triage,
    "true_bstar": true_bstar,
}

#: Policies that need a fitted oracle's predictions to run.
NEEDS_ORACLE = {"oracle_bhat", "oracle_greedy", "oracle_optimal", "oracle_triage"}

#: Policies that need measured labels, so they can only run offline.
NEEDS_LABELS = {"true_bstar"}


def build_policy(name: str) -> Callable[..., list[int]]:
    try:
        return POLICIES[name]
    except KeyError:
        raise ValueError(
            f"unknown allocation policy {name!r}. Available: {sorted(POLICIES)}"
        ) from None


def allocate(
    name: str,
    cands: list[Candidate],
    grid: list[int],
    total: int,
    triage_threshold: float = 0.3,
) -> list[int]:
    """Run one named policy. The single entry point both callers use."""
    caps = build_policy(name)(
        cands, sorted(grid), total, triage_threshold=triage_threshold
    )
    if sum(caps) > total:
        raise AssertionError(
            f"policy {name!r} allocated {sum(caps)} tokens against a total of {total} -- "
            "a matched-compute comparison against an arm that overspends is not matched"
        )
    return caps
