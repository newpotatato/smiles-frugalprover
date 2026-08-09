"""The solving agent interface.

The contract is deliberately narrow: given problems and a token budget, return
one completion per sample -- the text plus what it cost. The agent does not
grade, does not know what a budget sweep is, and does not decide when to stop.
Keeping it that dumb is what lets Stage 2 sweep the budget over any agent -- a
single model call, or a propose-verify-repair loop -- without either side
knowing about the other.

The budget may be *one cap for the whole batch* (what Stage 2 sweeps) or *one
cap per problem* (what an allocation policy hands down -- see allocate/). Both
go through the same argument, because to the agent they are the same thing: a
ceiling on what an attempt may generate.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from frugalprover.common.records import ProblemRecord


def stable_seed(problem_id: str, sample_index: int = 0) -> int:
    """A decoding seed derived from the problem id, stable across processes.

    Two runs of the same problem under *different budgets* then sample the same
    trajectory until the smaller cap truncates it, which is what makes an
    allocation A/B a paired comparison instead of two independent draws. Python's
    builtin `hash` is salted per process and would silently break that across
    runs; crc32 is not.

    `sample_index` is added in so `n_samples > 1` still draws distinct
    completions -- without it every sample of a problem would be identical and
    the self-consistency baseline would collapse to a point mass.
    """
    return (zlib.crc32(problem_id.encode("utf-8")) + sample_index) % (2 ** 31 - 1)


@dataclass
class Sample:
    """One completion returned by an agent, paired with its cost.

    `text` is the generated solution (never the prompt -- if the gold answer
    leaks in from the prompt, grading counts it as the agent's own output and
    every problem looks solved). `tokens` is the *total* the agent generated to
    produce it, summed across every internal model call (for a verify-repair
    loop that is prover + verifiers + corrector over all rounds), not just the
    length of `text`.

    Cost rides back with the text because Stage 2 needs both -- the text to grade
    and the tokens for the A2 `tokens_spent` field -- and returning them together
    keeps that on the protocol instead of some concrete agent's private state
    (which a caller couldn't read without reaching past the interface, and which
    a second `solve_batch` call would overwrite).
    """

    text: str
    tokens: int


@dataclass
class RoundTrace:
    """One verify(+repair) round of an attempt, as it actually happened.

    `corrector_tokens is None` means no repair ran after this round -- the round
    cap was reached, or the attempt was already over budget. Either way it cannot
    run at a *smaller* budget either, so the absence needs no further flag.
    """

    verifier_tokens: int
    accepted: bool
    corrector_tokens: int | None = None
    candidate_after: str | None = None


@dataclass
class AttemptTrace:
    """One attempt's trajectory, replayable at any budget it ran at or below.

    The premise is that a smaller budget yields a *prefix* of a larger one: the
    same calls, in the same order, stopping sooner. That holds only while the
    per-call caps don't change with the budget, so the caller owns that guard
    (see oracle/budget/sweep.py). Given it, one expensive pass at the largest
    budget answers the whole sweep, instead of re-solving from scratch per budget.
    """

    prover_tokens: int
    candidate_0: str
    rounds: list[RoundTrace] = field(default_factory=list)

    def replay(self, cap: int | None) -> Sample:
        """The Sample this attempt would have produced under a cap of `cap`.

        Mirrors the stop conditions in verify_repair.py exactly: finalize before
        an audit once the running total reaches the cap, stop on acceptance, and
        stop when no repair followed.
        """
        tokens, candidate = self.prover_tokens, self.candidate_0
        for r in self.rounds:
            if cap is not None and tokens >= cap:
                break                       # spent its budget before this audit
            tokens += r.verifier_tokens
            if r.accepted:
                break
            if r.corrector_tokens is None:
                break                       # no repair followed at any budget
            if cap is not None and tokens >= cap:
                break                       # audit consumed the rest of the cap
            tokens += r.corrector_tokens
            candidate = r.candidate_after
        return Sample(text=candidate, tokens=tokens)


@runtime_checkable
class TracingAgent(Protocol):
    """Optional capability: solve once, return replayable trajectories.

    An agent implements this when its behaviour under a smaller token cap is a
    prefix of its behaviour under a larger one. Stage 2 uses it to reconstruct a
    whole budget sweep from a single pass; agents that don't implement it are
    swept the ordinary way, one full pass per budget.
    """

    def solve_batch_traced(
        self,
        problems: list[ProblemRecord],
        max_new_tokens: int | list[int],
        n_samples: int,
    ) -> list[list[AttemptTrace]]:
        """Like `solve_batch`, but returns traces instead of finished Samples."""
        ...


@runtime_checkable
class SolverAgent(Protocol):
    """Attempts to solve problems under a token budget."""

    def setup(self) -> None:
        """Load models. Called once before the first batch."""

    def solve_batch(
        self,
        problems: list[ProblemRecord],
        max_new_tokens: int | list[int],
        n_samples: int,
    ) -> list[list[Sample]]:
        """`n_samples` samples for each problem, in input order.

        Each :class:`Sample` carries the generated text and the tokens spent
        producing it.

        `max_new_tokens` is the budget being measured. An agent that internally
        makes several model calls must account for its *total* generated tokens
        against this cap, otherwise the budget axis measures nothing.

        It is either an int applying to every problem (0 = uncapped) or a list
        of one cap per problem, parallel to `problems` -- the per-problem form
        is how an allocation policy spends a fixed total budget unevenly. A
        non-positive entry in the list means that problem is uncapped.
        """
        ...

    def teardown(self) -> None:
        """Free GPU memory."""

    @property
    def spec(self) -> dict:
        """Model name and decoding params -- recorded in the A2 sidecar."""
        ...
