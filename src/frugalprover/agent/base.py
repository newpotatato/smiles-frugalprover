"""The solving agent interface.

The contract is deliberately narrow: given problems and a token budget, return
one completion per sample -- the text plus what it cost. The agent does not
grade, does not know what a budget sweep is, and does not decide when to stop.
Keeping it that dumb is what lets Stage 2 sweep the budget over any agent -- a
single model call, or a propose-verify-repair loop -- without either side
knowing about the other.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from frugalprover.common.records import ProblemRecord


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


@runtime_checkable
class SolverAgent(Protocol):
    """Attempts to solve problems under a token budget."""

    def setup(self) -> None:
        """Load models. Called once before the first batch."""

    def solve_batch(
        self,
        problems: list[ProblemRecord],
        max_new_tokens: int,
        n_samples: int,
    ) -> list[list[Sample]]:
        """`n_samples` samples for each problem, in input order.

        Each :class:`Sample` carries the generated text and the tokens spent
        producing it.

        `max_new_tokens` is the budget being measured. An agent that internally
        makes several model calls must account for its *total* generated tokens
        against this cap, otherwise the budget axis measures nothing.
        """
        ...

    def teardown(self) -> None:
        """Free GPU memory."""

    @property
    def spec(self) -> dict:
        """Model name and decoding params -- recorded in the A2 sidecar."""
        ...
