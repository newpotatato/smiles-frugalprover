"""The solving agent interface.

Nothing implements this yet -- see agent/README.md for why it exists now.

The contract is deliberately narrow: given problems and a token budget, return
raw completions. The agent does not grade, does not know what a budget sweep
is, and does not decide when to stop. Keeping it that dumb is what lets Stage 2
sweep the budget over any agent -- a single model call today, a
propose-verify-repair loop later -- without either side knowing about the other.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from frugalprover.common.records import ProblemRecord


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
    ) -> list[list[str]]:
        """`n_samples` raw completions for each problem, in input order.

        Returns generated text only -- not the prompt. Leaving the prompt in
        means that if the gold answer appears there, grading will count it as
        the agent's own output and every problem will look solved.

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
