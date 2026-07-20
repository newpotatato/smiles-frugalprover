"""Stage 2 interface: measure how much effort each problem costs.

The interface is batched because on a GPU that is where all the throughput is —
generating 3 completions for 6 problems at once is not 18x the cost of doing one.
Baking batching in now means a real implementation doesn't have to change shape
later.

`setup` / `teardown` exist so model loading happens when the stage actually
runs, not at construction. That keeps `--dry-run`, `--help`, and config
validation from pulling several GB of weights onto the GPU.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from frugalprover.common.records import BudgetRecord, ProblemRecord


@runtime_checkable
class BudgetEstimator(Protocol):
    """Maps problems to budget labels (A2 records)."""

    def setup(self) -> None:
        """Load models and tokenizers here. Called once before the first batch."""

    def estimate_batch(self, problems: list[ProblemRecord]) -> list[BudgetRecord]:
        """One record per input problem, in the same order."""
        ...

    def teardown(self) -> None:
        """Free GPU memory. Called once after the last batch, even on error."""
