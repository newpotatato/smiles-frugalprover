"""Stage 3 interface: turn problems into fixed-size feature rows.

To swap in a different signal source -- another backbone, an embedding API, or a
random baseline for a null test -- implement this, or just write a parquet with
an `id` column and at least one `L<n>_<pooling>` column of equal-length arrays.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from frugalprover.common.records import ProblemRecord


@runtime_checkable
class HiddenStateExtractor(Protocol):
    """Produces one feature dict per problem: {column_name: vector_or_scalar}."""

    def setup(self) -> None:
        """Load the model. Called once before the first batch."""

    def extract_batch(self, problems: list[ProblemRecord]) -> list[dict[str, Any]]:
        """One dict per input problem, in the same order."""
        ...

    def teardown(self) -> None:
        """Free GPU memory."""

    @property
    def spec(self) -> dict:
        """Model name, hidden size, layer count, dtype -- recorded in the
        sidecar so a bare parquet of float arrays can say where it came from."""
        ...
