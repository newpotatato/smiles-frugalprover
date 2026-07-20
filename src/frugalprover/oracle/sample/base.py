"""Stage 1 interface: produce problems.

To add a new source (GSM8K, a Lean corpus, a hand-curated file), implement
this and register it — or skip the library entirely and write the A1 columns
yourself. Nothing downstream imports the sampler.
"""
from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

from frugalprover.common.records import ProblemRecord


@runtime_checkable
class DatasetSampler(Protocol):
    """Yields :class:`ProblemRecord`s. Must be deterministic given its config —
    reruns should produce identical ids so downstream artifacts stay joinable."""

    def sample(self) -> Iterator[ProblemRecord]: ...
