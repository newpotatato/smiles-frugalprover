"""How the k verifiers' verdicts combine into an accept/reject decision.

Pure functions, no model calls -- this is the "central safety metric" dial the
architecture turns on, so it lives in one small, unit-tested place.

- unanimity: every verifier must accept. Fewer wrongly-accepted proofs, more
  rounds and more rejections.
- majority: strictly more than half accept. Fewer rejections, higher false
  accepts.
"""
from __future__ import annotations

from frugalprover.agent.roles import Critique


def aggregate(critiques: list[Critique], rule: str) -> bool:
    """True if the ensemble accepts under `rule` ("unanimity" | "majority")."""
    if not critiques:
        raise ValueError("aggregate() needs at least one critique")
    accepts = sum(1 for c in critiques if c.accept)
    if rule == "unanimity":
        return accepts == len(critiques)
    if rule == "majority":
        return accepts * 2 > len(critiques)
    raise ValueError(f"unknown aggregation rule {rule!r}; use 'unanimity' or 'majority'")


def collect_flaws(critiques: list[Critique]) -> list[str]:
    """De-duplicated flaws from the rejecting verifiers, preserving first-seen order.

    This is exactly what the corrector is asked to fix. Accepting verifiers
    contribute nothing -- their silence isn't a flaw.
    """
    seen: dict[str, None] = {}
    for c in critiques:
        if c.accept:
            continue
        for flaw in c.flaws:
            seen.setdefault(flaw, None)
    return list(seen)
