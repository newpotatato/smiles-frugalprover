"""Budget allocation: spending a fixed total across problems (H2).

The Budget Oracle (`oracle/`) predicts how much effort a problem needs; the
solving agent (`agent/`) spends effort. This package is the join between them --
the only place that imports both -- and it exists because neither should know
about the other: the oracle never learns what a prover is, and the agent only
ever sees a number of tokens.

    oracle -> p_hat_i(B) --> [ policy ] --> caps --> agent

`policies.py` holds the allocators, and is pure so the offline simulation and
the live run share one implementation. `run.py` drives the arms. `metrics.py`
scores them.
"""
from frugalprover.allocate.policies import (
    NEEDS_LABELS,
    NEEDS_ORACLE,
    POLICIES,
    Candidate,
    allocate,
    build_policy,
)

__all__ = [
    "POLICIES", "NEEDS_ORACLE", "NEEDS_LABELS",
    "Candidate", "allocate", "build_policy", "run_allocate",
]


def run_allocate(cfg):
    """Run the allocation experiment. Imported lazily -- it pulls in the agent."""
    from frugalprover.allocate.run import run_allocate as _run

    return _run(cfg)
