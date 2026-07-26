"""The solving agent whose effort the oracle predicts.

`SolverAgent` (base.py) is the base class; concrete realizations live beside it:

- ``single``        -- one prover call (SingleCallAgent), the baseline.
- ``verify_repair`` -- prover -> k verifiers -> corrector, looped to consensus.

Select one with :func:`build_agent`; each role picks its own model via config
(:class:`frugalprover.common.config.AgentConfig`). Nothing here imports from
`oracle/` -- the dependency runs one way (see README.md).
"""
from __future__ import annotations

from frugalprover.agent.base import SolverAgent

__all__ = ["SolverAgent", "build_agent", "AGENTS"]

#: type name -> "module:class", imported lazily so `import frugalprover.agent`
#: never drags in a model backend.
AGENTS: dict[str, str] = {
    "single": "frugalprover.agent.verify_repair:SingleCallAgent",
    "verify_repair": "frugalprover.agent.verify_repair:VerifyRepairAgent",
}


def build_agent(cfg, **kwargs) -> SolverAgent:
    """Construct the agent named by ``cfg.type``.

    `cfg` is an :class:`AgentConfig` (or anything with a ``.type`` and the role
    fields). Extra kwargs (e.g. ``client_factory``) pass through to the agent.
    """
    import importlib

    try:
        target = AGENTS[cfg.type]
    except KeyError:
        raise ValueError(
            f"unknown agent type {cfg.type!r}. Available: {sorted(AGENTS)}"
        ) from None
    module_name, _, class_name = target.partition(":")
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(cfg, **kwargs)
