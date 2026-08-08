"""The verify-repair solving agent, and a single-call baseline.

`VerifyRepairAgent` is one realization of the `SolverAgent` protocol: a prover
proposes a candidate, k skeptical verifiers audit it, a corrector repairs what
they flag, and the loop iterates until the verifiers concur (by the configured
aggregation rule) or a round cap is hit. The solver is held untrusted until the
verifiers pass.

**Batched.** The loop advances every attempt (problem x sample) in lockstep and
batches each role's `generate` call across all still-active attempts, so a
batching backend processes the whole active set at once. Attempts drop out of
the active set as they are accepted, exhaust their rounds, or reach the token
cap; each remaining round runs on a smaller batch.

The one hard constraint from agent/README.md: whatever the loop does internally,
an attempt's *total generated tokens* must respect `max_new_tokens`, or the
budget axis Stage 2 measures becomes meaningless. When a positive
`max_new_tokens` is given, an attempt is finalized once its running token total
reaches the cap -- checked at each round boundary -- so it gets no further
generation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from frugalprover.agent.aggregation import aggregate, collect_flaws
from frugalprover.agent.base import Sample
from frugalprover.agent.model import ModelClient, build_model_client
from frugalprover.agent.roles import Corrector, Critique, Prover, Verifier
from frugalprover.common.config import AgentConfig, ModelSpec
from frugalprover.common.logging import get_logger
from frugalprover.common.records import ProblemRecord

log = get_logger(__name__)

#: A client factory: ModelSpec -> ModelClient. Swappable so tests can inject
#: scripted mocks without a config round-trip.
ClientFactory = Callable[[ModelSpec], ModelClient]


@dataclass
class _Attempt:
    """One problem-sample working its way through the loop."""

    prob_idx: int
    problem: str
    candidate: str = ""
    rounds: int = 0
    flaws: list[str] = field(default_factory=list)
    tokens: int = 0
    accepted: bool = False
    done: bool = False


class VerifyRepairAgent:
    """Prover -> k verifiers -> corrector, looped to consensus. A `SolverAgent`."""

    def __init__(self, cfg: AgentConfig, client_factory: ClientFactory = build_model_client):
        self.cfg = cfg
        self._factory = client_factory
        self.prover: Prover | None = None
        self.corrector: Corrector | None = None
        self.verifiers: list[Verifier] = []
        #: Per-attempt diagnostics, list[list[dict]] parallel to solve_batch's
        #: return (one inner list per problem, one dict per sample). Consumed by
        #: the `prove` CLI; not part of the SolverAgent contract.
        self.last_traces: list[list[dict]] = []

    # -- lifecycle ---------------------------------------------------------

    def setup(self) -> None:
        c = self.cfg
        self.prover = self._build(c.prover, c.prover_prompt, "prover")
        self.corrector = self._build(c.corrector, c.corrector_prompt, "corrector")
        self.verifiers = [
            self._build(spec, c.verifier_prompt, "verifier") for spec in c.verifiers
        ]

    def _build(self, spec: ModelSpec, prompt: str, role: str):
        client = self._factory(spec)
        client.setup()
        cls = {"prover": Prover, "corrector": Corrector, "verifier": Verifier}[role]
        return cls(client, spec, prompt, role)

    def teardown(self) -> None:
        for role in [self.prover, self.corrector, *self.verifiers]:
            if role is not None:
                role.client.teardown()

    # -- solving -----------------------------------------------------------

    def solve_batch(
        self,
        problems: list[ProblemRecord],
        max_new_tokens: int,
        n_samples: int,
    ) -> list[list[Sample]]:
        if self.prover is None:
            raise RuntimeError("call setup() before solve_batch()")
        cap = max_new_tokens if max_new_tokens and max_new_tokens > 0 else None

        # Flatten problems x samples into one batch of attempts, in an order that
        # regroups cleanly: all samples of problem 0, then problem 1, ...
        attempts = [
            _Attempt(prob_idx=i, problem=p.problem)
            for i, p in enumerate(problems)
            for _ in range(n_samples)
        ]

        # Prover: one batched call for the whole set.
        log.info("proposing %d attempts (%d problems x %d samples), cap=%s, max_rounds=%d",
                 len(attempts), len(problems), n_samples,
                 cap if cap is not None else "none", self.cfg.max_rounds)
        for a, text in zip(attempts, self.prover.propose([a.problem for a in attempts])):
            a.candidate = text
            a.tokens += self.prover.count_tokens(text)

        for round_i in range(self.cfg.max_rounds):
            active = self._active_under_budget(attempts, cap)
            if not active:
                break

            # Audit: each verifier runs once across all active attempts.
            accepted_this_round = 0
            for a, crits in zip(active, self._audit_batch(active)):
                a.rounds += 1
                a.tokens += sum(
                    v.count_tokens(c.raw) for v, c in zip(self.verifiers, crits)
                )
                if aggregate(crits, self.cfg.aggregation):
                    a.accepted = a.done = True
                    accepted_this_round += 1
                else:
                    a.flaws = collect_flaws(crits)
            log.info("round %d/%d: audited %d active, %d newly accepted",
                     round_i + 1, self.cfg.max_rounds, len(active), accepted_this_round)

            # Repair the survivors -- unless this was the last round, or they hit
            # the cap during the audit, in which case they're finalized instead.
            survivors = [a for a in active if not a.done]
            last_round = round_i == self.cfg.max_rounds - 1
            repairable = [] if last_round else [
                a for a in survivors if cap is None or a.tokens < cap
            ]
            repairable_ids = {id(a) for a in repairable}
            for a in survivors:
                if id(a) not in repairable_ids:
                    a.done = True  # exhausted rounds or over budget -> not accepted

            for a, text in zip(
                repairable,
                self.corrector.repair([(a.problem, a.candidate, a.flaws) for a in repairable]),
            ):
                a.candidate = text
                a.tokens += self.corrector.count_tokens(text)

        for a in attempts:  # anything the loop left hanging
            a.done = True

        return self._regroup(problems, attempts, n_samples)

    def _active_under_budget(self, attempts: list[_Attempt], cap: int | None) -> list[_Attempt]:
        """Not-yet-done attempts still under the token cap; finalize the rest."""
        active = []
        for a in attempts:
            if a.done:
                continue
            if cap is not None and a.tokens >= cap:
                a.done = True  # spent its budget without acceptance
            else:
                active.append(a)
        return active

    def _audit_batch(self, active: list[_Attempt]) -> list[list[Critique]]:
        """Per-attempt list of the k critiques. Verifiers batch across attempts.

        Blind (default): every verifier audits independently. Debate: verifier i
        sees the critiques verifiers 0..i-1 already gave for the same attempt --
        still one batched call per verifier, sequential across the k.
        """
        debate = self.cfg.independence == "debate"
        items = [(a.problem, a.candidate) for a in active]
        per_attempt: list[list[Critique]] = [[] for _ in active]
        for v in self.verifiers:
            crits = v.audit(items, per_attempt if debate else None)
            for i, c in enumerate(crits):
                per_attempt[i].append(c)
        return per_attempt

    def _regroup(
        self, problems: list[ProblemRecord], attempts: list[_Attempt], n_samples: int
    ) -> list[list[Sample]]:
        out: list[list[Sample]] = [[] for _ in problems]
        traces: list[list[dict]] = [[] for _ in problems]
        for a in attempts:
            out[a.prob_idx].append(Sample(text=a.candidate, tokens=a.tokens))
            traces[a.prob_idx].append({
                "accepted": a.accepted,
                "status": self._status(a.accepted),
                "rounds": a.rounds,
                "flaws": a.flaws,
                "tokens": a.tokens,
            })
        self.last_traces = traces
        return out

    def _status(self, accepted: bool) -> str:
        if accepted:
            return "accepted"
        return "flagged" if self.cfg.on_nonconvergence == "flag" else "rejected"

    @property
    def spec(self) -> dict:
        c = self.cfg
        return {
            "type": "verify_repair",
            "prover": dict(c.prover.__dict__),
            "corrector": dict(c.corrector.__dict__),
            "verifiers": [dict(v.__dict__) for v in c.verifiers],
            "aggregation": c.aggregation,
            "independence": c.independence,
            "max_rounds": c.max_rounds,
            "on_nonconvergence": c.on_nonconvergence,
            "k": len(c.verifiers),
        }


class SingleCallAgent:
    """One prover call per sample -- the baseline the loop has to beat.

    A `SolverAgent` with no verification: `status` is "unverified".
    """

    def __init__(self, cfg: AgentConfig, client_factory: ClientFactory = build_model_client):
        self.cfg = cfg
        self._factory = client_factory
        self.prover: Prover | None = None
        self.last_traces: list[list[dict]] = []

    def setup(self) -> None:
        client = self._factory(self.cfg.prover)
        client.setup()
        self.prover = Prover(client, self.cfg.prover, self.cfg.prover_prompt, "prover")

    def teardown(self) -> None:
        if self.prover is not None:
            self.prover.client.teardown()

    def solve_batch(self, problems, max_new_tokens, n_samples) -> list[list[Sample]]:
        if self.prover is None:
            raise RuntimeError("call setup() before solve_batch()")
        # One batched prover call for every problem x sample.
        texts = [p.problem for p in problems for _ in range(n_samples)]
        completions = self.prover.propose(texts)

        out: list[list[Sample]] = []
        traces: list[list[dict]] = []
        for i in range(len(problems)):
            chunk = completions[i * n_samples:(i + 1) * n_samples]
            samples = [Sample(text=c, tokens=self.prover.count_tokens(c)) for c in chunk]
            out.append(samples)
            traces.append([
                {"accepted": None, "status": "unverified", "rounds": 0,
                 "flaws": [], "tokens": s.tokens}
                for s in samples
            ])
        self.last_traces = traces
        return out

    @property
    def spec(self) -> dict:
        return {"type": "single", "prover": self.cfg.prover.__dict__}
