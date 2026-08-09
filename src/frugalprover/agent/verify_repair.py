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

The cap is **per attempt**, not per batch: `max_new_tokens` may be a list of one
budget per problem, which is how an allocation policy (see allocate/) spends a
fixed total unevenly. Attempts on different budgets still advance in the same
lockstep and ride in the same batched calls; only their own ceiling differs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from frugalprover.agent.aggregation import aggregate, collect_flaws
from frugalprover.agent.base import AttemptTrace, RoundTrace, Sample, stable_seed
from frugalprover.agent.model import ModelClient, build_model_client
from frugalprover.agent.roles import Corrector, Critique, Prover, Verifier
from frugalprover.common.config import AgentConfig, ModelSpec
from frugalprover.common.logging import get_logger
from frugalprover.common.records import ProblemRecord

log = get_logger(__name__)

#: A client factory: ModelSpec -> ModelClient. Swappable so tests can inject
#: scripted mocks without a config round-trip.
ClientFactory = Callable[[ModelSpec], ModelClient]


def _normalize_caps(max_new_tokens: int | list[int], n: int) -> list[int | None]:
    """One cap per problem, from either a scalar or a per-problem list.

    A non-positive cap means uncapped, in either form -- `solve_batch(...,
    max_new_tokens=0)` is the documented way to run the loop standalone, and a
    policy that allocates 0 tokens to a problem should say so by skipping it
    rather than by passing 0 here.
    """
    if isinstance(max_new_tokens, (list, tuple)):
        if len(max_new_tokens) != n:
            raise ValueError(
                f"max_new_tokens has {len(max_new_tokens)} cap(s) for {n} problem(s); "
                "pass one per problem, or a single int for all of them"
            )
        return [c if c and c > 0 else None for c in max_new_tokens]
    cap = max_new_tokens if max_new_tokens and max_new_tokens > 0 else None
    return [cap] * n


def _describe_caps(caps: list[int | None]) -> str:
    distinct = sorted({c for c in caps if c is not None})
    if not distinct:
        return "none"
    if len(distinct) == 1:
        return str(distinct[0])
    return f"{distinct[0]}-{distinct[-1]} ({len(distinct)} distinct)"


def _caps(attempts: list[_Attempt]) -> list[int | None]:
    return [a.cap for a in attempts]


def _seeds(attempts: list[_Attempt]) -> list[int] | None:
    """Per-attempt seeds, or None when this run isn't seeded at all."""
    seeds = [a.seed for a in attempts]
    return None if any(s is None for s in seeds) else seeds


@dataclass
class _Attempt:
    """One problem-sample working its way through the loop."""

    prob_idx: int
    problem: str
    #: This attempt's own total-token ceiling; None = uncapped.
    cap: int | None = None
    #: Decoding seed, when the config asks for reproducible sampling.
    seed: int | None = None
    candidate: str = ""
    rounds: int = 0
    flaws: list[str] = field(default_factory=list)
    tokens: int = 0
    accepted: bool = False
    done: bool = False
    #: Replayable trajectory, filled in as the loop advances. Lets Stage 2
    #: reconstruct smaller budgets from one pass -- see agent/base.py.
    trace: AttemptTrace | None = None


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
        max_new_tokens: int | list[int],
        n_samples: int,
    ) -> list[list[Sample]]:
        attempts = self._run(problems, max_new_tokens, n_samples)
        return self._regroup(problems, attempts, n_samples)

    def solve_batch_traced(
        self,
        problems: list[ProblemRecord],
        max_new_tokens: int | list[int],
        n_samples: int,
    ) -> list[list[AttemptTrace]]:
        """The `TracingAgent` capability: one pass, replayable at any lower cap.

        Same loop as `solve_batch` -- it just hands back each attempt's
        trajectory instead of only its endpoint, so Stage 2 can read off what a
        smaller budget would have produced without paying for another pass.
        """
        attempts = self._run(problems, max_new_tokens, n_samples)
        self._regroup(problems, attempts, n_samples)  # keeps last_traces populated
        out: list[list[AttemptTrace]] = [[] for _ in problems]
        for a in attempts:
            out[a.prob_idx].append(a.trace)
        return out

    def _run(
        self,
        problems: list[ProblemRecord],
        max_new_tokens: int | list[int],
        n_samples: int,
    ) -> list[_Attempt]:
        if self.prover is None:
            raise RuntimeError("call setup() before solve_batch()")
        caps = _normalize_caps(max_new_tokens, len(problems))
        seeded = self.cfg.seed_mode == "per_problem"

        # Flatten problems x samples into one batch of attempts, in an order that
        # regroups cleanly: all samples of problem 0, then problem 1, ...
        attempts = [
            _Attempt(
                prob_idx=i,
                problem=p.problem,
                cap=caps[i],
                seed=stable_seed(p.id, s) if seeded else None,
            )
            for i, p in enumerate(problems)
            for s in range(n_samples)
        ]

        # Prover: one batched call for the whole set.
        log.info("proposing %d attempts (%d problems x %d samples), cap=%s, max_rounds=%d",
                 len(attempts), len(problems), n_samples,
                 _describe_caps(caps), self.cfg.max_rounds)
        for a, text in zip(attempts, self.prover.propose(
            [a.problem for a in attempts], _caps(attempts), _seeds(attempts),
        )):
            n_tok = self.prover.count_tokens(text)
            a.candidate = text
            a.tokens += n_tok
            a.trace = AttemptTrace(prover_tokens=n_tok, candidate_0=text)

        for round_i in range(self.cfg.max_rounds):
            active = self._active_under_budget(attempts)
            if not active:
                break

            # Audit: each verifier runs once across all active attempts.
            accepted_this_round = 0
            for a, crits in zip(active, self._audit_batch(active)):
                a.rounds += 1
                n_tok = sum(
                    v.count_tokens(c.raw) for v, c in zip(self.verifiers, crits)
                )
                a.tokens += n_tok
                ok = aggregate(crits, self.cfg.aggregation)
                a.trace.rounds.append(RoundTrace(verifier_tokens=n_tok, accepted=ok))
                if ok:
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
                a for a in survivors if a.cap is None or a.tokens < a.cap
            ]
            repairable_ids = {id(a) for a in repairable}
            for a in survivors:
                if id(a) not in repairable_ids:
                    a.done = True  # exhausted rounds or over budget -> not accepted

            for a, text in zip(
                repairable,
                self.corrector.repair(
                    [(a.problem, a.candidate, a.flaws) for a in repairable],
                    _caps(repairable), _seeds(repairable),
                ),
            ):
                n_tok = self.corrector.count_tokens(text)
                a.candidate = text
                a.tokens += n_tok
                # Survivors that were NOT repaired keep corrector_tokens=None,
                # which is exactly what replay() reads as "stop here".
                a.trace.rounds[-1].corrector_tokens = n_tok
                a.trace.rounds[-1].candidate_after = text

        for a in attempts:  # anything the loop left hanging
            a.done = True

        return attempts

    def _active_under_budget(self, attempts: list[_Attempt]) -> list[_Attempt]:
        """Not-yet-done attempts still under their own token cap; finalize the rest."""
        active = []
        for a in attempts:
            if a.done:
                continue
            if a.cap is not None and a.tokens >= a.cap:
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
        caps, seeds = _caps(active), _seeds(active)
        per_attempt: list[list[Critique]] = [[] for _ in active]
        for v in self.verifiers:
            crits = v.audit(items, per_attempt if debate else None, caps, seeds)
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
            "seed_mode": c.seed_mode,
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
        # One batched prover call for every problem x sample. With a single call
        # per attempt the budget is just that call's cap, so honouring a
        # per-problem allocation needs no accounting -- unlike the loop, this
        # agent cannot overshoot at a round boundary because it has no rounds.
        caps = _normalize_caps(max_new_tokens, len(problems))
        seeded = self.cfg.seed_mode == "per_problem"
        texts = [p.problem for p in problems for _ in range(n_samples)]
        per_attempt_caps = [caps[i] for i in range(len(problems)) for _ in range(n_samples)]
        seeds = (
            [stable_seed(p.id, s) for p in problems for s in range(n_samples)]
            if seeded else None
        )
        completions = self.prover.propose(texts, per_attempt_caps, seeds)

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

    def solve_batch_traced(self, problems, max_new_tokens, n_samples) -> list[list[AttemptTrace]]:
        """`TracingAgent`: one call per sample, so the trace is just its cost.

        With no rounds there is nothing for a smaller budget to truncate, and
        `replay` correctly returns the same Sample at every cap -- which is the
        honest answer for an agent that ignores the budget.
        """
        solved = self.solve_batch(problems, max_new_tokens, n_samples)
        return [
            [AttemptTrace(prover_tokens=s.tokens, candidate_0=s.text) for s in samples]
            for samples in solved
        ]

    @property
    def spec(self) -> dict:
        return {
            "type": "single",
            "prover": self.cfg.prover.__dict__,
            "seed_mode": self.cfg.seed_mode,
        }
