"""The real budget estimator: sweep the token cap over a solving agent.

For each problem this runs the agent (`cfg.agent` of the pipeline -- the full
prover -> verifier -> corrector `SolverAgent`, or the single-call baseline) once
per budget in `cfg.budgets`, grades the completions, and records the smallest
budget that clears the success threshold as ``b_star``.

With `budget.single_pass_reconstruct` the sweep instead runs **one** pass at the
largest budget and replays each attempt's trajectory against the smaller ones.
That is exact whenever the per-role clamp below never binds -- the smaller budget
then produces the same calls in the same order, just stopping sooner -- and it is
guarded to fall back rather than assume. Besides the GPU time saved, it couples
the budgets to common random numbers, so `p(B2) - p(B1)` stops carrying the
resampling noise of two unrelated passes.

The agent owns its own decoding and prompt (each role's `ModelSpec` and the
`AgentConfig` prompts); this stage only chooses *how many tokens* it may spend.
So `budget.temperature`, `budget.top_p`, and `budget.prompt_template` are unused
under this estimator -- the agent config is authoritative. `budget.agent` (a bare
model name) is likewise superseded; the record's `agent` field is taken from the
agent's prover model.

See docs/reference_budget_notebook.ipynb for a single-GPU sketch of the same
loop written before the library existed, and docs/ARTIFACTS.md for the A2
contract.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from typing import TYPE_CHECKING

from frugalprover.common.config import AgentConfig, BudgetConfig
from frugalprover.common.grading import extract_answer, grade, normalize
from frugalprover.common.logging import get_logger
from frugalprover.common.records import BudgetRecord, ProblemRecord

log = get_logger(__name__)

if TYPE_CHECKING:
    # Type-only: never a runtime import, keeping the one-way oracle -> (no) agent
    # dependency intact. The concrete agent is built lazily in setup(), and the
    # reconstruction path only ever calls `.replay()` on traces the agent handed
    # back through the protocol -- no agent internals are reached into.
    from frugalprover.agent.base import AttemptTrace, Sample


class TokenSweepEstimator:
    """Measure solve effort by sweeping the generation token cap over an agent."""

    def __init__(self, cfg: BudgetConfig, agent_cfg: AgentConfig):
        self.cfg = cfg
        self.agent_cfg = agent_cfg
        self.agent = None
        #: The guard is re-checked per batch but its verdict can't change, so the
        #: explanation is logged once instead of once per batch (a long run is
        #: hundreds of batches).
        self._warned_fallback = False

    def setup(self) -> None:
        """Build the solving agent from the pipeline's `agent` config and load it.

        Going through `SolverAgent` (not a bare model) is the point of the
        indirection: swapping in a multi-step or tool-using agent needs no change
        here, because this stage only ever calls `solve_batch`.
        """
        from frugalprover.agent import build_agent

        self.agent = build_agent(self.agent_cfg)
        self.agent.setup()
        log.info("sweep agent: prover=%s, verifiers=[%s], corrector=%s",
                 self.agent_cfg.prover.model,
                 ", ".join(v.model for v in self.agent_cfg.verifiers),
                 self.agent_cfg.corrector.model)
        log.info("sweeping budgets=%s at n_samples=%d, success_threshold=%.2f",
                 sorted(self.cfg.budgets), self.cfg.n_samples, self.cfg.success_threshold)

    def estimate_batch(self, problems: list[ProblemRecord]) -> list[BudgetRecord]:
        if self.agent is None:
            raise RuntimeError("call setup() before estimate_batch()")
        budgets = sorted(self.cfg.budgets)
        if self._reconstructable(budgets):
            return self._estimate_reconstructed(problems, budgets)
        return self._estimate_independent(problems, budgets)

    def _role_cap(self) -> int:
        """Largest per-call max_tokens across the roles -- what the clamp tests."""
        return max(s.max_tokens for s in (self.agent_cfg.prover,
                                          self.agent_cfg.corrector,
                                          *self.agent_cfg.verifiers))

    def _reconstructable(self, budgets: list[int]) -> bool:
        """Whether one pass at max(budgets) can stand in for the whole sweep.

        Two conditions, both checked loudly rather than assumed: the agent must
        expose the tracing capability, and no budget may be small enough for
        `_solve_at`'s clamp to bind -- because a clamped role generates *less*
        per call, which is a different run, not a prefix of the same one.
        """
        if not self.cfg.single_pass_reconstruct:
            return False
        if not callable(getattr(self.agent, "solve_batch_traced", None)):
            self._warn_fallback(
                "budget.single_pass_reconstruct is set but agent %s has no "
                "solve_batch_traced(); falling back to %d independent passes.",
                type(self.agent).__name__, len(budgets),
            )
            return False
        cap = self._role_cap()
        if budgets[0] < cap:
            self._warn_fallback(
                "budget.single_pass_reconstruct is set but min(budgets)=%d < the largest role "
                "max_tokens=%d, so the per-role clamp WOULD bind and the smaller budgets are "
                "not prefixes of the largest. Falling back to %d independent passes -- raise "
                "the budgets or lower the role max_tokens to enable reconstruction.",
                budgets[0], cap, len(budgets),
            )
            return False
        return True

    def _warn_fallback(self, msg: str, *args) -> None:
        if not self._warned_fallback:
            log.warning(msg, *args)
            self._warned_fallback = True

    def _estimate_reconstructed(
        self, problems: list[ProblemRecord], budgets: list[int]
    ) -> list[BudgetRecord]:
        """One pass at the largest budget; read the rest off the trajectories."""
        cfg = self.cfg
        b_max = budgets[-1]
        log.info("single pass at B=%d, reconstructing %s: %d problems x %d samples",
                 b_max, budgets, len(problems), cfg.n_samples)
        traced = self._solve_traced_at(problems, b_max)

        n_success: list[dict[int, int]] = [{} for _ in problems]
        sc: list[dict[int, float]] = [{} for _ in problems]
        tokens_spent = [0 for _ in problems]
        generated = [0 for _ in problems]
        cleared = {b: 0 for b in budgets}

        for i, (p, traces) in enumerate(zip(problems, traced)):
            # What the pass really cost, as opposed to the sweep-shaped total below.
            generated[i] = sum(t.replay(b_max).tokens for t in traces)
            for b in budgets:
                samples = [t.replay(b) for t in traces]
                texts = [s.text for s in samples]
                n_ok = sum(grade(t, p.answer) for t in texts)
                n_success[i][b] = n_ok
                sc[i][b] = self._self_consistency(texts, p.answer)
                tokens_spent[i] += sum(s.tokens for s in samples)
                if cfg.n_samples and n_ok / cfg.n_samples >= cfg.success_threshold:
                    cleared[b] += 1

        for b in budgets:
            log.info("budget %d: %d/%d problems cleared (tau=%.2f) [reconstructed]",
                     b, cleared[b], len(problems), cfg.success_threshold)
        log.info("single pass generated %d tokens for %d problems (%.0f/problem)",
                 sum(generated), len(problems),
                 sum(generated) / max(1, len(problems)))

        records = self._records(problems, budgets, n_success, sc, tokens_spent)
        for r, n in zip(records, generated):
            # Marks a mixed corpus, and records the true single-pass cost next to
            # the sweep-shaped tokens_spent so the two aren't confused later.
            r.extra["reconstructed"] = True
            r.extra["tokens_generated"] = n
        return records

    def _estimate_independent(
        self, problems: list[ProblemRecord], budgets: list[int]
    ) -> list[BudgetRecord]:
        cfg = self.cfg

        # Per-problem accumulators, keyed by budget. Independent measurement per
        # budget; from_counts derives b_star as the smallest budget clearing tau.
        n_success: list[dict[int, int]] = [{} for _ in problems]
        sc: list[dict[int, float]] = [{} for _ in problems]
        tokens_spent = [0 for _ in problems]

        # Sweep by budget, not by problem: one solve_batch handles every problem
        # at budget B before moving to the next B (all at 128, then all at 256).
        for bi, budget in enumerate(budgets, 1):
            log.info("budget %d (%d/%d): solving %d problems x %d samples",
                     budget, bi, len(budgets), len(problems), cfg.n_samples)
            solved = self._solve_at(problems, budget)  # list[list[Sample]]
            cleared = 0
            budget_tokens = 0
            for i, (p, samples) in enumerate(zip(problems, solved)):
                texts = [s.text for s in samples]
                n_ok = sum(grade(t, p.answer) for t in texts)
                n_success[i][budget] = n_ok
                sc[i][budget] = self._self_consistency(texts, p.answer)
                b_tokens = sum(s.tokens for s in samples)
                tokens_spent[i] += b_tokens
                budget_tokens += b_tokens
                if cfg.n_samples and n_ok / cfg.n_samples >= cfg.success_threshold:
                    cleared += 1
            log.info("budget %d: %d/%d problems cleared (tau=%.2f), %d tokens this pass",
                     budget, cleared, len(problems), cfg.success_threshold, budget_tokens)

        return self._records(problems, budgets, n_success, sc, tokens_spent)

    def _records(
        self,
        problems: list[ProblemRecord],
        budgets: list[int],
        n_success: list[dict[int, int]],
        sc: list[dict[int, float]],
        tokens_spent: list[int],
    ) -> list[BudgetRecord]:
        """A2 records from the per-budget counts. Shared by both estimate paths
        so a reconstructed record is on exactly the same scale as a swept one."""
        agent_label = self.agent_cfg.prover.model
        return [
            BudgetRecord.from_counts(
                problem_id=p.id,
                agent=agent_label,
                budgets=budgets,
                n_samples=self.cfg.n_samples,
                n_success=n_success[i],
                success_threshold=self.cfg.success_threshold,
                sc=sc[i],
                tokens_spent=tokens_spent[i],
            )
            for i, p in enumerate(problems)
        ]

    def _solve_at(self, problems: list[ProblemRecord], budget: int) -> list[list[Sample]]:
        """Solve every problem under a total-token cap of `budget`.

        Each role's per-call `max_tokens` is temporarily lowered to `budget` so no
        single model call can overshoot it. The loop already finalizes an attempt
        once its running total reaches the cap (checked at round boundaries), but
        without capping the calls the prover's first generation would emit its full
        configured `max_tokens` regardless of B -- and an internal call that
        ignores the budget makes the budget axis meaningless (CLAUDE.md invariant).
        The role specs are shared with the built agent, so mutating them here is
        what the loop reads; originals are restored after the pass.
        """
        with self._clamped(budget):
            return self.agent.solve_batch(
                problems, max_new_tokens=budget, n_samples=self.cfg.n_samples
            )

    def _solve_traced_at(
        self, problems: list[ProblemRecord], budget: int
    ) -> list[list[AttemptTrace]]:
        """`_solve_at`'s tracing twin, for the reconstruction path.

        Still applies the clamp. Under `_reconstructable`'s guard it is a no-op by
        construction, but leaving it in means the invariant holds defensively
        rather than by the caller remembering to check.
        """
        with self._clamped(budget):
            return self.agent.solve_batch_traced(
                problems, max_new_tokens=budget, n_samples=self.cfg.n_samples
            )

    @contextmanager
    def _clamped(self, budget: int):
        """Temporarily lower every role's per-call cap to `budget`.

        The role specs are shared with the built agent, so mutating them here is
        what the loop reads; originals are restored on the way out.
        """
        specs = [
            self.agent_cfg.prover,
            self.agent_cfg.corrector,
            *self.agent_cfg.verifiers,
        ]
        saved = [s.max_tokens for s in specs]
        for s in specs:
            s.max_tokens = min(s.max_tokens, budget)
        try:
            yield
        finally:
            for s, original in zip(specs, saved):
                s.max_tokens = original

    @staticmethod
    def _self_consistency(completions: list[str], gold: str) -> float:
        """1.0 if the plurality answer across samples is correct, else 0.0.

        The natural allocation baseline: it costs nothing extra (the samples
        already exist) and is what the oracle has to beat.
        """
        answers = [normalize(extract_answer(c)) for c in completions]
        answers = [a for a in answers if a]
        if not answers:
            return 0.0
        top, _ = Counter(answers).most_common(1)[0]
        return 1.0 if top == normalize(gold) else 0.0

    def teardown(self) -> None:
        if self.agent is not None:
            self.agent.teardown()
