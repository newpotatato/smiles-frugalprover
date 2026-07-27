"""The real budget estimator: sweep the token cap over a solving agent.

For each problem this runs the agent (`cfg.agent` of the pipeline -- the full
prover -> verifier -> corrector `SolverAgent`, or the single-call baseline) once
per budget in `cfg.budgets`, grades the completions, and records the smallest
budget that clears the success threshold as ``b_star``.

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
from typing import TYPE_CHECKING

from frugalprover.common.config import AgentConfig, BudgetConfig
from frugalprover.common.grading import extract_answer, grade, normalize
from frugalprover.common.records import BudgetRecord, ProblemRecord

if TYPE_CHECKING:
    # Type-only: never a runtime import, keeping the one-way oracle -> (no) agent
    # dependency intact. The concrete agent is built lazily in setup().
    from frugalprover.agent.base import Sample


class TokenSweepEstimator:
    """Measure solve effort by sweeping the generation token cap over an agent."""

    def __init__(self, cfg: BudgetConfig, agent_cfg: AgentConfig):
        self.cfg = cfg
        self.agent_cfg = agent_cfg
        self.agent = None

    def setup(self) -> None:
        """Build the solving agent from the pipeline's `agent` config and load it.

        Going through `SolverAgent` (not a bare model) is the point of the
        indirection: swapping in a multi-step or tool-using agent needs no change
        here, because this stage only ever calls `solve_batch`.
        """
        from frugalprover.agent import build_agent

        self.agent = build_agent(self.agent_cfg)
        self.agent.setup()

    def estimate_batch(self, problems: list[ProblemRecord]) -> list[BudgetRecord]:
        if self.agent is None:
            raise RuntimeError("call setup() before estimate_batch()")
        cfg = self.cfg
        budgets = sorted(cfg.budgets)

        # Per-problem accumulators, keyed by budget. Independent measurement per
        # budget; from_counts derives b_star as the smallest budget clearing tau.
        n_success: list[dict[int, int]] = [{} for _ in problems]
        sc: list[dict[int, float]] = [{} for _ in problems]
        tokens_spent = [0 for _ in problems]

        # Sweep by budget, not by problem: one solve_batch handles every problem
        # at budget B before moving to the next B (all at 128, then all at 256).
        for budget in budgets:
            solved = self._solve_at(problems, budget)  # list[list[Sample]]
            for i, (p, samples) in enumerate(zip(problems, solved)):
                texts = [s.text for s in samples]
                n_success[i][budget] = sum(grade(t, p.answer) for t in texts)
                sc[i][budget] = self._self_consistency(texts, p.answer)
                tokens_spent[i] += sum(s.tokens for s in samples)

        agent_label = self.agent_cfg.prover.model
        return [
            BudgetRecord.from_counts(
                problem_id=p.id,
                agent=agent_label,
                budgets=budgets,
                n_samples=cfg.n_samples,
                n_success=n_success[i],
                success_threshold=cfg.success_threshold,
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
        specs = [
            self.agent_cfg.prover,
            self.agent_cfg.corrector,
            *self.agent_cfg.verifiers,
        ]
        saved = [s.max_tokens for s in specs]
        for s in specs:
            s.max_tokens = min(s.max_tokens, budget)
        try:
            return self.agent.solve_batch(
                problems, max_new_tokens=budget, n_samples=self.cfg.n_samples
            )
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
