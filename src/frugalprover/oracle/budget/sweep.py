"""The real budget estimator. NOT IMPLEMENTED YET.

This is the one expensive stage in the pipeline and the only one still missing.
Everything around it exists: the record schema, the resumable runner, the
config, and a mock that lets Stages 3-5 run today. What's left is one method.

Read :class:`TokenSweepEstimator.estimate_batch`'s docstring for the spec, and
docs/reference_budget_notebook.ipynb for a working single-GPU sketch of the
same loop written before the library existed.
"""
from __future__ import annotations

from frugalprover.common.config import BudgetConfig
from frugalprover.common.records import BudgetRecord, ProblemRecord


class TokenSweepEstimator:
    """Measure solve effort by sweeping the generation token cap.

    **Not implemented.** Set ``budget.estimator: mock`` to exercise the rest of
    the pipeline, or implement :meth:`estimate_batch` below.
    """

    def __init__(self, cfg: BudgetConfig):
        self.cfg = cfg

    def setup(self) -> None:
        """Load the solving agent (`cfg.agent`) and its tokenizer onto the GPU.

        This will eventually construct a `frugalprover.agent.SolverAgent` rather
        than a bare model, so that swapping in a multi-step or tool-using agent
        doesn't require touching this file.
        """
        raise NotImplementedError(_SPEC)

    def estimate_batch(self, problems: list[ProblemRecord]) -> list[BudgetRecord]:
        """Sweep token caps and return one A2 record per problem.

        What a conforming implementation must do, for each problem:

        1. Render the prompt: ``cfg.prompt_template.format(problem=p.problem)``.

        2. For each budget ``B`` in ``cfg.budgets`` (ascending), generate
           ``cfg.n_samples`` completions with ``max_new_tokens=B``,
           ``do_sample=True``, ``temperature=cfg.temperature``,
           ``top_p=cfg.top_p``. Decode only the newly generated tokens --
           leaving the prompt in means the gold answer, if it appears in the
           prompt, gets graded as the model's own output.

        3. Grade each completion with
           ``frugalprover.common.grading.grade(text, problem.answer)`` and count
           the successes at that budget.

        4. Optionally record self-consistency: the majority vote over the
           non-null extracted answers at each budget. It costs nothing extra --
           the samples already exist -- and it is the natural baseline the
           oracle has to beat on allocation.

        5. Build the record with
           ``BudgetRecord.from_counts(problem.id, cfg.agent, cfg.budgets,
           cfg.n_samples, n_success, cfg.success_threshold, sc, tokens_spent)``.
           That helper derives ``p``, the Wilson intervals, and ``b_star``, so
           the "smallest budget clearing tau" rule lives in exactly one place.

        Return records in the same order as `problems`.

        Things the runner already handles, so don't reimplement them here:
        resume from a partial file, append-and-flush per problem, sorting the
        final output, and writing the sidecar meta.

        Two things worth getting right, because they silently corrupt labels:

        - **Batch by budget, not by problem.** All problems at B=128, then all
          at B=256. Mixing budgets in one `generate` call means padding to the
          largest, which wastes most of the compute this stage is spending.

        - **Check the tokenizer's padding side.** Decoder-only models are
          usually left-padded for generation; slicing off the prompt with a
          fixed `input_ids.shape[1]` is only correct when padding is uniform.
        """
        raise NotImplementedError(_SPEC)

    def teardown(self) -> None:
        pass


_SPEC = (
    "Stage 2 (budget labeling) is intentionally not implemented yet.\n"
    "  - To exercise the rest of the pipeline now:  --set budget.estimator=mock\n"
    "  - To implement it: see the docstring of\n"
    "    frugalprover.oracle.budget.sweep.TokenSweepEstimator.estimate_batch,\n"
    "    the A2 contract in docs/ARTIFACTS.md, and the working sketch in\n"
    "    docs/reference_budget_notebook.ipynb."
)
