# `agent/` — the solving agent

The oracle predicts how much effort a *solver* needs; `agent/` is that solver.
Keeping it separate marks the boundary: `oracle/` is about predicting effort,
`agent/` is about spending it. Mixing them is how "the oracle works" quietly
becomes "the oracle works with this one hardcoded prompt."

`base.py` defines the `SolverAgent` protocol; concrete realizations live beside
it and are selected with `build_agent(cfg.agent)`.

## What's here

- `model.py` — `ModelClient` ABC, a scriptable `MockModelClient` that runs the
  loop on CPU with no models or network, and `HFClient` (local
  `transformers.generate`; same-model roles share one loaded copy, optionally
  bitsandbytes-quantized via `ModelSpec.quantization`). The `openai`
  (vLLM-served, OpenAI-compatible) backend is registered but raises
  `NotImplementedError` with a spec until implemented.
- `roles.py` — `Prover`, `Verifier`, `Corrector`, and `Critique` (verdict +
  specific diagnosed flaws — the feedback contract handed to the corrector).
- `aggregation.py` — how the k verdicts combine: `unanimity` vs `majority`.
- `verify_repair.py` — `VerifyRepairAgent` (the propose-verify-repair loop) and
  `SingleCallAgent` (the one-shot baseline).

Each role chooses its own model via `AgentConfig` in `common/config.py`. Run the
loop end-to-end on CPU with the mock backend:

    frugalprover prove -c configs/base.yaml -c configs/agent/mock.yaml --problems <problems.jsonl>

## The loop

`VerifyRepairAgent` holds a prover untrusted until verification passes: the
prover writes a candidate, k skeptical verifiers audit it, the corrector repairs
whatever they flag, and the loop iterates until the verifiers concur (by the
configured aggregation rule) or a round cap is hit. Non-convergence rejects or
flags. Four control-flow dials pin the architecture down, all in `AgentConfig`:
the stopping rule (`max_rounds`, `on_nonconvergence`), the aggregation rule
(`unanimity` vs `majority`), the feedback contract (`Critique.flaws`, the
specific diagnoses the corrector repairs against), and verifier independence
(`blind` vs `debate`).

The loop is **batched across tasks**, not run per task. All attempts
(problem × sample) advance in lockstep, and each role's `generate` runs once
across the whole active set — the prover for every attempt, then each verifier
for every active attempt, then the corrector for every survivor. Attempts leave
the active set as they are accepted, exhaust their rounds, or hit the token cap,
so later rounds run on a shrinking batch. A batching backend (vLLM, HF) sees one
big call per step instead of one prompt at a time.

## How it plugs in

`oracle/budget/sweep.py` constructs a `SolverAgent` in `setup()` and calls
`solve_batch(problems, max_new_tokens=B, n_samples=n)` once per budget. Each
returned `Sample` carries the completion text and the tokens it cost; Stage 2
grades the text with `frugalprover.common.grading.grade`, derives `B*`, and sums
the tokens for `tokens_spent`. Everything the stage needs rides back on the
protocol return — it never reaches into agent internals like `last_traces`
(that field is a `prove`-CLI convenience, not part of the contract). Stage 2 is
the only thing that calls an agent, and it calls it through `SolverAgent` — so
swapping solvers never touches the labeling loop.

Nothing in `oracle/` imports from `agent/`, and nothing in `agent/` imports from
`oracle/` — the dependency runs one way.

## The one constraint that matters

Whatever the agent does internally, an attempt's **total generated tokens must
respect `max_new_tokens`**. An agent that makes three model calls of 512 tokens
each while being labeled at "budget 512" makes the budget axis meaningless, and
every number downstream inherits that. `VerifyRepairAgent` checks the cap at each
round boundary: once an attempt's running token total reaches it, the attempt is
finalized and gets no further generation.
