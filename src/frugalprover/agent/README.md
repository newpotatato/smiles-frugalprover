# `agent/` — the solving agent

**Currently empty by design.** Only the protocol in `base.py` exists.

## Why it's here now

The oracle predicts how much effort a *solver* needs. That solver is a real
component with its own design space — a single model call, self-consistency
voting, propose-verify-repair — and it will eventually live here.

Reserving the space now does two things:

1. **It keeps the seam visible.** Stage 2 (`oracle/budget/`) is the only thing
   that will ever call an agent, and it calls it through `SolverAgent`. If the
   agent were built inside Stage 2, swapping solvers would mean editing the
   labeling loop.
2. **It marks the boundary.** `oracle/` is about predicting effort; `agent/` is
   about spending it. Mixing them is how "the oracle works" quietly becomes
   "the oracle works with this one hardcoded prompt."

## What goes here

- `hf_agent.py` — the baseline: one `model.generate` call per attempt with the
  configured temperature and top-p. This is what the pilot notebook does.
- `sc_agent.py` — self-consistency: sample *k* times, majority-vote the
  answers. Free from samples the sweep already draws, and the baseline the
  oracle's allocation has to beat.
- Later, if the research goes that way: a verify-and-repair loop.

## How it plugs in

`oracle/budget/sweep.py` will construct a `SolverAgent` in `setup()` and call
`solve_batch(problems, max_new_tokens=B, n_samples=n)` once per budget. The
agent returns raw completions; Stage 2 grades them with
`frugalprover.common.grading.grade` and derives `B*`.

Nothing in `oracle/` imports from `agent/` today, and nothing in `agent/`
should ever import from `oracle/` — the dependency runs one way.

## The one constraint that matters

Whatever the agent does internally, its **total generated tokens must respect
`max_new_tokens`**. An agent that makes three model calls of 512 tokens each
while being labeled at "budget 512" makes the budget axis meaningless, and
every number downstream inherits that.
