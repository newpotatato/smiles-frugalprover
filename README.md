# FrugalProver — pilot

Research pilot around [FrugalProver](frugalproverBarannikov.pdf) (Serguei
Barannikov, Skoltech): a compact multi-agent theorem-proving system that
budgets its own compute per problem, predicting effort from a small model's
internal activations before attempting a solution.

This repo is the literature review + a runnable pilot testing the system's
core premise (H1: "effort is predictable in advance from activations") on a
small scale, ahead of the full project.

## Where to start

- **[pilot/RESEARCH_PLAN.md](pilot/RESEARCH_PLAN.md)** — what we're actually
  testing, by direction: layer-wise activations, oracle architectures,
  dimensionality-reduction/ID vs. difficulty. Read this first.
- **[pilot/TEAM_PLAN.md](pilot/TEAM_PLAN.md)** — who runs what, in what order.
- **[references/PIPELINE.md](references/PIPELINE.md)** — full breakdown of the
  proposed system (budget oracle, prover, verifiers, corrector, human gate,
  orchestrator), risks, and open gaps.
- **[references/MANIFEST.md](references/MANIFEST.md)** — how each of the 13
  cited papers maps to the system's components and hypotheses.

## Repo layout

```
frugalproverBarannikov.pdf/.md   the proposal itself
references/                      13 cited papers (PDF) + literature synthesis
pilot/                            the runnable pipeline (see below)
```

## Running the pilot

Local steps need only `pip install -r requirements.txt` (CPU, no GPU). The
notebook step needs a Colab GPU runtime.

1. **`python pilot/data_prep.py`** — pulls MATH from Hugging Face, builds a
   large balanced pool (`math_pool_large.json`, ~300 problems) plus a smaller
   subset nested inside it (`math_labeling_subset.json`, ~80 problems) for the
   expensive full budget-sweep labeling.
2. **`python pilot/split_subset.py <file> <N>`** — splits either file into N
   parts (stratified by level/type) for parallel runs.
3. **`pilot/colab_budget_oracle_pilot.ipynb`** (upload to Colab) — set
   `RUN_MODE` to one of three tiers before running:
   - `activations_only` — cheap, forward-pass-only, scales to hundreds of problems.
   - `full_sweep` — expensive, the real budget/B* labels, run only on the labeling subset.
   - `baseline_single` — cheap-ish middle ground, one budget level at larger scale.
4. **`python pilot/merge_results.py`** — joins all downloaded tier outputs by
   problem id into one `pilot_results.jsonl`.
5. Analysis, pick what you need:
   - `python pilot/analyze.py pilot_results.jsonl` — layer-depth sweep, baseline comparison, bootstrap/permutation checks.
   - `python pilot/oracle.py pilot_results.jsonl --mode classification --compare-models --demo` — fit and compare oracle model architectures.
   - `python pilot/id_vs_difficulty.py pilot_results.jsonl` — ID/compression-method vs. difficulty hypothesis.
   - `python pilot/calibration_cost.py pilot_results.jsonl` — measured cost of labeling vs. an oracle-guided projection.
   - `python pilot/learning_curve.py pilot_results.jsonl` — does the oracle improve as labeled data accrues.

`pilot/smoke_test.py` is a CPU-only, tiny-model sanity check for the harness
logic (grading, activation shapes) -- run it before burning Colab time on
changes to the notebook.

## Data note

`pilot/pilot_*.jsonl` (raw per-problem activation dumps) are **not** committed
-- activations at every layer for hundreds of problems serialize to well over
GitHub's 100MB file limit. Regenerate them by running the notebook, or ask
whoever ran it to share the file directly (Drive/Slack) if you just want to
run the analysis scripts without re-running Colab yourself.
