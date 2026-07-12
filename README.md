# FrugalProver — pilot

Research pilot around [FrugalProver](docs/proposal.pdf) (Serguei Barannikov,
Skoltech): a compact multi-agent theorem-proving system that budgets its own
compute per problem, predicting effort from a small model's internal
activations before attempting a solution.

This repo is the literature review + a runnable pilot testing the system's core
premise (H1: "effort is predictable in advance from activations") on a small
scale, ahead of the full project.

## Where to start

- **[docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md)** — what we're actually
  testing, organized as go/no-go phases (H1 → H2 → H3 + geometry): what each
  phase tests, how, and what counts as a useful result. Read this first.
- **[references/PIPELINE.md](references/PIPELINE.md)** — full breakdown of the
  proposed system (budget oracle, prover, verifiers, corrector, human gate,
  orchestrator), risks, and open gaps.
- **[references/MANIFEST.md](references/MANIFEST.md)** — how each of the 13
  cited papers maps to the system's components and hypotheses.

## Repo layout

The repo is organized so the three research directions can be developed
independently: each owns one folder under `experiments/`, and they share one
library (`frugalprover`), one data pool (`data/`), and one extraction step
(`extraction/`).

```
docs/                 the proposal (proposal.pdf) + RESEARCH_PLAN
references/            13 cited papers (PDF) + literature synthesis
src/frugalprover/      shared library (pip-installed, see Setup)
  paths.py              repo-anchored DATA_DIR / RESULTS_DIR
  id_estimators.py      TwoNN + MLE intrinsic-dimension estimators
data/                  MATH problem pools + subsets (+ gitignored raw dumps)
pipeline/             data prep / split / merge / CPU smoke test (shared infra)
extraction/           Colab GPU notebooks that produce the activations
  configs/              layer-pooling YAML configs
experiments/          one folder per research direction (see RESEARCH_PLAN phases)
  A_oracle/             oracle architectures + calibration/learning-curve (Phase 1)
  B_layers/             layer-depth empirics (Phase 1)
  C_geometry/           ID / geometry vs. difficulty (Phase 4)
results/              generated plots / tables / oracle_model.joblib (gitignored dumps)
slides/               LaTeX deck (build output under slides/out/ is gitignored)
```

## Setup

Local steps are CPU-only (no GPU). Install the shared library once, in editable
mode, so every script and notebook can `import frugalprover` regardless of the
folder it runs from:

```
pip install -r requirements.txt
pip install -e .
```

The extraction notebooks (`extraction/*.ipynb`) run on a Colab GPU runtime and
install their own GPU stack in their first cell.

## Running the pilot

1. **`python pipeline/data_prep.py`** — pulls MATH from Hugging Face, builds a
   large balanced pool (`data/math_pool_large.json`, ~300 problems) plus a
   smaller subset nested inside it (`data/math_labeling_subset.json`, ~80
   problems) for the expensive full budget-sweep labeling.
2. **`python pipeline/split_subset.py <file> <N>`** — splits either file (by
   name, resolved in `data/`) into N parts (stratified by level/type) for
   parallel runs.
3. **`extraction/colab_budget_oracle_pilot.ipynb`** (upload to Colab) — set
   `RUN_MODE` to one of three tiers before running:
   - `activations_only` — cheap, forward-pass-only, scales to hundreds of problems.
   - `full_sweep` — expensive, the real budget/B* labels, run only on the labeling subset.
   - `baseline_single` — cheap-ish middle ground, one budget level at larger scale.
4. **`python pipeline/merge_results.py`** — joins all downloaded tier outputs
   (placed in `data/`) by problem id into one `data/pilot_results.jsonl`.
5. Analysis, pick what you need (each takes the merged jsonl; plots/artifacts
   are written to the current directory):
   - `python experiments/B_layers/analyze.py data/pilot_results.jsonl` — layer-depth sweep, baseline comparison, bootstrap/permutation checks.
   - `python experiments/A_oracle/oracle.py data/pilot_results.jsonl --mode classification --compare-models --demo` — fit and compare oracle model architectures.
   - `python experiments/C_geometry/id_vs_difficulty.py data/pilot_results.jsonl` — ID/compression-method vs. difficulty hypothesis.
   - `python experiments/A_oracle/calibration_cost.py data/pilot_results.jsonl` — measured cost of labeling vs. an oracle-guided projection.
   - `python experiments/A_oracle/learning_curve.py data/pilot_results.jsonl` — does the oracle improve as labeled data accrues.

`pipeline/smoke_test.py` is a CPU-only, tiny-model sanity check for the harness
logic (grading, activation shapes) -- run it before burning Colab time on
changes to the notebook.

There's also a separate feature-extraction track for layer-pooled hidden states
(parquet output), documented in
[extraction/README_layer_poolings.md](extraction/README_layer_poolings.md).

## Data note

`data/pilot_*.jsonl` (raw per-problem activation dumps and the merged
`pilot_results.jsonl`) are **not** committed -- activations at every layer for
hundreds of problems serialize to well over GitHub's 100MB file limit.
Regenerate them by running the notebook, or ask whoever ran it to share the
file directly (Drive/Slack) if you just want to run the analysis scripts
without re-running Colab yourself.
