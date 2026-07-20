# Artifact contracts

The pipeline's five stages talk **only through files**. This document is the
contract between them.

```
sample ──> problems.jsonl ──┬──> budget  ──> budgets.jsonl ──┐
                            │                                ├──> train ──> oracle.joblib
                            └──> extract ──> hidden_states.parquet ─┘         metrics.json
                                                                                   │
                                                                     report <──────┘
                                                                        │
                                                                 results/<run>/
```

## The one rule

**The contract is the columns. Nothing else is required.**

If you produce a dataset somewhere else — another script, a colleague's export,
a different benchmark entirely — reformat it to the columns below and the
pipeline will read it. There is no version field to match, no hash to satisfy,
no registration step.

The library *writes* a `<filename>.meta.json` next to each artifact it produces,
recording the config, model name, git SHA and timestamp. **Nothing ever reads it
as a precondition.** No stage refuses to run because a meta is missing, stale,
or from a different config. It exists so that six weeks from now a bare parquet
of float arrays can tell you where it came from. Files from elsewhere don't need
one.

## `id` — the join key

A string derived from the source dataset:

```
{subject}/{split}/{row_index}      e.g.  "algebra/test/1423"
```

Every artifact carries it; every join is on it. Because it's derived from the
source rather than from position in a file, two people sampling the same
benchmark on different machines get the same ids, and joining files that were
never produced together just works.

If you're bringing your own data, any stable string is fine — it only has to be
consistent across your own files.

---

## A1 — `problems.jsonl`

Produced by Stage 1 (`sample`). Read by Stages 2 and 3. One JSON object per line.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `id` | str | yes | Join key |
| `problem` | str | yes | Problem statement, raw (no prompt template applied) |
| `answer` | str | yes | Gold answer, from the last `\boxed{...}` |
| `type` | str | yes | Subject, e.g. `"Algebra"` |
| `level` | int \| null | no | Human difficulty 1–5 — see the warning below |
| `solution` | str \| null | no | Full reference solution |

```jsonc
{"id": "algebra/test/1423", "problem": "What is the sum of...", "answer": "4",
 "type": "Algebra", "level": 4, "solution": "We begin by..."}
```

> **`level` is never a model feature.** It's kept for stratifying the sample and
> for plots. A problem encountered in the wild has no human difficulty
> annotation, so training on it would inflate every score and stop the result
> from answering the question the project is asking. This is enforced by there
> being no `level` feature block.

---

## A2 — `budgets.jsonl`

Produced by Stage 2 (`budget`). Read by Stage 4. **Stage 2 is not implemented
yet** — see [Implementing Stage 2](#implementing-stage-2).

### What this file means

Take a problem. Give the solver a token cap `B` and let it try `n` times. Count
how many attempts got the right answer — that fraction is `p(B)`, the
probability of solving within `B` tokens. Repeat for several caps. You get a
curve: `p` rises as the solver gets more room to think.

**`success_threshold` (τ)** turns that curve into one number, because the oracle
predicts a number:

```
B* = the smallest budget B where p(B) >= tau
```

At τ = 0.5, `B*` is the smallest budget that solves the problem at least half
the time. τ = 0.5 is pre-registered in `RESEARCH_PLAN.md` Phase 0 — chosen
before seeing results, so it can't be tuned to flatter them. It's stored per
record so changing it later doesn't make old runs ambiguous.

**Censoring.** Some problems are never solved at any budget swept. `b_star` is
`null` for those. That means "above the largest budget I tried", not "missing".

| Field | Type | Required | Meaning |
|---|---|---|---|
| `id` | str | yes | Join key |
| `agent` | str | yes | What solved it: a model id, or `"mock"` |
| `budgets` | list[int] | yes | Token caps tried, ascending |
| `n_samples` | int | yes | Attempts per (problem, budget) |
| `success_threshold` | float | yes | τ |
| `n_success` | dict[str,int] | yes | Successes per budget, **string** keys |
| `p` | dict[str,float] | yes | `n_success / n_samples` |
| `p_ci` | dict[str,[float,float]] | no | Wilson 95% interval |
| `b_star` | int \| null | yes | Smallest budget with `p >= tau`; null = censored |
| `sc` | dict[str,float] | no | Self-consistency (majority vote) success |
| `tokens_spent` | int | no | Tokens actually generated, for cost analysis |

Budget dict keys are **strings** throughout (`"128"`, not `128`).

### Full sweep and single pass are the same schema

A single fixed-budget run is just a sweep of length one:

```jsonc
// full sweep
{"id": "algebra/test/1423", "agent": "Qwen/Qwen2.5-1.5B-Instruct",
 "budgets": [128, 256, 512], "n_samples": 3, "success_threshold": 0.5,
 "n_success": {"128": 0, "256": 1, "512": 3},
 "p": {"128": 0.0, "256": 0.33, "512": 1.0},
 "b_star": 512}

// single pass -- identical fields, one entry
{"id": "algebra/test/1423", "agent": "Qwen/Qwen2.5-1.5B-Instruct",
 "budgets": [256], "n_samples": 3, "success_threshold": 0.5,
 "n_success": {"256": 2}, "p": {"256": 0.67},
 "b_star": 256}
```

They are not equally useful, and the pipeline says so rather than pretending
otherwise:

| | classification | regression |
|---|---|---|
| full sweep | works | works |
| single pass | works — one row per problem instead of several | **refuses**, with a message |

In the single-pass case `b_star` degenerates to a solved/unsolved label rather
than an effort estimate, so regressing onto it would fit a constant.
`RegressionOracle.fit` checks for this and tells you to use classification.

Files of either shape concatenate freely as long as `n_samples` and
`success_threshold` agree. A problem missing from some budgets is fine —
classification just gets fewer rows for it.

### Implementing Stage 2

`TokenSweepEstimator.estimate_batch` raises `NotImplementedError`. Its docstring
is the spec. Summary:

1. Render the prompt from `budget.prompt_template`.
2. For each budget `B` ascending, sample `n_samples` completions with
   `max_new_tokens=B`. Decode **only the new tokens** — leaving the prompt in
   means a gold answer appearing there gets graded as the model's own output.
3. Grade each with `frugalprover.common.grading.grade(text, problem.answer)`.
4. Build the record with `BudgetRecord.from_counts(...)`, which derives `p`, the
   Wilson intervals, and `b_star` — so the τ rule lives in one place.

The runner already handles resume, per-record flushing, ordering and the
sidecar. You write one method.

Two things that silently corrupt labels: **batch by budget, not by problem**
(mixing budgets in one `generate` call pads everything to the largest), and
**check the tokenizer's padding side** before slicing off the prompt.

---

## A3 — `hidden_states.parquet`

Produced by Stage 3 (`extract`). Read by Stage 4 and the analyses. One row per
problem.

| Column | Type | Required | Meaning |
|---|---|---|---|
| `id` | str | **yes** | Join key |
| `type` | str | no | Denormalized so Stage 4 can build its one-hot from this file alone |
| `L{n}_{pooling}` | list\<float32\> | ≥1 | Pooled vector, length = hidden size |
| `L{n}_{metric}` | float32 | no | Geometry scalar |

`n` is the **absolute** layer index. Negative config indices are resolved at
extraction time, so `-1` on a 28-layer model becomes `L28_mean` — a column
literally named `L-1_mean` never exists.

- `pooling` ∈ `mean` `sum` `std` `max` `last`
- `metric` ∈ `l2_norm` `mean_token_norm` `token_norm_std` `anisotropy` `effective_rank`

Pooled columns and geometry columns are told apart by their *values* (array vs
scalar), not by parsing the suffix — so a custom extractor can invent metric
names without breaking the reader.

The sidecar records `model_name`, `hidden_size`, `num_hidden_layers`,
`prompt_template`, `max_input_tokens`, `dtype` and the full column map. A bare
parquet of 1536-dim float arrays is otherwise unidentifiable.

**To replace this stage**, write a parquet with `id` and at least one
`L{n}_{pooling}` column of equal-length arrays. A different backbone or an
embedding API drops straight in — `oracle/states/synthetic.py` is a worked
example of a non-transformer extractor.

That synthetic extractor (`extract.extractor: synthetic`) is also the **null
baseline**: with `synthetic_signal_strength: 0` it emits pure noise, and the
oracle should score at chance. If it doesn't, the score is coming from the
surface features or a leak, not from activations. Worth running once next to
any positive result.

---

## A4 — `oracle.joblib` + `metrics.json`

Produced by Stage 4 (`train`).

`oracle.joblib` is a pickled `OracleModel`. It carries its own fitted feature
blocks, so it's the whole checkpoint — there's no separate bundle of scalers and
PCA rotations to reassemble:

```python
from frugalprover import OracleModel
model = OracleModel.load("results/pilot/oracle.joblib")
```

`metrics.json` is the same information without the pickle:

```jsonc
{"mode": "classification", "layer": "L14_mean",
 "cv_metric": "auc", "cv_score": 0.71,
 "layer_scores": {"L0_mean": 0.51, "L7_mean": 0.58, "L14_mean": 0.71, "L28_mean": 0.60},
 "feature_blocks": ["surface", "subject", "activations"],
 "feature_columns": ["surface.char_len", ..., "L14_mean.pc0", ...],
 "n_problems": 80, "n_censored": 31, "budgets_seen": [128, 256, 512]}
```

### Why `layer_scores` keeps every layer

Training fits the oracle once per candidate layer and cross-validates each; the
best one is saved as the model. But **all** the scores are recorded, because the
shape of that curve is the actual finding.

One good number out of 29 tried is weak evidence — somebody had to win. A smooth
hump peaking mid-network, with neighbours agreeing, is a different claim
entirely. `plots/layer_sweep.png` draws it.

---

## A5 — `results/<run_name>/`

Produced by Stage 5 (`report`). Collects; recomputes nothing.

```
config.yaml         the exact config that trained the model
metrics.json        from Stage 4
oracle.joblib       from Stage 4
predictions.jsonl   {id, b_hat, p_by_budget_pred, mode}
plots/
  layer_sweep.png   score vs depth
  calibration.png   predicted vs actual
  budget_hist.png   B* distribution, censored included
README.md           auto-generated summary
```

`config.yaml` is written by **`train`**, not `report` — `report` is a separate
invocation with its own flags, and a config recorded there could disagree with
what actually produced the model.

`frugalprover runs` tabulates every run alongside the config keys that differ
between them.

---

## Swapping a stage

Because stages only share files, replacing one means writing its output columns.
Nothing else reruns:

```bash
# retrain with different features; stages 1-3 untouched
frugalprover train --config configs/pipeline.yaml --set 'train.features=[surface,subject]'

# reuse another run's expensive artifacts
frugalprover train --config configs/pipeline.yaml --run-name ablation \
    --set train.hidden_states=pilot/hidden_states.parquet \
    --set train.budgets=pilot/budgets.jsonl \
    --set train.problems=pilot/problems.jsonl
```

Path resolution: a bare filename resolves inside the current run's directory, a
relative path resolves against `data_dir`/`results_dir` (which is how the second
example reaches into `pilot/`), and an absolute path is used as-is.

### `run-all --skip-existing`

Skips a stage whose output file already exists — deliberately dumb, no hashing.
Two cases where it declines to skip, because skipping would silently hand back
a stale result:

- **The stage's config changed** since its output was written (compared against
  `config` in the sidecar). It says which keys differ and reruns.
- **An upstream stage just re-ran**, so this stage's inputs are new.

```
$ frugalprover run-all -c configs/pipeline.yaml --skip-existing --set train.mode=regression
[sample]  skipped - problems.jsonl already exists
[budget]  skipped - budgets.jsonl already exists
[extract] skipped - hidden_states.parquet already exists
[train]   config changed since oracle.joblib was written (mode) -- rerunning rather than skipping
[report]  rerunning - an upstream stage was re-run
```

An artifact with no sidecar (one you brought from elsewhere) is never
considered stale — there's nothing to compare against, and the contract doesn't
require a sidecar.
