# The Budget Oracle — implementation

How the oracle is actually built, as of `frugalprover` 0.2.0.

This is the *implementation* document. For the file formats between stages see
[ARTIFACTS.md](ARTIFACTS.md); for what the experiment is testing and why, see
[RESEARCH_PLAN.md](RESEARCH_PLAN.md); for the oracle's place in the larger
system, [../references/PIPELINE.md](../references/PIPELINE.md).

---

## 1. What it computes

The oracle is a function

```
f_θ( φ(x), g_ℓ(x) )  ->  B̂(x)
```

- `x` — a problem statement, text only. No solution, no attempt.
- `φ(x)` — cheap surface features of the text (length, LaTeX density, …).
- `g_ℓ(x)` — the pooled hidden state of a **small** model at layer `ℓ`, from a
  single forward pass. Nothing is generated.
- `B̂(x)` — predicted token budget needed to solve it.

The load-bearing property is that `g_ℓ` comes from a *cheap* model while `B*` is
defined against the *prover*. If that cross-model transfer doesn't hold, the
whole approach collapses — this is the "linchpin" flagged in RESEARCH_PLAN §0,
and it is not yet tested, because Stage 2 is unimplemented.

## 2. Inputs

`OracleDataset` ([`model/dataset.py`](../src/frugalprover/oracle/model/dataset.py))
is the only module that knows what the artifact files look like. It inner-joins
three files on `id`:

```python
OracleDataset.load(
    problems="problems.jsonl",              # A1: text, answer, type, level
    hidden_states="hidden_states.parquet",  # A3: L{n}_{pooling} vectors
    budgets="budgets.jsonl",                # A2: b_star, p(B) -- optional
)
```

The join is loud about what it drops. Extracting states for 300 problems and
labeling 80 gives you 80 — reported at load time. A silent mismatch (an id
format change, two different sampling runs) would otherwise produce a
plausible-looking number from twelve problems.

Accessors the rest of the code uses: `activations(layer)` → `(n, hidden_size)`,
`b_star()` → `(n,)` with NaN for censored, `groups()` → problem ids for CV,
`swept_budgets()`, `pooled_columns()`, `solved_subset()`.

Pooled vectors and geometry scalars are distinguished by **inspecting the cell
type** (array vs scalar), not by parsing the column suffix — so a custom
extractor can invent metric names without breaking the reader.

## 3. Features

Composable blocks in
[`model/features.py`](../src/frugalprover/oracle/model/features.py). Each maps
the dataset to an `(n_problems, k)` matrix; the oracle horizontally stacks
whichever the config names. Blocks are *fitted objects* — they hold their
scalers and PCA rotations and are pickled with the model, which is what makes
`predict` apply exactly the transform training used.

| Block | Columns | What it is |
|---|---|---|
| `surface` | 7 | char/word length, LaTeX command count, `$` count, brace depth, digit count, `=` count. Standardized. |
| `subject` | n_types + 1 | One-hot over MATH subject, plus a reserved `__other__` |
| `activations` | ≤ `n_pcs` | Standardize `g_ℓ`, then PCA |
| `geometry` | n_metrics | The `L{n}_{metric}` scalars (norms, anisotropy, effective rank) |

Default stack: `[surface, subject, activations]`.

**Why `surface` stays in by default.** It's the confound, not a nicety. Longer
problems plausibly need more tokens for reasons unrelated to reasoning
difficulty. Keeping it in the model means the activation block has to earn its
place *on top of* length — and running `features=[surface,subject]` alone gives
the number that has to be beaten.

**Why PCA is not optional.** Hidden vectors are 1536-dimensional; there are ~80
labeled problems. Anything fitted on raw activations interpolates the training
set and generalizes to nothing. `n_pcs` defaults to 10, guarded to
`min(n_pcs, n_samples - 1, n_features)`.

**Why `__other__` is reserved at fit time.** Training typically runs on the small
labeled subset while prediction runs on the full pool, so a subject unseen
during training is expected. Without the reserved column, such a problem gets an
all-zero one-hot that looks like valid input; with it, "unknown subject" is an
explicit feature.

**There is deliberately no `level` block.** MATH's human difficulty annotation
is available in the data and is used for stratification and plots, but never as
a model input — a problem in the wild has no such label, so training on it would
inflate every score and answer a different question than the one being asked.

## 4. The two framings

Both subclass `OracleModel`
([`model/base.py`](../src/frugalprover/oracle/model/base.py)).

### 4.1 Classification (default)

[`model/classification.py`](../src/frugalprover/oracle/model/classification.py).
Models

```
P(solved | φ(x), g_ℓ(x), B)
```

**Row expansion.** One training row per *(problem, budget)* pair, with
`log(B)` appended as a feature:

```
row(i, B) = [ φᵢ ‖ subjectᵢ ‖ PCA(g_ℓ(xᵢ)) ‖ log B ]
label      = 1{ B ≥ B*ᵢ }
group      = xᵢ.id
```

20 problems × 3 budgets = 60 rows.

`log B` rather than `B` because budgets are geometric (128/256/512) and success
is roughly linear in log-compute; on the raw scale a linear model would have to
bend a straight line.

**Censoring is free.** A problem never solved contributes label 0 at every
budget. No imputed `B*`, no dropped rows.

**Final estimator:** `LogisticRegressionCV(Cs=logspace(-3,3,13), cv=5)`, which
tunes regularization internally. (Note `Cs` is *inverse* regularization, the
opposite of Ridge's `alphas`; the grid is symmetric in log space so it spans the
same models, but don't read a chosen `C` as an `alpha`.)

**Inference.** `B̂(x)` is the smallest swept budget whose predicted probability
clears τ:

```python
B̂(x) = min{ B ∈ budgets_seen : P̂(solved | x, B) ≥ τ }     # NaN if none clear it
```

NaN means "more than any budget I've seen" — the model declining to extrapolate.

### 4.2 Regression

[`model/regression.py`](../src/frugalprover/oracle/model/regression.py). Predicts
`B*` directly with `RidgeCV(alphas=logspace(-3,3,13))`.

Simpler to read, but it pays twice:

- **Censored problems are unusable** — no target to regress onto, so they're
  dropped. Since censored problems are exactly the hard ones, the model trains
  on a systematically easy subset. `min_solved_for_regression` (default 15)
  refuses to fit when too few remain.
- **It needs a real sweep.** With one budget, every solved problem has the same
  `B*` and the fit is a constant. `fit()` checks `len(swept_budgets) > 1` and
  raises with a message pointing at classification.

**Inference.** `predict_budget` clips to `[min, max]` of the swept budgets — a
linear model will otherwise extrapolate to negative token counts on an easy
problem, and outside the swept range there's no evidence either way.

`predict_success(ds, B)` is a hard 0/1 step (`B̂ ≤ B`), not a probability. This
framing predicts a point, not a distribution; if you need calibrated
probabilities, use classification.

### 4.3 Choosing between them

| | classification | regression |
|---|---|---|
| target | `P(solved \| ·, B)` | `B*` |
| rows | n_problems × n_budgets | n_solved |
| censored problems | used as zeros | dropped |
| single fixed budget | works | refuses |
| metric | AUC | R² |
| calibrated probabilities | yes | no |

Classification is the default and the one that matches the research plan's own
formalization in terms of `pᵢ(B)`. Regression is a cross-check.

## 5. Layer selection

`OracleModel.select_layer`. Candidates are every pooled column in the parquet
(`L0_mean … L28_mean` with `all_layers: true`).

```
for each candidate layer ℓ:
    build features with ActivationPCA reading ℓ   (surface/subject unchanged)
    cross-validate a fixed, simple estimator
    record the score
pick argmax, then refit the final tuned estimator there
```

Only the activation block varies between candidates, so score differences are
attributable to the activations.

**Two different estimators, deliberately.** The sweep scores with plain
`LogisticRegression` / `RidgeCV` — fast, fixed, a consistent yardstick across
~29 comparisons. The final fit uses the CV variants that tune regularization.
Tuning inside the sweep would confound "this layer is better" with "this layer
drew a luckier hyperparameter."

**Every score is kept**, not just the winner, in `layer_scores` →
`metrics.json` → `plots/layer_sweep.png`. See §8 for why that matters.

Pin the sweep with `train.layers: [L14_mean]` to skip it entirely.

## 6. Cross-validation

**Classification: `LeaveOneGroupOut`, grouped by problem id.** Non-negotiable.
A problem's rows at 128/256/512 share identical activations; splitting them
across train and test lets the model see the same features on both sides and the
AUC becomes fiction.

**Regression: leave-one-out below 60 problems, else 5-fold** (`cv: auto`). At
n=80 a 5-fold split trains on 64 and tests on 16, and the variance of that
estimate swamps the effect being measured. LOO uses every problem as a test case
exactly once.

Predictions come from `cross_val_predict`, so every scored prediction comes from
a model that did not see that problem.

## 7. What gets persisted

`joblib.dump(model)` — one object. The fitted feature blocks live on the model,
so there is no separate bundle of scalers and rotations to reassemble:

```python
from frugalprover import OracleModel
model = OracleModel.load("results/pilot/oracle.joblib")
model.predict_budget(new_dataset)
```

`metrics.json` carries the same information without the pickle: `mode`, `layer`,
`cv_metric`, `cv_score`, `layer_scores`, `feature_blocks`, `feature_columns`,
`n_pcs`, `n_train`, `budgets_seen`, `success_threshold`, `n_problems`,
`n_censored`.

The sidecar `oracle.joblib.meta.json` additionally stores the resolved
`train` config, which is what lets `run-all --skip-existing` notice that the
training settings changed and rerun instead of returning a stale model.

## 8. Known limitations

Read any result with these in mind.

**`cv_score` is optimistically biased.** It's the maximum over ~29
cross-validated estimates. Picking the best and reporting it as *the* score
means the winner is partly winning on noise, and the bias grows with the number
of candidates swept. At n=80 that is not a small effect.

To measure it rather than assume it, run the null baseline — `--set
extract.extractor=synthetic --set extract.synthetic_signal_strength=0` replaces
activations with pure noise, so the reported "best layer" score is entirely
selection bias. Do it at the same n and the same layer count as your real run;
the smoke config's 4 layers and 20 problems are too small to say anything.

This is why the full sweep is retained — a single peak is weak evidence, a
smooth curve peaking mid-network with neighbours agreeing is a real claim. For
anything going on a slide, use
[`analysis/layer_probe.py`](../src/frugalprover/analysis/layer_probe.py), which
runs a **confirmatory** test at an a-priori layer (the last one, per Ansuini et
al.) plus a Bonferroni-corrected **exploratory** test at the argmax.

**The final model is fit on all data.** By design — it's the deployed artifact,
not the evaluation. `cv_score` is the honest number; the fitted coefficients are
not held out from anything.

**`B̂` is quantized to swept budgets** in classification mode. With
`[128, 256, 512]` there are four possible outputs including NaN. Finer
allocation needs a finer sweep, which costs linearly in Stage 2.

**Small n, high dimension.** Everything here is regularized linear models on
~80 problems because that's what the sample size supports. `compare_models: true`
fits MLP and Gaussian Process alternatives under the same CV — the research
plan's rule is to prefer the simplest model unless a complex one wins by more
than ~0.02.

**Grading is string comparison.** `common/grading.py` normalizes both sides and
compares. It under-counts algebraically-equal-but-textually-different answers,
which biases `p(B)` down roughly uniformly — acceptable for *ranking* problems
by effort, worth remembering before quoting an absolute accuracy.

**Untested end-to-end on real labels.** Every number produced so far came from
`budget.estimator: mock`. The oracle machinery is verified; the science is not.

## 9. Configuration

```yaml
train:
  mode: classification            # | regression
  layers: all                     # | [L14_mean]
  features: [surface, subject, activations]   # | + geometry
  n_pcs: 10
  cv: auto                        # | loo | <int>
  compare_models: false
  min_solved_for_regression: 15
  problems: problems.jsonl
  budgets: budgets.jsonl
  hidden_states: hidden_states.parquet
  output: oracle.joblib
  metrics: metrics.json
```

## 10. Extending it

**A new feature block** — implement `fit`/`transform`/`column_names`, register
it in `build_blocks`. It becomes available by name in `train.features`.

**A new framing** — subclass `OracleModel`, implement `fit`, `score`,
`predict_budget`, `predict_success`; register in `ORACLES`. `select_layer` and
the CV splitter come free.

**A different signal source** — nothing here cares that the vectors come from a
transformer. Write a parquet with `id` and one `L{n}_{pooling}` column per A3
and it works;
[`states/synthetic.py`](../src/frugalprover/oracle/states/synthetic.py) is a
worked example of a non-transformer extractor.

## 11. File map

| File | Role |
|---|---|
| `model/base.py` | `OracleModel` ABC, layer sweep, CV splitter, save/load, `summary()` |
| `model/dataset.py` | `OracleDataset` — the artifact join, the only format-aware module |
| `model/features.py` | Feature blocks and the `build_blocks` factory |
| `model/classification.py` | `ClassificationOracle` — row expansion, grouped CV |
| `model/regression.py` | `RegressionOracle` — solved-subset fit, clipping, guards |
| `model/compare.py` | Model-family comparison under identical CV splits |
| `model/__init__.py` | `run_train` — Stage 4 entry point, writes A4 |
