# FrugalProver — Research Plan

Full reach (**H1 → H2 → H3** + geometry/topology) on a **compute ramp**: cheap
early phases, more resources as the premise proves out. The whole system rests
on one empirical bet — **H1: a small model's internal activations can predict
how much effort a larger prover will need** — so the work is organized as
**go/no-go phases**: each phase's result decides whether the next is worth
running. We do not build allocation (H2) or self-improvement (H3) until H1
clears.

This is the scientific plan (what we test, how, and what counts as a useful
result). For how to actually run each piece, see the top-level
[README](../README.md) and the scripts under `experiments/` and `pipeline/`.
The slide deck (`slides/frugalprover.tex`) is the presentation-level version of
this same phase structure.

---

## 0. Guiding logic

- **The linchpin is cross-model prediction.** The budget `B*` is defined w.r.t.
  the *prover* (which writes proofs), but the predictive signal `g_ℓ(x)` comes
  from a *cheap small model* that only reads the statement. H1 is really: *can a
  small model's representation of a problem statement predict the large prover's
  required budget?* This is the non-obvious claim and the main failure mode —
  the first thing we try to falsify.
- **Cheapest-first.** The H1 probe study needs `B*` labels and a small model,
  not the full multi-agent loop. Build the minimum needed to generate labels,
  run H1, then scale only if it works.
- **Deconfounding is the common denominator.** No result counts without a
  comparison to a real baseline (surface features `φ` and length) and without
  checks across independent methods or layers. A single significant number on a
  single layer with a single estimator is not a finding.
- **Everything logged.** Full traces, per-agent token counts, seeds, and
  model/version hashes from day one — the basis of later "matched-compute"
  comparisons and of governance (H3).

---

## Phase 0 — Harness + ground-truth `B*` labels *(cheap)*

**Objective.** For a stratified subset of MATH, produce the empirical budget
`B*ᵢ` a successful attempt warrants — the regression target for H1.

**Define `B*` operationally (pick one, pre-register it):**
- *Threshold (recommended).* Estimate the success-vs-budget curve `pᵢ(B)` by
  sampling `n` completions at increasing token caps; set
  `B*ᵢ = min{ B : p̂ᵢ(B) ≥ τ }` with `τ = 0.5`, denoised by a logistic fit in
  `log B`. Report a **Wilson confidence interval** on each `p̂ᵢ(B)` so the
  label noise at small `n` is explicit rather than hidden behind a point
  estimate.
- *First-correct (cheaper, noisier).* `B*ᵢ` = median token length of the first
  correct sample. Smoke-test only; upgrade to the threshold definition before
  trusting H1.

**Cost-tiered labeling (how the pilot actually realizes this).** Activations are
free (one forward pass) and scale to hundreds of problems; a full budget sweep
is expensive. So the pool is split into tiers and joined by problem id, never
double-labeling the same work:
- `activations_only` — `φ(x)` + `g_ℓ(x)` at **all layers**, no generation.
  Runs on the whole large pool (~300 problems).
- `full_sweep` — the honest `B*` / `pᵢ(B)` with Wilson CIs, on a smaller
  **labeling subset (~80)** whose ids are a strict subset of the large pool.
- `baseline_single` — one budget × several samples: a cheap solvability signal
  at larger scale than `full_sweep`.

`pipeline/merge_results.py` joins tiers by id, so one problem can carry
activations from tier 1 and `B*` from tier 2 in a single record.

**Models (cheap tier).**
- *Prover:* GLM-4.5-Air via vLLM (thinking / non-thinking logged separately);
  drop to a 7–14B model for the first end-to-end smoke test if GPU is tight.
- *Small activation model `g_ℓ`:* a small **dense** model — GLM-4-9B or
  Qwen2.5-7B — that only encodes the statement.
  *(The current pilot run used `Qwen2.5-1.5B-Instruct` for the
  `activations_only` tier; the choice of small model is an open decision, see
  §Open decisions.)*
- Keep the multi-agent loop minimal (a single answer-check is enough to *label*;
  the full k≥3 verifier + corrector loop matters for H2/H3, not for `B*`).

**Deliverable.** A labeled table: `(problem, φ(x), g_ℓ(x) per layer, B*, solved?,
full trace)`.

**Pitfalls.** `B*` is stochastic (sample enough, report Wilson CIs);
de-duplicate near-identical problems to avoid train/test leakage; log tokens for
*every* agent so later matched-compute comparisons are honest.

---

## Phase 1 — H1: effort is predictable *(cheap; the go/no-go)*

**Objective.** Show `g_ℓ` (+ surface features `φ`) predicts `B*` substantially
better than a length baseline, and that `Î(g_ℓ; B*) > 0`.

**Features.**
- `φ(x)` — surface features on the statement:
  `char_len, word_len, latex_cmd_count, dollar_count, brace_depth, digit_count,
  eq_count`, plus subject and level tags.
- `g_ℓ(x)` — the small model's hidden state at layer `ℓ`, pooled (last-token
  and/or mean over statement tokens), PCA-compressed; **`ℓ` is swept over all
  layers**, not pre-picked.

**Predictor `f_θ` — two framings.**
- *Regression* on `log B*` (Ridge / ElasticNet / MLP / Gaussian Process). Simple,
  but `B*` is censored (undefined when a problem is never solved).
- *Classification (recommended)* — model `P(success | φ, g_ℓ, budget)` with
  budget as an input feature, trained on all `(problem, budget)` pairs. This
  matches the paper's own `pᵢ(B)` formalization and sidesteps censoring; `B̂(x)`
  is then the smallest budget whose predicted success clears the threshold.
- *Default rule:* prefer the simplest model (Ridge / Logistic) unless a more
  complex one beats it by more than ~0.02 CV metric — complexity must earn its
  place. A Gaussian Process is kept because its calibrated uncertainty is useful
  for capping conservatively.

**The core comparison table.**

| # | Features | Purpose |
|---|---|---|
| 1 | Length only | weakest baseline |
| 2 | Surface `φ` only | are cheap hand features enough? |
| 3 | Activations `g_ℓ` only | signal in representation alone |
| 4 | **Combined `φ + g_ℓ`** | the hypothesis |
| 5 | + informative-layer selection | which `ℓ`, chosen honestly (below) |
| 6 | Ceilings | MATH-level label (leaky upper bound); prover's **own** activations (self-prediction upper bound — the price of using a cheap probe instead of the prover itself) |

**Deconfounding discipline (this is what makes a number a finding).**
- *Stronger baseline than length:* `φ + TF-IDF` bag-of-words on the problem
  text. Lexical cues ("prove that…", "find the number of…") already leak a lot
  of difficulty, so this is the honest bar for activations to beat.
- *Confirmatory vs exploratory layer, to keep multiple comparisons honest:*
  - **confirmatory** — the **last** hidden layer, chosen a priori from Ansuini
    et al.'s headline finding; tested at `α = 0.05`, no cherry-picking penalty.
  - **exploratory** — the argmax layer over the full sweep, reported with a
    **Bonferroni-adjusted** `α = 0.05 / n_layers`, since "best of N noisy
    estimates" inflates false positives.
- *Is any improvement real at n ≈ 40–80 with thousands of activation dims?*
  Answered four ways: cross-validated `R²` (LeaveOneOut for small n, grouped by
  problem id for the classification framing so a problem's rows don't leak
  across folds); a bootstrap 95% CI on `ΔR²` (augmented − baseline); a
  permutation test (shuffle activations across problems); and a partial
  correlation (residualize `B*` and the activation PC1 on the baseline features)
  as a practical stand-in for `I(g_ℓ; B* | φ)`.

**Metrics.** MAE/RMSE on `log B*`, calibration, Spearman rank; `Î(g_ℓ; B*)`
estimated after dimensionality reduction and compared against a
**label-shuffled control** with CIs (MI estimators are biased in high
dimension — the shuffle control is essential); downstream, cap the prover at
`B̂(x)` and measure tokens saved vs accuracy retained.

**What counts as a useful result.** Not only significant/not: the **shape** of
`R²(layer)` is itself informative — monotone rise to the end, a mid-depth hump
(à la Ansuini), or structureless noise are three different slide stories. If the
confirmatory layer shows nothing and only an exploratory layer does but fails
Bonferroni, that is an **honest negative** — reportable, not a prompt to fish
for another layer.

**✅ Gate.** Proceed to H2 only if `φ + g_ℓ` beats length **and** the
success-vs-compute curve, with `Î` clearly above the shuffle control. Otherwise:
a clean, publishable negative result — the premise fails here, cheaply.

**Pitfalls.** MI estimator bias (reduce dim, use controls); cross-model mismatch
(the small model may not "see" difficulty like the prover — the self-prediction
ceiling quantifies this); label noise propagating into an apparent error floor.

*Implemented in the pilot:* `experiments/B_layers/analyze.py` (layer sweep +
baselines + confirmatory/exploratory diagnostics), `experiments/A_oracle/`
(`oracle.py` architectures, `calibration_cost.py`, `learning_curve.py`).

---

## Phase 2 — H2: allocation improves efficiency *(medium)*

**Objective.** Use `B̂(x)` to allocate a **fixed total** budget better than
uniform/length, raising the *entire* success-vs-compute curve.

**Method.** Fit per-problem `pᵢ(B)` (logistic in `log B`) from Phase 0 samples;
allocate greedily by marginal gain `dp/dB` under `Σ Bᵢ ≤ B_tot` (the
knapsack/marginal-gain rule). Run with **predicted `B̂`** and, as reference,
**oracle `B*`** — the gap is the cost of prediction error.

**Baselines.** Uniform budget; length-based budget; repeated sampling /
self-consistency at matched compute; oracle-disabled fixed cap; reduced verifier
count.

**Also validated here:** the multi-agent loop — ablate verifier count
(k = 1 / 3 / 5) and corrector on/off; track the wrongly-accepted-proof rate vs
compute.

**Metrics.** Problems-solved-per-token (area under success-vs-compute);
tokens-to-first-valid-solution; sweep `B_tot` to confirm the *whole* curve
lifts, not one operating point.

**Scale-up.** Full MATH; introduce GLM-4.5 (355B) as prover if resources allow;
keep the small model as the cheap probe.

**✅ Gate.** Budget-aware allocation should Pareto-dominate uniform and length
across the sweep. If predicted-`B̂` collapses toward the oracle line, the probe
is production-worthy.

**Pitfalls.** Distribution shift between the set used to fit `pᵢ(B)` and the eval
set; "matched compute" must count *all* agent tokens (prover + verifiers +
corrector), not just prover output.

*(Not yet implemented — conditional on H1 clearing.)*

---

## Phase 3 — H3: safe self-improvement *(more resources)*

**Objective.** Over rounds, online tuning + human-approved tactic updates raise
problems-solved-per-token **without** raising the false-accept rate at the gate.

**Design.**
- **Rounds:** update `f_θ` online as outcomes accrue; an orchestrator proposes
  tactics, **sandbox-tested** before deployment; a **human gate** is the sole
  sign-off.
- **A/B:** sandbox-validated tactic adoption vs a frozen control policy.
- A **frozen audited eval set** carried across all rounds for apples-to-apples
  comparison.

**Metrics.** Efficiency (problems-solved-per-token, round-N vs round-0);
**safety (load-bearing):** wrongly-accepted-proof rate at the gate over rounds —
**pre-register a threshold**; the claim is that it *does not rise*; improvement
magnitude across rounds.

**Governance.** Log every tactic change and gate decision; risk-based / sampled
human auditing; reproducible toolkit; human-audited case study on well-known
tractable problems.

**Pitfalls.** Online overfitting of the predictor (regularize; keep the frozen
eval honest); **verifier gaming** — prover and verifiers from the same family
may collude, so use diverse verifiers and periodic adversarial audits; gate
throughput vs coverage.

*(Not yet implemented — conditional on H2.)*

---

## Phase 4 — Geometry & topology of the difficulty representation *(parallel)*

Runs alongside once Phase 0 activations exist — analysis, not a gate. It asks
whether the **geometry** of the activation manifold (its dimension and density),
not just its raw predictive power from Phase 1, is meaningfully tied to
difficulty — and whether that tie is robust to the choice of estimator.

- **Intrinsic dimension by difficulty level.** Three estimators, computed per
  MATH level (1–5) on the confirmatory (last) and exploratory (middle) layers:
  - **TwoNN** (Facco et al. 2017) — nonlinear, robust to non-uniform density.
  - **MLE / Levina–Bickel** — nonlinear, an independent derivation.
  - **PCA participation ratio** — a linear effective rank.
  Agreement between the two *independent nonlinear* estimators (TwoNN, MLE) is
  far stronger evidence than any single method; where they diverge in direction,
  we say so in the output rather than pick the prettier one.
- **Per-problem local density.** Mean distance to the `k` nearest neighbors in
  activation space, per problem (vs aggregate per-level ID). Spearman-correlated
  with difficulty **level** over **all** records — geometry needs no budget
  labels at all, so this runs on the full `activations_only` pool.
- **Link to budget (the H1-relevant half).** A separate Spearman of local
  density with accuracy@max-budget, restricted to the `full_sweep` tier — does
  the geometry track actual **solvability**, not just the human level label?
  *This half is currently blocked on Phase 0 `full_sweep` labels.*
- **Online accumulation.** Following the paper's "updated online as outcomes
  accrue": fix a hold-out, simulate problems arriving in random order, and check
  whether the oracle's AUC grows with `n` — i.e. whether the compressed geometry
  becomes *more* legible as data accrues, or is just noisy raw signal.
- **Exploratory topology.** Persistent-homology / barcode descriptors of
  activation clouds stratified by difficulty. Clearly labeled exploratory (no
  implementation yet).

**What counts as a useful result.** TwoNN-vs-MLE agreement is a result on its
own (how stable is ID at this n and in this space?). At n ≈ 40–80 per level the
aggregate per-level ID is the noisiest piece; per-problem local density (which
uses the whole n) is the more reliable signal at this scale, and the script says
so. A growing *or* flat learning curve is presentable: growth supports online
updating, a plateau is an honest "signal saturates early / n still too small".

*Implemented in the pilot:* `experiments/C_geometry/id_vs_difficulty.py`,
`src/frugalprover/id_estimators.py` (TwoNN + MLE).

---

## Current status (honest read)

First real `activations_only` run: **100 MATH problems, `Qwen2.5-1.5B-Instruct`,
all layers** — a Phase 4 (geometry) early signal, since Phase 0 `B*` labels do
not exist yet.

- **Layer 14 (middle):** local density vs difficulty **level** —
  Spearman `ρ = −0.298, p = 0.003`. Harder problems sit in a denser manifold
  mid-forward-pass.
- **Layer 28 (last):** `ρ = −0.113, p = 0.262` — not significant.

Takeaway for the probe: **middle layers look most promising.** Three caveats,
per the discipline above, keep this from being oversold:
1. This is density vs **difficulty level** (a human label), **not** vs `B*` —
   the cross-model budget claim at the heart of H1 has **not** been tested yet
   (it needs the `full_sweep` tier from Phase 0).
2. The signal is at the **exploratory** (middle) layer; the a-priori
   **confirmatory** (last) layer showed nothing. Report it as exploratory.
3. No baseline comparison yet — whether mid-layer density beats `φ + TF-IDF` is
   still to be run.

Phase 1 proper (predicting `B*`) and the oracle-architecture comparison
(`experiments/A_oracle/`) are waiting on Phase 0 labels.

---

## Compute ramp

| Phase | Models | Data | Compute |
|---|---|---|---|
| 0 Harness + labels | Air (or 7–14B) prover + 7–9B activation model | ~1–2k stratified MATH | Single / few GPUs |
| 1 H1 probe | same | same subset | Cheap — probe training is tiny |
| 2 H2 allocation | GLM-4.5 prover + small probe | Full MATH | Medium (sampling + sweeps dominate) |
| 3 H3 rounds | full pipeline (k≥3 verifiers + corrector) | MATH + olympiad | More (rounds × sampling) |
| 4 Geometry/topology | activations from above | reuse Phase 0–2 activations | Parallel; scales with activation-set size |

---

## Minimal viable experiment (run this first)

On the ~1–2k stratified MATH subset with the cheap tier: generate `B*` labels
(Phase 0), then run the H1 core table — `φ + g_ℓ` vs the length baseline on
prediction error plus `Î(g_ℓ; B*)` vs the shuffle control. This single
experiment falsifies-or-confirms the whole premise for the least compute;
everything else is conditional on it.

## Open design decisions to lock before starting

1. Exact `B*` definition (threshold `τ` vs first-correct) — pre-register.
2. Which small model supplies `g_ℓ` — same family as the prover (representational
   alignment) or a different family (tests transfer). The current pilot used
   `Qwen2.5-1.5B-Instruct`; the config track also explores the base
   `Qwen2.5-Math-1.5B`. Settle on one and justify it.
3. Pooling for `g_ℓ` (last-token vs mean-pool) and the candidate layers for the
   sweep.
4. Pre-registered false-accept threshold for H3.
