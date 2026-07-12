# FrugalProver — detailed breakdown of the pipeline, calls, and technologies

A synthesis of the proposal paper + the 13 literature sources. Structure:
end-to-end data flow → node-by-node breakdown (purpose / required technologies /
key challenges / what the literature provides) → cross-cutting project risks →
final stack → what the sources do NOT cover (open gaps).

---

## 1. End-to-end data flow

```
problem x (theorem / problem statement)
        │
        ▼
┌───────────────────┐
│   BUDGET ORACLE O  │  ← φ(x) surface features + gℓ(x) small-model activations
│  B̂(x) = fθ(φ,gℓ)  │
└─────────┬──────────┘
          │ budget B̂(x) (tokens)
          ▼
┌───────────────────┐
│      PROVER        │  generates a candidate proof under the cap B̂(x)
└─────────┬──────────┘
          │ candidate proof
          ▼
┌───────────────────┐
│  VERIFIERS (k≥3)   │  an ensemble of skeptical checkers
└─────────┬──────────┘
          │ concur? ──no──┐
          │ yes            ▼
          │         ┌─────────────┐
          │         │  CORRECTOR   │  repairs the gaps found
          │         └──────┬───────┘
          │                │ (back to verifiers, loop until they concur
          │◄───────────────┘  or the budget is exhausted)
          ▼
┌───────────────────┐
│    HUMAN GATE       │  the single point of manual sign-off
└─────────┬──────────┘
          ▼
   accepted proof

In parallel, offline:
┌───────────────────┐
│   ORCHESTRATOR      │  proposes new tactics → sandbox-test → deploy
└────────────────────┘
```

Formally, the system solves a resource-allocation problem:
**max Σᵢ pᵢ(Bᵢ)** s.t. **Σᵢ Bᵢ ≤ B_tot**, where pᵢ(B) is the probability of
solving problem i within B tokens. The budget oracle predicts not the fact of a
solution but how many tokens to allocate for the greatest marginal return.

---

## 2. Node by node

### 2.1 Budget Oracle

**Purpose.** Before each backbone call — estimate how many tokens the problem
"deserves" and set a cap on generation.

**Required technologies:**
- A small open model to extract gℓ(x) — activations at some intermediate layer ℓ
  on the problem statement (without a full solution, i.e. a forward pass, not
  generation).
- A surface-feature extractor φ(x): statement length, presence of special
  notation / LaTeX constructs, a heuristic subject/topic classifier.
- A method to compress/analyze the activation geometry — an **intrinsic
  dimension** estimate (the **TwoNN** estimator, Facco et al. 2017). Transformer
  activations have thousands of dimensions and a raw regression over them
  overfits easily; ID gives a compact summary robust to curvature.
- A light trainable head fθ (a probe: linear / shallow-MLP regression) on top of
  (φ, gℓ) → B̂(x), updated online as outcomes accrue.

**Key challenges:**
1. **Choice of layer ℓ and pooling** of activations — non-trivial; the wrong
   choice can give a signal no stronger than text length.
2. **Deconfounding against a length baseline.** H1 explicitly requires showing
   *additional* information: I(gℓ; B*) > 0 beyond what statement length already
   gives. If this is not shown rigorously, the whole point of "prediction from
   activations" collapses.
3. **Chicken-and-egg in labeling.** To train fθ to predict the "minimally
   sufficient budget" B*, you need (problem, B*) examples — and to get B* for one
   problem you must run the solver at several budget levels and find the smallest
   successful one. This is expensive and itself consumes the very budget the
   project is trying to save.
4. **Generalization gap.** The calibration dataset is MATH (competitive but still
   "school" olympiad math); the target "harder olympiad problems" may have a
   different difficulty distribution, and activations calibrated on MATH need not
   transfer.
5. **MI estimation in high dimension** — an open statistical problem in itself,
   not solved explicitly by any of the 13 sources (see §4).

**What the sources provide:**
- *Ansuini 2019* — an empirical precedent: the ID of a CNN's last hidden layer
  predicts test classification accuracy. A direct analogy for H1
  (representation ID/geometry ↔ success) — but carried over from computer vision
  to LLMs/math; the domain differs, so relying on a direct transfer is risky.
- *Facco/TwoNN 2017* — the concrete ID-measurement tool, robust at a small number
  of points and on curved manifolds (i.e. suitable even for a modest number of
  calibration problems).
- *Snell 2024* — the formalization of "where the next token is best spent" — this
  is the work giving the mathematical frame max ΣBᵢpᵢ(Bᵢ) that the paper cites
  directly (reference [7]).

---

### 2.2 Prover

**Purpose.** Generate a candidate proof within the allotted budget.

**Required technologies:**
- An open-weight backbone with a "thinking mode" — GLM-4.5 / GLM-4.5-Air (Zeng et
  al. 2025), able to switch between a "thinking" and a direct-answer mode.
- The base generation technique: **Chain-of-Thought prompting** (Wei et al.
  2022) — step-by-step reasoning via few-shot demonstrations.
- Optionally — guided search instead of linear generation: **Tree-of-Thoughts**
  (Yao et al. 2023) — branching, self-evaluation of intermediate "thoughts",
  backtracking. Especially valuable under a hard budget: instead of spending the
  whole cap on one linear path that may hit a dead end, you can explore several
  branches and stop at the best one when the budget runs out.
- External tool calls: **Toolformer**-style self-supervised tool use (Schick et
  al. 2023) or **ReAct**-style "reason → act → observe" interleaving (Yao et al.
  2023, ICLR) — for a calculator, symbolic algebra, possibly integration with a
  formal assistant (Lean/Coq/Isabelle) for the subclass of problems where formal
  verification is reachable.
- A generation-capping mechanism tied to B̂(x) — length control at the decoding
  level.

**Key challenges:**
1. **A single budget vs the strategy for spending it.** Given a fixed B̂(x),
   what's better: one long attempt, several short parallel ones with a vote
   (**self-consistency**, Wang et al. 2022 — a baseline in the validation
   protocol), or tree search (ToT)? This is not decided in advance — it's exactly
   what the baseline experiment is meant to show.
2. **The solver stays offline** ("keeping the solver offline until an
   independent verifier validates each attempt") — the prover is not tuned via RL
   during the project, so all quality gains must come through
   prompting/tactics (the orchestrator), not through weights. This strongly caps
   the achievable quality within two weeks.
3. **Tool calls eat budget** — calling an external tool also costs tokens to
   format the request / parse the response; a naive integration can wipe out the
   savings.

**What the sources provide:** Wei (CoT — the base), Yao/ToT (search),
Schick/Toolformer and Yao/ReAct (tools), Snell and Wang (what to compare against
at a fixed compute).

---

### 2.3 Verifiers (ensemble, k≥3)

**Purpose.** Skeptically check a candidate before it reaches the
corrector/human gate.

**Required technologies:**
- Several independent checker instances: either one backbone with different
  prompts/seeds, or (ideally) a separate **process reward model (PRM)** — a model
  that checks *every step* of the reasoning, not just the final answer.
- An aggregation rule for the k≥3 verifiers' decisions (majority? unanimity?
  weighted vote?).
- A separate verification strategy for two problem types: automatic final-answer
  checking (MATH, where the answer is known) vs verifying the reasoning chain
  itself (open olympiad problems with no numeric "answer").

**Key challenges:**
1. **Step-level vs final-answer checking.** *Lightman et al. 2024* show
   empirically that process supervision substantially beats outcome supervision
   (78% of a MATH subset solved vs a markedly weaker outcome-only result). But
   their PRM is trained on **PRM800K** — 800K human-labeled steps. Such a dataset
   can't be built in two weeks, so FrugalProver will almost certainly have to use
   a *prompt-instructed* LLM judge instead of an honestly-trained PRM — a
   qualitatively weaker, noisier verification signal.
2. **Correlation of "independent" verifiers.** If all k≥3 verifiers are the same
   backbone with different seeds, their errors are correlated (the same model
   blind spots), and the ensemble offers far less protection than the nominal k
   suggests.
3. **Ensemble cost.** Each extra verifier is tokens subtracted from the total
   budget; a growing k directly conflicts with the "problems-solved-per-token"
   metric.
4. **Non-terminating loop.** "Loop iterates until verifiers concur" — an explicit
   cap on the number of iterations is needed, or the prover↔corrector↔verifiers
   loop may not converge within the problem's budget.

**What the sources provide:** Lightman/PRM — the direct methodological basis;
indirectly, Snell/Wang give the "verification/voting as an axis of test-time
compute" context.

---

### 2.4 Corrector

**Purpose.** Locally fix the gaps the verifiers found, without regenerating the
proof from scratch.

**Required technologies:**
- A **feedback → refine** loop in the spirit of **Self-Refine** (Madaan et al.
  2023): the same model that generated the candidate uses the critique (ideally
  from the verifiers, localized per step) and rewrites the problematic parts.
- A structured feedback channel from the verifiers: the feedback must point to a
  **specific step**, not just say "wrong" — otherwise the corrector is no
  different from regenerating from scratch.

**Key challenges:**
1. *Madaan et al.* show that most of the gain from iterative refine is in the
   first 1–2 iterations, after which returns drop sharply. So it's sensible to
   cap the corrector's iterations low rather than loop "to the bitter end" —
   which directly contradicts the original paper's "loop iterates until verifiers
   concur" phrasing unless an explicit limit is added.
2. **Remaining budget.** If the prover has used most of B̂(x), the corrector may
   lack tokens for a meaningful repair — logic is needed to reserve part of the
   budget for correction up front.
3. **Oscillation.** Without a convergence guarantee, the corrector may "fix" one
   error while introducing another, bouncing between wrong versions.

**What the sources provide:** Madaan/Self-Refine — the direct algorithmic
precursor (their Algorithm 1 literally describes a generate→feedback→refine loop
with a stop condition — exactly what the corrector needs).

---

### 2.5 Human Gate

**Purpose.** The single point of manual review for candidates that passed all
verifiers.

**Required technologies (engineering, not described in the sources directly):**
- An interface to view the candidate + the history of its checking/correction.
- Full logging of the entire path of a problem through the pipeline (stated
  explicitly in Goal G1 — "with complete logging") — needed for auditing and for
  later analysis of where the system most often breaks.

**Key challenges:**
1. The "wrongly accepted proofs at the gate" metric implies that *even a human*
   can miss an error — especially on hard olympiad problems, where the reviewer
   needs high-level expertise, not just attentiveness.
2. The human gate is a scaling bottleneck: if the verifier ensemble works poorly,
   too many candidates reach the human, and "the single point of human sign-off"
   stops being practical as the problem stream grows.
3. A validation protocol via "well-known but tractable problems" (G3) is needed —
   i.e. control problems with independently checkable ground truth, without which
   you can't calibrate how reliable the human gate itself is.

---

### 2.6 Orchestrator (self-improvement)

**Purpose.** In parallel with the main flow — propose new solving/checking
tactics, test them in a sandbox, and deploy them if they pass validation.

**Required technologies:**
- A versioned library of tactics/prompts.
- An isolated sandbox environment for testing a new tactic on a held-out problem
  subset before deployment.
- A "sandbox-validated" statistical criterion — a significance threshold for the
  improvement, or it's easy to accept a noise improvement as real.
- Bootstrapping in the spirit of **STaR** (Zelikman et al. 2022): successful,
  already-verified solution trajectories become new few-shot examples/tactics for
  future attempts — removing the need for a large pre-labeled reasoning dataset.

**Key challenges:**
1. **Reward hacking / verifier gaming.** This is perhaps the most serious risk
   not fully closed by the sources: if the orchestrator optimizes tactics against
   the same verifier ensemble the main pipeline uses, the system risks learning
   not to "solve better" but to "pass the check more convincingly" — i.e. to
   exploit the verifiers' blind spots. Neither STaR nor Self-Refine explicitly
   considers this adversarial regime (both assume benign, non-adversarial
   feedback).
2. H3 explicitly frames this as a hypothesis to test ("the human gate prevents
   any rise in the rate of wrongly accepted proofs") but gives no ready mechanism
   to ensure it — an open engineering question: do you need rotation/diversity of
   verifiers, periodic human spot-checks of tactics, or something else?
3. **The two-week horizon.** A full "propose a tactic → sandbox-test on a
   statistically meaningful sample → human approval → deploy → measure the
   effect" cycle is itself a multi-day process; fitting several such rounds into
   two weeks alongside the rest of the system is a very tight schedule.

**What the sources provide:** Zelikman/STaR (bootstrapping), Madaan/Self-Refine
(the iterative-improvement mechanism); but, as noted, none solves the safety of a
self-improving generator against its own verifier.

---

## 3. Cross-cutting (project-level) challenges

1. **Timeline density.** Two weeks for the harness + budget predictor +
   allocation experiments (week 1) and for self-improvement + geometry/topology
   analysis (week 2) — when even a single component (e.g. an honest PRM à la
   Lightman) required hundreds of thousands of labeled examples in the original
   work. A realistic scale is a heavily trimmed version of each component.
2. **Backbone compute cost.** GLM-4.5 is 355B parameters (32B active, MoE); even
   the compact GLM-4.5-Air is 106B. Running the full pipeline (prover + k≥3
   verifiers + corrector, possibly over several rounds per problem) on such models
   within the project's available compute is a non-trivial logistical
   constraint, separate from the algorithmic questions.
3. **Circular dependency in budget labeling** (see 2.1, point 3) — the data to
   train the oracle itself requires "extra" budget to collect.
4. **The definition of the "minimally sufficient budget" B*** is ambiguous: the
   solve probability p(B) is usually not a step function but a smoothly rising,
   noisy one (partly due to sampling randomness) — so defining the target
   variable for fθ requires statistical smoothing / multiple runs.
5. **Mixing of problem domains.** The representation-geometry cluster
   (Ansuini/Facco) is from computer vision (CNNs, images), whereas the
   application is language models and mathematical text. Transferring the
   "hunchback" ID-vs-depth intuition from CNNs to theorem-proving transformers is
   an extrapolation not directly confirmed by any of the sources.

---

## 4. What the sources do NOT cover (open gaps)

- **Mutual-information estimation** I(gℓ; B*) in high dimension — the paper
  frames this as part of H1, but none of the 13 sources provides a ready robust
  MI estimator for this specific setting (TwoNN gives ID, not MI).
- **The adversarial generator-vs-verifier dynamic** during self-improvement (see
  2.6, point 1) — not covered by STaR, Self-Refine, or Lightman's PRM work (all
  three assume a benign/non-manipulative generator).
- **Formal verification** (Lean/Coq/Isabelle integration) is mentioned in the
  paper only indirectly via "tool use" (Toolformer/ReAct), but none of the
  sources describes a concrete integration of an LLM prover with a formal math
  assistant — a separate body of work (Metamath/GPT-f and the like, absent from
  the reference list though indirectly mentioned as related work in the MATH
  paper).
- **Calibration of the probability estimate** p(B) and the statistical
  robustness of measuring it (sampling noise, multiple testing when choosing the
  "minimal successful budget") — a methodological issue not addressed directly by
  any of the works.

---

## 5. Final technology stack

| Layer | Technology | Source |
|---|---|---|
| Backbone LLM | GLM-4.5 / GLM-4.5-Air (open weights, thinking mode) | Zeng et al. 2025 |
| Reasoning generation | Chain-of-Thought prompting | Wei et al. 2022 |
| Reasoning search (opt.) | Tree-of-Thoughts | Yao et al. 2023 (NeurIPS) |
| Tool calls | Toolformer / ReAct pattern | Schick et al. 2023; Yao et al. 2023 (ICLR) |
| Voting/sampling (baseline) | Self-Consistency | Wang et al. 2022 |
| Budget allocation | Compute-optimal test-time scaling formalization | Snell et al. 2024 |
| Verification | Process-level checking (PRM approach) | Lightman et al. 2023 |
| Self-correction | Self-Refine (feedback→refine loop) | Madaan et al. 2023 |
| Self-improvement | STaR tactic bootstrapping | Zelikman et al. 2022 |
| Representation geometry | Intrinsic Dimension analysis | Ansuini et al. 2019 |
| ID estimator | TwoNN | Facco et al. 2017 |
| Benchmark | MATH dataset + harder olympiad problems | Hendrycks et al. 2021 |
| Infrastructure (not from sources) | tactic sandbox, full logging, human-review interface, harness for multi-round runs | — |
