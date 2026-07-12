# FrugalProver — literature context & manifest

Links the 13 sources from `docs/proposal.pdf` to the components of the proposed
system (budget oracle, prover, verifiers, corrector, human gate, orchestrator)
and to the three hypotheses (H1 effort is predictable, H2 budgeting pays off,
H3 self-improvement is safe).

## Cluster 1 — Geometry of internal representations (backs H1: "effort is predictable")

This is the only cluster not directly tied to LLMs/mathematics — it supplies the
measurement method the budget oracle is built on: the hypothesis that "the
reasonable amount of effort can be estimated from internal activations" reduces
formally to estimating the *intrinsic dimension* (ID) of the activation manifold.

| # | File | What it shows | Role in FrugalProver |
|---|---|---|---|
| 1 | `01_Ansuini_IntrinsicDimension_NeurIPS2019.pdf` | The ID of hidden-layer representations in CNNs forms a characteristic "hunchback" curve over network depth; the ID of the last hidden layer correlates with test classification accuracy. | Direct support for G2 ("characterize the geometry/topology of the internal representation of problem difficulty"): if last-layer ID predicts classification success, an analogous ID↔difficulty link may underpin the budget oracle. |
| 2 | `02_Facco_TwoNN_SciReports2017.pdf` | Introduces the TwoNN estimator — ID of a dataset from just the distances to the 1st and 2nd nearest neighbors, robust to curvature and non-uniform density. | The concrete tool G2 uses to measure the "intrinsic dimension" of the activations gℓ(x) — this method is cited as a *technique*, not merely as motivation. |

## Cluster 2 — Benchmark and compute-scaling method (backs H2: "budgeting is more efficient")

| # | File | What it shows | Role in FrugalProver |
|---|---|---|---|
| 3 | `03_Hendrycks_MATH_NeurIPS2021.pdf` | Introduces the MATH dataset — 12,500 competition-math problems with step-by-step solutions; shows that simply scaling model size barely helps on MATH (unlike most NLP tasks). | The main validation benchmark (Datasets/Metrics, §1.3). Also motivates the project itself: since parameter scale doesn't solve the task, you need smart compute control — a budget oracle, not just "a bigger model". |
| 7 | `07_Snell_ScalingTestTimeCompute_ICLR2025.pdf` | Shows that optimal (adaptive, per-problem) allocation of test-time compute yields up to 4× efficiency over best-of-N; sometimes offsets a 14× model-size gap. | Direct source of the "Formulation" — max ΣBᵢ pᵢ(Bᵢ) s.t. Σ Bᵢ ≤ Btot — which is essentially the same adaptive-compute-allocation problem this paper solves (cited as [7]). The core of hypothesis H2. |
| 8 | `08_Wang_SelfConsistency_ICLR2023.pdf` | Self-consistency: sample several reasoning chains and pick the answer by majority vote instead of greedy decoding; a large accuracy gain on arithmetic/common-sense. | One of the baselines in the validation protocol ("repeated sampling / self-consistency at matched compute", cited as [8]) — what budget-aware allocation is compared against. |

## Cluster 3 — propose→verify→repair pipelines (the prover/verifiers/corrector architecture)

| # | File | What it shows | Role in FrugalProver |
|---|---|---|---|
| 4 | `04_Lightman_LetsVerifyStepByStep_ICLR2024.pdf` | Process supervision (step-by-step checking of reasoning) substantially beats outcome supervision; a PRM trained on 800K step labels (PRM800K) solves 78% of a MATH subset. | Direct precursor of the "skeptical verifiers": the idea of step-level, not just final-answer, checking of a proof — the methodological basis of the system's verification layer. |
| 5 | `05_Madaan_SelfRefine_NeurIPS2023.pdf` | SELF-REFINE: the same LLM generates, critiques, and rewrites its own answer iteratively, with no fine-tuning; ~20% gains across tasks. | Precursor of the "corrector": the feedback→refine pattern — how the model repairs its own proof gaps until the verifiers concur. |
| 12 | `12_Zelikman_STaR_NeurIPS2022.pdf` | STaR: bootstrapping reasoning ability — the model generates rationales, fine-tunes on the successful ones, iteratively improving chain quality without a large labeled dataset. | Basis for the "self-improving" part of the system (adopting new tactics, G3) — iterative self-training on the model's own successful trajectories. |

## Cluster 4 — Reasoning techniques and agency (the "prover" itself + tool use)

| # | File | What it shows | Role in FrugalProver |
|---|---|---|---|
| 9 | `09_Wei_ChainOfThought_NeurIPS2022.pdf` | Chain-of-thought prompting: a few step-by-step reasoning demonstrations in the prompt sharply improve reasoning in sufficiently large models. | The base technique for generating candidate proofs in the prover; the foundation almost all other methods in clusters 3–4 build on. |
| 10 | `10_Yao_TreeOfThoughts_NeurIPS2023.pdf` | Tree-of-Thoughts: generalizes CoT to a tree of intermediate "thoughts" with selection, backtracking, and lookahead during a global search for a solution. | An alternative/extension of the proof-generation strategy — guided search instead of linear generation, applicable to the prover when exploring tactics. |
| 11 | `11_Yao_ReAct_ICLR2023.pdf` | ReAct: interleaves verbal reasoning with actions (tool/environment calls), a foundation for many later LLM agents. | Backs "tool use" (cited as [6] to Toolformer, but ReAct is the methodological analog) and the overall agentic architecture that reaches for external resources (e.g. a formal proof checker). |
| 6 | `06_Schick_Toolformer_NeurIPS2023.pdf` | Toolformer: the model self-teaches when and how to call external APIs (calculator, search, etc.) in a self-supervised way. | The direct [6] "tool use" reference in the Introduction — justification that the prover/corrector may use external tools (e.g. symbolic solvers) within a budget. |

## Cluster 5 — Backbone model

| # | File | What it shows | Role in FrugalProver |
|---|---|---|---|
| 13 | `13_Zeng_GLM4.5_arXiv2025.pdf` | GLM-4.5 — an open MoE model (355B / 32B active) with a hybrid "thinking/direct response" mode, strong on agentic/reasoning/coding benchmarks (incl. AIME, SWE-bench). | The concrete open-weight backbone (cited as [13]) intended to run the whole pipeline — satisfying the abstract's "runs entirely on open-weight models" condition. |

## How the clusters map to the paper's three hypotheses

- **H1 (effort is predictable from activations)** → Cluster 1 (Ansuini, Facco/TwoNN) supplies the measurement method; Cluster 2 (MATH, Snell) supplies the empirical arena to test it.
- **H2 (budgeting beats uniform/length-based)** → Snell (adaptive compute scaling) is the direct methodological predecessor; Wang (self-consistency) and Hendrycks (MATH) are the comparison baselines.
- **H3 (self-improvement is safe under a human gate)** → Zelikman (STaR, bootstrapping), Madaan (Self-Refine, iterative correction), Lightman (process supervision as a per-step check that lowers the false-accept risk).

## Note on parsing completeness

The first 5 files (Ansuini, Facco, Hendrycks, Lightman, Madaan) were parsed in
full via `documentor` (structure, sections, tables/figures cached in
`.documentor/`). For the other 8, this manifest uses annotations from
authoritative sources (arXiv/NeurIPS/ICLR abstract pages) — they can be read in
full too if needed.
