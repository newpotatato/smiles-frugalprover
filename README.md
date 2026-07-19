# FrugalProver

A compact multi-agent theorem-proving system that **budgets its own compute per
problem** — predicting how much effort a problem needs from a small model's
internal activations, *before* attempting to solve it, and spending accordingly.

Based on the [FrugalProver proposal](docs/proposal.pdf) by Serguei Barannikov
(Skoltech). This repository is the implementation.

## The idea

Given a fixed total compute budget `B_tot` across a set of problems, spend it
where it buys the most:

```
maximize  Σ pᵢ(Bᵢ)     subject to  Σ Bᵢ ≤ B_tot
```

where `pᵢ(B)` is the probability of solving problem `i` within budget `B`.
Uniform allocation wastes tokens on problems that were easy and starves the ones
that were nearly solved. Doing better requires knowing something about a problem
*before* solving it — which is the bet the whole system rests on.

## System

```
         ┌──────────────┐
   x ───►│ Budget Oracle│──── B̂(x) ────┐
         └──────────────┘               │
                                        ▼
                                  ┌──────────┐
                                  │  Prover  │
                                  └────┬─────┘
                                       │ candidate proof
                                       ▼
                              ┌──────────────────┐
                              │ Verifiers (k≥3)  │
                              └────┬────────┬────┘
                                   │ pass   │ fail
                                   ▼        ▼
                            ┌───────────┐  ┌───────────┐
                            │Human Gate │  │ Corrector │──┐
                            └───────────┘  └───────────┘  │
                                                 ▲        │
                                                 └────────┘
                              ┌──────────────┐
                              │ Orchestrator │  (offline: online oracle
                              └──────────────┘   updates + tactic proposals)
```

| Component | Status | Where |
|---|---|---|
| **Budget Oracle** — predicts `B̂(x)` from activations | **built** | `oracle/` |
| **Prover** — solves under a budget | protocol only | `agent/` |
| **Verifiers** — ensemble check, k≥3 | not started | — |
| **Corrector** — repair loop on failure | not started | — |
| **Human Gate** — sole sign-off on accepted proofs | not started | — |
| **Orchestrator** — online updates, tactic proposals | not started | — |

Full component breakdown, including risks and open gaps:
[references/PIPELINE.md](references/PIPELINE.md).

## Roadmap

The project is staged so each phase is a go/no-go for the next — details and
gates in [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md).

| Phase | Claim | Status |
|---|---|---|
| **0** | Harness + ground-truth `B*` labels | infrastructure built; **budget labeling not implemented** |
| **1** | **H1**: solve effort is predictable from activations | pipeline built, awaiting real labels |
| **2** | **H2**: budget-aware allocation beats uniform at matched compute | not started |
| **3** | **H3**: self-improvement without raising the false-accept rate | not started |
| **4** | Geometry of the difficulty representation | analyses built, pilot results in `docs/geometry_pilot/` |

Phase 1 is the linchpin. If effort isn't predictable from a cheap model's
activations, allocation has nothing to allocate on and Phases 2–3 don't follow.

> **The one blocking gap:** Phase 0's budget labeling (Stage 2 below) isn't
> implemented. Its interface, record schema and resumable runner all exist —
> what's missing is the generation loop. See
> [docs/ARTIFACTS.md](docs/ARTIFACTS.md#implementing-stage-2).

## What runs today: the Budget Oracle pipeline

Five stages, each independently runnable and swappable, talking only through
documented files:

```
sample ──> problems.jsonl ──┬──> budget  ──> budgets.jsonl ──┐
                            │                                ├──> train ──> oracle
                            └──> extract ──> hidden_states ──┘        │
                                                                      └──> report
```

The file contracts are in **[docs/ARTIFACTS.md](docs/ARTIFACTS.md)** — read that
before replacing a stage or bringing in data produced elsewhere.

### Setup

```bash
pip install -e .            # CPU: sampling, oracle training, analysis, reporting
pip install -e ".[gpu]"     # adds torch + transformers, needed for extraction
```

`frugalprover` lands on your PATH; `python -m frugalprover` works identically if
it doesn't.

### Quick start

```bash
frugalprover run-all --config configs/smoke.yaml
frugalprover runs
```

No GPU, no model download: budget labels are mocked and hidden states are
synthetic, with a planted signal in one layer so you can watch the layer sweep
find it. Nothing it produces is a research result — it proves the plumbing
works. The real config is `configs/pipeline.yaml`.

### A real pilot

```bash
# 1. Balanced MATH sample, stratified by subject x level
frugalprover sample --config configs/pipeline.yaml

# 2. Budget labeling -- NOT IMPLEMENTED. Mock it for now:
frugalprover budget --config configs/pipeline.yaml --set budget.estimator=mock

# 3. Hidden states (needs a GPU -- see docs/extract_hidden_states_colab.ipynb)
frugalprover extract --config configs/pipeline.yaml

# 4. Fit the oracle, sweeping every layer
frugalprover train --config configs/pipeline.yaml

# 5. Metrics, predictions and plots into results/pilot/
frugalprover report --config configs/pipeline.yaml
```

Override any key without editing the config:

```bash
frugalprover train -c configs/pipeline.yaml --set train.mode=regression
```

### The two oracle framings

| | predicts | uses censored problems | single fixed budget |
|---|---|---|---|
| `classification` (default) | `P(solved \| features, budget)` | yes, as honest zeros | works |
| `regression` | `B*` directly | no — drops them | refuses |

Censored problems — never solved at any budget tried — are exactly the hard
ones, which is why classification is the default. Regression is a cross-check
when you have a real multi-budget sweep.

### As a library

```python
from frugalprover import OracleDataset, ClassificationOracle

ds = OracleDataset.load(
    problems="data/pilot/problems.jsonl",
    hidden_states="data/pilot/hidden_states.parquet",
    budgets="data/pilot/budgets.jsonl",
)
oracle = ClassificationOracle().fit(ds)     # sweeps layers, keeps the best
print(oracle.layer, oracle.cv_score, oracle.layer_scores)
oracle.save("oracle.joblib")
```

`import frugalprover` doesn't pull in torch, transformers or datasets — the
analysis path works on a laptop with no GPU stack.

## Layout

```
src/frugalprover/
  common/         config, artifact I/O, records, grading
  agent/          Prover -- PROTOCOL ONLY, see its README
  oracle/         the Budget Oracle, as five stages
    sample/         Stage 1  problem sampling
    budget/         Stage 2  budget labeling (unimplemented + mock)
    states/         Stage 3  hidden-state extraction
    model/          Stage 4  the oracle: feature blocks, both framings
    reporting/      Stage 5  results collection, run comparison
  analysis/       research analyses (layer probe, geometry, calibration,
                  learning curve)
configs/          pipeline.yaml (real), smoke.yaml (fast, CPU, fake labels)
docs/             ARTIFACTS.md (contracts), RESEARCH_PLAN.md, proposal,
                  Colab notebooks, geometry pilot results
references/       13 cited papers + literature synthesis
slides/           LaTeX deck
```

Later components get their own top-level package next to `agent/` and `oracle/`
— `verify/`, `correct/`, `orchestrate/`. The dependency direction is one-way:
`oracle/` never imports `agent/`, and nothing imports `orchestrate/`.

## Extending it

**A new pipeline stage implementation** — a different sampler, solver, or
extractor — implements the Protocol in that stage's `base.py` and registers
itself. Nothing downstream changes, because stages only share files.

**A new system component** — verifiers, corrector — gets a package, a Protocol
in `base.py`, and a documented artifact contract in `docs/ARTIFACTS.md` before
any implementation. The contract is what makes a component replaceable.

**Bringing data from elsewhere** — match the columns in
[docs/ARTIFACTS.md](docs/ARTIFACTS.md). There's no version to satisfy and no
registration step; sidecar `.meta.json` files are written for provenance but
never read as a precondition.

## Methodology notes

Three conventions that affect how results should be read:

**`level` is never a model feature.** MATH's human difficulty annotation is used
for stratifying and plotting only. A problem in the wild carries no such label,
so training on it would inflate every score and stop the experiment answering
its own question.

**The surface-feature baseline is the bar.** Longer problems plausibly need more
tokens for reasons unrelated to reasoning difficulty. If activations don't beat
`train.features=[surface,subject]`, there is no result. Run both, compare with
`frugalprover runs`. There's a null baseline too — `--set
extract.extractor=synthetic --set extract.synthetic_signal_strength=0` replaces
activations with noise, and the oracle should score at chance.

**The pipeline's `cv_score` is exploratory.** It's the maximum over ~29 layers,
so it's optimistically biased — measure how much with the null baseline above,
at your real n and layer count. For anything going on a slide, use
`analysis/layer_probe.py`, which runs a confirmatory test at an a-priori layer
and Bonferroni-corrects the exploratory one.
[docs/ORACLE.md](docs/ORACLE.md#8-known-limitations) has the full list.

## Analyses

```bash
python -m frugalprover.analysis.layer_probe \
    --problems data/pilot/problems.jsonl \
    --hidden-states data/pilot/hidden_states.parquet \
    --budgets data/pilot/budgets.jsonl \
    --out-dir results/analysis
```

Also `geometry` (intrinsic dimension vs difficulty — Phase 4), `calibration`
(labeling cost vs oracle-guided cost), and `learning_curve` (does the oracle
improve as labels accrue — Phase 3's online-update claim).

## Reading order

1. **[docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md)** — what's being tested, as
   go/no-go phases, and what counts as a result. Start here.
2. **[references/PIPELINE.md](references/PIPELINE.md)** — the full system:
   every node, its technology, its risks.
3. **[docs/ORACLE.md](docs/ORACLE.md)** — how the Budget Oracle is implemented:
   features, both framings, layer selection, CV protocol, known limitations.
4. **[docs/ARTIFACTS.md](docs/ARTIFACTS.md)** — the file contracts between
   pipeline stages.
5. **[references/MANIFEST.md](references/MANIFEST.md)** — how each cited paper
   maps to a component.

## Data note

Hidden-state dumps aren't committed — pooled vectors at every layer for a few
hundred problems run to tens of MB, and the raw equivalent exceeded GitHub's
file limit. Regenerate with `frugalprover extract`, or get the parquet from
whoever ran it. Runs are reproduced from the `config.yaml` in their results
directory.

## Authors

Created during **SMILES 2026**.

| Author | Affiliation | Contact |
| --- | --- | --- |
| Michael Leontiev | HSE University | [michlea@yandex.ru](mailto:michlea@yandex.ru) |
| Kirill Chernikov | ITMO University | [kmchernikov@itmo.ru](mailto:kmchernikov@itmo.ru) |
| Nika Smirnova | AIRI | [nika646470@gmail.com](mailto:nika646470@gmail.com) |
| Alima Chekueva | — | — |

Based on the FrugalProver proposal by Serguei Barannikov (Skoltech).
