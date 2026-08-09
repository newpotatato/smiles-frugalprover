# H2 — budget-aware allocation at matched compute

2× RTX 4090, DeepSeek-R1-Distill-Qwen-7B served by vLLM (`--data-parallel-size 2`),
one **seeded** sample per problem. Four arms solved the *same* 200 held-out MATH
problems with the *same* agent under the *same* total budget
`B_tot = 2896 × 400`; only the split differed. 1.97M generated tokens, 48 min.

Eval problems are drawn from the 4320 in `data/label5h/problems.jsonl` that the
sweep never labelled, so they are genuinely out-of-sample for the oracle.
`agent.seed_mode: per_problem` means a problem follows the same trajectory in
every arm until its own cap truncates it — the comparison is paired, and the
difference between arms is the budget, not the sampling.

## Result

| arm | solved | rate | 95% Wilson | tokens spent | solved / 1M tok | cap util |
|---|---|---|---|---|---|---|
| `uniform` | 122/200 | 0.610 | [0.541, 0.675] | 465,535 | **262.1** | 0.80 |
| `length` | 120/200 | 0.600 | [0.531, 0.665] | 485,892 | 247.0 | 0.88 |
| `oracle_greedy` | 126/200 | 0.630 | [0.561, 0.694] | 514,154 | 245.1 | 0.90 |
| `oracle_triage` | **127/200** | **0.635** | [0.566, 0.699] | 505,021 | 251.5 | 0.88 |

Paired McNemar against `uniform` — problems exactly one arm solved:

| arm | arm only | uniform only | p |
|---|---|---|---|
| `length` | 18 | 20 | 0.871 |
| `oracle_greedy` | 19 | 15 | 0.608 |
| `oracle_triage` | 21 | 16 | 0.511 |

## Reading it

**The direction is right and the magnitude is not resolvable at n=200.** Both
oracle arms beat uniform (+4 and +5 problems, +2.0 and +2.5 points) and `length`
does not — exactly the ordering the offline simulation predicted. But the
confidence intervals overlap almost completely and no paired test comes close to
significance. The simulation predicted +4.3 points; we measured +2.5 with a
standard error around ±3. **This run is consistent with the predicted effect and
equally consistent with no effect.** Separating them needs roughly 800–1000
problems, not 200.

**At matched *spent* compute, the oracle advantage disappears.** This is the
finding that matters most, and it is only visible because the runner records
both numbers. Every arm was *allotted* the same `B_tot`, but the oracle arms
*used* more of it — 0.90 cap utilisation against uniform's 0.80 — because
concentrating budget on problems predicted to be hard puts tokens exactly where
attempts run long instead of stopping early. So the 5 extra problems cost 40k
extra tokens, and on problems-per-token uniform is ahead of every other arm
(262 vs 245–252 per million). RESEARCH_PLAN's own pitfall note ("matched compute
must count *all* agent tokens") turns out to bite in a subtler way than
expected: the arms were matched on the budget *offered* and differed on the
budget *taken*.

**Abstention was free.** `oracle_triage` declined 5 problems outright, and
`uniform` solved **0** of them. That is the one unambiguous positive here: on
this sample the oracle identified genuinely hopeless problems without
sacrificing a single solvable one. It is also 5 problems, so it is an
encouraging anecdote rather than a result.

**The verifier behaved.** Acceptance 43–52% across arms, above the healthy
20–50% band and nowhere near the never-accepts failure mode.

## What the offline simulation predicted

On the 276 labelled problems with out-of-fold predictions from this same oracle
(`L16_mean`, CV AUC 0.7997), at this operating point (of 276 problems):

| arm | vs uniform |
|---|---|
| `true_bstar` (ceiling) | +54.7 |
| `oracle_triage` | +12.0 |
| `oracle_greedy` | +11.7 |
| `oracle_optimal` | +11.3 |
| `length` | −10.3 |
| `oracle_bhat` | −11.0 |

Ceiling 216.7 against uniform's 162.0, of which ~9.3 is label noise rather than
real headroom (at `b_bar = max(grid)` no allocation should beat uniform, and the
ceiling does — that gap bounds the artifact).

The live run reproduces the *sign* of every one of these and about **half** the
magnitude for the oracle arms. Simulation scores an allocation against measured
`p(B)` under the labelling distribution (τ = 0.5 of 3 samples); the live run is
one seeded sample per problem against a fresh agent. Half is a reasonable
attenuation for that gap, not evidence the simulation was wrong.

## Provenance

- Oracle: `results/h2/metrics.json` — refit on the pod from freshly extracted
  hidden states. The extraction reproduced the lost original to within 1e-4
  (fresh CV AUC 0.7997 vs the committed 0.7996 at the same layer, `L16_mean`),
  which is what licenses comparing this run to `deepseek_r1_qwen7b_n3`.
- Activations still do **not** earn their place: surface+subject alone predicts
  +11.3 problems in simulation against +12.0 with activations, mirroring the AUC
  ablation (0.8043 vs 0.7996). Both are in
  `results/analysis/` (surface only) and `analysis_activations/` (with).
- `alloc_rows.jsonl` is the per-(arm, problem) record without the candidate
  solution texts, which stayed on the pod. Everything in the tables above
  recomputes from it via `frugalprover.allocate.metrics.summarize`.

## Reproducing

```bash
frugalprover allocate -c configs/base.yaml \
    -c configs/agent/DeepSeek_R1_Distill_Qwen_7B_vllm.yaml \
    -c configs/agent/DeepSeek_R1_Distill_Qwen_7B_vllm_2gpu.yaml \
    -c configs/allocate/h2_matched_compute.yaml --run-name h2 \
    --set allocate.problems=h2/problems.jsonl \
    --set allocate.hidden_states=h2/hidden_states.parquet \
    --set allocate.oracle=h2/oracle.joblib \
    --set allocate.exclude_labeled=label5h/budgets.jsonl \
    --set allocate.chunk_size=50 --set allocate.max_problems=400
```

`docs/allocate_runpod.ipynb` is the full pod runbook, including the extraction
that has to happen *before* vLLM claims the cards.

**If you rerun this, change one thing: give it more problems.** Keep
`max_problems=400` (it fixes `B_tot` and therefore every arm's caps, so the run
resumes into the same experiment) and raise `time_budget_s` — at the 585 tok/s
this pod sustained, 200 problems cost 48 minutes and n=800 would need about
three hours.
