# H2 offline: allocation policies scored on the `label5h` labels

Produced by

```bash
python -m frugalprover.analysis.allocation_sim \
    --problems data/label5h/problems.jsonl \
    --budgets  data/label5h/budgets.jsonl \
    --b-bar 2896 --sweep-triage --out-dir results/analysis
```

276 problems, grid `[1024, 2048, 4096, 8192]`, τ=0.5, n=3. Predicted curves are
**out-of-fold** (LeaveOneGroupOut by problem id, the same protocol as the
pipeline's `cv_score`); they are scored against each problem's *measured* `p(B)`
from the A2 record. No GPU was involved: a budget sweep already measured what
every problem does at every budget, so an allocation can be scored without
re-solving anything.

Features are **surface + subject only**. `data/label5h/hidden_states.parquet`
was lost with the pod that produced it (`.gitignore` excludes `data/*/`), so the
activation version of this table has to be regenerated on a GPU — that is §2–3
of `docs/allocate_runpod.ipynb`. Note that the committed run's own ablation puts
surface+subject *ahead* of activations on AUC (0.8043 vs 0.7996,
`results/deepseek_r1_qwen7b_n3/baselines.json`), so this is the stronger of the
two predictors, not a fallback.

## Expected problems solved, of 276

| `b_bar` | uniform | length | `oracle_bhat` | `oracle_greedy` | `oracle_triage` | ceiling |
|---|---|---|---|---|---|---|
| 1024 | 111.0 | 111.0 | 111.0 | 111.0 | 111.7 | 158.0 |
| 2048 | 147.0 | 130.7 | 142.3 | 144.7 | 145.3 | 204.0 |
| 2233 | 150.0 | 134.0 | 144.0 | 153.7 | 154.7 | 208.0 |
| **2896** | **163.6 ± 2.3** | 151.7 | 150.3 | 173.0 | **174.0** | 216.7 |
| 4096 | 189.3 | 177.7 | 177.7 | 191.0 | 190.7 | 216.7 |
| 8192 | 207.3 | 203.7 | 207.3 | 207.3 | 206.3 | 216.7 |

Triage threshold swept at `b_bar=2896`: 0.1→173.3, 0.2→173.3, 0.3→173.3,
**0.4→174.0**, 0.5→172.0. Flat then falling — abstention pays until it starts
declining problems that were winnable.

## Reading it

**A gain exists, and it is small.** `oracle_triage` beats uniform by 11.3
problems out of 276 (+7%) at the best operating point, capturing 21% of the gap
to the ceiling. `uniform` there is a mean over 12 tie-break seeds (σ=2.3,
max=166.7), so this is roughly 4σ — an effect, not a lucky draw.

**It lives off-grid, and that is not an artifact.** At `b_bar` on the grid
(1024, 2048, 4096, 8192) every arm can afford the same cap for every problem and
the arms come within noise of each other; at 2048 uniform actually wins. Off
grid the total does not divide evenly and a policy must decide *who* gets the
spare step — which is both where the oracle earns its keep and the realistic
case, since a real compute budget is not `n` times a power of two. **A live
experiment run at an on-grid `b_bar` cannot answer its own question.**

**`length` loses everywhere**, which is the result that matters most: the
deconfounder the README sets as the bar is not what is driving the oracle arms.

**`oracle_bhat` loses too.** Budget ∝ B̂ ignores the shape of the success curve
and overspends on problems predicted hard that were never going to be solved.
The marginal-gain rule (`oracle_greedy`) and abstention (`oracle_triage`) are
what work, and they are within 0.3 problems of each other and of the exact
knapsack solution — so the approximation is not what is limiting this, the
predictor is.

**The ceiling is inflated.** At `b_bar = 8192` no allocation should be able to
beat uniform, because everyone can already afford the largest budget. The
ceiling gains 9.3 problems anyway, by banking cases where measured
`p(2048) > p(8192)` through pure n=3 sampling noise. So of the 54.7-problem
headroom above uniform, perhaps 45 is real. Still large: a *better* oracle is
worth building, which is the useful half of this negative-ish result.

## Files

- `allocation_sim.json` — the full `b_bar` sweep, every arm, plus the verdict
  block and the triage sweep.
- `allocation_sim.png` — success-vs-compute, all policies, ceiling dashed.
