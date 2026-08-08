# Run: deepseek_r1_qwen7b_n3

## Result

- mode: `classification`
- layer: `L16_mean`
- CV AUC: **0.7996**
- problems: 276 (57 censored)
- features: surface, subject, activations
- solver: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (verify_repair, n_samples=3, budgets=[1024, 2048, 4096, 8192])
- extractor: `Qwen/Qwen2.5-Math-1.5B` (activation source, separate from the solver above)

## Baselines (see `baselines.json`)

| features | CV AUC |
|---|---|
| chance | 0.500 |
| length only (`char_len`) | 0.734 |
| surface + subject (no activations) | **0.804** |
| surface + subject + activations (L16_mean) | 0.800 |

**Activations do not beat the surface+subject baseline** (0.7996 < 0.8043). Per
this repo's own bar for a result, that means this run has none to report for
activations -- consistent with the flat, humpless `layer_sweep.png` curve
(0.785-0.800 across all 29 layers).

## Files

| file | what |
|---|---|
| `config.yaml` | the exact config that produced this run |
| `metrics.json` | scores, including the full per-layer sweep |
| `predictions.jsonl` | per-problem B-hat and predicted success probabilities |
| `plots/layer_sweep.png` | score vs depth - look for structure, not just the max |
| `plots/calibration.png` | predicted vs actual |
| `plots/budget_hist.png` | B* distribution, censored problems included |

## Reproduce

```
frugalprover run-all --config configs/base.yaml --run-name deepseek_r1_qwen7b_n3
```