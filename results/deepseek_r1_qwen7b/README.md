# Run: deepseek_r1_qwen7b

## Result

- mode: `regression`
- layer: `L6_mean`
- CV R2: **0.117**
- problems: 100 (52 censored)
- features: surface, subject, activations
- solver: `unknown`

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
frugalprover run-all --config configs/pipeline.yaml --run-name deepseek_r1_qwen7b
```