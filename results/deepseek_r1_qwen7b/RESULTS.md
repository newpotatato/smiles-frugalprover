# Budget Oracle — run `deepseek_r1_qwen7b`

End-to-end pilot on **real** budget labels from `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
(100 MATH problems, 2048-token cap), with oracle features from the small sibling
**DeepSeek-R1-Distill-Qwen-1.5B** (all 29 layers, mean pooling).

## Data provenance

- **Budgets** ([`data/deepseek_r1_qwen7b/budgets.jsonl`](../../data/deepseek_r1_qwen7b/budgets.jsonl)):
  derived from an external token-usage run. `b_star = completion_tokens` when the
  generation finished naturally; **censored** (`b_star = null`) when it hit the 2048
  cap. **48 solved / 52 censored.** The source has no answer grading, so "finished
  within the cap" is the solved proxy — good for *ranking* effort, not for absolute
  solve accuracy.
- **Problems**: reconstructed from `EleutherAI/hendrycks_math`, matched by
  `(type, level, exact char_len)` → `input_tokens` (chat-templated length) →
  `response_text` overlap, then re-keyed to canonical `{subject}/{split}/{index}` ids
  (`id_map.json` keeps the original `math_*` mapping). **100/100 matched.**

## Results (grouped 5-fold / LOO CV, `frugalprover runs`)

| run | framing | features | score | best layer |
|---|---|---|---|---|
| `deepseek_r1_qwen7b`          | regression (B*)          | surface+subject+**activations** | **R² = +0.117** | L6_mean |
| `deepseek_r1_qwen7b_baseline` | regression (B*)          | surface+subject                 | R² = −0.156 | — |
| `deepseek_r1_qwen7b_null`     | regression (B*)          | **noise** + surface+subject     | R² = −0.026 | — |
| `deepseek_r1_qwen7b_clf`      | classification P(solv\|B)| surface+subject+**activations** | AUC = 0.888 | L5_mean |
| `deepseek_r1_qwen7b_clf_baseline` | classification       | surface+subject                 | AUC = 0.932 | — |

## Reading

- **Regression shows a weak but real signal.** Predicting B* directly, the 1.5B
  activations reach R² = +0.117, clearly above the surface/subject length baseline
  (−0.156) and above the pure-noise null (−0.026, i.e. the layer-sweep selection-bias
  floor at this n). The peak sits **mid-network (L5–L6)**, where difficulty
  representations are expected — not at an edge layer, which is mild evidence it's real.
  But it's small (explains ~12% of B* variance, n = 48 solved).
- **Classification shows no activation benefit.** There, `log B` is itself a feature
  and mechanically predicts success, so surface+subject already reaches AUC 0.932;
  adding activations *drops* it to 0.888. The high AUC is the budget axis, not the
  problem representation.
- **Bottom line (H1 = "solve effort is predictable from cheap activations"):** weak,
  framing-dependent support. Positive in the regression framing that actually asks the
  H1 question; absent in classification. Not yet a confident result.

## Caveats

1. **fp16 extraction overflowed at depth.** Layers L25–L27 were dropped from the sweep
   (mean-pooled fp16 → inf). The winning layer (L6) is unaffected, but a clean final
   number — and the deep-layer / `layer_probe` analyses — need a **float32 (or bf16)
   re-extract**. Re-run the Colab notebook with `--set extract.dtype=float32`.
2. **Small n.** 100 problems / 48 solved. `cv_score` is the max over ~26 usable layers,
   so it's optimistically biased; the null run brackets that bias near zero here.
3. **`layer_probe` not applicable as-is.** Its accuracy@budget section assumes a uniform
   budget grid shared across problems; this encoding gives each solved problem its own
   natural budget, so that section errors (`KeyError: '2048'`). The pipeline's own layer
   sweep + null baseline cover the selection-bias check.
4. **Solved = natural finish proxy** (no answer grading in the source).

## Reproduce (CPU, from the committed parquet)

```bash
frugalprover train  -c configs/pipeline.yaml --run-name deepseek_r1_qwen7b --set train.mode=regression
frugalprover report -c configs/pipeline.yaml --run-name deepseek_r1_qwen7b
# baselines: --set 'train.features=[surface,subject]' ; --set train.mode=classification ;
# null: extract --set extract.extractor=synthetic --set extract.synthetic_signal_strength=0
```
