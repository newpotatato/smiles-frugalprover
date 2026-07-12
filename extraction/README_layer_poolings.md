# layer_poolings

A pilot for turning MATH problems into **feature vectors extracted from a language
model's hidden states**, so downstream models can be trained cheaply (e.g. to predict
problem *hardness* / `level`) without running full generation.

The pipeline samples problems from the MATH benchmark, runs each through a HuggingFace
model with `output_hidden_states=True`, and collapses the per-token hidden states at
chosen layers into:

- **pooled vectors** — one `(hidden_size,)` vector per `(layer, pooling)` entry
- **geometry scalars** — cheap shape statistics of the per-token "token cloud" per `(layer, metric)` entry

Everything is written to a single Parquet file, one row per problem.

## Layout

| Path | What it is |
|---|---|
| [`generate_hidden_states.ipynb`](generate_hidden_states.ipynb) | The pipeline. Runs on a Colab GPU: install → config → load model → sample MATH → extract → save Parquet. |
| [`configs/config.example.yaml`](configs/config.example.yaml) | Annotated template config. Documents every field, pooling type, and geometry metric. |
| [`configs/Qwen1_5B_math_mid_mean.yaml`](configs/Qwen1_5B_math_mid_mean.yaml) | Config tuned for predicting problem hardness with `Qwen2.5-Math-1.5B` (mid/late/final layers). |
| [`../data/Qwen1_5B_math_mid_mean.parquet`](../data/Qwen1_5B_math_mid_mean.parquet) | Output produced by the config above: 200 algebra problems × 3 pooled + 6 geometry features. |
| [`WORKING_WITH_PARQUET.md`](WORKING_WITH_PARQUET.md) | How to load the Parquet and use it (schema, stacking pooled columns into matrices, probes, caveats). |

## Quick start

1. Open [`generate_hidden_states.ipynb`](generate_hidden_states.ipynb) in Colab and set the runtime to **GPU** (`Runtime → Change runtime type → GPU`).
2. Pick a config:
   - Edit the inline `CONFIG` dict (cell 4), **or**
   - Set `USE_YAML = True` and upload a YAML (e.g. `configs/Qwen1_5B_math_mid_mean.yaml`) to `YAML_PATH`.
3. Run all cells. The final cell writes `CONFIG["output_path"]` (a `.parquet`).
4. Load and use the result — see [`WORKING_WITH_PARQUET.md`](WORKING_WITH_PARQUET.md).

## Configuration

Config lives in a YAML file (or the inline `CONFIG` dict). Key fields:

- `model_name` — HuggingFace model id, loaded with `AutoModel` (decoder-only base models work best for math semantics).
- `dataset_name` / `dataset_config` / `dataset_split` / `num_problems` — MATH source and sampling.
- `features` — list of `{layer, pooling}`. `pooling ∈ {mean, std, max, last, sum}`.
- `geometry` — list of `{layer, metric}`. `metric ∈ {l2_norm, mean_token_norm, token_norm_std, anisotropy, effective_rank}`.
- `batch_size`, `max_input_tokens` (input truncation only — nothing is generated), `dtype`, `device`.

`layer` indexes `output_hidden_states`: `0` = embeddings, `1..N` = transformer block
outputs, negatives count from the end (`-1` = last layer). See
[`config.example.yaml`](configs/config.example.yaml) for the fully annotated reference.

## Output schema (brief)

Each row is one problem. Columns:

- `problem`, `level`, `type` — metadata (`level`/`type` may be `null`).
- `L{layer}_{pooling}` — a **column of arrays**; each cell is a `(hidden_size,)` vector.
- `L{layer}_{metric}` — a plain scalar column.

The pooled columns hold arrays, not scalars — stack them with `np.stack(df[col])`
before any linear algebra. Full details, gotchas, and example probes are in
[`WORKING_WITH_PARQUET.md`](WORKING_WITH_PARQUET.md).
