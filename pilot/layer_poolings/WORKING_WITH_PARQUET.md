# Working with the hidden-states Parquet

This file is produced by [`generate_hidden_states.ipynb`](generate_hidden_states.ipynb)
(see cell 7, `df.to_parquet(...)`). Default name: `hidden_states_dataset.parquet`
(set by `CONFIG["output_path"]`).

Each **row = one MATH problem**. Columns fall into three groups: metadata, pooled
hidden-state vectors, and geometry scalars.

## Schema

| Column | Type | Notes |
|---|---|---|
| `problem` | `str` | The problem text (after `prompt_template` was applied). |
| `level` | `str` / `null` | MATH difficulty, e.g. `"Level 3"`. `null` if the source lacked it. |
| `type` | `str` / `null` | MATH subject, e.g. `"Algebra"`. `null` if absent. |
| `L{layer}_{pooling}` | `list<float>` | **Pooled vector**, one per `features` entry. Each cell is a 1-D array of length `hidden_size`. |
| `L{layer}_{metric}` | `float` | **Geometry scalar**, one per `geometry` entry. One number per row. |

Column names come directly from the config (cell 12):

- pooled: `f"L{layer}_{pooling}"` → e.g. `L3_max`, `L5_last`, `L-1_mean`
- geometry: `f"L{layer}_{metric}"` → e.g. `L3_l2_norm`, `L5_anisotropy`, `L-1_effective_rank`

`layer` indexes `output_hidden_states`: `0` = embeddings, `1..N` = transformer block
outputs, negatives count from the end (`-1` = last layer). A `-1` in the name is
literal text, not an error.

> **Key gotcha:** pooled columns are **columns of arrays**, not scalars. Every cell in
> `L3_max` holds a full `(hidden_size,)` vector (768, 896, … depending on the model).
> You must stack them into a 2-D matrix before doing any ML/linear algebra.

## Load it

```python
import pandas as pd

df = pd.read_parquet("hidden_states_dataset.parquet")
print(df.shape)
print(df.columns.tolist())
df.head(3)
```

Pandas needs `pyarrow` (already installed by the notebook). If reading elsewhere:
`pip install pandas pyarrow`.

## Turn a pooled column into a matrix

Each cell is a 1-D array; stack them into `(num_rows, hidden_size)`:

```python
import numpy as np

X = np.stack(df["L-1_mean"].to_numpy())   # shape: (n_rows, hidden_size)
print(X.shape)
```

`np.stack` over the object column is the reliable way — plain `df["L-1_mean"].values`
gives you an array of arrays, which most libraries won't accept.

Concatenate several pooled layers into one feature block:

```python
pooled_cols = [c for c in df.columns
               if c.startswith("L") and df[c].dtype == object
               and isinstance(df[c].iloc[0], np.ndarray)]
X = np.concatenate([np.stack(df[c]) for c in pooled_cols], axis=1)
```

## Use the geometry scalars

These are already plain numeric columns — use them directly:

```python
geom_cols = ["L3_l2_norm", "L5_mean_token_norm", "L5_anisotropy", "L-1_effective_rank"]
G = df[geom_cols].to_numpy()          # (n_rows, n_metrics)

# e.g. correlate effective rank with difficulty
df.groupby("level")["L-1_effective_rank"].mean()
```

## Common tasks

**Probe / classifier** (predict `level` or `type` from pooled hidden states):

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = np.stack(df["L-1_mean"])
y = df["type"].fillna("unknown")

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
print(clf.score(Xte, yte))
```

**Dimensionality reduction / visualization:**

```python
from sklearn.decomposition import PCA
emb2d = PCA(n_components=2).fit_transform(np.stack(df["L5_last"]))
```

**Similarity search between problems:**

```python
from sklearn.metrics.pairwise import cosine_similarity
X = np.stack(df["L-1_mean"])
sims = cosine_similarity(X[:1], X)[0]     # problem 0 vs. all
top = sims.argsort()[::-1][1:6]
df.iloc[top]["problem"]
```

## Notes & caveats

- **Padding is already handled.** Pooling and geometry respect the attention mask
  (cell 10), so pad tokens never contribute — no masking needed downstream.
- **Vector length = model `hidden_size`.** It differs across models; don't hardcode it.
  Read it as `np.stack(df[col]).shape[1]`.
- **dtype.** Vectors were cast to `float32` on save (`.float().cpu().numpy()`), even when
  the model ran in `float16`/`bfloat16`.
- **Config isn't stored in the file.** Which model / dataset / layers produced these
  columns lives only in the notebook `CONFIG`. Keep the config (or the `config.yaml`)
  alongside the parquet if provenance matters.
- **Appending/merging files:** two parquets are only compatible if generated with the
  same `features`/`geometry` config (identical column names) **and** the same model
  (identical `hidden_size`). Otherwise `pd.concat` will misalign or error.
- **Missing metadata.** `level`/`type` are `null` when the source dataset lacks those
  fields — filter with `df.dropna(subset=["level"])` before grouping on them.
