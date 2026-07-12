# Activations, intrinsic dimension, and compression methods — working notes

A companion to `PIPELINE.md`. Zooms in on the Budget Oracle node: what the
activations gℓ(x) are, why an intrinsic-dimension (ID) estimate is needed, how
TwoNN works, what exactly hypothesis H1 claims (and where the naive phrasing
breaks), and why UMAP is not a replacement for TwoNN.

---

## 1. What activations are

Activations are the internal numeric representations the model builds at each
layer while processing an input. For a problem statement x and layer ℓ, gℓ(x) is
a vector (typically hundreds to a few thousand numbers per token), taken in **a
single forward pass, without generating an answer**. It is not the model's
answer but its "internal state" while reading the statement — a snapshot of what
happens inside before the model has started to solve the problem.

## 2. It is feature engineering — but of a special kind

Unlike classical feature engineering (φ(x): text length, presence of formulas,
etc. — hand-crafted features), gℓ(x) is a feature **extracted from an
already-learned representation** the model built for another purpose
(next-token prediction). It is essentially analogous to using embeddings from a
pretrained network as features for a separate downstream task.

Working with it still requires the classic feature-engineering decisions:
- which layer ℓ,
- how to aggregate over tokens (mean? last token?),
- dimensionality compression (thousands of numbers → a compact feature),
- training a light head fθ on top of the result.

## 3. Intrinsic Dimension (ID) — what it is and why

ID is the minimum number of independent "degrees of freedom" needed to describe
the geometry of a point cloud, even if it formally lives in a space of thousands
of dimensions. It is **not** the same as "the dimension that best preserves
information" (as in an information bottleneck or a PCA explained-variance
threshold) — ID is defined geometrically, through how fast the number of
neighbors around a point grows as the radius increases.

**Why not PCA.** PCA is a linear method: it finds the plane that best
approximates the data. If the manifold is curved (e.g. rolled up like a "swiss
roll"), PCA either overestimates the dimension or misses the structure entirely.
Ansuini et al. show explicitly that the effect of interest is **not**
reproduced by a linear estimate — a nonlinear method is required.

**Classic nonlinear alternatives** (for context, not used directly in the
project):
- Correlation dimension (Grassberger–Procaccia) — from the power-law growth of
  the number of point pairs inside a radius-r ball;
- The Levina–Bickel MLE estimator — from the likelihood of distances to the k
  nearest neighbors.

Both are sensitive to non-uniform density — they confuse "a sparse region" with
"dimension".

## 4. TwoNN — the algorithm (Facco, D'Errico, Rodriguez, Laio, 2017)

The key trick: use not absolute distances but the **ratio** of the distances to
the two nearest neighbors — the local point density cancels out mathematically in
that ratio.

**Step 1.** For each point i, find the distance to the first neighbor r₁ and to
the second neighbor r₂ (only 2 neighbors per point are needed).

**Step 2.** Compute μᵢ = r₂ / r₁.

**Step 3 (the key fact).** If the data is locally distributed as a homogeneous
Poisson process on a d-dimensional manifold with some local density ρ, then
μ = r₂/r₁ follows a Pareto distribution with parameter d:

f(μ) = d · μ^(−d−1), μ ≥ 1

ρ is **absent** from this formula — the density cancels entirely when moving to
the ratio. Hence the method's robustness to non-uniform density and to
curvature: no need to assume constant density across the dataset, only local
homogeneity in the tiny neighborhood of each point.

**Step 4.** From the theoretical CDF it follows that −log(1 − F(μ)) = d · log(μ).
Build the empirical F̂(μ) from the sorted μᵢ, plot −log(1−F̂(μᵢ)) against
log(μᵢ) — it should be a straight line through the origin with slope d. The
estimate d = an ordinary linear regression through the origin.

**Why it is robust at small N and on curved manifolds:**
- only 2 neighbors per point are used → short distances are locally almost
  linear even on a curved surface;
- the distance ratio is invariant to the local density scale;
- the final estimate is a regression over all points at once, so noise averages
  out.

**How Ansuini et al. apply it.** They do not invent a new estimator — they take
TwoNN off the shelf and apply it layer-wise: for each layer ℓ they compute the
ID of the activation cloud over many inputs. The resulting "ID vs depth" curve is
hunchback-shaped: it rises in the early layers and falls toward the output. The
ID of the last hidden layer correlates with final test accuracy — but that
comparison is **between different trained networks** (architectures/training
regimes), not between different inputs within one fixed network.

## 5. Where the naive phrasing of the hypothesis breaks

A tempting but imprecise phrasing: "the ID of the final layers directly
correlates with prediction quality for a specific input x". The problem — **ID
cannot be computed from a single point**: TwoNN needs a point cloud (it needs
neighbors to get r₁, r₂). A lone vector gℓ(x) has no "ID of its own".

Hence two different, non-equivalent ways to use ID in the project:

**(a) Diagnostic (direct analogy with Ansuini, corresponds to G2).** Take the
whole problem corpus, compute the gℓ cloud over all x, and compare the geometry
of sub-clouds — e.g. the ID of "easy" problems vs the ID of "hard" ones. This
characterizes the corpus as a whole, not a prediction for a new problem.

**(b) Predictive (what the oracle itself needs).** A per-problem signal is
required. Here fθ is fed not an ID number but the raw vector gℓ(x) (or a local
characteristic — e.g. the neighbor density around x, also from r₁/r₂ but without
aggregating into a single global d). The testable condition is I(gℓ; B*) > 0
(mutual information beyond what φ(x) provides), not a correlation with ID.

**Conclusion:** TwoNN/ID in the project is more likely a tool for G2 (explaining,
via the geometry of difficulty clusters, *why* the predictor works) than a
feature fed directly into fθ to predict a specific B̂(x).

## 6. UMAP — how it works and how it differs from TwoNN

UMAP (McInnes, Healy, Melville, 2018) is not a dimension estimator but a
**projection** algorithm into a pre-specified dimension.

**Step 1 — a graph in the original space.**
1. For each point i, find the k nearest neighbors (`n_neighbors`).
2. ρᵢ — distance to the closest neighbor (guarantees connectivity).
3. σᵢ — a local scale, calibrated so that the sum of weights to the k neighbors
   is fixed (analogous to perplexity in t-SNE).
4. Directed weight: w(i→j) = exp(−(d(i,j) − ρᵢ) / σᵢ).
5. Symmetrization via a fuzzy union: w(i,j) = w(i→j) + w(j→i) − w(i→j)·w(j→i).

The theoretical basis is Riemannian geometry + fuzzy simplicial sets (algebraic
topology): the data lies on a manifold with a locally varying metric, and the
graph approximates its topology, robustly to non-uniform sampling density.

**Step 2 — optimizing the low-dimensional representation.**
1. Choose the target dimension `n_components` (usually 2 for visualization).
2. Initialize — usually via a spectral embedding (Laplacian Eigenmaps).
3. Low-dim weight: w_low(i,j) = 1 / (1 + a·dist(i,j)^(2b)), with a and b tuned to
   `min_dist`.
4. Minimize the cross-entropy between the high-dim and low-dim graph weights:
   attraction for neighbors, repulsion for non-neighbors.
5. Optimize with SGD + negative sampling (as in word2vec) instead of the full
   O(N²) pass over pairs — the source of UMAP's speed at large N.

**Key hyperparameters:** `n_neighbors` (local vs global structure), `min_dist`
(packing density in the projection), `n_components` (target dimension —
**set by the user**).

## 7. TwoNN vs UMAP — not interchangeable

| | TwoNN | UMAP |
|---|---|---|
| What it does | Estimates d — the minimum dimension needed to describe the data | Projects the data into d dimensions that you specify in advance |
| Output | A number (the ID estimate) | A set of coordinates in low dimension |
| If you set n_components=2 but the true ID = 15 | Not applicable — the method *is* the way to find the true ID | Silently squeezes the data into 2D with distortion, no warning |
| Role in the project | Formal test of H1 (G2) | Auxiliary compression/visualization |

**Where UMAP is still useful in the project** — not instead of TwoNN, but
alongside it:
- visual diagnostics for G2: project gℓ(x) into 2D and see whether "easy"/"hard"
  problems separate into clusters — a quick qualitative check before the formal
  TwoNN analysis;
- feature compression for fθ: if the raw gℓ(x) is too large for a light probe
  regression, compress it with UMAP (e.g. to 20–50 dimensions) and train fθ on
  that — an alternative engineering choice, competing with using the raw
  activations directly.

In short: TwoNN is about measuring the data's true complexity, UMAP is about
compression/visualization at a chosen dimension. Rigorously testing H1 needs the
former; the latter can complement the pipeline but does not replace the
measurement logic.
