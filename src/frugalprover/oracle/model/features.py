"""Composable feature blocks.

Each block turns an :class:`OracleDataset` into a ``(n_problems, k)`` matrix.
The oracle stacks whichever blocks the config names, so an ablation -- surface
only, activations only, both -- is a config change rather than an edit to a
hardcoded `hstack`.

Blocks are fitted objects: they hold their scalers and PCA rotations, and they
are pickled with the model. That's what makes `predict` on new problems apply
exactly the same transform as training did.

**`level` is deliberately not available as a block.** The human difficulty
annotation is what the oracle is supposed to recover from activations; feeding
it in would inflate every score and answer a different question. It stays in
the data for stratification and plots.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from frugalprover.common.grading import SURFACE_KEYS, surface_features

#: Marker column for a subject unseen at fit time. Reserved up front so a
#: problem from a new subject produces an explicit 1 here rather than a silently
#: all-zero one-hot that looks like valid input.
OTHER_SUBJECT = "__other__"


@runtime_checkable
class FeatureBlock(Protocol):
    name: str

    def fit(self, ds) -> "FeatureBlock": ...
    def transform(self, ds) -> np.ndarray: ...
    def column_names(self) -> list[str]: ...


class SurfaceFeatures:
    """Cheap text statistics: length, LaTeX density, brace depth, digits, `=`.

    This is the baseline that matters. Longer problems plausibly need more
    tokens for reasons having nothing to do with reasoning difficulty, so an
    oracle that beats these seven numbers has shown something and one that
    doesn't hasn't. Keep it in the stack even when studying activations.
    """

    name = "surface"

    def __init__(self):
        self.scaler = None

    def _raw(self, ds) -> np.ndarray:
        return np.array(
            [[surface_features(p.problem)[k] for k in SURFACE_KEYS] for p in ds.problems],
            dtype=float,
        )

    def fit(self, ds) -> SurfaceFeatures:
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler().fit(self._raw(ds))
        return self

    def transform(self, ds) -> np.ndarray:
        return self.scaler.transform(self._raw(ds))

    def column_names(self) -> list[str]:
        return [f"surface.{k}" for k in SURFACE_KEYS]


class SubjectOneHot:
    """One-hot over MATH subject, plus an `__other__` column."""

    name = "subject"

    def __init__(self):
        self.types: list[str] = []

    def fit(self, ds) -> SubjectOneHot:
        self.types = sorted({p.type for p in ds.problems})
        return self

    def transform(self, ds) -> np.ndarray:
        cols = self.types + [OTHER_SUBJECT]
        out = np.zeros((len(ds.problems), len(cols)))
        index = {t: i for i, t in enumerate(self.types)}
        for row, p in enumerate(ds.problems):
            out[row, index.get(p.type, len(self.types))] = 1.0
        return out

    def column_names(self) -> list[str]:
        return [f"subject.{t}" for t in self.types + [OTHER_SUBJECT]]


class ActivationPCA:
    """Standardize a pooled hidden-state column, then reduce it with PCA.

    Reduction isn't optional at this scale: the vectors are 1536-dimensional
    and there are ~80 labeled problems. Fitting anything on raw activations
    would interpolate the training set perfectly and generalize to nothing.
    Ten components is the pilot's choice, kept configurable.
    """

    name = "activations"

    def __init__(self, layer: str, n_components: int = 10):
        self.layer = layer
        self.n_components = n_components
        self.scaler = None
        self.pca = None
        self.n_fitted = 0

    def fit(self, ds) -> ActivationPCA:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        acts = ds.activations(self.layer)
        self.scaler = StandardScaler().fit(acts)
        scaled = self.scaler.transform(acts)
        # PCA can't produce more components than samples-1 or features
        n = max(1, min(self.n_components, len(acts) - 1, scaled.shape[1]))
        self.pca = PCA(n_components=n).fit(scaled)
        self.n_fitted = n
        return self

    def transform(self, ds) -> np.ndarray:
        return self.pca.transform(self.scaler.transform(ds.activations(self.layer)))

    def column_names(self) -> list[str]:
        return [f"{self.layer}.pc{i}" for i in range(self.n_fitted)]

    def explained_variance(self) -> float:
        return float(self.pca.explained_variance_ratio_.sum()) if self.pca else 0.0


class GeometryScalars:
    """The per-layer geometry scalars (norms, anisotropy, effective rank).

    Cheap to carry -- they're already in the parquet -- and they're the direct
    test of the geometric hypothesis: if the shape of the token cloud predicts
    difficulty, these columns should help where raw PCA components don't.
    """

    name = "geometry"

    def __init__(self):
        self.columns: list[str] = []
        self.scaler = None

    def fit(self, ds) -> GeometryScalars:
        from sklearn.preprocessing import StandardScaler

        self.columns = ds.geometry_columns()
        if self.columns:
            self.scaler = StandardScaler().fit(ds.scalars(self.columns))
        return self

    def transform(self, ds) -> np.ndarray:
        if not self.columns:
            return np.zeros((len(ds.problems), 0))
        return self.scaler.transform(ds.scalars(self.columns))

    def column_names(self) -> list[str]:
        return [f"geom.{c}" for c in self.columns]


def build_blocks(names: list[str], layer: str | None, n_pcs: int = 10) -> list[FeatureBlock]:
    """Instantiate the named blocks. `layer` is which pooled column the
    activation block should read; None drops that block."""
    blocks: list[FeatureBlock] = []
    for name in names:
        if name == "surface":
            blocks.append(SurfaceFeatures())
        elif name == "subject":
            blocks.append(SubjectOneHot())
        elif name == "activations":
            if layer is not None:
                blocks.append(ActivationPCA(layer, n_pcs))
        elif name == "geometry":
            blocks.append(GeometryScalars())
        else:
            raise ValueError(
                f"unknown feature block {name!r}. "
                "Available: surface, subject, activations, geometry"
            )
    if not blocks:
        raise ValueError(f"no usable feature blocks from {names!r}")
    return blocks


def fit_transform(blocks: list[FeatureBlock], ds) -> np.ndarray:
    return np.hstack([b.fit(ds).transform(ds) for b in blocks])


def transform(blocks: list[FeatureBlock], ds) -> np.ndarray:
    return np.hstack([b.transform(ds) for b in blocks])


def column_names(blocks: list[FeatureBlock]) -> list[str]:
    return [c for b in blocks for c in b.column_names()]
