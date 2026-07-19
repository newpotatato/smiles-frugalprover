"""The oracle interface.

A fitted model carries its own feature blocks, so `joblib.dump(model)` is the
whole checkpoint -- there is no separate bundle of loose scalers and rotations
to reassemble by hand at predict time.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from frugalprover.oracle.model.dataset import OracleDataset


class OracleModel(ABC):
    """Predicts how much budget a problem needs, or whether a budget suffices.

    Two framings, both fitted the same way:

    - **classification** models ``P(solved | features, budget)`` with budget as
      an input. Uses every problem including censored ones, and works on
      single-fixed-budget data.
    - **regression** predicts ``B*`` directly. Simpler, but only usable on a
      real multi-budget sweep and only trainable on solved problems.
    """

    mode: ClassVar[str]
    cv_metric: ClassVar[str]

    def __init__(self, feature_names: list[str] | None = None, n_pcs: int = 10, cv: str = "auto"):
        self.feature_names = feature_names or ["surface", "subject", "activations"]
        self.n_pcs = n_pcs
        self.cv = cv
        self.blocks: list = []
        self.model: Any = None
        self.layer: str | None = None
        self.layer_scores: dict[str, float] = {}
        self.cv_score: float = float("nan")
        self.budgets_seen: list[int] = []
        self.success_threshold: float = 0.5
        self.n_train: int = 0

    # ------------------------------------------------------------- interface

    @abstractmethod
    def fit(self, ds: OracleDataset, layer: str | None = None) -> OracleModel:
        """Fit on `ds`. If `layer` is None, sweep all pooled columns and keep
        the best-scoring one."""

    @abstractmethod
    def score(self, ds: OracleDataset, layer: str) -> float:
        """Cross-validated score at one layer. Used by the sweep, so it must not
        leak: every prediction has to come from a model that didn't see that
        problem."""

    @abstractmethod
    def predict_budget(self, ds: OracleDataset) -> np.ndarray:
        """Predicted B* per problem. NaN means no swept budget is predicted to
        suffice -- the model's way of saying "more than I've seen"."""

    @abstractmethod
    def predict_success(self, ds: OracleDataset, budget: int) -> np.ndarray:
        """P(solve within `budget`) per problem."""

    # -------------------------------------------------------------- defaults

    def select_layer(self, ds: OracleDataset, layers: list[str] | None = None) -> tuple[str, dict]:
        """Score every candidate layer, return the best plus the full sweep.

        The whole sweep is kept, not just the winner: one good number out of 29
        tried is weak evidence, but a smooth curve peaking mid-network is the
        actual finding. Reporting reads `layer_scores` to plot it.
        """
        candidates = layers or ds.pooled_columns()
        if not candidates:
            raise ValueError(
                "no pooled activation columns in the hidden states - "
                "check that extraction produced L<n>_<pooling> columns"
            )

        scores: dict[str, float] = {}
        for layer in candidates:
            try:
                scores[layer] = self.score(ds, layer)
            except ValueError as e:
                print(f"  {layer}: skipped ({e})")
                continue
            print(f"  {layer}: CV {self.cv_metric} = {scores[layer]:.3f}")

        usable = {k: v for k, v in scores.items() if not np.isnan(v)}
        if not usable:
            raise ValueError(
                f"every layer scored NaN. Usually this means the labels are "
                f"degenerate (all solved, or all censored) rather than a bug in "
                f"the features."
            )
        best = max(usable, key=usable.get)
        return best, scores

    def cv_splitter(self, n: int):
        """Leave-one-out below 60 problems, else 5-fold.

        LOO because at n=80 a 5-fold split trains on 64 problems and tests on
        16, and the variance of that estimate swamps the effect being measured.
        LOO uses every problem as a test case exactly once.
        """
        from sklearn.model_selection import KFold, LeaveOneOut

        if self.cv == "loo":
            return LeaveOneOut()
        if isinstance(self.cv, int):
            return KFold(n_splits=self.cv, shuffle=True, random_state=0)
        return LeaveOneOut() if n <= 60 else KFold(n_splits=5, shuffle=True, random_state=0)

    def predict_rows(self, ds: OracleDataset) -> list[dict]:
        """Predictions as A5 records, one per problem."""
        b_hat = self.predict_budget(ds)
        probs = {b: self.predict_success(ds, b) for b in self.budgets_seen}
        rows = []
        for i, p in enumerate(ds.problems):
            rows.append({
                "id": p.id,
                "b_hat": None if np.isnan(b_hat[i]) else int(b_hat[i]),
                "p_by_budget_pred": {str(b): round(float(probs[b][i]), 4) for b in self.budgets_seen},
                "mode": self.mode,
            })
        return rows

    def summary(self) -> dict:
        """The pickle-free record of this fit, written to metrics.json."""
        from frugalprover.oracle.model.features import column_names

        return {
            "mode": self.mode,
            "layer": self.layer,
            "cv_metric": self.cv_metric,
            "cv_score": None if np.isnan(self.cv_score) else round(float(self.cv_score), 4),
            "layer_scores": {k: (None if np.isnan(v) else round(float(v), 4))
                             for k, v in self.layer_scores.items()},
            "feature_blocks": self.feature_names,
            "feature_columns": column_names(self.blocks) if self.blocks else [],
            "n_pcs": self.n_pcs,
            "n_train": self.n_train,
            "budgets_seen": self.budgets_seen,
            "success_threshold": self.success_threshold,
        }

    def save(self, path: str | Path, config: dict | None = None) -> Path:
        from frugalprover.common.io import save_model

        meta = {"artifact": "oracle", **self.summary()}
        if config is not None:
            meta["config"] = config
        return save_model(self, path, meta=meta)

    @staticmethod
    def load(path: str | Path) -> OracleModel:
        from frugalprover.common.io import load_model

        return load_model(path)
