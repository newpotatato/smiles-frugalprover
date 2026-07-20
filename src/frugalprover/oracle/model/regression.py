"""Regression framing: predict B* directly.

Simpler than the classification framing and easier to read off, but it pays for
that twice:

- **Censored problems are unusable.** A problem never solved has no target, so
  it's dropped. Since censored problems are exactly the hard ones, the model
  trains on a systematically easy subset.
- **It needs a real sweep.** Regressing onto a B* that can only take one value
  fits a constant. `fit` refuses on single-budget data instead of returning a
  meaningless model.

Keep it as a cross-check on classification, not as the primary result.
"""
from __future__ import annotations

import numpy as np

from frugalprover.oracle.model.base import OracleModel
from frugalprover.oracle.model.dataset import OracleDataset
from frugalprover.oracle.model.features import build_blocks, fit_transform, transform

ALPHAS = np.logspace(-3, 3, 13)


class RegressionOracle(OracleModel):
    mode = "regression"
    cv_metric = "r2"

    def __init__(self, *args, min_solved: int = 15, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_solved = min_solved

    def _check(self, ds: OracleDataset) -> OracleDataset:
        if not ds.has_budgets():
            raise ValueError("regression needs budget labels - run `frugalprover budget` first")
        if ds.single_pass():
            raise ValueError(
                f"regression needs a multi-budget sweep, but only {ds.swept_budgets()} "
                f"was tried. Every solved problem would have the same B*, so the fit "
                f"would be a constant. Use --set train.mode=classification, which "
                f"handles single-budget data."
            )
        solved = ds.solved_subset()
        if len(solved) < self.min_solved:
            raise ValueError(
                f"only {len(solved)}/{len(ds)} problems were solved within the swept "
                f"budgets, below the minimum of {self.min_solved}. Regression drops "
                f"censored problems entirely; classification uses them as zeros - "
                f"try --set train.mode=classification."
            )
        return solved

    def score(self, ds: OracleDataset, layer: str) -> float:
        from sklearn.linear_model import RidgeCV
        from sklearn.model_selection import cross_val_predict

        solved = ds.solved_subset()
        y = solved.b_star()
        self.blocks = build_blocks(self.feature_names, layer, self.n_pcs)
        X = fit_transform(self.blocks, solved)

        preds = cross_val_predict(RidgeCV(alphas=ALPHAS), X, y, cv=self.cv_splitter(len(y)))
        ss_res = float(np.sum((y - preds) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        # R2 <= 0 means "worse than predicting the mean" -- reported as-is
        # rather than clipped, because a negative number is informative here
        return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    def fit(self, ds: OracleDataset, layer: str | None = None) -> RegressionOracle:
        from sklearn.linear_model import RidgeCV

        solved = self._check(ds)
        print(f"{len(solved)}/{len(ds)} problems solved within the swept budgets "
              f"(censored ones are dropped by this framing)")

        if layer is None:
            print(f"sweeping {len(ds.pooled_columns())} layers by CV R^2:")
            layer, self.layer_scores = self.select_layer(ds)
            self.cv_score = self.layer_scores[layer]
            print(f"-> {layer} (CV R^2 = {self.cv_score:.3f})")
        else:
            self.cv_score = self.score(ds, layer)
            self.layer_scores = {layer: self.cv_score}

        self.layer = layer
        self.blocks = build_blocks(self.feature_names, layer, self.n_pcs)
        X = fit_transform(self.blocks, solved)
        y = solved.b_star()
        self.budgets_seen = ds.swept_budgets()
        self.success_threshold = next(iter(ds.budgets.values())).success_threshold
        self.n_train = len(X)
        self.model = RidgeCV(alphas=ALPHAS).fit(X, y)
        return self

    def predict_budget(self, ds: OracleDataset) -> np.ndarray:
        """Predicted B*, clipped to the swept range.

        Clipping because the model is linear and will happily extrapolate to
        negative token counts on an easy problem. Outside the swept range there
        is no evidence either way, so the endpoints are the honest answer.
        """
        raw = self.model.predict(transform(self.blocks, ds))
        return np.clip(raw, min(self.budgets_seen), max(self.budgets_seen))

    def predict_success(self, ds: OracleDataset, budget: int) -> np.ndarray:
        """Derived, not modeled: 1 where the predicted B* fits inside `budget`.

        This framing has no probability to report -- it predicts a point, not a
        distribution -- so this is a hard 0/1 step. If you need calibrated
        probabilities, use the classification oracle.
        """
        return (self.predict_budget(ds) <= budget).astype(float)
