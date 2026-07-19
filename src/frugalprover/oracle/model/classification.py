"""Classification framing: P(solved | features, budget).

This is the default, for three reasons:

1. **Censored problems are usable.** A problem never solved at any budget
   contributes an honest 0 at every budget, instead of being dropped (as
   regression must) or given an invented B*.
2. **It matches the formalization.** The research plan is written in terms of
   p_i(B), the probability of solving problem i within B tokens. This models
   that directly.
3. **Single-fixed-budget data works.** One budget just means one row per
   problem rather than three.

B̂ falls out as the smallest swept budget whose predicted probability clears the
threshold.
"""
from __future__ import annotations

import numpy as np

from frugalprover.oracle.model.base import OracleModel
from frugalprover.oracle.model.dataset import OracleDataset
from frugalprover.oracle.model.features import build_blocks, fit_transform, transform

#: Regularization grid. Note sklearn's `Cs` is INVERSE regularization strength
#: (C = 1/alpha), the opposite of Ridge's `alphas`. The grid is symmetric in log
#: space around 1, so it spans the same range of models either way -- but don't
#: read a chosen C as if it were an alpha.
ALPHAS = np.logspace(-3, 3, 13)


class ClassificationOracle(OracleModel):
    mode = "classification"
    cv_metric = "auc"

    # ------------------------------------------------------------- row layout

    def _expand(self, X_static: np.ndarray, ds: OracleDataset, budgets: list[int]):
        """One row per (problem, budget), with log(budget) appended.

        `log` rather than raw budget because the budgets are geometric
        (128/256/512) and success is roughly linear in log-compute; on the raw
        scale the model would have to bend a straight line.

        `groups` carries the problem id so cross-validation can keep a problem's
        rows together. Without that, a problem's 128-row trains the model that
        scores its 512-row -- same activations on both sides of the split, and
        the AUC becomes fiction.
        """
        rows, labels, groups = [], [], []
        for i, p in enumerate(ds.problems):
            record = ds.budgets[p.id]
            for b in budgets:
                if b not in record.budgets:
                    continue  # this problem wasn't tried at this budget
                rows.append(np.append(X_static[i], np.log(b)))
                labels.append(float(record.solved_at(b)))
                groups.append(p.id)
        if not rows:
            raise ValueError("no (problem, budget) rows - are there any budget labels?")
        return np.array(rows), np.array(labels), np.array(groups)

    def _build(self, ds: OracleDataset, layer: str, fit: bool):
        budgets = ds.swept_budgets()
        if fit:
            self.blocks = build_blocks(self.feature_names, layer, self.n_pcs)
            X_static = fit_transform(self.blocks, ds)
        else:
            X_static = transform(self.blocks, ds)
        return self._expand(X_static, ds, budgets) + (budgets,)

    # ------------------------------------------------------------------ score

    def score(self, ds: OracleDataset, layer: str) -> float:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

        X, y, groups, _ = self._build(ds, layer, fit=True)
        if len(set(y)) < 2:
            raise ValueError(
                "labels are all one class - every problem was solved at every "
                "budget, or none was. Nothing to learn; widen the budget range."
            )
        splits = list(LeaveOneGroupOut().split(X, y, groups))
        proba = cross_val_predict(
            LogisticRegression(max_iter=1000), X, y, cv=splits, method="predict_proba"
        )[:, 1]
        return float(roc_auc_score(y, proba))

    # -------------------------------------------------------------------- fit

    def fit(self, ds: OracleDataset, layer: str | None = None) -> ClassificationOracle:
        from sklearn.linear_model import LogisticRegressionCV

        if not ds.has_budgets():
            raise ValueError("classification needs budget labels - run `frugalprover budget` first")

        if layer is None:
            print(f"sweeping {len(ds.pooled_columns())} layers by grouped-CV AUC:")
            layer, self.layer_scores = self.select_layer(ds)
            self.cv_score = self.layer_scores[layer]
            print(f"-> {layer} (CV AUC = {self.cv_score:.3f})")
        else:
            self.cv_score = self.score(ds, layer)
            self.layer_scores = {layer: self.cv_score}

        self.layer = layer
        X, y, _, budgets = self._build(ds, layer, fit=True)
        self.budgets_seen = budgets
        self.success_threshold = next(iter(ds.budgets.values())).success_threshold
        self.n_train = len(X)
        # LogisticRegressionCV picks the regularization strength internally, so
        # there's no separate hyperparameter to tune by hand
        self.model = LogisticRegressionCV(Cs=ALPHAS, max_iter=1000, cv=5).fit(X, y)
        return self

    # -------------------------------------------------------------- inference

    def predict_success(self, ds: OracleDataset, budget: int) -> np.ndarray:
        X_static = transform(self.blocks, ds)
        X = np.column_stack([X_static, np.full(len(X_static), np.log(budget))])
        return self.model.predict_proba(X)[:, 1]

    def predict_budget(self, ds: OracleDataset) -> np.ndarray:
        """Smallest swept budget whose predicted success clears the threshold."""
        probs = np.column_stack([self.predict_success(ds, b) for b in self.budgets_seen])
        out = np.full(len(ds), np.nan)
        for i in range(len(ds)):
            cleared = [b for b, p in zip(self.budgets_seen, probs[i]) if p >= self.success_threshold]
            if cleared:
                out[i] = min(cleared)
        return out
