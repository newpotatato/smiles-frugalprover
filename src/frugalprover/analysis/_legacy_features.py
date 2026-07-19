"""Feature building against the flat record dicts, for the analysis scripts.

These are the functions the analyses used to import from the old
`experiments/A_oracle/oracle.py`. They are kept verbatim in behaviour --
surface features, subject one-hot, PCA of one activation layer -- so that the
statistics in `layer_probe.py`, `calibration.py` and `learning_curve.py` keep
producing what they produced before the restructure.

The pipeline itself does NOT use these. It uses the composable blocks in
`frugalprover.oracle.model.features`, which do the same thing but are fitted
objects that travel with the model. Don't build new work on this module.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from frugalprover.common.grading import SURFACE_KEYS

PHI_KEYS = list(SURFACE_KEYS)
ALPHAS = np.logspace(-3, 3, 13)
N_PCS = 10


def build_features(records, type_names, layer, scaler=None, pca=None, fit=False):
    """phi + subject one-hot + PCA(activations at `layer`).

    Fits the scaler/PCA when `fit=True`, otherwise reuses the ones passed in so
    new problems get the same transform.
    """
    phi = np.array([[r["phi"][k] for k in PHI_KEYS] for r in records], dtype=float)
    type_oh = np.array([[1.0 if r["type"] == t else 0.0 for t in type_names] for r in records])
    acts = np.array([r["activations"][layer] for r in records], dtype=float)

    if fit:
        scaler = {"phi": StandardScaler().fit(phi), "acts": StandardScaler().fit(acts)}
        phi_n = scaler["phi"].transform(phi)
        acts_n = scaler["acts"].transform(acts)
        pca = PCA(n_components=min(N_PCS, len(records) - 2, acts_n.shape[1])).fit(acts_n)
    else:
        phi_n = scaler["phi"].transform(phi)
        acts_n = scaler["acts"].transform(acts)

    pcs = pca.transform(acts_n)
    return np.hstack([phi_n, type_oh, pcs]), scaler, pca


def build_classification_dataset(records, type_names, layer, budgets,
                                 scaler=None, pca=None, fit=False):
    """One row per (problem, budget). Label = 1{budget >= b_star}.

    `groups` is returned so CV can hold a problem's rows together
    (LeaveOneGroupOut) -- rows from one problem share activations and must not
    be split across train and test.
    """
    X_static, scaler, pca = build_features(records, type_names, layer, scaler, pca, fit)
    log_budgets = np.log(np.array(budgets, dtype=float))

    X_rows, y_rows, groups = [], [], []
    for i, r in enumerate(records):
        for b, log_b in zip(budgets, log_budgets):
            X_rows.append(np.append(X_static[i], log_b))
            solved = r.get("b_star") is not None and b >= r["b_star"]
            y_rows.append(float(solved))
            groups.append(r["id"])
    return np.array(X_rows), np.array(y_rows), np.array(groups), scaler, pca


def layer_names(records):
    """Activation layer keys sorted by depth (`layer_0`, `layer_1`, ...)."""
    return sorted(records[0]["activations"].keys(), key=lambda s: int(s.split("_")[1]))


def pick_best_layer_classification(records, type_names, budgets):
    """Layer with the best grouped-CV AUC."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

    best_layer, best_auc = None, -np.inf
    for layer in layer_names(records):
        X, y, groups, _, _ = build_classification_dataset(
            records, type_names, layer, budgets, fit=True
        )
        splits = list(LeaveOneGroupOut().split(X, y, groups))
        proba = cross_val_predict(
            LogisticRegression(max_iter=1000), X, y, cv=splits, method="predict_proba"
        )[:, 1]
        auc = roc_auc_score(y, proba) if len(set(y)) > 1 else float("nan")
        print(f"  {layer}: CV AUC (grouped by problem) = {auc:.3f}")
        if auc > best_auc:
            best_auc, best_layer = auc, layer
    return best_layer, best_auc
