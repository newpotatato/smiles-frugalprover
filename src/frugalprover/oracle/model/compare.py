"""Head-to-head comparison of model families under the same CV splits.

Empirics for "which oracle method", rather than an assertion that Ridge is
fine. Each family answers a different question: MLP asks whether nonlinearity
helps, ElasticNet whether sparsity helps, GP whether calibrated uncertainty is
available (useful if you later want to allocate budget conservatively).

Read the output with the sample size in mind. At n=80, a 0.01 gap is noise.
The rule of thumb from the research plan: prefer the simplest model unless a
more complex one beats it by more than ~0.02.
"""
from __future__ import annotations

import warnings

import numpy as np

ALPHAS = np.logspace(-3, 3, 13)


def compare(model, ds) -> dict[str, float]:
    """Score several families on `model`'s chosen layer. Returns {name: score}."""
    from sklearn.exceptions import ConvergenceWarning

    # small-n MLP/ElasticNet fits warn constantly and it isn't informative
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        if model.mode == "classification":
            results = _compare_classification(model, ds)
        else:
            results = _compare_regression(model, ds)

    print(f"\n--- model families at {model.layer} (same CV splits) ---")
    for name, score in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<20} CV {model.cv_metric} = {score:.3f}")
    best = max(results, key=results.get)
    print(f"-> best: {best} ({results[best]:.3f}). At this n, prefer the simplest "
          f"model within ~0.02 of the top, not the top number itself.")
    return {k: round(float(v), 4) for k, v in results.items()}


def _compare_classification(model, ds) -> dict[str, float]:
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
    from sklearn.neural_network import MLPClassifier

    X, y, groups, _ = model._build(ds, model.layer, fit=True)
    splits = list(LeaveOneGroupOut().split(X, y, groups))
    families = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "MLP (16,)": MLPClassifier(hidden_layer_sizes=(16,), alpha=1.0, max_iter=3000, random_state=0),
        "GaussianProcess": GaussianProcessClassifier(kernel=RBF(), random_state=0),
    }
    out = {}
    for name, est in families.items():
        proba = cross_val_predict(est, X, y, cv=splits, method="predict_proba")[:, 1]
        out[name] = roc_auc_score(y, proba) if len(set(y)) > 1 else float("nan")
    return out


def _compare_regression(model, ds) -> dict[str, float]:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    from sklearn.linear_model import ElasticNetCV, RidgeCV
    from sklearn.model_selection import cross_val_predict
    from sklearn.neural_network import MLPRegressor

    from frugalprover.oracle.model.features import build_blocks, fit_transform

    solved = ds.solved_subset()
    y = solved.b_star()
    blocks = build_blocks(model.feature_names, model.layer, model.n_pcs)
    X = fit_transform(blocks, solved)
    cv = model.cv_splitter(len(y))

    families = {
        "Ridge": RidgeCV(alphas=ALPHAS),
        "ElasticNet": ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], alphas=ALPHAS, max_iter=5000),
        "MLP (16,)": MLPRegressor(hidden_layer_sizes=(16,), alpha=1.0, max_iter=3000, random_state=0),
        "GaussianProcess": GaussianProcessRegressor(
            kernel=RBF() + WhiteKernel(), normalize_y=True, n_restarts_optimizer=2, random_state=0
        ),
    }
    out = {}
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    for name, est in families.items():
        preds = cross_val_predict(est, X, y, cv=cv)
        ss_res = float(np.sum((y - preds) ** 2))
        out[name] = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return out
