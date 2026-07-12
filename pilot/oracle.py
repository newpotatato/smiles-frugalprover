"""
The Budget Oracle f_theta(phi(x), g_ell(x)) -> B_hat(x), fit on the merged
pilot data. Two framings, pick with --mode:

  --mode regression (default): regress the point estimate B* directly.
      Simple, but B* is a derived, somewhat noisy/censored quantity (undefined
      whenever a problem wasn't solved at any swept budget).

  --mode classification: model P(success | phi, g_ell, budget) directly, with
      budget as an input feature, trained on ALL (problem, budget) pairs --
      this is a more direct match to the paper's own formalization
      ("let p_i(B) be the probability of solving x_i within B tokens") and
      sidesteps the censoring problem entirely: an unsolved problem just
      contributes label=0 at every swept budget instead of needing a fake B*.
      B_hat(x) is then the smallest swept budget whose predicted P(success)
      clears SUCCESS_THRESHOLD. Recommended as the primary framing; keep
      regression around as a simpler cross-check.

Both modes pick the best-scoring activation layer, fit a final model on ALL
data (not held out -- this is the deployed model, not the evaluation), and
save to oracle_model.joblib. This is the thing for the 4th person to poke at:
swap features, try regularization strength, try combining layers, see how
predictions move on individual problems.

Usage:
    python oracle.py pilot_results.jsonl                              # regression (default)
    python oracle.py pilot_results.jsonl --mode classification        # p(B) framing
    python oracle.py pilot_results.jsonl --demo                       # + example predictions
    python oracle.py pilot_results.jsonl --compare-models             # Ridge/ElasticNet/MLP/GP head-to-head
"""
import argparse
import json
import sys
import warnings

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.linear_model import ElasticNetCV, LogisticRegression, LogisticRegressionCV, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut, cross_val_predict
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)  # small-n MLP/ElasticNet fits often warn; not fatal

PHI_KEYS = ["char_len", "word_len", "latex_cmd_count", "dollar_count",
            "brace_depth", "digit_count", "eq_count"]
ALPHAS = np.logspace(-3, 3, 13)
N_PCS = 10


def load_records(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def build_features(records, type_names, layer, scaler=None, pca=None, fit=False):
    """phi + type one-hot + PCA(activations at `layer`). Fits scaler/pca if fit=True,
    otherwise reuses the ones passed in (for consistent transform of new problems)."""
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
    X = np.hstack([phi_n, type_oh, pcs])
    return X, scaler, pca


def compare_regression_models(X, y, cv):
    """Same CV protocol, several model families -- empirics for 'which oracle
    method', not just an assertion that Ridge is fine. GP gives calibrated
    uncertainty (useful for capping conservatively), MLP checks whether
    nonlinearity helps, ElasticNet checks whether sparsity helps."""
    models = {
        "Ridge": RidgeCV(alphas=ALPHAS),
        "ElasticNet": ElasticNetCV(l1_ratio=[.1, .5, .9], alphas=ALPHAS, max_iter=5000),
        "MLP (16,)": MLPRegressor(hidden_layer_sizes=(16,), alpha=1.0, max_iter=3000, random_state=0),
        "GaussianProcess": GaussianProcessRegressor(
            kernel=RBF() + WhiteKernel(), normalize_y=True, n_restarts_optimizer=2, random_state=0),
    }
    print("\n--- comparing oracle model families (regression, same CV splits) ---")
    results = {}
    for name, model in models.items():
        preds = cross_val_predict(model, X, y, cv=cv)
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        results[name] = r2
        print(f"  {name:<16}: CV R^2 = {r2:.3f}")
    best = max(results, key=results.get)
    print(f"-> best by CV R^2: {best} ({results[best]:.3f}). At this n, prefer the "
          "simplest model within ~0.02 of the best (Ridge/ElasticNet), not just the top number.")
    return results


def compare_classification_models(X, y, groups):
    splits = list(LeaveOneGroupOut().split(X, y, groups))
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "MLP (16,)": MLPClassifier(hidden_layer_sizes=(16,), alpha=1.0, max_iter=3000, random_state=0),
        "GaussianProcess": GaussianProcessClassifier(kernel=RBF(), random_state=0),
    }
    print("\n--- comparing oracle model families (classification, grouped CV, same splits) ---")
    results = {}
    for name, model in models.items():
        proba = cross_val_predict(model, X, y, cv=splits, method="predict_proba")[:, 1]
        auc = roc_auc_score(y, proba) if len(set(y)) > 1 else float("nan")
        results[name] = auc
        print(f"  {name:<20}: CV AUC = {auc:.3f}")
    best = max(results, key=results.get)
    print(f"-> best by CV AUC: {best} ({results[best]:.3f}). At this n, prefer the "
          "simplest model within ~0.02 of the best (LogisticRegression), not just the top number.")
    return results


# ---------------------------------------------------------------- regression

def pick_best_layer_regression(solved, type_names, y, cv):
    layer_names = sorted(solved[0]["activations"].keys(), key=lambda s: int(s.split("_")[1]))
    best_layer, best_r2 = None, -np.inf
    for layer in layer_names:
        X, _, _ = build_features(solved, type_names, layer, fit=True)
        preds = cross_val_predict(RidgeCV(alphas=ALPHAS), X, y, cv=cv)
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else -np.inf
        print(f"  {layer}: CV R^2 = {r2:.3f}")
        if r2 > best_r2:
            best_r2, best_layer = r2, layer
    return best_layer, best_r2


def fit_regression(records, args):
    # after merge_results.py's id-based join, most records will be
    # activations_only (no "p_by_budget"/"b_star" at all) -- only the smaller
    # full_sweep tier has them. Restrict to that tier before filtering solved/
    # unsolved, so the denominator below isn't misleadingly inflated by
    # problems that were simply never swept.
    swept = [r for r in records if "p_by_budget" in r]
    solved = [r for r in swept if r.get("b_star") is not None]
    n = len(solved)
    print(f"{len(swept)}/{len(records)} records have full_sweep budget data; "
          f"{n}/{len(swept)} of those were solved within the tested budgets")
    if n < 15:
        print("too few solved problems for point-regression on B* -- try --mode "
              "classification instead (it uses ALL problems, solved or not).")
        sys.exit(1)

    y = np.array([r["b_star"] for r in solved], dtype=float)
    type_names = sorted(set(r["type"] for r in solved))
    cv = LeaveOneOut() if n <= 60 else 5

    print("\npicking the layer with the best CV R^2:")
    best_layer, best_r2 = pick_best_layer_regression(solved, type_names, y, cv)
    print(f"-> using {best_layer} (CV R^2 = {best_r2:.3f})")

    X, scaler, pca = build_features(solved, type_names, best_layer, fit=True)

    if args.compare_models:
        compare_regression_models(X, y, cv)

    model = RidgeCV(alphas=ALPHAS).fit(X, y)

    bundle = {
        "mode": "regression", "model": model, "scaler": scaler, "pca": pca,
        "layer": best_layer, "type_names": type_names, "phi_keys": PHI_KEYS,
        "cv_score": best_r2,
        "budgets_seen": sorted(set(int(b) for r in solved for b in r["p_by_budget"].keys())),
    }

    if args.demo:
        print("\n--- demo: predicted vs actual B* on the first 5 solved problems ---")
        preds = model.predict(X)
        for r, pred, actual in list(zip(solved, preds, y))[:5]:
            print(f"id={r['id']:>3} type={r['type']:<22} actual B*={actual:>5.0f}  predicted={pred:>7.1f}")

    return bundle


# ----------------------------------------------------------- classification

def build_classification_dataset(records, type_names, layer, budgets, scaler=None, pca=None, fit=False):
    """One row per (problem, swept budget). Label = 1{budget >= b_star} (0 for
    every budget if the problem was never solved -- no censoring special-case
    needed). `groups` returned so CV can be done per-problem (LeaveOneGroupOut),
    since rows from the same problem across budgets share activations and
    must not be split across train/test."""
    X_static, scaler, pca = build_features(records, type_names, layer, scaler, pca, fit)
    log_budgets = np.log(np.array(budgets, dtype=float))

    X_rows, y_rows, groups = [], [], []
    for i, r in enumerate(records):
        for b, log_b in zip(budgets, log_budgets):
            X_rows.append(np.append(X_static[i], log_b))
            solved_at_b = r["b_star"] is not None and b >= r["b_star"]
            y_rows.append(float(solved_at_b))
            groups.append(r["id"])
    return np.array(X_rows), np.array(y_rows), np.array(groups), scaler, pca


def pick_best_layer_classification(records, type_names, budgets):
    layer_names = sorted(records[0]["activations"].keys(), key=lambda s: int(s.split("_")[1]))
    best_layer, best_auc = None, -np.inf
    for layer in layer_names:
        X, y, groups, _, _ = build_classification_dataset(records, type_names, layer, budgets, fit=True)
        splits = list(LeaveOneGroupOut().split(X, y, groups))
        proba = cross_val_predict(LogisticRegression(max_iter=1000), X, y, cv=splits, method="predict_proba")[:, 1]
        auc = roc_auc_score(y, proba) if len(set(y)) > 1 else float("nan")
        print(f"  {layer}: CV AUC (grouped by problem) = {auc:.3f}")
        if auc > best_auc:
            best_auc, best_layer = auc, layer
    return best_layer, best_auc


def fit_classification(records, args):
    # same restriction as fit_regression: only the full_sweep tier has budget
    # fields at all -- activations_only/baseline_single records would KeyError
    # on r["p_by_budget"] below otherwise.
    swept = [r for r in records if "p_by_budget" in r]
    if len(swept) < len(records):
        print(f"{len(swept)}/{len(records)} records have full_sweep budget data "
              f"-- restricting the classification framing to those (the rest are "
              f"activations_only/baseline_single, used elsewhere, not here)")
    if not swept:
        print("no full_sweep records found -- run RUN_MODE='full_sweep' on the "
              "labeling subset first.")
        sys.exit(1)

    budgets = sorted(set(int(b) for r in swept for b in r["p_by_budget"].keys()))
    n_solved = sum(1 for r in swept if r.get("b_star") is not None)
    print(f"{n_solved}/{len(swept)} problems solved within swept budgets "
          f"-- using ALL {len(swept)} problems x {len(budgets)} budgets = "
          f"{len(swept) * len(budgets)} training rows (censoring is not a special case here)")

    type_names = sorted(set(r["type"] for r in swept))
    print("\npicking the layer with the best grouped-CV AUC:")
    best_layer, best_auc = pick_best_layer_classification(swept, type_names, budgets)
    print(f"-> using {best_layer} (CV AUC = {best_auc:.3f})")

    X, y, groups, scaler, pca = build_classification_dataset(swept, type_names, best_layer, budgets, fit=True)

    if args.compare_models:
        compare_classification_models(X, y, groups)

    model = LogisticRegressionCV(Cs=ALPHAS, max_iter=1000, cv=5).fit(X, y)

    bundle = {
        "mode": "classification", "model": model, "scaler": scaler, "pca": pca,
        "layer": best_layer, "type_names": type_names, "phi_keys": PHI_KEYS,
        "cv_score": best_auc, "budgets_seen": budgets,
        "success_threshold": 0.5,  # must match SUCCESS_THRESHOLD used in the notebook
    }

    if args.demo:
        print("\n--- demo: B_hat(x) = smallest swept budget with predicted P(success) >= threshold ---")
        X_static, _, _ = build_features(swept[:5], type_names, best_layer, scaler=scaler, pca=pca, fit=False)
        for i, r in enumerate(swept[:5]):
            probs = model.predict_proba(np.array([np.append(X_static[i], np.log(b)) for b in budgets]))[:, 1]
            solved = [b for b, p in zip(budgets, probs) if p >= 0.5]
            b_hat = min(solved) if solved else f">{max(budgets)}"
            print(f"id={r['id']:>3} type={r['type']:<22} actual B*={r['b_star']}  "
                  f"P(success) by budget={dict(zip(budgets, probs.round(2)))}  predicted B_hat={b_hat}")

    return bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="pilot_results.jsonl")
    parser.add_argument("--mode", choices=["regression", "classification"], default="regression")
    parser.add_argument("--demo", action="store_true", help="show predictions on a few example problems")
    parser.add_argument("--compare-models", action="store_true",
                         help="fit Ridge/ElasticNet/MLP/GaussianProcess (or Logistic/MLP/GP for "
                              "classification) under the same CV and print a comparison table")
    args = parser.parse_args()

    records = load_records(args.path)
    print(f"loaded {len(records)} records from {args.path} (mode={args.mode})")

    bundle = fit_regression(records, args) if args.mode == "regression" else fit_classification(records, args)

    joblib.dump(bundle, "oracle_model.joblib")
    print(f"\nsaved oracle_model.joblib (mode={bundle['mode']}, layer={bundle['layer']}, "
          f"cv_score={bundle['cv_score']:.3f})")
    print("\nTo predict on a NEW problem: build phi(x) with the same PHI_KEYS, extract "
          f"activations at {bundle['layer']} the same way as the notebook, then use "
          "build_features(..., fit=False) with bundle['scaler']/bundle['pca'], and either "
          "bundle['model'].predict(X) (regression) or scan bundle['budgets_seen'] with "
          "predict_proba and bundle['success_threshold'] (classification).")


if __name__ == "__main__":
    main()
