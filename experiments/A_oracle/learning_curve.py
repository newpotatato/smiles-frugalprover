"""
Checks the paper's claim that the budget oracle is "updated online as
outcomes accrue" (abstract / Formulation section) -- oracle.py fits ONE
static model on the whole pilot dataset and stops there; this script instead
simulates outcomes arriving one problem at a time and asks: does the
oracle's held-out AUC actually improve as more outcomes accrue, or does it
plateau / not help at all (a legitimate finding at this n)?

A fixed validation set is carved out ONCE, up front, and never used for
training at any checkpoint size -- that's the yardstick the "online" curve
is measured against. Repeated over multiple random arrival orders and
averaged, since a single random order is noisy at pilot scale.

Usage:
    python learning_curve.py pilot_results.jsonl
"""
import json
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from oracle import build_classification_dataset, pick_best_layer_classification

N_REPEATS = 30
VAL_FRACTION = 0.25
CHECKPOINT_FRACTIONS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # of the accrual pool
SEED = 0


def evaluate_at_size(pool, val, type_names, layer, k, rng):
    idx = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
    train_subset = [pool[i] for i in idx]
    budgets = sorted(set(int(b) for r in (train_subset + val) for b in r["p_by_budget"].keys()))

    X_train, y_train, _, scaler, pca = build_classification_dataset(
        train_subset, type_names, layer, budgets, fit=True)
    if len(set(y_train)) < 2:
        return float("nan")  # this random draw has only one class -- can't fit/score AUC

    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)

    X_val, y_val, _, _, _ = build_classification_dataset(
        val, type_names, layer, budgets, scaler=scaler, pca=pca, fit=False)
    if len(set(y_val)) < 2:
        return float("nan")
    return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "pilot_results.jsonl"
    all_records = [json.loads(line) for line in open(path, encoding="utf-8")]
    records = [r for r in all_records if "p_by_budget" in r]
    print(f"loaded {len(all_records)} records from {path}; {len(records)} have full_sweep "
          f"budget data (this script simulates accrual of LABELED outcomes, so only that "
          f"tier is usable -- activations_only/baseline_single records are skipped here)")
    if len(records) < 20:
        print("too few full_sweep records for a meaningful accrual curve -- run RUN_MODE="
              "'full_sweep' on more of the labeling subset first.")
        return

    rng = np.random.default_rng(SEED)
    shuffled = list(records)
    rng.shuffle(shuffled)
    n_val = max(10, int(len(shuffled) * VAL_FRACTION))
    val, pool = shuffled[:n_val], shuffled[n_val:]
    print(f"held-out validation set: {n_val} problems (fixed, never trained on)")
    print(f"accrual pool: {len(pool)} problems")

    type_names = sorted(set(r["type"] for r in records))
    budgets_all = sorted(set(int(b) for r in records for b in r["p_by_budget"].keys()))
    print("\npicking a fixed layer once on the full pool (kept fixed across all checkpoint sizes "
          "-- this curve is about data quantity, not re-tuning the layer each time):")
    best_layer, _ = pick_best_layer_classification(pool, type_names, budgets_all)
    print(f"-> using {best_layer}\n")

    sizes = sorted(set(max(10, int(len(pool) * f)) for f in CHECKPOINT_FRACTIONS))
    curve = {k: [] for k in sizes}
    for rep in range(N_REPEATS):
        rep_rng = np.random.default_rng(rep)
        for k in sizes:
            auc = evaluate_at_size(pool, val, type_names, best_layer, k, rep_rng)
            if not np.isnan(auc):
                curve[k].append(auc)

    print("accrued training size -> held-out AUC (mean +/- std over repeats):")
    means, stds = [], []
    for k in sizes:
        vals = curve[k]
        m, s = (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), float("nan"))
        means.append(m)
        stds.append(s)
        print(f"  n={k:>3}: AUC = {m:.3f} +/- {s:.3f}  ({len(vals)} valid repeats)")

    plt.figure(figsize=(5.5, 4))
    plt.errorbar(sizes, means, yerr=stds, marker="o", capsize=3)
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="random guessing")
    plt.xlabel("accrued problems used to fit the oracle")
    plt.ylabel("held-out AUC")
    plt.title("Does the oracle improve as outcomes accrue?")
    plt.legend()
    plt.tight_layout()
    plt.savefig("online_learning_curve.png", dpi=150)
    print("\nsaved online_learning_curve.png")

    valid = [(m, s) for m, s in zip(means, stds) if not np.isnan(m)]
    if valid and valid[-1][0] > valid[0][0] + 0.03:
        print("-> AUC trends up with more accrued data: supports keeping the oracle "
              "updated online rather than freezing it after the calibration batch.")
    else:
        print("-> no clear improvement from more accrued data at this n -- an honest "
              "finding: either the signal saturates early, or n is too small to see the "
              "online-update benefit yet. Worth presenting as-is rather than assuming "
              "online updating helps just because the abstract asserts it.")


if __name__ == "__main__":
    main()
