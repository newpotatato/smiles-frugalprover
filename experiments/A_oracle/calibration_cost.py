"""
Turns the "chicken-and-egg" labeling cost (PIPELINE.md 2.1 p.3: getting B*
for one problem requires sweeping several budget levels, which itself spends
the very budget the project is trying to save) into an actual measured
number for the pre-defense slide, instead of just acknowledging it in words.

With the tiered data collection (activations_only on a LARGE pool + full_sweep
labeling only on a smaller nested subset -- see data_prep.py), this now reports
THREE numbers instead of two:

  1. Calibration cost: tokens ACTUALLY spent producing B* labels on the
     full_sweep tier only -- exact, known in advance regardless of outcome.
  2. Oracle-guided cost projected on that SAME labeled tier (honest
     cross-validated / out-of-sample predictions) -- the apples-to-apples
     comparison, same n on both sides.
  3. Oracle-guided cost projected across the FULL large pool -- using a
     model fit on the labeled tier but applied to every OTHER problem's
     already-free activations. This is the actual amortization story: the
     calibration cost is a fixed, one-time bill on the small labeled subset,
     while the deployed oracle's benefit scales with however many more
     problems you have activations for, at no extra labeling cost.

Run locally, after merge_results.py:
    python calibration_cost.py pilot_results.jsonl
"""
import json
import sys

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut, cross_val_predict

from oracle import ALPHAS, build_features


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "pilot_results.jsonl"
    records = [json.loads(line) for line in open(path, encoding="utf-8")]
    print(f"loaded {len(records)} records from {path}")

    swept = [r for r in records if "p_by_budget" in r]
    print(f"{len(swept)}/{len(records)} records have full_sweep budget data "
          f"(calibration cost is only paid on this tier)")
    if not swept:
        print("no full_sweep records found -- run RUN_MODE='full_sweep' on the "
              "labeling subset first.")
        return

    budgets = sorted(int(b) for b in swept[0]["p_by_budget"].keys())
    print(f"budgets swept per problem: {budgets}")

    # --- 1. calibration cost: exact, deterministic, paid only on `swept` ---
    tokens_per_problem_per_sample = sum(budgets)
    print(f"\ntokens spent per (problem, sample) across the full sweep: sum({budgets}) = "
          f"{tokens_per_problem_per_sample}")
    print("NOTE: multiply by N_SAMPLES (set in the notebook, e.g. 3) for the true total -- "
          "this script reports the per-sample figure since N_SAMPLES isn't saved per-record; "
          "state your actual N_SAMPLES on the slide.")
    calibration_cost_per_problem = tokens_per_problem_per_sample  # x N_SAMPLES, noted above
    total_calibration_cost = calibration_cost_per_problem * len(swept)
    print(f"total calibration cost (x1 sample): {total_calibration_cost} tokens "
          f"across {len(swept)} LABELED problems")

    # --- 2. oracle-guided projected cost on the SAME labeled tier: honest
    # out-of-sample B_hat, apples-to-apples comparison (same n) ---
    solved = [r for r in swept if r.get("b_star") is not None]
    n = len(solved)
    if n < 15:
        print("\ntoo few solved problems to fit an oracle for the cost projection -- "
              "report the calibration cost alone on the slide.")
        return

    y = np.array([r["b_star"] for r in solved], dtype=float)
    type_names = sorted(set(r["type"] for r in solved))
    cv = LeaveOneOut() if n <= 60 else 5
    layer_names = sorted(solved[0]["activations"].keys(), key=lambda s: int(s.split("_")[1]))

    best_layer, best_preds, best_r2 = None, None, -np.inf
    for layer in layer_names:
        X, _, _ = build_features(solved, type_names, layer, fit=True)
        preds = cross_val_predict(RidgeCV(alphas=ALPHAS), X, y, cv=cv)
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else -np.inf
        if r2 > best_r2:
            best_r2, best_layer, best_preds = r2, layer, preds

    clipped_preds = np.clip(best_preds, min(budgets), max(budgets))
    oracle_guided_cost_labeled = clipped_preds.sum()
    n_censored = len(swept) - n  # unsolved within the labeled tier
    oracle_guided_cost_labeled += n_censored * max(budgets)  # charge worst case observed

    ratio_same_n = total_calibration_cost / oracle_guided_cost_labeled if oracle_guided_cost_labeled > 0 else float("nan")
    print(f"\noracle-guided cost on the SAME {len(swept)} labeled problems "
          f"(out-of-sample B_hat, layer={best_layer}, CV R^2={best_r2:.3f}): "
          f"{oracle_guided_cost_labeled:.0f} tokens ({ratio_same_n:.1f}x less than calibration, "
          f"1 attempt each)")

    # --- 3. amortization: fit a FINAL model on all `solved` data, project onto
    # every OTHER problem in the merged file that has activations but was never
    # labeled -- their cost is "free" in labeling terms, only oracle-guided
    # solving cost applies. This is the actual scale-up story. ---
    unlabeled = [r for r in records if "p_by_budget" not in r]  # never swept at all
    if unlabeled:
        X_solved, scaler, pca = build_features(solved, type_names, best_layer, fit=True)
        final_model = RidgeCV(alphas=ALPHAS).fit(X_solved, y)
        X_unlabeled, _, _ = build_features(unlabeled, type_names, best_layer, scaler=scaler, pca=pca, fit=False)
        preds_unlabeled = np.clip(final_model.predict(X_unlabeled), min(budgets), max(budgets))
        oracle_guided_cost_pool = oracle_guided_cost_labeled + preds_unlabeled.sum()
        print(f"\nAMORTIZATION: applying that same oracle to the {len(unlabeled)} OTHER problems "
              f"in the pool (never swept, activations already free) projects "
              f"~{preds_unlabeled.sum():.0f} more tokens to solve them once -- "
              f"total oracle-guided cost across all {len(records)} pool problems: "
              f"~{oracle_guided_cost_pool:.0f} tokens, against a FIXED calibration bill of "
              f"{total_calibration_cost} tokens paid only once on the {len(swept)} labeled problems.")
    else:
        print("\nno unlabeled pool problems found (large pool == labeled tier here) -- "
              "amortization projection needs a bigger activations_only pool than the "
              "labeling subset (see data_prep.py).")

    print("\nCAVEAT: even a zero-signal oracle looks 'cheaper' than the full sweep here, "
          "purely because guessing once is mechanically cheaper than testing every budget "
          "level -- don't present the ratio alone as evidence the oracle is smart. Compare "
          "against a naive 'always predict the median B*' baseline to isolate the actual "
          "value of the learned signal (CV R^2 above is the honest number for that).")


if __name__ == "__main__":
    main()
