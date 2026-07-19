"""
Third workstream: is the GEOMETRY of the activation manifold (not just its raw
predictive power, which is analyze.py's job) actually related to problem
DIFFICULTY? Ansuini et al.'s headline finding was ID-of-last-layer vs
classification accuracy on CNNs; this is the direct analogue for this
backbone: does ID (or another compression method) trend with MATH's own
difficulty levels?

Works directly on a pipeline run's artifacts (problems.jsonl +
hidden_states.parquet + budgets.jsonl). Problems without budget labels simply
have no budget fields. Parts 1, 2
and the level_num half of part 3 use ALL records (pure geometry, no budget
needed -- and now benefit from the much larger activations_only pool); only
the solvability half of part 3 needs budget labels and is restricted to
it explicitly.

Four sub-questions, all cheap (pure numpy/sklearn, no GPU, reuses data
already collected -- no extra Colab time needed):

  1. Per-level global ID: split problems into MATH's level_num buckets (1-5),
     compute ID of each bucket's activation cloud, per layer. Does ID trend
     with difficulty, the way Ansuini's ID-vs-accuracy trend did?

  2. Robustness to the compression METHOD: three independent estimators --
     TwoNN (nonlinear, 2 neighbors), MLE/Levina-Bickel (nonlinear, k
     neighbors, different derivation), and PCA "participation ratio"
     (effective rank from the eigenvalue spectrum -- a LINEAR notion of
     dimensionality). If TwoNN and MLE (both nonlinear) agree, that's real
     evidence; if either disagrees with participation ratio, that's a
     linear-vs-nonlinear distinction, not necessarily a contradiction.
     Disagreement is reported, not hidden.

  3. Per-problem local density: unlike global ID estimators (one number per
     GROUP), a per-problem proxy -- mean distance to k nearest neighbors in
     activation space -- CAN be correlated directly, per problem, with
     level_num (uses ALL records) and with accuracy/solvability (restricted
     to labeled problems, the only ones with outcomes).

Run locally (no GPU needed), against a pipeline run's artifacts:
    python -m frugalprover.analysis.geometry \
        --problems data/pilot/problems.jsonl \
        --hidden-states data/pilot/hidden_states.parquet \
        --budgets data/pilot/budgets.jsonl \
        --out-dir results/analysis
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from frugalprover.analysis._records import add_paths_arg, load_records
from frugalprover.analysis.id_estimators import mle_dimension, twonn_dimension

MIN_GROUP_SIZE = 10  # TwoNN/MLE need a reasonable number of points per level bucket
K_NEIGHBORS = 5


def participation_ratio(X):
    """Effective rank from the eigenvalue spectrum of the covariance matrix --
    a LINEAR notion of dimensionality, unlike TwoNN/MLE's nonlinear one."""
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = np.cov(Xc, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0, None)
    s1, s2 = eigvals.sum(), (eigvals ** 2).sum()
    return (s1 ** 2) / s2 if s2 > 0 else float("nan")


def local_density(X, k=K_NEIGHBORS):
    """Per-point mean distance to its k nearest neighbors -- a per-PROBLEM
    proxy (unlike the global estimators above, which only yield one number
    per group), so it can be correlated directly against level_num / outcome."""
    n = X.shape[0]
    k = min(k, n - 1)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, _ = nbrs.kneighbors(X)
    return dist[:, 1:].mean(axis=1)  # exclude self (distance 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_paths_arg(ap)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.problems, args.hidden_states, args.budgets, args.pooling)
    print(f"loaded {len(records)} records")

    swept = [r for r in records if "p_by_budget" in r]
    print(f"{len(swept)}/{len(records)} records have budget labels "
          f"(used only for the solvability half of part 3 below; everything "
          f"else here uses all {len(records)} records -- pure activation geometry)")

    levels = sorted(set(r["level_num"] for r in records))
    counts = {lv: sum(1 for r in records if r["level_num"] == lv) for lv in levels}
    print(f"problems per level (all tiers combined): {counts}")
    if min(counts.values()) < MIN_GROUP_SIZE:
        print(f"WARNING: smallest level bucket has {min(counts.values())} problems "
              f"(< {MIN_GROUP_SIZE}) -- per-level ID estimates below will be noisy. "
              f"The per-problem local-density analysis (part 3) doesn't need "
              f"per-level grouping and is more reliable at this n.")

    layer_names = sorted(records[0]["activations"].keys(), key=lambda s: int(s.split("_")[1]))
    # Reuse the same confirmatory/exploratory convention as analyze.py: last
    # layer (a priori, Ansuini-motivated) + middle layer (exploratory contrast).
    candidate_layers = [layer_names[-1], layer_names[len(layer_names) // 2]]
    print(f"focusing on layers: {candidate_layers} (last=confirmatory, middle=exploratory contrast)")

    # --- 1 & 2: per-level ID (TwoNN, MLE) and participation ratio, per candidate layer ---
    for layer in candidate_layers:
        print(f"\n--- {layer}: ID (TwoNN, MLE) and participation ratio by MATH difficulty level ---")
        twonn_ids, mle_ids, prs = [], [], []
        for lv in levels:
            X = np.array([r["activations"][layer] for r in records if r["level_num"] == lv])
            if len(X) < 6:
                print(f"  level {lv}: only {len(X)} problems, skipping (too few for a stable estimate)")
                twonn_ids.append(float("nan"))
                mle_ids.append(float("nan"))
                prs.append(float("nan"))
                continue
            enough = len(X) >= MIN_GROUP_SIZE
            d_two = twonn_dimension(X) if enough else float("nan")
            d_mle = mle_dimension(X) if enough else float("nan")
            pr = participation_ratio(X)
            twonn_ids.append(d_two)
            mle_ids.append(d_mle)
            prs.append(pr)
            two_str = "n/a" if np.isnan(d_two) else f"{d_two:.2f}"
            mle_str = "n/a" if np.isnan(d_mle) else f"{d_mle:.2f}"
            print(f"  level {lv} (n={len(X)}): TwoNN ID = {two_str}  MLE ID = {mle_str}  "
                  f"participation ratio = {pr:.2f}")

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        axes[0].plot(levels, twonn_ids, "o-")
        axes[0].set_xlabel("MATH level")
        axes[0].set_ylabel("TwoNN intrinsic dimension")
        axes[0].set_title(f"{layer}: TwoNN ID vs difficulty")
        axes[1].plot(levels, mle_ids, "o-", color="seagreen")
        axes[1].set_xlabel("MATH level")
        axes[1].set_ylabel("MLE (Levina-Bickel) ID")
        axes[1].set_title(f"{layer}: MLE ID vs difficulty")
        axes[2].plot(levels, prs, "o-", color="darkorange")
        axes[2].set_xlabel("MATH level")
        axes[2].set_ylabel("PCA participation ratio")
        axes[2].set_title(f"{layer}: effective rank vs difficulty")
        plt.tight_layout()
        fname = f"id_vs_difficulty_{layer}.png"
        plt.savefig(out_dir / fname, dpi=150)
        print(f"saved {fname}")

        valid = [(i, t, m, p) for i, (t, m, p) in enumerate(zip(twonn_ids, mle_ids, prs)) if not np.isnan(t)]
        if len(valid) >= 3:
            idx, tvals, mvals, pvals = zip(*valid)
            lv_valid = [levels[i] for i in idx]
            r_two, p_two = spearmanr(lv_valid, tvals)
            r_mle, p_mle = spearmanr(lv_valid, mvals)
            r_pr, p_pr = spearmanr(lv_valid, pvals)
            print(f"  Spearman(level, TwoNN ID) = {r_two:+.3f} (p={p_two:.3f})")
            print(f"  Spearman(level, MLE ID)   = {r_mle:+.3f} (p={p_mle:.3f})")
            print(f"  Spearman(level, participation ratio) = {r_pr:+.3f} (p={p_pr:.3f})")
            if np.sign(r_two) == np.sign(r_mle) and min(abs(r_two), abs(r_mle)) > 0.1:
                print("  -> TwoNN and MLE (both nonlinear, different derivations) AGREE on "
                      "direction -- meaningfully stronger evidence than either alone.")
            elif np.sign(r_two) != np.sign(r_mle) and min(abs(r_two), abs(r_mle)) > 0.1:
                print("  -> TwoNN and MLE DISAGREE on direction -- treat the nonlinear-ID "
                      "trend here as unstable, not a real effect, regardless of p-values.")
            if np.sign(r_two) != np.sign(r_pr) and min(abs(r_two), abs(r_pr)) > 0.1:
                print("  -> nonlinear ID (TwoNN) and linear participation ratio disagree on "
                      "direction -- report both, don't just quote the one that tells a nicer story.")

    # --- 3: per-problem local density vs level_num (all records) and vs
    # solvability (restricted to labeled problems, the only ones
    # with outcome labels at all) ---
    print("\n--- per-problem local density (mean dist to k-NN) vs difficulty / outcome ---")
    for layer in candidate_layers:
        X_all = StandardScaler().fit_transform(np.array([r["activations"][layer] for r in records]))
        density_all = local_density(X_all)
        level_arr = np.array([r["level_num"] for r in records], dtype=float)
        r_lvl, p_lvl = spearmanr(density_all, level_arr)
        print(f"{layer}: Spearman(local density [n={len(records)}], level_num) = "
              f"{r_lvl:+.3f} (p={p_lvl:.3f})")

        swept_idx = [i for i, r in enumerate(records) if "p_by_budget" in r]
        if not swept_idx:
            print("  no full_sweep records -- skipping density-vs-solvability "
                  "(run `frugalprover budget` first)")
        else:
            max_budget = max(int(b) for b in records[swept_idx[0]]["p_by_budget"].keys())
            solvability = np.array([records[i]["p_by_budget"][str(max_budget)] for i in swept_idx])
            density_swept = density_all[swept_idx]
            if np.std(solvability) < 1e-9:
                print(f"  Spearman(local density, accuracy@max_budget) = n/a "
                      f"(constant across the {len(swept_idx)} swept problems)")
            else:
                r_solv, p_solv = spearmanr(density_swept, solvability)
                print(f"  Spearman(local density [n={len(swept_idx)} swept], "
                      f"accuracy@max_budget) = {r_solv:+.3f} (p={p_solv:.3f})")

        plt.figure(figsize=(5, 4))
        plt.scatter(level_arr, density_all, alpha=0.5)
        plt.xlabel("MATH level")
        plt.ylabel(f"local density ({layer}, mean dist to {K_NEIGHBORS}-NN)")
        plt.title(f"{layer}: per-problem local density vs difficulty (n={len(records)})")
        plt.tight_layout()
        fname = f"local_density_vs_level_{layer}.png"
        plt.savefig(out_dir / fname, dpi=150)
        print(f"saved {fname}")

    print(f"\nNote: parts 1-2 and the level_num half of part 3 use all {len(records)} "
          f"records (activations_only + full_sweep + baseline_single combined); the "
          f"solvability half of part 3 is restricted to the {len(swept)} full_sweep "
          f"records, since that's the only tier with outcome labels.")


if __name__ == "__main__":
    main()
