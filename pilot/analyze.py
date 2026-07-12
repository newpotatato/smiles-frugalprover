"""
Trek A / step 3 (local, no GPU needed): analyze pilot_results.jsonl produced by
colab_budget_oracle_pilot.ipynb.

Answers four questions for the pre-defense slide:
  1. ID-vs-layer profile for this backbone on math problem statements
     (Ansuini-style replication).
  2. Do activations beat TWO baselines at predicting B*: (a) hand-picked
     surface features phi(x) + subject type, and (b) phi(x) + a TF-IDF /
     bag-of-words model on the problem text -- a much harder baseline to
     beat than length/LaTeX counts, since lexical cues ("prove that...",
     "find the number of...") can already leak a lot of difficulty signal.
  3. WHICH layer, chosen how? The notebook now saves EVERY layer, not 3
     pre-picked depths, so this script first does a cheap CV-R^2 sweep across
     all of them (a real Ansuini-style depth curve, not 3 bars) and THEN runs
     the expensive diagnostics (bootstrap/permutation/partial-correlation)
     only on two layers, to keep the multiple-comparisons problem honest:
       - a *confirmatory* pick made a priori from the literature (the final
         hidden layer, matching Ansuini et al.'s own headline finding) --
         no cherry-picking penalty, this one was chosen before looking at
         the data.
       - an *exploratory* pick (argmax of the full sweep) -- reported WITH
         a Bonferroni-adjusted significance threshold (0.05 / n_layers),
         since "best of ~30 noisy estimates" inflates false positives if
         treated like a single planned test.
  4. Is any observed improvement real or noise, given n ~ 40-80 problems and
     activation vectors in the thousands of dimensions (the exact overfitting
     risk flagged in PIPELINE.md 2.1 p.2)? Answered three ways:
       - cross-validated R^2 (LeaveOneOut for small n), not in-sample fit
       - bootstrap 95% CI on delta R^2 (augmented - baseline)
       - permutation test: shuffle activations across problems, see how often
         a spurious delta this large shows up by chance
       - partial correlation: residualize both B* and the activations' PC1
         on the baseline features, correlate the residuals -- a practical
         stand-in for I(g_ell; B* | phi), sidestepping the open problem of
         estimating mutual information directly in high dimensions
         (see PIPELINE.md section 4).

Run locally:
    python analyze.py pilot_results.jsonl
"""
import json
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler

from twonn import twonn_dimension

PHI_KEYS = ["char_len", "word_len", "latex_cmd_count", "dollar_count",
            "brace_depth", "digit_count", "eq_count"]
ALPHAS = np.logspace(-3, 3, 13)
N_BOOT = 300
N_PERM = 200
RNG = np.random.default_rng(0)


def load_records(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def type_one_hot(records):
    types = sorted(set(r["type"] for r in records))
    return np.array([[1.0 if r["type"] == t else 0.0 for t in types] for r in records]), types


def cv_r2(X, y, cv):
    model = RidgeCV(alphas=ALPHAS)
    preds = cross_val_predict(model, X, y, cv=cv)
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def fit_eval(Xtr, ytr, Xte, yte):
    model = RidgeCV(alphas=ALPHAS).fit(Xtr, ytr)
    pred = model.predict(Xte)
    ss_res = np.sum((yte - pred) ** 2)
    ss_tot = np.sum((yte - yte.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def bootstrap_delta_r2(X_base, X_aug, y, n_boot=N_BOOT):
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        oob = np.setdiff1d(np.arange(n), idx)
        if len(oob) < 5:
            continue
        r2b = fit_eval(X_base[idx], y[idx], X_base[oob], y[oob])
        r2a = fit_eval(X_aug[idx], y[idx], X_aug[oob], y[oob])
        if np.isnan(r2b) or np.isnan(r2a):
            continue
        deltas.append(r2a - r2b)
    return np.array(deltas)


def permutation_test(X_base_only, extra_block, y, cv, observed_delta, n_perm=N_PERM):
    """Shuffle `extra_block` (e.g. activation PCs) across problems, keeping
    X_base_only/y pairing intact, and see how often the resulting delta R^2
    is >= the one actually observed."""
    r2_base = cv_r2(X_base_only, y, cv)
    null_deltas = np.empty(n_perm)
    n = len(y)
    for i in range(n_perm):
        perm = RNG.permutation(n)
        X_perm = np.hstack([X_base_only, extra_block[perm]])
        null_deltas[i] = cv_r2(X_perm, y, cv) - r2_base
    p_value = (1 + np.sum(null_deltas >= observed_delta)) / (n_perm + 1)
    return p_value, null_deltas


def partial_correlation(baseline_feats, y, activation_pcs, cv):
    """Residualize y and PC1 of activations on the baseline features, then
    correlate the residuals -- practical proxy for I(g_ell; B* | phi)."""
    ridge = RidgeCV(alphas=ALPHAS)
    y_resid = y - cross_val_predict(ridge, baseline_feats, y, cv=cv)
    pc1 = activation_pcs[:, 0]
    pc1_resid = pc1 - cross_val_predict(ridge, baseline_feats, pc1, cv=cv)
    r_pearson, p_pearson = pearsonr(y_resid, pc1_resid)
    r_spearman, p_spearman = spearmanr(y_resid, pc1_resid)
    return r_pearson, p_pearson, r_spearman, p_spearman


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "pilot_results.jsonl"
    records = load_records(path)
    print(f"loaded {len(records)} records from {path}")

    # After merge_results.py's id-based join across tiers, most records will
    # be activations_only (phi + activations, no budget fields at all) -- only
    # the smaller full_sweep tier has p_by_budget/sc_by_budget/b_star. That's
    # expected, not an error; filter to the tier that actually has them.
    swept = [r for r in records if "p_by_budget" in r]
    print(f"{len(swept)}/{len(records)} records have full_sweep budget data "
          f"(the rest are activations_only/baseline_single -- used for the "
          f"layer/ID analysis below, but not for the B*/accuracy@budget sections)")
    if not swept:
        print("no full_sweep records found -- nothing more to do (run RUN_MODE="
              "'full_sweep' on the labeling subset first).")
        return

    # --- Track B, free of charge: single-sample vs self-consistency accuracy@budget,
    # aggregated over the full_sweep tier only (defined regardless of censoring) ---
    print("\n--- Track B baseline: accuracy@budget, single sample vs self-consistency ---")
    budgets = sorted(int(b) for b in swept[0]["p_by_budget"].keys())
    single_acc, sc_acc = {}, {}
    for b in budgets:
        single_acc[b] = np.mean([r["p_by_budget"][str(b)] for r in swept])
        sc_acc[b] = np.mean([r["sc_by_budget"][str(b)] for r in swept])
        print(f"budget={b:>4}: single-sample acc={single_acc[b]:.3f}  self-consistency acc={sc_acc[b]:.3f}")

    plt.figure(figsize=(5, 4))
    plt.plot(budgets, [single_acc[b] for b in budgets], "o-", label="single sample")
    plt.plot(budgets, [sc_acc[b] for b in budgets], "o-", label="self-consistency (majority vote)")
    plt.xlabel("token budget")
    plt.ylabel("accuracy")
    plt.title("Accuracy vs budget: single-sample vs self-consistency")
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy_vs_budget.png", dpi=150)
    print("saved accuracy_vs_budget.png\n")

    # censoring is only meaningful WITHIN the full_sweep tier -- a problem that
    # was never swept at all (activations_only/baseline_single) isn't "censored",
    # it just wasn't tested, so the denominator here is `swept`, not `records`.
    solved = [r for r in swept if r.get("b_star") is not None]
    censored_frac = 1 - len(solved) / len(swept) if swept else float("nan")
    print(f"{len(solved)}/{len(swept)} swept problems solved within tested budgets "
          f"({censored_frac:.0%} censored)")
    n = len(solved)
    if n < 15:
        print("WARNING: too few solved problems for a meaningful regression. "
              "Report the censoring rate itself as a finding, or widen BUDGETS "
              "/ lower SUCCESS_THRESHOLD in the notebook and re-run.")
        return

    y = np.array([r["b_star"] for r in solved], dtype=float)
    cv = LeaveOneOut() if n <= 60 else 5

    # --- external / "surface" features: hand-picked phi + subject type one-hot ---
    phi_numeric = np.array([[r["phi"][k] for k in PHI_KEYS] for r in solved], dtype=float)
    phi_numeric = StandardScaler().fit_transform(phi_numeric)
    type_oh, type_names = type_one_hot(solved)
    phi = np.hstack([phi_numeric, type_oh])
    print(f"phi(x): {PHI_KEYS} + type one-hot {type_names} -> {phi.shape[1]} dims")

    r2_phi = cv_r2(phi, y, cv)
    print(f"\nB0: phi only                       CV R^2 = {r2_phi:.3f}")

    # --- stronger baseline: phi + TF-IDF bag-of-words on the problem text ---
    texts = [r.get("problem", "") for r in solved]
    if all(t == "" for t in texts):
        print("NOTE: records have no 'problem' text (older notebook run) -- "
              "skipping the TF-IDF baseline. Re-run the notebook to get it.")
        baseline = phi
        r2_baseline = r2_phi
        baseline_label = "phi only"
    else:
        tfidf = TfidfVectorizer(max_features=300, stop_words="english").fit_transform(texts).toarray()
        n_pcs_tfidf = min(10, n - 2)
        tfidf_pcs = PCA(n_components=n_pcs_tfidf).fit_transform(StandardScaler().fit_transform(tfidf))
        baseline = np.hstack([phi, tfidf_pcs])
        r2_baseline = cv_r2(baseline, y, cv)
        baseline_label = "phi + TF-IDF"
        print(f"B1: phi + TF-IDF ({n_pcs_tfidf} PCs)   CV R^2 = {r2_baseline:.3f}")

    # --- ID-vs-layer profile (Ansuini-style), computed over ALL records ---
    layer_names = sorted(solved[0]["activations"].keys(), key=lambda s: int(s.split("_")[1]))
    print("\n--- Intrinsic dimension vs layer (TwoNN) ---")
    ids_by_layer = []
    for layer in layer_names:
        X_layer = np.array([r["activations"][layer] for r in records])
        d = twonn_dimension(X_layer)
        ids_by_layer.append(d)
        print(f"{layer}: ID = {d:.2f}  (ambient dim = {X_layer.shape[1]})")

    plt.figure(figsize=(5, 4))
    plt.plot([int(l.split("_")[1]) for l in layer_names], ids_by_layer, "o-")
    plt.xlabel("layer index")
    plt.ylabel("intrinsic dimension (TwoNN)")
    plt.title("ID of problem-statement activations vs. layer")
    plt.tight_layout()
    plt.savefig("id_vs_layer.png", dpi=150)
    print("saved id_vs_layer.png")

    # --- STEP 1: cheap full-depth sweep -- plain CV R^2 for every layer, no
    # bootstrap/permutation yet. This is the actual "which layer" empirics:
    # an Ansuini-style curve over the WHOLE depth, not 3 pre-picked points. ---
    print(f"\n--- Layer-depth sweep: CV R^2 on top of [{baseline_label}], all {len(layer_names)} layers ---")
    layer_pcs = {}
    r2_by_layer = {}
    for layer in layer_names:
        acts = StandardScaler().fit_transform(np.array([r["activations"][layer] for r in solved]))
        n_pcs = min(10, n - 2, acts.shape[1])
        pcs = PCA(n_components=n_pcs).fit_transform(acts)
        layer_pcs[layer] = pcs
        r2_by_layer[layer] = cv_r2(np.hstack([baseline, pcs]), y, cv)
        print(f"  {layer}: CV R^2 = {r2_by_layer[layer]:.3f}")

    layer_idx_sorted = [int(l.split("_")[1]) for l in layer_names]
    plt.figure(figsize=(6, 4))
    plt.plot(layer_idx_sorted, [r2_by_layer[l] for l in layer_names], "o-")
    plt.axhline(r2_baseline, color="gray", linestyle="--", linewidth=1, label=f"baseline ({baseline_label})")
    plt.xlabel("layer index (0 = embeddings)")
    plt.ylabel("cross-validated R^2 (predicting B*)")
    plt.title("Full layer-depth sweep: is there a real peak, or just noise?")
    plt.legend()
    plt.tight_layout()
    plt.savefig("r2_vs_layer_sweep.png", dpi=150)
    print("saved r2_vs_layer_sweep.png")

    # --- STEP 2: expensive diagnostics on just TWO layers, to keep multiple
    # comparisons honest --
    #   confirmatory = last layer, picked a priori from Ansuini et al.'s own
    #                  finding (no cherry-picking penalty)
    #   exploratory  = argmax of the step-1 sweep, reported with a
    #                  Bonferroni-adjusted alpha since it's "best of N" ---
    confirmatory_layer = layer_names[-1]
    exploratory_layer = max(layer_names, key=lambda l: r2_by_layer[l])
    bonferroni_alpha = 0.05 / len(layer_names)
    print(f"\nconfirmatory layer (a priori, last hidden state): {confirmatory_layer}")
    print(f"exploratory layer (argmax of {len(layer_names)} layers): {exploratory_layer}  "
          f"-> Bonferroni-adjusted alpha = {bonferroni_alpha:.4f} (not 0.05)")

    diagnostics = {}
    picks = {"confirmatory": confirmatory_layer}
    if exploratory_layer != confirmatory_layer:
        picks["exploratory"] = exploratory_layer
    for tag, layer in picks.items():
        pcs = layer_pcs[layer]
        X_aug = np.hstack([baseline, pcs])
        r2_aug = r2_by_layer[layer]
        observed_delta = r2_aug - r2_baseline

        deltas = bootstrap_delta_r2(baseline, X_aug, y)
        ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])

        p_value, _ = permutation_test(baseline, pcs, y, cv, observed_delta)
        r_p, p_p, r_s, p_s = partial_correlation(baseline, y, pcs, cv)
        diagnostics[tag] = dict(layer=layer, r2=r2_aug, delta=observed_delta,
                                 ci=(ci_lo, ci_hi), p_value=p_value,
                                 pearson=(r_p, p_p), spearman=(r_s, p_s))

        alpha = bonferroni_alpha if tag == "exploratory" else 0.05
        sig = "***" if (ci_lo > 0 and p_value < alpha) else ("(overlaps 0)" if ci_lo < 0 < ci_hi else "")
        print(f"\n[{tag}] {layer}: CV R^2={r2_aug:.3f}  delta={observed_delta:+.3f} "
              f"95% CI=[{ci_lo:+.3f}, {ci_hi:+.3f}] {sig}  perm p={p_value:.3f} "
              f"(alpha used: {alpha:.4f})  "
              f"partial corr (Pearson)={r_p:+.3f} (p={p_p:.3f}), (Spearman)={r_s:+.3f} (p={p_s:.3f})")

    plt.figure(figsize=(6, 4))
    labels = [baseline_label] + [f"{tag}\n({d['layer']})" for tag, d in diagnostics.items()]
    values = [r2_baseline] + [d["r2"] for d in diagnostics.values()]
    errs = [0] + [(d["ci"][1] - d["ci"][0]) / 2 for d in diagnostics.values()]
    plt.bar(labels, values, yerr=errs, capsize=4)
    plt.ylabel("cross-validated R^2 (predicting B*)")
    plt.title("Confirmatory (a priori) vs exploratory (best-of-N) layer pick")
    plt.tight_layout()
    plt.savefig("baseline_vs_activations_r2.png", dpi=150)
    print("\nsaved baseline_vs_activations_r2.png")

    conf = diagnostics["confirmatory"]
    print(f"\nSUMMARY: baseline ({baseline_label}) R^2={r2_baseline:.3f}; "
          f"confirmatory layer ({conf['layer']}) R^2={conf['r2']:.3f} "
          f"(delta={conf['delta']:+.3f}, 95% CI={conf['ci']}, perm p={conf['p_value']:.3f})")
    if conf["ci"][0] > 0 and conf["p_value"] < 0.05:
        print("-> the a priori (literature-motivated) layer shows signal beyond "
              f"{baseline_label} that survives bootstrap + permutation checks: "
              "supports H1's deconfounding claim without a multiple-comparisons caveat.")
    else:
        print("-> the a priori layer shows no signal distinguishable from noise at this n -- "
              "an honest negative/inconclusive result (see PIPELINE.md 2.1 risk #2).")
    if "exploratory" in diagnostics:
        exp = diagnostics["exploratory"]
        if exp["ci"][0] > 0 and exp["p_value"] < bonferroni_alpha:
            print(f"-> the exploratory best-of-{len(layer_names)} layer ({exp['layer']}) ALSO survives "
                  "the stricter Bonferroni-adjusted threshold -- worth reporting as a secondary finding, "
                  "but flag it as exploratory since the layer was chosen after seeing the data.")
        else:
            print(f"-> the exploratory best-of-{len(layer_names)} layer ({exp['layer']}) does NOT survive "
                  "Bonferroni correction -- treat its raw p-value as likely fishing, not a real effect.")


if __name__ == "__main__":
    main()
