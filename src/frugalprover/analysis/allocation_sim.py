"""H2 offline: does budget-aware allocation beat uniform, on the labels we have?

The live experiment (`frugalprover allocate`) buys one operating point for one
GPU-hour. This buys the whole success-vs-compute curve for nothing, because a
budget sweep already measured what every problem does at every grid budget: the
A2 `p` field IS p_i(B). So an allocation can be *scored* without re-solving
anything -- hand each problem a cap, look up its measured success at that cap,
and sum.

    expected solved = sum_i p_measured_i(B_i)   subject to  sum_i B_i <= B_tot

The policies are imported from `allocate/policies.py`, not reimplemented here.
That is the point of keeping them pure: if the simulation and the live run
disagree, it is the world that differs, not the allocator.

Run it before renting a GPU. It answers three things:

  1. Does *any* policy beat uniform on this data? If not, the live run has
     nothing to find and the GPU hours are better spent on more Stage 2 labels.
  2. Which triage threshold? (`--sweep-triage`) -- a free sweep offline, one
     whole arm each on the pod.
  3. How much of the gap to the perfect-information ceiling (`true_bstar`) does
     prediction actually close?

    python -m frugalprover.analysis.allocation_sim \
        --problems data/label5h/problems.jsonl \
        --budgets  data/label5h/budgets.jsonl \
        --out-dir  results/analysis

`--hidden-states` is optional: without it the oracle is fitted on
surface+subject alone, which is the strongest baseline anyway
(results/deepseek_r1_qwen7b_n3/baselines.json: 0.8043 vs 0.7996 with
activations). Pass it to run the same simulation with the activation oracle.

CAVEAT, and it is the same one analysis/calibration.py ends on: an allocation
that merely spends *less* is not evidence of a smart oracle. Read the curves,
not one ratio -- the honest comparison is uniform at the same B_tot, and the
honest ceiling is `true_bstar`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from frugalprover.allocate.policies import POLICIES, Candidate, allocate
from frugalprover.common.grading import surface_features
from frugalprover.common.logging import configure as configure_logging
from frugalprover.common.logging import get_logger
from frugalprover.oracle.model.classification import ClassificationOracle
from frugalprover.oracle.model.dataset import OracleDataset

log = get_logger(__name__)

#: Arms to simulate. `true_bstar` is the perfect-information ceiling and only
#: exists offline; everything else is exactly what the live run executes.
ARMS = ["uniform", "length", "oracle_bhat", "oracle_greedy", "oracle_optimal",
        "oracle_triage", "true_bstar"]


def out_of_fold_curves(
    ds: OracleDataset, features: list[str], layer: str | None, n_pcs: int
) -> dict[str, dict[int, float]]:
    """Honest predicted success curves: p_hat_i(B) from a model that never saw i.

    Same protocol as the pipeline's own `cv_score` -- LeaveOneGroupOut grouped by
    problem id over one row per (problem, budget), plain LogisticRegression --
    so the numbers here and in metrics.json are commensurable. Anything less
    (fitting on everything and predicting in-sample) would make the oracle arms
    look good for the one reason that cannot transfer to the pod.

    The feature blocks are still fitted on all problems, exactly as
    `ClassificationOracle.score` does. It is a small optimistic leak through the
    scaler and PCA rotation, shared by every number this project reports, and
    correcting it here alone would make this incomparable with the rest.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

    oracle = ClassificationOracle(feature_names=features, n_pcs=n_pcs)
    X, y, groups, budgets = oracle._build(ds, layer, fit=True)
    splits = list(LeaveOneGroupOut().split(X, y, groups))
    proba = cross_val_predict(
        LogisticRegression(max_iter=1000), X, y, cv=splits, method="predict_proba"
    )[:, 1]

    # Rebuild `_expand`'s row order to attach each prediction to its (id, budget).
    curves: dict[str, dict[int, float]] = {}
    row = 0
    for p in ds.problems:
        record = ds.budgets[p.id]
        for b in budgets:
            if b not in record.budgets:
                continue
            curves.setdefault(p.id, {})[b] = float(proba[row])
            row += 1
    assert row == len(proba), "row order diverged from ClassificationOracle._expand"
    return curves


def build_candidates(
    ds: OracleDataset, curves: dict[str, dict[int, float]], tau: float
) -> tuple[list[Candidate], list[Candidate]]:
    """(predicted, measured) candidate lists -- same problems, different curves.

    The measured list drives `true_bstar`, so the ceiling is the identical greedy
    rule run with the labels instead of a prediction.
    """
    predicted, measured = [], []
    for p in ds.problems:
        record = ds.budgets[p.id]
        char_len = surface_features(p.problem)["char_len"]
        p_hat = curves[p.id]
        b_hat = min((b for b, v in sorted(p_hat.items()) if v >= tau), default=None)
        predicted.append(Candidate(id=p.id, char_len=char_len, p_by_budget=p_hat, b_hat=b_hat))
        p_true = {int(b): float(v) for b, v in record.p.items()}
        measured.append(Candidate(
            id=p.id, char_len=char_len, p_by_budget=p_true, b_hat=record.b_star,
        ))
    return predicted, measured


def score(caps: list[int], measured: list[Candidate]) -> tuple[float, int]:
    """Expected problems solved, and tokens allotted, for one cap vector."""
    return sum(c.p(cap) for cap, c in zip(caps, measured)), sum(caps)


def simulate(
    predicted: list[Candidate],
    measured: list[Candidate],
    grid: list[int],
    b_bars: list[float],
    triage_threshold: float,
    arms: list[str],
) -> list[dict]:
    """One row per (arm, B_tot) across the compute sweep."""
    n = len(predicted)
    rows = []
    for b_bar in b_bars:
        total = int(round(b_bar * n))
        for arm in arms:
            cands = measured if arm == "true_bstar" else predicted
            caps = allocate(arm, cands, grid, total, triage_threshold)
            solved, allotted = score(caps, measured)
            rows.append({
                "arm": arm,
                "b_bar": b_bar,
                "b_tot": total,
                "allotted": allotted,
                "expected_solved": round(solved, 3),
                "solve_rate": round(solved / n, 4),
                "n_skipped": sum(1 for c in caps if c <= 0),
                "solved_per_1m_allotted": round(1e6 * solved / allotted, 2) if allotted else 0.0,
            })
    return rows


def sweep_triage(
    predicted: list[Candidate],
    measured: list[Candidate],
    grid: list[int],
    b_bar: float,
    thresholds: list[float],
) -> list[dict]:
    """Pick the abstention threshold offline, where it costs nothing."""
    n = len(predicted)
    total = int(round(b_bar * n))
    out = []
    for t in thresholds:
        caps = allocate("oracle_triage", predicted, grid, total, t)
        solved, allotted = score(caps, measured)
        out.append({
            "triage_threshold": t,
            "expected_solved": round(solved, 3),
            "n_skipped": sum(1 for c in caps if c <= 0),
            "allotted": allotted,
        })
    return out


def ceiling_noise_bias(rows: list[dict], grid: list[int]) -> float:
    """How much of the ceiling is winner's curse rather than real headroom.

    At `b_bar = max(grid)` every problem can afford the top budget, so no
    allocation choice should be able to beat uniform. `true_bstar` does anyway,
    because measured `p` at n=3 takes values in {0, 1/3, 2/3, 1} and is not
    monotone in B: for some problems `p(2048) > p(8192)` by pure sampling noise,
    and an optimizer handed those numbers will happily bank the noise. The gap
    at that point is a floor on how much of the ceiling everywhere else is the
    same artifact. It is not subtracted automatically -- it is reported, because
    the honest statement is "the ceiling is inflated by about this much".
    """
    top = max(r["b_bar"] for r in rows)
    at = {r["arm"]: r["expected_solved"] for r in rows if abs(r["b_bar"] - top) < 1e-9}
    if "true_bstar" not in at or "uniform" not in at:
        return 0.0
    return round(at["true_bstar"] - at["uniform"], 3)


def verdict(rows: list[dict], b_bar: float, grid: list[int]) -> dict:
    """Does anything beat uniform -- at the chosen operating point, and anywhere?"""
    at = {r["arm"]: r for r in rows if abs(r["b_bar"] - b_bar) < 1e-9}
    if "uniform" not in at:
        return {}
    base = at["uniform"]["expected_solved"]
    ceiling = at.get("true_bstar", {}).get("expected_solved", base)
    headroom = ceiling - base
    gains = {
        arm: round(r["expected_solved"] - base, 3)
        for arm, r in at.items() if arm != "uniform"
    }
    learned = [a for a in gains if a != "true_bstar"]
    best = max(learned, key=gains.get, default=None)

    # The same question across the whole compute sweep. A policy that wins only
    # at one B_tot has not "raised the entire success-vs-compute curve"
    # (RESEARCH_PLAN Phase 2), and picking that point after the fact would be
    # exactly the cherry-picking the layer-sweep discipline exists to avoid --
    # so report where the gain lives and how wide the winning region is.
    by_bbar = {}
    for r in rows:
        by_bbar.setdefault(r["b_bar"], {})[r["arm"]] = r["expected_solved"]
    sweep = {
        b: round(max((v[a] for a in learned if a in v), default=0.0) - v.get("uniform", 0.0), 3)
        for b, v in sorted(by_bbar.items())
    }
    winning = [b for b, g in sweep.items() if g > 0]

    return {
        "b_bar": b_bar,
        "uniform_expected_solved": base,
        "ceiling_expected_solved": ceiling,
        "headroom_over_uniform": round(headroom, 3),
        "ceiling_noise_bias": ceiling_noise_bias(rows, grid),
        "gain_over_uniform": gains,
        "best_learned_arm": best,
        "fraction_of_headroom_captured": (
            round(gains[best] / headroom, 3) if best and headroom > 0 else None
        ),
        "best_gain_over_sweep": sweep,
        "winning_region_b_bar": [min(winning), max(winning)] if winning else [],
        "best_operating_point": max(sweep, key=sweep.get) if sweep else None,
        "go": bool(best and gains[best] > 0),
    }


def plot(rows: list[dict], arms: list[str], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6.5, 4.5))
    for arm in arms:
        pts = sorted(
            ((r["b_tot"], r["expected_solved"]) for r in rows if r["arm"] == arm),
            key=lambda t: t[0],
        )
        style = "k--" if arm == "true_bstar" else ("o-" if arm.startswith("oracle") else "s-")
        plt.plot([p[0] for p in pts], [p[1] for p in pts], style, label=arm, markersize=4)
    plt.xscale("log")
    plt.xlabel("total token budget B_tot (log scale)")
    plt.ylabel("expected problems solved")
    plt.title("Allocation policies at matched compute (measured p, out-of-fold p-hat)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "allocation_sim.png", dpi=150)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problems", required=True)
    ap.add_argument("--budgets", required=True)
    ap.add_argument("--hidden-states", default=None,
                    help="optional; adds the activation block to the oracle")
    ap.add_argument("--layer", default=None,
                    help="pooled column for the activation block (default: the "
                         "best-scoring one, swept)")
    ap.add_argument("--n-pcs", type=int, default=10)
    ap.add_argument("--triage-threshold", type=float, default=0.3)
    ap.add_argument("--sweep-triage", action="store_true",
                    help="also sweep the abstention threshold at --b-bar")
    ap.add_argument("--b-bar", type=float, default=2048.0,
                    help="the operating point the verdict is read at")
    ap.add_argument("--out-dir", default="results/analysis")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    configure_logging("INFO")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = OracleDataset.load(args.problems, args.hidden_states, args.budgets)
    grid = ds.swept_budgets()
    tau = next(iter(ds.budgets.values())).success_threshold
    log.info("%d labeled problems, grid=%s, tau=%.2f", len(ds), grid, tau)
    if len(grid) < 2:
        raise SystemExit("a single-budget sweep has nothing to allocate over - "
                         "every policy would return the same caps.")

    features = ["surface", "subject"]
    layer = None
    if args.hidden_states:
        features.append("activations")
        layer = args.layer or ClassificationOracle(
            feature_names=features, n_pcs=args.n_pcs
        ).select_layer(ds)[0]
        log.info("activation block at layer %s", layer)
    log.info("fitting out-of-fold curves on features=%s", features)

    curves = out_of_fold_curves(ds, features, layer, args.n_pcs)
    predicted, measured = build_candidates(ds, curves, tau)

    # The sweep spans the grid: at b_bar = min(grid) every arm is starved and at
    # max(grid) every arm can afford the top budget for everyone, so the
    # interesting region is strictly between -- which is also where the whole
    # curve has to lift for H2 to hold, not one point of it.
    b_bars = sorted({float(b) for b in grid} | {
        float(np.exp(x)) for x in np.linspace(np.log(grid[0]), np.log(grid[-1]), 9)
    } | {float(args.b_bar)})
    rows = simulate(predicted, measured, grid, b_bars, args.triage_threshold, ARMS)

    report = {
        "n_problems": len(ds),
        "grid": grid,
        "tau": tau,
        "features": features,
        "layer": layer,
        "triage_threshold": args.triage_threshold,
        "curve": rows,
        "verdict": verdict(rows, args.b_bar, grid),
    }
    if args.sweep_triage:
        report["triage_sweep"] = sweep_triage(
            predicted, measured, grid, args.b_bar, [0.1, 0.2, 0.3, 0.4, 0.5]
        )

    (out_dir / "allocation_sim.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if not args.no_plot:
        plot(rows, ARMS, out_dir)

    v = report["verdict"]
    log.info("\nAt b_bar=%.0f over %d problems (expected solved, out of %d):",
             args.b_bar, len(ds), len(ds))
    for r in sorted((r for r in rows if abs(r["b_bar"] - args.b_bar) < 1e-9),
                    key=lambda r: -r["expected_solved"]):
        log.info("  %-15s %7.2f   (%+.2f vs uniform)%s",
                 r["arm"], r["expected_solved"],
                 r["expected_solved"] - v["uniform_expected_solved"],
                 f"   [{r['n_skipped']} skipped]" if r["n_skipped"] else "")
    if v.get("go"):
        log.info("\nGO: %r beats uniform by %.2f problems, capturing %s of the %.2f "
                 "available above uniform (the ceiling, of which ~%.1f is label noise).",
                 v["best_learned_arm"], v["gain_over_uniform"][v["best_learned_arm"]],
                 f"{100 * v['fraction_of_headroom_captured']:.0f}%"
                 if v["fraction_of_headroom_captured"] is not None else "n/a",
                 v["headroom_over_uniform"], v["ceiling_noise_bias"])
    else:
        log.info("\nNO-GO at b_bar=%.0f: nothing beats uniform there.", args.b_bar)
    if v.get("winning_region_b_bar"):
        lo, hi = v["winning_region_b_bar"]
        log.info("Across the sweep, some learned arm beats uniform for b_bar in "
                 "[%.0f, %.0f]; the widest margin is at b_bar=%.0f (+%.2f problems). "
                 "Run the live experiment there.",
                 lo, hi, v["best_operating_point"], max(v["best_gain_over_sweep"].values()))
    else:
        log.info("No operating point in the sweep favours any learned arm. The live "
                 "run has nothing to find -- spend the GPU window on more Stage 2 "
                 "labels instead.")
    log.info("wrote %s", out_dir / "allocation_sim.json")
    if args.sweep_triage:
        log.info("triage sweep: %s", report["triage_sweep"])


if __name__ == "__main__":
    main()
