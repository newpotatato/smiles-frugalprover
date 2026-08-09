"""Scoring an allocation run: per-arm summaries and the paired arm-vs-arm test.

Two rules shape everything here.

**Only ids every arm finished are scored.** A run stopped by its time budget
leaves the last chunk uneven, and comparing an arm that saw 300 problems against
one that saw 275 would attribute the difference to the policy. Restricting to
the common set costs a few problems and keeps the comparison paired.

**Compute is what was spent, not what was allotted.** Arms stop early at
different rates, so the cap budget and the actual generated tokens are different
numbers. Both are reported; the matched-compute claim is made on the second
(RESEARCH_PLAN Phase 2: "matched compute must count *all* agent tokens").
"""
from __future__ import annotations

from frugalprover.common.logging import get_logger

log = get_logger(__name__)


def common_ids(rows: list[dict], arms: list[str]) -> list[str]:
    """Ids present in every arm, in sorted order."""
    per_arm = [{r["id"] for r in rows if r["arm"] == a} for a in arms]
    if not per_arm:
        return []
    keep = set.intersection(*per_arm)
    return sorted(keep)


def summarize_arm(rows: list[dict], arm: str, ids: set[str]) -> dict:
    mine = [r for r in rows if r["arm"] == arm and r["id"] in ids]
    n = len(mine)
    solved = sum(1 for r in mine if r["solved"])
    tokens = sum(r["tokens"] for r in mine)
    cap_budget = sum(r["cap"] for r in mine)
    skipped = sum(1 for r in mine if r["cap"] <= 0)
    audited = [r for r in mine if r["accepted"] is not None]
    return {
        "arm": arm,
        "n": n,
        "n_solved": solved,
        "solve_rate": round(solved / n, 4) if n else 0.0,
        "n_skipped": skipped,
        "tokens": tokens,
        "cap_budget": cap_budget,
        # How much of the allowance the arm actually burned. A low number is not
        # waste -- it means attempts finished early -- but a big gap between arms
        # is the first thing to check before believing a token saving.
        "cap_utilisation": round(tokens / cap_budget, 4) if cap_budget else 0.0,
        "solved_per_1m_tokens": round(1e6 * solved / tokens, 2) if tokens else 0.0,
        "accept_rate": round(sum(1 for r in audited if r["accepted"]) / len(audited), 4)
        if audited else None,
        "mean_cap": round(cap_budget / n, 1) if n else 0.0,
    }


def mcnemar(rows: list[dict], baseline: str, arm: str, ids: set[str]) -> dict:
    """Paired test on per-problem outcomes: does `arm` solve a different set?

    `b` counts problems the baseline solved and the arm didn't, `c` the reverse.
    Under the null (the policy changes nothing) each discordant pair is a coin
    flip, so the exact binomial test on `c` out of `b + c` is the p-value. Exact
    rather than the chi-square approximation because `b + c` is routinely under
    25 at these sample sizes.
    """
    base = {r["id"]: bool(r["solved"]) for r in rows if r["arm"] == baseline and r["id"] in ids}
    other = {r["id"]: bool(r["solved"]) for r in rows if r["arm"] == arm and r["id"] in ids}
    shared = sorted(set(base) & set(other))
    b = sum(1 for i in shared if base[i] and not other[i])
    c = sum(1 for i in shared if other[i] and not base[i])
    out = {"baseline": baseline, "arm": arm, "n_paired": len(shared),
           "baseline_only": b, "arm_only": c, "p_value": None}
    if b + c == 0:
        return out
    try:
        from scipy.stats import binomtest

        out["p_value"] = round(float(binomtest(c, b + c, 0.5).pvalue), 4)
    except ImportError:  # scipy is a core dep, but don't lose the counts over it
        log.warning("scipy unavailable -- reporting discordant counts without a p-value")
    return out


def summarize(rows: list[dict], arms: list[str], baseline: str = "uniform") -> dict:
    """The whole report: per-arm rows, paired tests, and the triage accounting."""
    ids = set(common_ids(rows, arms))
    dropped = {r["id"] for r in rows} - ids
    if dropped:
        log.info("scoring %d problem(s) completed by every arm (%d dropped as "
                 "partial -- the run stopped mid-chunk)", len(ids), len(dropped))

    per_arm = [summarize_arm(rows, a, ids) for a in arms]
    tests = [mcnemar(rows, baseline, a, ids) for a in arms if a != baseline]

    report = {
        "n_scored": len(ids),
        "n_dropped_partial": len(dropped),
        "baseline": baseline,
        "arms": per_arm,
        "paired_tests": tests,
    }

    # What abstention cost: problems an arm skipped that the baseline solved.
    # Without this an arm can look efficient purely by declining hard problems.
    base_solved = {r["id"] for r in rows if r["arm"] == baseline and r["id"] in ids and r["solved"]}
    for arm in arms:
        skipped = {r["id"] for r in rows if r["arm"] == arm and r["id"] in ids and r["cap"] <= 0}
        if skipped:
            report.setdefault("abstention", []).append({
                "arm": arm,
                "n_skipped": len(skipped),
                "skipped_but_baseline_solved": len(skipped & base_solved),
            })
    return report


def format_table(report: dict) -> str:
    """The one-screen version, for the run log."""
    head = f"{'arm':<16}{'n':>5}{'solved':>8}{'rate':>8}{'tokens':>12}{'/1M tok':>9}{'util':>7}"
    lines = [head, "-" * len(head)]
    for a in report["arms"]:
        lines.append(
            f"{a['arm']:<16}{a['n']:>5}{a['n_solved']:>8}{a['solve_rate']:>8.3f}"
            f"{a['tokens']:>12,}{a['solved_per_1m_tokens']:>9.1f}{a['cap_utilisation']:>7.2f}"
        )
    for t in report["paired_tests"]:
        p = "n/a" if t["p_value"] is None else f"{t['p_value']:.4f}"
        lines.append(
            f"  {t['arm']} vs {t['baseline']}: +{t['arm_only']} / -{t['baseline_only']} "
            f"discordant, p={p}"
        )
    return "\n".join(lines)
