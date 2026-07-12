"""
Joins records from possibly DIFFERENT tiers (activations_only, full_sweep,
baseline_single -- see colab_budget_oracle_pilot.ipynb) BY PROBLEM ID into one
unified pilot_results.jsonl. This is a join, not a plain concatenation: a
problem that appears in the activations_only pool AND in the full_sweep
labeling subset (they're nested by construction, see data_prep.py) ends up
as ONE record with both its activations/phi AND its budget/B* fields.

Downstream scripts (analyze.py, oracle.py, id_vs_difficulty.py,
calibration_cost.py, learning_curve.py) already guard for missing fields --
most records will have phi/activations but NOT p_by_budget/b_star (only the
smaller full_sweep tier does), and that's expected, not an error.

Run locally, after downloading each person's/tier's output:
    python merge_results.py pilot_activations_only_part1.jsonl pilot_activations_only_part2.jsonl \
        pilot_activations_only_part3.jsonl pilot_full_sweep_part1.jsonl pilot_full_sweep_part2.jsonl \
        pilot_full_sweep_part3.jsonl
    # or just glob everything:
    python merge_results.py
"""
import json
import sys
from pathlib import Path


def main():
    inputs = sys.argv[1:]
    if not inputs:
        inputs = sorted(str(p) for p in Path(__file__).parent.glob("pilot_*_part*.jsonl"))
    if not inputs:
        print("no pilot_<mode>_part*.jsonl files found -- pass paths explicitly")
        return

    print(f"merging: {inputs}")
    merged = {}  # id -> record, fields updated/unioned as more tiers are read
    per_file_new_ids = {}
    tiers_seen = {}

    for path in inputs:
        new_ids = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                pid = rec["id"]
                if pid not in merged:
                    merged[pid] = {}
                    new_ids += 1
                merged[pid].update(rec)  # later files can add fields, but won't erase earlier ones
                merged[pid]["id"] = pid  # guard: don't let a later file's partial dict clobber id
                tiers_seen.setdefault(pid, set()).add(rec.get("run_mode", "unknown"))
        per_file_new_ids[path] = new_ids
        print(f"  {path}: introduced {new_ids} new ids (total so far: {len(merged)})")

    out_path = Path(__file__).parent / "pilot_results.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for pid in sorted(merged):
            f.write(json.dumps(merged[pid]) + "\n")

    n_with_activations = sum(1 for r in merged.values() if "activations" in r)
    n_with_full_sweep = sum(1 for r in merged.values() if "p_by_budget" in r)
    n_with_baseline = sum(1 for r in merged.values() if "p_baseline" in r)
    tier_counts = {}
    for tiers in tiers_seen.values():
        key = "+".join(sorted(tiers))
        tier_counts[key] = tier_counts.get(key, 0) + 1

    print(f"\nwrote {len(merged)} unified problems to {out_path}")
    print(f"  with activations:        {n_with_activations}")
    print(f"  with full_sweep (B*):    {n_with_full_sweep}")
    print(f"  with baseline_single:    {n_with_baseline}")
    print(f"  tier coverage breakdown: {tier_counts}")


if __name__ == "__main__":
    main()
