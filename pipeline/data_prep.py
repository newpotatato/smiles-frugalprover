"""
Builds a two-tier MATH sample:

  1. A LARGE, level/type-balanced pool (math_pool_large.json) for cheap
     activation-only extraction (forward pass only, no generation -- scales
     easily to hundreds of problems).
  2. A smaller, stratified subset NESTED inside that pool (same problem ids,
     math_labeling_subset.json) for the expensive full budget-sweep labeling.

Nesting matters: because the labeling subset's ids are a subset of the large
pool's ids, activations computed for the large pool already cover every
labeled problem too -- merge_results.py joins them by id, no duplicate work.

Run locally:
    python data_prep.py

Output:
    pilot/math_pool_large.json       -- ~300 problems, for activations_only
    pilot/math_labeling_subset.json  -- ~80 problems, subset of the above, for full_sweep
"""
import json
import random
import re

from datasets import load_dataset

from frugalprover.paths import DATA_DIR

SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

N_PER_LEVEL_PER_SUBJECT_LARGE = 12  # 7 subjects * 5 levels * 12 ~= 420, capped to TARGET_N_LARGE
TARGET_N_LARGE = 300

N_PER_LEVEL_PER_SUBJECT_LABEL = 3  # drawn FROM the large pool's own (type, level) groups
TARGET_N_LABEL = 80

SEED = 0

BOXED_RE = re.compile(r"\\boxed\{")


def extract_boxed(solution: str):
    """Extract the content of the last \\boxed{...}, respecting nested braces."""
    matches = list(BOXED_RE.finditer(solution))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    i = start
    while i < len(solution) and depth > 0:
        if solution[i] == "{":
            depth += 1
        elif solution[i] == "}":
            depth -= 1
        i += 1
    return solution[start : i - 1].strip()


def main():
    random.seed(SEED)

    # --- 1. large pool: over-sample per (subject, level), shuffle, cap ---
    large_pool = []
    for subj in SUBJECTS:
        ds = load_dataset("EleutherAI/hendrycks_math", subj, split="test")
        by_level = {}
        for row in ds:
            ans = extract_boxed(row["solution"])
            if ans is None:
                continue
            by_level.setdefault(row["level"], []).append(
                {"problem": row["problem"], "level": row["level"], "type": row["type"], "answer": ans}
            )
        for level, rows in by_level.items():
            random.shuffle(rows)
            large_pool.extend(rows[:N_PER_LEVEL_PER_SUBJECT_LARGE])

    random.shuffle(large_pool)
    large_pool = large_pool[:TARGET_N_LARGE]
    for i, row in enumerate(large_pool):
        row["id"] = i
        m = re.search(r"\d+", row["level"])
        row["level_num"] = int(m.group()) if m else None

    large_path = DATA_DIR / "math_pool_large.json"
    large_path.write_text(json.dumps(large_pool, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- 2. labeling subset: stratified sample drawn FROM the large pool, same ids ---
    by_group = {}
    for row in large_pool:
        by_group.setdefault((row["type"], row["level_num"]), []).append(row)

    labeling_subset = []
    for key, rows in by_group.items():
        rows = rows[:]
        random.shuffle(rows)
        labeling_subset.extend(rows[:N_PER_LEVEL_PER_SUBJECT_LABEL])
    random.shuffle(labeling_subset)
    labeling_subset = labeling_subset[:TARGET_N_LABEL]

    label_path = DATA_DIR / "math_labeling_subset.json"
    label_path.write_text(json.dumps(labeling_subset, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(large_pool)} problems to {large_path}")
    print(f"Wrote {len(labeling_subset)} problems to {label_path} (ids are a subset of the large pool)")

    for name, rows in [("large pool", large_pool), ("labeling subset", labeling_subset)]:
        levels = sorted(set(r["level_num"] for r in rows if r["level_num"] is not None))
        types = sorted(set(r["type"] for r in rows))
        print(f"\n{name} ({len(rows)} problems):")
        print("  level distribution:", {lv: sum(1 for r in rows if r["level_num"] == lv) for lv in levels})
        print("  type distribution:", {t: sum(1 for r in rows if r["type"] == t) for t in types})

    label_ids = set(r["id"] for r in labeling_subset)
    large_ids = set(r["id"] for r in large_pool)
    assert label_ids <= large_ids, "labeling subset must be nested inside the large pool"


if __name__ == "__main__":
    main()
