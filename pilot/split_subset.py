"""
Split a MATH problem file into N parts for N parallel Colab runs, preserving
the level/type balance in each part (round-robin within each (type, level)
group, not a random/contiguous slice) -- otherwise a naive slice risks one
part skewing toward a subset of difficulty levels.

Run locally:
    python split_subset.py math_pool_large.json 3
    python split_subset.py math_labeling_subset.json 3
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

SEED = 0


def main():
    src_name = sys.argv[1] if len(sys.argv) > 1 else "math_pool_large.json"
    n_parts = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    random.seed(SEED)
    src = Path(__file__).parent / src_name
    data = json.loads(src.read_text(encoding="utf-8"))
    stem = src.stem  # e.g. "math_pool_large" -> parts named math_pool_large_part1.json

    groups = defaultdict(list)
    for row in data:
        groups[(row["type"], row["level_num"])].append(row)

    # a running counter (not reset per group) so small groups don't all land
    # in the same part -- naive `i % n_parts` per group starves later parts
    # whenever most groups have fewer than n_parts members.
    parts = [[] for _ in range(n_parts)]
    counter = 0
    for key, rows in groups.items():
        rows = rows[:]
        random.shuffle(rows)
        for row in rows:
            parts[counter % n_parts].append(row)
            counter += 1

    for i, part in enumerate(parts, start=1):
        out_path = Path(__file__).parent / f"{stem}_part{i}.json"
        out_path.write_text(json.dumps(part, ensure_ascii=False, indent=2), encoding="utf-8")
        levels = {lv: sum(1 for r in part if r["level_num"] == lv) for lv in sorted(set(r["level_num"] for r in part))}
        types = {t: sum(1 for r in part if r["type"] == t) for t in sorted(set(r["type"] for r in part))}
        print(f"part{i}: {len(part)} problems -> {out_path.name}")
        print(f"  levels: {levels}")
        print(f"  types:  {types}")


if __name__ == "__main__":
    main()
