# Configs

Configs are **layered**, not copied. Instead of a full file per experiment,
start from a base and stack small fragments that each touch one stage:

```bash
frugalprover run-all \
    -c configs/base.yaml \
    -c configs/train/regression.yaml \
    --run-name pilot_regression
```

`-c` is repeatable. Files deep-merge left-to-right onto the dataclass
defaults, then `--set key=value` wins over everything.

## Precedence (low → high)

1. dataclass defaults (in `common/config.py`)
2. each `-c` file, in the order given
3. `--set dotted.key=value` overrides
4. `--run-name` / `--data-dir` / `--results-dir` flags

## Merge rules

- **Mappings** merge key-by-key, so a fragment only needs the keys it changes.
- **Lists and scalars** are *replaced* by the later layer — a fragment that
  sets `budget.budgets: [512]` means that one budget, not an append onto the
  base's `[128, 256, 512]`. To tweak one entry, restate the list.
- Unknown keys still raise (a typo'd key is an error, not a silent default).

## Layout

```
configs/
  base.yaml        full pipeline with every key at its default — the reference
  smoke.yaml       standalone tiny CPU run (also expressible as base + fragments)
  sample/          Stage 1 fragments   e.g. small.yaml
  budget/          Stage 2 fragments   e.g. mock.yaml
  extract/         Stage 3 fragments   e.g. synthetic.yaml, null_baseline.yaml
  train/           Stage 4 fragments   e.g. regression.yaml, compare_models.yaml
  report/          Stage 5 fragments
```

Each fragment is an ordinary partial config; by convention it only sets keys in
its own stage, so any combination composes. `run_name` deliberately lives in
`base.yaml` (or `--run-name`) rather than the fragments, so the same fragment
is reusable across runs without clobbering another run's artifacts.

The fully-resolved config for any run is dumped to
`results/<run_name>/config.yaml`, so a run is always reproducible from that one
file regardless of how many fragments composed it.
