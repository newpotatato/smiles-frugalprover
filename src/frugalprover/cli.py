"""Console entrypoint: `frugalprover <stage> --config cfg.yaml [--set k=v ...]`.

Stages are importable as plain functions too (`run_sample(cfg)` etc.); the CLI
is a thin wrapper so nothing here is load-bearing. Stage runners are imported
lazily inside each handler, so `frugalprover sample` doesn't drag in torch and
`frugalprover train` doesn't drag in `datasets`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from frugalprover.common.config import PipelineConfig, load_config

#: Pipeline order. `run-all --from/--to` slices this list.
STAGES = ["sample", "budget", "extract", "train", "report"]


def _resolve(args: argparse.Namespace) -> PipelineConfig:
    cfg = load_config(args.config, args.set or [])
    if args.run_name:
        cfg.run_name = args.run_name
    if args.data_dir:
        cfg.data_dir = Path(args.data_dir)
    if args.results_dir:
        cfg.results_dir = Path(args.results_dir)
    return cfg


# ------------------------------------------------------------------- stages

def _stage_output(cfg: PipelineConfig, stage: str) -> Path:
    """Where a stage writes, for the --skip-existing check."""
    return {
        "sample": cfg.data_path(cfg.sample.output),
        "budget": cfg.data_path(cfg.budget.output),
        "extract": cfg.data_path(cfg.extract.output),
        "train": cfg.result_path(cfg.train.output),
        "report": cfg.run_results_dir / "metrics.json",
    }[stage]


def _run_stage(cfg: PipelineConfig, stage: str) -> None:
    if stage == "sample":
        from frugalprover.oracle.sample import run_sample

        run_sample(cfg)
    elif stage == "budget":
        from frugalprover.oracle.budget import run_budget

        run_budget(cfg)
    elif stage == "extract":
        from frugalprover.oracle.states import run_extract

        run_extract(cfg)
    elif stage == "train":
        from frugalprover.oracle.model import run_train

        run_train(cfg)
    elif stage == "report":
        from frugalprover.oracle.reporting import run_report

        run_report(cfg)
    else:  # pragma: no cover
        raise ValueError(f"unknown stage {stage!r}")


def cmd_stage(args: argparse.Namespace) -> int:
    cfg = _resolve(args)
    _run_stage(cfg, args.command)
    return 0


def _stage_config(cfg: PipelineConfig, stage: str) -> dict:
    section = {"sample": cfg.sample, "budget": cfg.budget, "extract": cfg.extract,
               "train": cfg.train, "report": cfg.report}[stage]
    return dict(section.__dict__)


def _is_stale(cfg: PipelineConfig, stage: str, out) -> list[str]:
    """Config keys that changed since `out` was written.

    `--skip-existing` is deliberately dumb -- if the file is there, skip it --
    because hash-gating every stage adds a lot of machinery for a research
    pipeline. But a silently stale skip is the one failure this can't be
    allowed to cause: change `train.mode` and rerun, get the old model back,
    and believe it. So the skip stays dumb and just says when it looks wrong.
    """
    from frugalprover.common.io import read_meta

    meta = read_meta(out)
    if not meta or "config" not in meta:
        return []
    old, new = meta["config"], _stage_config(cfg, stage)
    return sorted(k for k in new if k in old and _norm(old[k]) != _norm(new[k]))


def _norm(v):
    return str(v)


def cmd_run_all(args: argparse.Namespace) -> int:
    cfg = _resolve(args)
    start = STAGES.index(args.start) if args.start else 0
    stop = STAGES.index(args.stop) + 1 if args.stop else len(STAGES)

    upstream_ran = False
    for stage in STAGES[start:stop]:
        out = _stage_output(cfg, stage)
        if args.skip_existing and out.exists() and not upstream_ran:
            changed = _is_stale(cfg, stage, out)
            if changed:
                print(f"[{stage}] config changed since {out.name} was written "
                      f"({', '.join(changed)}) -- rerunning rather than skipping")
            else:
                print(f"[{stage}] skipped - {out} already exists")
                continue
        elif args.skip_existing and out.exists() and upstream_ran:
            # its inputs were just regenerated, so the existing output is stale
            # by definition -- skipping here is what silently mixes a new
            # dataset with an old model
            print(f"[{stage}] rerunning - an upstream stage was re-run")
        print(f"\n=== {stage} ===")
        _run_stage(cfg, stage)
        upstream_ran = True
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from frugalprover.common.io import load_model, write_jsonl
    from frugalprover.oracle.model.dataset import OracleDataset

    model = load_model(args.model)
    ds = OracleDataset.load(problems=args.problems, hidden_states=args.hidden)
    rows = model.predict_rows(ds)
    write_jsonl(args.out, rows, meta={"artifact": "predictions", "model": str(args.model)})
    print(f"wrote {len(rows)} predictions -> {args.out}")
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    from frugalprover.oracle.reporting.runs import print_runs_table

    cfg = _resolve(args)
    print_runs_table(cfg.results_dir)
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    import importlib
    import platform

    print(f"python       {platform.python_version()} ({platform.system()})")
    for mod in ["frugalprover", "numpy", "sklearn", "pandas", "pyarrow", "yaml",
                "datasets", "torch", "transformers"]:
        try:
            m = importlib.import_module(mod)
            print(f"{mod:<12} {getattr(m, '__version__', '?')}")
        except ImportError:
            note = "  (optional: pip install 'frugalprover[gpu]')" if mod in ("torch", "transformers") else ""
            print(f"{mod:<12} not installed{note}")

    try:
        import torch

        print(f"\ncuda available: {torch.cuda.is_available()}")
    except ImportError:
        print("\ncuda available: unknown (torch not installed)")
    return 0


# --------------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frugalprover",
        description="Budget oracle pipeline: sample -> budget -> extract -> train -> report.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", "-c", help="YAML config file")
        p.add_argument("--set", "-s", action="append", metavar="KEY=VALUE",
                       help="override a config key, e.g. --set train.mode=regression (repeatable)")
        p.add_argument("--run-name", help="override run_name")
        p.add_argument("--data-dir", help="override data_dir")
        p.add_argument("--results-dir", help="override results_dir")

    descriptions = {
        "sample": "Stage 1: draw a balanced MATH sample -> problems.jsonl",
        "budget": "Stage 2: label problems with solve effort -> budgets.jsonl",
        "extract": "Stage 3: pool model hidden states -> hidden_states.parquet",
        "train": "Stage 4: fit the oracle -> oracle.joblib + metrics.json",
        "report": "Stage 5: collect everything into results/<run_name>/",
    }
    for stage in STAGES:
        p = sub.add_parser(stage, help=descriptions[stage], description=descriptions[stage])
        add_common(p)
        p.set_defaults(func=cmd_stage)

    p = sub.add_parser("run-all", help="run every stage in order")
    add_common(p)
    p.add_argument("--from", dest="start", choices=STAGES, help="start at this stage")
    p.add_argument("--to", dest="stop", choices=STAGES, help="stop after this stage")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip a stage whose output file already exists")
    p.set_defaults(func=cmd_run_all)

    p = sub.add_parser("predict", help="apply a fitted oracle to new problems")
    add_common(p)
    p.add_argument("--model", required=True, help="path to oracle.joblib")
    p.add_argument("--problems", required=True, help="problems.jsonl")
    p.add_argument("--hidden", required=True, help="hidden_states.parquet")
    p.add_argument("--out", default="predictions.jsonl")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("runs", help="compare runs under results/")
    add_common(p)
    p.set_defaults(func=cmd_runs)

    p = sub.add_parser("info", help="show versions and torch/cuda availability")
    add_common(p)
    p.set_defaults(func=cmd_info)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except NotImplementedError as e:
        # Stage 2's expected path until someone implements the sweep.
        print(f"\nnot implemented: {e}", file=sys.stderr)
        return 2
    except (ValueError, FileNotFoundError) as e:
        print(f"\nerror: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
