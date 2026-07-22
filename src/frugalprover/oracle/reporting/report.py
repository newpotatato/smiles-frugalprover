"""Stage 5 - collect a run into results/<run_name>/.

This stage recomputes nothing. It reads the fitted model and writes down what
happened, in formats you can read without importing anything: the resolved
config, the metrics, per-problem predictions, and plots.

The plots are the point. A CV score is one number and hides the two things you
actually need to see: whether the layer sweep has structure (a hump beating its
neighbours) or is just noise with a lucky maximum, and whether the predictions
track the labels or collapse onto the mean.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from frugalprover.common.config import PipelineConfig
from frugalprover.common.io import read_json, write_json, write_jsonl
from frugalprover.oracle.model.dataset import OracleDataset


def run_report(cfg: PipelineConfig) -> Path:
    """Assemble results/<run_name>/. Returns the directory."""
    out_dir = cfg.run_results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = cfg.result_path(cfg.train.output)
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found - run `frugalprover train` before `report`."
        )

    from frugalprover.oracle.model.base import OracleModel

    model = OracleModel.load(model_path)
    ds = OracleDataset.load(
        problems=cfg.data_path(cfg.train.problems),
        hidden_states=cfg.data_path(cfg.train.hidden_states),
        budgets=cfg.data_path(cfg.train.budgets),
        verbose=False,
    )

    # `train` already wrote config.yaml with the settings that actually fitted
    # the model. This invocation may carry different --set flags, so don't
    # overwrite it -- only fill it in if training predates this behaviour.
    config_path = out_dir / "config.yaml"
    if not config_path.exists():
        _write_yaml(config_path, cfg.to_dict())

    metrics_src = cfg.result_path(cfg.train.metrics)
    metrics = read_json(metrics_src) if metrics_src.exists() else model.summary()
    write_json(out_dir / "metrics.json", metrics)

    predictions = model.predict_rows(ds)
    write_jsonl(out_dir / "predictions.jsonl", predictions,
                meta={"artifact": "predictions", "mode": model.mode, "layer": model.layer})

    if cfg.report.plots:
        _plots(model, ds, out_dir / "plots")

    _write_readme(out_dir, cfg, model, metrics)

    print(f"\nresults -> {out_dir}")
    print(f"  metrics.json, predictions.jsonl ({len(predictions)} rows), config.yaml"
          + (", plots/" if cfg.report.plots else ""))
    return out_dir


# ---------------------------------------------------------------------- plots

def _plots(model, ds: OracleDataset, plot_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")  # no display in Colab/CI
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)

    _plot_layer_sweep(model, plot_dir / "layer_sweep.png", plt)
    _plot_calibration(model, ds, plot_dir / "calibration.png", plt)
    _plot_budget_hist(ds, plot_dir / "budget_hist.png", plt)
    print(f"  wrote 3 plots -> {plot_dir}")


def _layer_index(name: str) -> int:
    import re

    m = re.match(r"^L(\d+)_", name)
    return int(m.group(1)) if m else -1


def _plot_layer_sweep(model, path: Path, plt) -> None:
    """Score vs depth. The shape is the finding, not the maximum."""
    scores = {k: v for k, v in model.layer_scores.items() if v is not None and not np.isnan(v)}
    if not scores:
        return
    items = sorted(scores.items(), key=lambda kv: _layer_index(kv[0]))
    xs = [_layer_index(k) for k, _ in items]
    ys = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, "o-", color="#3b6ea5")
    best = max(scores, key=scores.get)
    ax.scatter([_layer_index(best)], [scores[best]], color="#c0392b", zorder=5,
               label=f"best: {best} ({scores[best]:.3f})")
    # chance level: AUC 0.5 means no signal; R2 0 means no better than the mean
    chance = 0.5 if model.cv_metric == "auc" else 0.0
    ax.axhline(chance, ls="--", lw=1, color="grey",
               label="chance" if model.cv_metric == "auc" else "predicting the mean")
    ax.set_xlabel("layer")
    ax.set_ylabel(f"CV {model.cv_metric.upper()}")
    # layers are integers; matplotlib's default ticks land on 0.5 steps
    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(f"Oracle score by layer ({model.mode})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_calibration(model, ds: OracleDataset, path: Path, plt) -> None:
    """Predicted vs actual. Regression: a scatter against the diagonal.
    Classification: observed success rate within predicted-probability bins."""
    fig, ax = plt.subplots(figsize=(5, 5))

    if model.mode == "regression":
        solved = ds.solved_subset()
        if len(solved) < 2:
            plt.close(fig)
            return
        actual = solved.b_star()
        pred = model.predict_budget(solved)
        ax.scatter(actual, pred, alpha=0.7, color="#3b6ea5")
        lo, hi = min(actual.min(), pred.min()), max(actual.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "--", color="grey", lw=1, label="perfect")
        ax.set_xlabel("actual B*")
        ax.set_ylabel("predicted B*")
        ax.legend(fontsize=8)
    else:
        probs, actual = [], []
        for budget in model.budgets_seen:
            probs.extend(model.predict_success(ds, budget))
            actual.extend(float(ds.budgets[p.id].solved_at(budget)) for p in ds.problems)
        probs, actual = np.array(probs), np.array(actual)
        edges = np.linspace(0, 1, 6)
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (probs >= lo) & (probs < hi if hi < 1 else probs <= 1)
            if sel.sum():
                xs.append(probs[sel].mean())
                ys.append(actual[sel].mean())
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfectly calibrated")
        ax.plot(xs, ys, "o-", color="#3b6ea5", label="observed")
        ax.set_xlabel("predicted P(solved)")
        ax.set_ylabel("observed success rate")
        ax.legend(fontsize=8)

    ax.set_title(f"Calibration ({model.mode})")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_budget_hist(ds: OracleDataset, path: Path, plt) -> None:
    """B* distribution, with censored problems as their own bar.

    Worth looking at before trusting any score: if nearly everything lands in
    one bar, the budget range was badly chosen and the labels carry little
    information regardless of how the model does.
    """
    if not ds.has_budgets():
        return
    budgets = ds.swept_budgets()
    counts = [sum(1 for r in ds.budgets.values() if r.b_star == b) for b in budgets]
    n_censored = sum(1 for r in ds.budgets.values() if r.censored)

    labels = [str(b) for b in budgets] + ["censored"]
    values = counts + [n_censored]
    colors = ["#3b6ea5"] * len(budgets) + ["#b0b0b0"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=colors)
    ax.set_xlabel("B* (smallest budget that solved it)")
    ax.set_ylabel("problems")
    ax.set_title("Budget label distribution")
    for i, v in enumerate(values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------- writing

def _write_yaml(path: Path, obj: dict) -> None:
    import yaml

    path.write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _write_readme(out_dir: Path, cfg: PipelineConfig, model, metrics: dict) -> None:
    agent = "unknown"
    budget_meta = None
    try:
        from frugalprover.common.io import read_meta

        budget_meta = read_meta(cfg.data_path(cfg.train.budgets))
        if budget_meta:
            agent = budget_meta.get("config", {}).get("agent", "unknown")
    except Exception:
        pass

    warning = ""
    if agent == "mock":
        warning = (
            "\n> **These budget labels are synthetic.** They came from "
            "`budget.estimator: mock`, not from running a real solver. "
            "Nothing here is a research result.\n"
        )

    score = metrics.get("cv_score")
    lines = [
        f"# Run: {cfg.run_name}",
        warning,
        "## Result",
        "",
        f"- mode: `{model.mode}`",
        f"- layer: `{model.layer}`",
        f"- CV {model.cv_metric.upper()}: **{score if score is not None else 'n/a'}**",
        f"- problems: {metrics.get('n_problems', '?')} "
        f"({metrics.get('n_censored', '?')} censored)",
        f"- features: {', '.join(metrics.get('feature_blocks', []))}",
        f"- solver: `{agent}`",
        "",
        "## Files",
        "",
        "| file | what |",
        "|---|---|",
        "| `config.yaml` | the exact config that produced this run |",
        "| `metrics.json` | scores, including the full per-layer sweep |",
        "| `predictions.jsonl` | per-problem B-hat and predicted success probabilities |",
        "| `plots/layer_sweep.png` | score vs depth - look for structure, not just the max |",
        "| `plots/calibration.png` | predicted vs actual |",
        "| `plots/budget_hist.png` | B* distribution, censored problems included |",
        "",
        "## Reproduce",
        "",
        "```",
        f"frugalprover run-all --config configs/base.yaml --run-name {cfg.run_name}",
        "```",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
