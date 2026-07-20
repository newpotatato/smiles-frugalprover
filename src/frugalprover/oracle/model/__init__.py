"""Stage 4 - fit the oracle. Produces `oracle.joblib` + `metrics.json` (A4)."""
from __future__ import annotations

from frugalprover.common.config import PipelineConfig
from frugalprover.common.io import write_json
from frugalprover.oracle.model.base import OracleModel
from frugalprover.oracle.model.classification import ClassificationOracle
from frugalprover.oracle.model.dataset import OracleDataset
from frugalprover.oracle.model.features import (
    ActivationPCA,
    FeatureBlock,
    GeometryScalars,
    SubjectOneHot,
    SurfaceFeatures,
    build_blocks,
)
from frugalprover.oracle.model.regression import RegressionOracle

__all__ = [
    "OracleModel", "ClassificationOracle", "RegressionOracle", "OracleDataset",
    "FeatureBlock", "SurfaceFeatures", "SubjectOneHot", "ActivationPCA",
    "GeometryScalars", "build_blocks", "ORACLES", "build_oracle", "run_train",
]

ORACLES = {
    "classification": ClassificationOracle,
    "regression": RegressionOracle,
}


def build_oracle(cfg: PipelineConfig) -> OracleModel:
    tc = cfg.train
    kwargs = dict(feature_names=tc.features, n_pcs=tc.n_pcs, cv=tc.cv)
    if tc.mode == "regression":
        return RegressionOracle(min_solved=tc.min_solved_for_regression, **kwargs)
    return ClassificationOracle(**kwargs)


def run_train(cfg: PipelineConfig) -> OracleModel:
    """Fit the oracle and write A4."""
    tc = cfg.train
    ds = OracleDataset.load(
        problems=cfg.data_path(tc.problems),
        hidden_states=cfg.data_path(tc.hidden_states),
        budgets=cfg.data_path(tc.budgets),
    )

    layers = None if tc.layers == "all" else (
        [tc.layers] if isinstance(tc.layers, str) else list(tc.layers)
    )
    if layers is not None and len(layers) == 1:
        model = build_oracle(cfg).fit(ds, layer=layers[0])
    else:
        model = build_oracle(cfg)
        if layers is not None:
            # pin the sweep to the named subset
            best, scores = model.select_layer(ds, layers)
            model.layer_scores = scores
            model.fit(ds, layer=best)
        else:
            model.fit(ds)

    if tc.compare_models:
        from frugalprover.oracle.model.compare import compare

        comparison = compare(model, ds)
    else:
        comparison = None

    model_path = cfg.result_path(tc.output)
    # `config` in the sidecar is what lets `run-all --skip-existing` notice
    # that the training settings changed and rerun instead of handing back a
    # stale model
    model.save(model_path, config=dict(tc.__dict__))

    # Record the config HERE, not in the report stage. `report` is a separate
    # invocation with its own --set flags, so a config.yaml written there can
    # silently disagree with what actually trained the model.
    _write_config(cfg)

    summary = model.summary()
    summary["n_problems"] = len(ds)
    summary["n_censored"] = sum(1 for r in ds.budgets.values() if r.censored)
    if comparison:
        summary["model_comparison"] = comparison
    write_json(cfg.result_path(tc.metrics), summary)

    print(f"\nsaved {model.mode} oracle -> {model_path}")
    print(f"  layer={model.layer}  CV {model.cv_metric}={model.cv_score:.3f}  "
          f"n_train_rows={model.n_train}")
    return model


def _write_config(cfg: PipelineConfig) -> None:
    import yaml

    out = cfg.run_results_dir / "config.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
