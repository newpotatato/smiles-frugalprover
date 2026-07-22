"""FrugalProver: predicting math-problem solve effort from a small model's hidden states.

A five-stage pipeline, each stage independently runnable and swappable:

    sample -> budget -> extract -> train -> report

Stages talk only through files, documented in docs/ARTIFACTS.md. Use the CLI::

    frugalprover run-all --config configs/base.yaml

or the library::

    from frugalprover import OracleDataset, ClassificationOracle

    ds = OracleDataset.load("problems.jsonl", "hidden_states.parquet", "budgets.jsonl")
    oracle = ClassificationOracle().fit(ds)
    oracle.save("oracle.joblib")

Importing this package is cheap: torch, transformers and datasets are pulled in
only by the stages that need them, so the analysis path works on a laptop
without a GPU stack.
"""
from __future__ import annotations

__version__ = "0.2.0"

from frugalprover.agent.base import SolverAgent
from frugalprover.common.config import (
    BudgetConfig,
    ExtractConfig,
    PipelineConfig,
    ReportConfig,
    SampleConfig,
    TrainConfig,
    load_config,
)
from frugalprover.common.grading import (
    extract_answer,
    extract_boxed,
    grade,
    normalize,
    surface_features,
    wilson_ci,
)
from frugalprover.common.io import (
    read_jsonl,
    read_meta,
    read_table,
    write_jsonl,
    write_table,
)
from frugalprover.common.records import BudgetRecord, ProblemRecord
from frugalprover.oracle.budget.base import BudgetEstimator
from frugalprover.oracle.budget.mock import MockEstimator
from frugalprover.oracle.budget.sweep import TokenSweepEstimator
from frugalprover.oracle.model.base import OracleModel
from frugalprover.oracle.model.classification import ClassificationOracle
from frugalprover.oracle.model.dataset import OracleDataset
from frugalprover.oracle.model.features import (
    ActivationPCA,
    FeatureBlock,
    GeometryScalars,
    SubjectOneHot,
    SurfaceFeatures,
)
from frugalprover.oracle.model.regression import RegressionOracle
from frugalprover.oracle.sample.base import DatasetSampler
from frugalprover.oracle.sample.math_sampler import MathStratifiedSampler
from frugalprover.oracle.states.base import HiddenStateExtractor
from frugalprover.oracle.states.pooling import geometry, pool, resolve_layer

__all__ = [
    "__version__",
    # config
    "PipelineConfig", "SampleConfig", "BudgetConfig", "ExtractConfig",
    "TrainConfig", "ReportConfig", "load_config",
    # records
    "ProblemRecord", "BudgetRecord",
    # protocols / ABCs
    "DatasetSampler", "BudgetEstimator", "HiddenStateExtractor", "SolverAgent",
    "OracleModel", "FeatureBlock",
    # implementations
    "MathStratifiedSampler", "MockEstimator", "TokenSweepEstimator",
    "ClassificationOracle", "RegressionOracle", "OracleDataset",
    "SurfaceFeatures", "SubjectOneHot", "ActivationPCA", "GeometryScalars",
    # io
    "read_jsonl", "write_jsonl", "read_table", "write_table", "read_meta",
    # helpers
    "grade", "extract_boxed", "extract_answer", "normalize", "surface_features",
    "wilson_ci", "pool", "geometry", "resolve_layer",
    # stage runners (lazy, see __getattr__)
    "run_sample", "run_budget", "run_extract", "run_train", "run_report",
    "TransformerHiddenStateExtractor",
]


def __getattr__(name: str):
    """Stage runners and the torch-backed extractor, imported on first use.

    Kept out of the eager imports above so `import frugalprover` doesn't pull in
    torch, transformers, or datasets. A laptop with only scikit-learn can still
    load a fitted oracle and run analyses.
    """
    lazy = {
        "run_sample": ("frugalprover.oracle.sample", "run_sample"),
        "run_budget": ("frugalprover.oracle.budget", "run_budget"),
        "run_extract": ("frugalprover.oracle.states", "run_extract"),
        "run_train": ("frugalprover.oracle.model", "run_train"),
        "run_report": ("frugalprover.oracle.reporting", "run_report"),
        "TransformerHiddenStateExtractor": (
            "frugalprover.oracle.states.hf_extractor",
            "TransformerHiddenStateExtractor",
        ),
    }
    if name in lazy:
        import importlib

        module, attr = lazy[name]
        return getattr(importlib.import_module(module), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
