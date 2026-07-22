"""Typed configuration: layered YAML files plus `--set dotted.key=value`.

One dataclass per stage, all hanging off :class:`PipelineConfig`. Loading is
strict about *names* (a typo'd key raises instead of being silently ignored,
which is how you lose an afternoon to a sweep that ran with defaults) but
permissive about everything else.

**Layering.** :func:`load_config` takes any number of config files and
deep-merges them left-to-right onto the dataclass defaults, so a variant is a
tiny file that touches one stage rather than a full copy of the pipeline::

    load_config(["configs/base.yaml", "configs/train/regression.yaml"])

Each file is an ordinary (partial) :class:`PipelineConfig`; by convention the
fragments under ``configs/<stage>/`` only set keys in their own stage. Mappings
merge key-by-key; lists and scalars are *replaced* wholesale by the later
layer -- a fragment that sets ``budget.budgets: [512]`` means that one budget,
not an append onto the base's ``[128, 256, 512]``.

Values on the right of `--set` go through the YAML parser (and win over every
file), so `--set train.n_pcs=4` gives an int, `--set extract.device=cpu` a
string, and `--set budget.budgets=[128,512]` a list.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Sequence

from frugalprover.common.paths import DATA_DIR, RESULTS_DIR

#: The seven MATH subjects on the Hub.
SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

POOLINGS = ["mean", "sum", "std", "max", "last"]
METRICS = ["l2_norm", "mean_token_norm", "token_norm_std", "anisotropy", "effective_rank"]

SOLVE_PROMPT = (
    "Problem:\n{problem}\n\n"
    "Solve step by step, then give the final answer in \\boxed{{}}.\n\nSolution:"
)


@dataclass
class SampleConfig:
    """Stage 1 — draw a level/type-balanced sample from MATH."""

    dataset_name: str = "EleutherAI/hendrycks_math"
    subjects: list[str] = field(default_factory=lambda: list(SUBJECTS))
    split: str = "test"
    seed: int = 0
    #: Over-sample this many per (subject, level) before capping at `n_problems`.
    per_level_per_subject: int = 12
    n_problems: int = 300
    #: Skip problems whose reference solution has no \boxed{} — no gold answer
    #: means no way to grade, so they're useless for budget labeling.
    require_boxed_answer: bool = True
    output: str = "problems.jsonl"


@dataclass
class BudgetConfig:
    """Stage 2 — label problems with the effort needed to solve them.

    `estimator: sweep` is the real thing and is not implemented yet; it exits
    with a message pointing at the spec. `mock` produces deterministic fake
    labels so Stages 3-5 are runnable today.
    """

    estimator: str = "sweep"
    agent: str = "Qwen/Qwen2.5-1.5B-Instruct"
    #: Token caps to try. A single entry means a single fixed-budget pass,
    #: which classification can use but regression cannot.
    budgets: list[int] = field(default_factory=lambda: [128, 256, 512])
    n_samples: int = 3
    #: tau. B* is the smallest budget solved at least this often.
    success_threshold: float = 0.5
    temperature: float = 0.7
    top_p: float = 0.9
    batch_size: int = 6
    prompt_template: str = SOLVE_PROMPT
    #: Which A1 file to label. Budget labeling is the expensive stage, so this
    #: often points at a smaller file than Stage 3 uses.
    problems: str = "problems.jsonl"
    max_problems: int | None = None
    seed: int = 0
    output: str = "budgets.jsonl"


@dataclass
class LayerPooling:
    layer: int
    pooling: str = "mean"


@dataclass
class LayerMetric:
    layer: int
    metric: str


@dataclass
class ExtractConfig:
    """Stage 3 — forward-pass a small model and pool its hidden states.

    No generation happens here; this is the cheap signal the oracle reads.
    """

    #: "transformer" runs a real model; "synthetic" fabricates vectors with no
    #: model at all -- a null baseline, and the way the smoke config runs
    #: end-to-end without a GPU. See states/synthetic.py.
    extractor: str = "transformer"
    model_name: str = "Qwen/Qwen2.5-Math-1.5B"
    prompt_template: str = "{problem}"
    features: list[LayerPooling] = field(default_factory=list)
    geometry: list[LayerMetric] = field(default_factory=list)
    #: Shorthand for "every layer with `all_layers_pooling`", appended to
    #: `features`. Layer choice is the open question the sweep exists to
    #: answer, so extracting all of them at once is usually right — they're
    #: computed anyway by output_hidden_states.
    all_layers: bool = False
    all_layers_pooling: str = "mean"
    batch_size: int = 8
    max_input_tokens: int = 512
    dtype: str = "float16"
    device: str = "cuda"
    problems: str = "problems.jsonl"
    output: str = "hidden_states.parquet"

    # -- `extractor: synthetic` only
    synthetic_layers: int = 4
    synthetic_hidden_size: int = 32
    synthetic_signal_layer: int = 1
    #: 0 = pure noise (a true null baseline); >0 plants a difficulty-correlated
    #: direction in `synthetic_signal_layer`, so the layer sweep has something
    #: findable and a broken pipeline is distinguishable from a working one.
    synthetic_signal_strength: float = 2.5
    synthetic_seed: int = 0


@dataclass
class TrainConfig:
    """Stage 4 — fit the oracle."""

    #: "classification" predicts P(solved | features, budget) and works on
    #: single-pass data; "regression" predicts B* directly and needs a sweep.
    mode: str = "classification"
    #: "all" sweeps every pooled column in the parquet; a list pins specific ones.
    layers: str | list[str] = "all"
    #: Feature blocks to stack. See oracle/model/features.py.
    features: list[str] = field(default_factory=lambda: ["surface", "subject", "activations"])
    n_pcs: int = 10
    #: "auto" -> leave-one-out when n <= 60, else 5-fold.
    cv: str = "auto"
    compare_models: bool = False
    min_solved_for_regression: int = 15
    budgets: str = "budgets.jsonl"
    problems: str = "problems.jsonl"
    hidden_states: str = "hidden_states.parquet"
    output: str = "oracle.joblib"
    metrics: str = "metrics.json"


@dataclass
class ReportConfig:
    """Stage 5 — collect everything into results/<run_name>/."""

    plots: bool = True


@dataclass
class PipelineConfig:
    run_name: str = "default"
    seed: int = 0
    data_dir: Path = DATA_DIR
    results_dir: Path = RESULTS_DIR
    sample: SampleConfig = field(default_factory=SampleConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    # -- path helpers: stage configs hold bare filenames, resolved against the run

    @property
    def run_data_dir(self) -> Path:
        return Path(self.data_dir) / self.run_name

    @property
    def run_results_dir(self) -> Path:
        return Path(self.results_dir) / self.run_name

    def data_path(self, name: str) -> Path:
        """Resolve an artifact name to a path.

        - bare filename  -> this run's dir      (``problems.jsonl``)
        - relative path  -> relative to data_dir (``pilot/problems.jsonl``),
          which is how one run reuses another's output: point Stage 4 at
          ``smoke/hidden_states.parquet`` and it trains on the smoke run's
          states without re-extracting them
        - absolute path  -> used as-is
        """
        return _resolve(name, self.run_data_dir, Path(self.data_dir))

    def result_path(self, name: str) -> Path:
        """Same rules as :meth:`data_path`, rooted at results_dir."""
        return _resolve(name, self.run_results_dir, Path(self.results_dir))

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)


def _resolve(name: str, run_dir: Path, root: Path) -> Path:
    p = Path(name)
    if p.is_absolute():
        return p
    return (root / p) if len(p.parts) > 1 else (run_dir / p)


# ------------------------------------------------------------------ loading

def load_config(
    paths: str | Path | Sequence[str | Path] | None = None,
    overrides: list[str] | None = None,
) -> PipelineConfig:
    """Build a config from one or more YAML files plus `key.path=value` strings.

    `paths` may be a single path or a sequence; later files deep-merge onto
    earlier ones (see the module docstring), then `--set` overrides win over
    all of them.
    """
    if paths is None:
        paths = []
    elif isinstance(paths, (str, Path)):
        paths = [paths]

    raw: dict[str, Any] = {}
    for p in paths:
        import yaml

        loaded = yaml.safe_load(Path(p).read_text(encoding="utf-8"))
        if loaded is None:
            continue
        if not isinstance(loaded, dict):
            raise TypeError(f"config file {p} must be a mapping at the top level, "
                            f"got {type(loaded).__name__}")
        raw = _deep_merge(raw, loaded)

    for override in overrides or []:
        _apply_override(raw, override)

    cfg = _from_dict(PipelineConfig, raw, path="")
    _validate(cfg)
    return cfg


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge `override` onto `base`, recursing into nested mappings.

    Two mappings merge key-by-key; anything else -- lists, scalars, or a key
    whose value is a mapping on one side but not the other -- is replaced
    wholesale by `override`. Replacing lists rather than concatenating is the
    least-surprising rule: a layer that restates `budget.budgets` gets exactly
    the list it wrote. Returns a new dict; neither argument is mutated.
    """
    out = dict(base)
    for key, val in override.items():
        if isinstance(out.get(key), dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _apply_override(raw: dict[str, Any], override: str) -> None:
    import yaml

    if "=" not in override:
        raise ValueError(f"--set expects key.path=value, got {override!r}")
    key, _, value = override.partition("=")
    node = raw
    parts = key.strip().split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"--set {key}: {part!r} is not a section")
    node[parts[-1]] = yaml.safe_load(value)


#: Which fields hold nested dataclasses. Declared explicitly rather than read
#: off `field.type`, because `from __future__ import annotations` turns
#: annotations into strings and `is_dataclass("SampleConfig")` is False.
_NESTED: dict[type, dict[str, type]] = {}
_NESTED_LISTS: dict[type, dict[str, type]] = {}


def _from_dict(cls: type, data: Any, path: str) -> Any:
    """Recursively build a dataclass from a dict, rejecting unknown keys."""
    if not is_dataclass(cls):
        return data
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise TypeError(
            f"config section {path or '<root>'!r} should be a mapping, got {type(data).__name__}"
        )

    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(
            f"unknown config key(s) in {path or '<root>'}: {sorted(unknown)}. "
            f"Valid keys: {sorted(known)}"
        )

    nested = _NESTED.get(cls, {})
    nested_lists = _NESTED_LISTS.get(cls, {})

    kwargs: dict[str, Any] = {}
    for name in known:
        sub = f"{path}.{name}".lstrip(".")
        if name in nested:
            # always construct, so absent sections still get their defaults
            kwargs[name] = _from_dict(nested[name], data.get(name), sub)
        elif name not in data:
            continue
        elif name in nested_lists:
            kwargs[name] = [_from_dict(nested_lists[name], v, sub) for v in (data[name] or [])]
        elif name in ("data_dir", "results_dir"):
            kwargs[name] = Path(data[name])
        else:
            kwargs[name] = data[name]

    return cls(**kwargs)


def _to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _validate(cfg: PipelineConfig) -> None:
    """Fail fast on bad enum values, listing what's legal."""
    def check(value: str, legal: list[str], where: str) -> None:
        if value not in legal:
            raise ValueError(f"{where}: {value!r} is not valid. Choose one of {legal}.")

    check(cfg.train.mode, ["classification", "regression"], "train.mode")
    check(cfg.budget.estimator, ["sweep", "mock"], "budget.estimator")
    check(cfg.extract.extractor, ["transformer", "synthetic"], "extract.extractor")
    check(cfg.extract.all_layers_pooling, POOLINGS, "extract.all_layers_pooling")
    for i, f in enumerate(cfg.extract.features):
        check(f.pooling, POOLINGS, f"extract.features[{i}].pooling")
    for i, g in enumerate(cfg.extract.geometry):
        check(g.metric, METRICS, f"extract.geometry[{i}].metric")

    if not cfg.budget.budgets:
        raise ValueError("budget.budgets is empty — at least one token cap is required.")
    if sorted(cfg.budget.budgets) != list(cfg.budget.budgets):
        raise ValueError(f"budget.budgets must be ascending, got {cfg.budget.budgets}")
    if not 0.0 < cfg.budget.success_threshold <= 1.0:
        raise ValueError(
            f"budget.success_threshold must be in (0, 1], got {cfg.budget.success_threshold}"
        )


_NESTED[PipelineConfig] = {
    "sample": SampleConfig,
    "budget": BudgetConfig,
    "extract": ExtractConfig,
    "train": TrainConfig,
    "report": ReportConfig,
}
_NESTED_LISTS[ExtractConfig] = {
    "features": LayerPooling,
    "geometry": LayerMetric,
}
