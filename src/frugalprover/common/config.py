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
#: bitsandbytes load modes for the `hf` agent client (see ModelSpec.quantization).
QUANTIZATIONS = ["none", "8bit", "4bit"]

SOLVE_PROMPT = (
    "Problem:\n{problem}\n\n"
    "Solve step by step, then give the final answer in \\boxed{{}}.\n\nSolution:"
)

#: Prompts for the verify-repair agent's three roles. They are ordinary config
#: (overridable per run), but the *shapes* matter: the verifier is asked for a
#: verdict AND specific diagnosed flaws, because those flaws are the only thing
#: the corrector gets -- pass/fail alone can't be repaired against.
PROVE_PROMPT = (
    "Problem:\n{problem}\n\n"
    "Write a complete, rigorous solution. End with the final answer in "
    "\\boxed{{}}.\n\nSolution:"
)
VERIFY_PROMPT = (
    "You are a skeptical proof-checker. Audit the candidate solution against the "
    "problem. Look for wrong steps, unjustified claims, and arithmetic errors.\n\n"
    "Problem:\n{problem}\n\n"
    "Candidate solution:\n{candidate}\n\n"
    "Respond in exactly this form:\n"
    "VERDICT: ACCEPT or REJECT\n"
    "FLAWS:\n- <one specific flaw per line, or 'none' if you accept>\n"
)
REPAIR_PROMPT = (
    "Your previous solution was audited and found flawed. Produce a corrected, "
    "complete solution that fixes every flaw below. End with the final answer in "
    "\\boxed{{}}.\n\n"
    "Problem:\n{problem}\n\n"
    "Previous solution:\n{candidate}\n\n"
    "Flaws to fix:\n{flaws}\n\nCorrected solution:"
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
    #: Run a single pass at max(budgets) and reconstruct every smaller budget
    #: from the recorded trajectory, instead of one full re-solve per budget.
    #: Requires an agent exposing `solve_batch_traced` AND every budget >= every
    #: role's max_tokens, so sweep.py's per-role clamp never binds and the
    #: smaller budgets really are prefixes of the largest. Falls back to
    #: independent passes, with a warning, when either condition fails.
    single_pass_reconstruct: bool = False
    #: tau. B* is the smallest budget solved at least this often.
    success_threshold: float = 0.5
    temperature: float = 0.7
    top_p: float = 0.9
    batch_size: int = 6
    #: Stop cleanly once this many wall-clock seconds have elapsed, measured from
    #: the start of run_budget (model loading included). The in-flight batch is
    #: always finished and flushed, and the sorted output plus sidecar are still
    #: written; rerunning resumes the remainder. None = run to completion.
    time_budget_s: float | None = None
    #: Shuffle problem order with `seed` before labeling. problems.jsonl is
    #: written sorted by id, i.e. grouped by subject (oracle/sample/math_sampler.py),
    #: so any truncated run -- by `time_budget_s` or by `max_problems` -- would
    #: otherwise cover only the alphabetically-early subjects. Since `subject` is
    #: a training feature, that is a confound rather than an inconvenience.
    shuffle: bool = False
    #: Log and skip a batch that raises, instead of aborting the run. Worth
    #: setting for a long unattended run against a remote server, where one
    #: transient failure would otherwise discard the remaining hours; per-record
    #: flushing and resume bound the loss to the failed batch.
    continue_on_error: bool = False
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
class ModelSpec:
    """How one role (prover / verifier / corrector) reaches a model.

    ``mock`` runs on CPU with no weights; ``hf`` runs a local
    ``transformers.generate`` model (see agent/model.py:HFClient); ``openai``
    talks to a vLLM-served, OpenAI-compatible endpoint at ``base_url`` (see
    agent/model.py:OpenAIClient) and needs no torch in this process at all.
    """

    client: str = "mock"           # mock | openai | hf
    model: str = "mock"            # repo id / served-model-name
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048         # per-call generation cap for this role
    base_url: str | None = None    # openai-compatible endpoint
    api_key_env: str | None = None  # env var holding the key, never the key itself
    #: hf only: prompts per model.generate call, bounding peak GPU memory when the
    #: loop's active set is large. Ignored by mock; the openai client uses
    #: max_concurrency instead.
    max_batch_size: int = 8
    #: openai only: in-flight requests per generate() call. A served backend does
    #: its own continuous batching, so this is a *concurrency* limit rather than a
    #: memory bound like max_batch_size -- kept separate because the two defaults
    #: differ by an order of magnitude and reusing max_batch_size (default 8, and
    #: 4 in the local-hf configs) would silently serialize a 192-request active
    #: set into 48 waves. Set it >= budget.batch_size * budget.n_samples.
    max_concurrency: int = 64
    #: hf only: load the weights quantized via bitsandbytes (CUDA only), trading
    #: accuracy for memory -- 4bit puts a 7B in ~5GB against ~15GB at bf16, which
    #: is the difference between fitting a 16GB card and not. Roles sharing a
    #: model must agree: the loaded-weight cache keys on (model, quantization),
    #: so a mismatch loads a second copy and defeats the sharing.
    quantization: str = "none"     # none | 8bit | 4bit


@dataclass
class AgentConfig:
    """The solving agent: prover -> k verifiers -> corrector loop.

    `type: single` is a one-shot prover baseline; `verify_repair` runs the full
    audit-and-repair loop. The three control-flow dials -- `aggregation`,
    `independence`, `on_nonconvergence` -- are the architecture, not decoration;
    the defaults (unanimity / blind / reject) are the safety-first corner.
    """

    type: str = "verify_repair"                 # single | verify_repair
    prover: ModelSpec = field(default_factory=ModelSpec)
    corrector: ModelSpec = field(default_factory=ModelSpec)
    #: The k verifiers. Default is a k=3 ensemble; make them heterogeneous
    #: (different models) so the three don't get fooled by the same bad step.
    verifiers: list[ModelSpec] = field(
        default_factory=lambda: [ModelSpec(), ModelSpec(), ModelSpec()]
    )
    max_rounds: int = 4
    aggregation: str = "unanimity"              # unanimity | majority
    independence: str = "blind"                 # blind | debate
    on_nonconvergence: str = "reject"           # reject | flag
    #: `per_problem` derives each attempt's decoding seed from the problem id
    #: (agent/base.py:stable_seed) and sends it with every request. The same
    #: problem then follows the same trajectory under two different budgets, so
    #: an allocation A/B compares budgets rather than two independent draws.
    #: Only the `openai` client honours it; `hf` warns and ignores it. Leave at
    #: `none` for labeling runs, where independent draws are the point.
    seed_mode: str = "none"                     # none | per_problem
    prover_prompt: str = PROVE_PROMPT
    verifier_prompt: str = VERIFY_PROMPT
    corrector_prompt: str = REPAIR_PROMPT


@dataclass
class AllocateConfig:
    """H2 — spend a fixed total token budget across problems, several ways.

    Not a pipeline stage: it consumes a fitted oracle and drives the agent, so
    it runs after `train` and produces its own artifact rather than feeding the
    next stage. See allocate/policies.py for what each arm does.
    """

    #: The arms, run against the same problems at the same total budget. The
    #: first is usually `uniform` -- see `baseline`.
    policies: list[str] = field(
        default_factory=lambda: ["uniform", "length", "oracle_bhat", "oracle_triage"]
    )
    #: Token caps a policy may assign. Must be the budgets the oracle was fitted
    #: on: its success curve is only defined at these points.
    grid: list[int] = field(default_factory=lambda: [1024, 2048, 4096, 8192])
    #: Mean budget per problem. B_tot = b_bar * n_problems, identical for every
    #: arm -- this is what "matched compute" means here. Put it *on* the grid, or
    #: `uniform` has to round and stops being exactly uniform.
    b_bar: int = 2048
    n_samples: int = 1
    #: Fitted oracle, resolved against results/ then data/.
    oracle: str = "oracle.joblib"
    hidden_states: str | None = "hidden_states.parquet"
    #: `oracle_triage` abstains below this predicted success at max(grid).
    triage_threshold: float = 0.3
    #: Arm the paired tests compare against.
    baseline: str = "uniform"
    problems: str = "problems.jsonl"
    #: A budgets.jsonl whose ids are dropped from the eval set. The oracle was
    #: fitted on those problems, so leaving them in would score memorisation.
    exclude_labeled: str | None = None
    max_problems: int | None = None
    #: Problems per chunk. Every arm runs on a chunk before the next chunk
    #: starts, so a run stopped by `time_budget_s` leaves all arms on the same
    #: prefix and the paired comparison still holds.
    chunk_size: int = 25
    time_budget_s: float | None = None
    shuffle: bool = True
    seed: int = 0
    continue_on_error: bool = False
    output: str = "alloc.jsonl"
    report: str = "allocation.json"


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
    agent: AgentConfig = field(default_factory=AgentConfig)
    allocate: AllocateConfig = field(default_factory=AllocateConfig)

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

    a = cfg.agent
    check(a.type, ["single", "verify_repair"], "agent.type")
    check(a.aggregation, ["unanimity", "majority"], "agent.aggregation")
    check(a.independence, ["blind", "debate"], "agent.independence")
    check(a.on_nonconvergence, ["reject", "flag"], "agent.on_nonconvergence")
    check(a.seed_mode, ["none", "per_problem"], "agent.seed_mode")
    for role, spec in [("prover", a.prover), ("corrector", a.corrector)]:
        check(spec.client, ["mock", "openai", "hf"], f"agent.{role}.client")
        check(spec.quantization, QUANTIZATIONS, f"agent.{role}.quantization")
    for i, spec in enumerate(a.verifiers):
        check(spec.client, ["mock", "openai", "hf"], f"agent.verifiers[{i}].client")
        check(spec.quantization, QUANTIZATIONS, f"agent.verifiers[{i}].quantization")
    if a.max_rounds < 1:
        raise ValueError(f"agent.max_rounds must be >= 1, got {a.max_rounds}")
    if a.type == "verify_repair" and len(a.verifiers) < 1:
        raise ValueError("agent.verifiers is empty — the loop needs at least one verifier.")

    al = cfg.allocate
    if not al.grid:
        raise ValueError("allocate.grid is empty — at least one token cap is required.")
    if sorted(al.grid) != list(al.grid):
        raise ValueError(f"allocate.grid must be ascending, got {al.grid}")
    if not al.policies:
        raise ValueError("allocate.policies is empty — nothing to compare.")
    if al.baseline not in al.policies:
        raise ValueError(
            f"allocate.baseline={al.baseline!r} is not one of allocate.policies "
            f"{al.policies} — the paired tests have nothing to compare against."
        )
    if not min(al.grid) <= al.b_bar <= max(al.grid):
        raise ValueError(
            f"allocate.b_bar={al.b_bar} is outside allocate.grid {al.grid}. The mean "
            "budget has to be reachable, or no arm can spend its allowance."
        )
    if not 0.0 <= al.triage_threshold <= 1.0:
        raise ValueError(
            f"allocate.triage_threshold must be in [0, 1], got {al.triage_threshold}"
        )


_NESTED[PipelineConfig] = {
    "sample": SampleConfig,
    "budget": BudgetConfig,
    "extract": ExtractConfig,
    "train": TrainConfig,
    "report": ReportConfig,
    "agent": AgentConfig,
    "allocate": AllocateConfig,
}
_NESTED[AgentConfig] = {
    "prover": ModelSpec,
    "corrector": ModelSpec,
}
_NESTED_LISTS[ExtractConfig] = {
    "features": LayerPooling,
    "geometry": LayerMetric,
}
_NESTED_LISTS[AgentConfig] = {
    "verifiers": ModelSpec,
}
