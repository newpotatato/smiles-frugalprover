"""Model backends for the agent's roles.

A role (prover, verifier, corrector) talks to a model through a `ModelClient`.
The interface is deliberately thin -- text in, text out -- so the loop in
`verify_repair.py` never knows whether it's hitting a mock, a vLLM endpoint, or
a local transformer.

`MockModelClient` needs no torch and no network, so the whole loop runs and is
testable on a laptop. `HFClient` runs an open model locally via `transformers`.
`openai` is still a registered stub that raises `NotImplementedError` with a spec
-- the same "seam visible, body later" pattern as oracle/budget/sweep.py.

Nothing here imports from `oracle/`; the dependency runs one way (see
agent/README.md).
"""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from frugalprover.common.config import ModelSpec
from frugalprover.common.logging import get_logger

log = get_logger(__name__)


@dataclass
class _LoadedModel:
    """One (model, tokenizer) pair loaded once and shared across roles."""

    model: Any
    tokenizer: Any
    device: str
    refcount: int = 0


class ModelClient(ABC):
    """Generates completions for a batch of prompts under decoding params.

    Concrete clients hold whatever they need (an endpoint, a loaded model) and
    are built from a :class:`ModelSpec` via :func:`build_model_client`.
    """

    def __init__(self, spec: ModelSpec):
        self.spec = spec

    def setup(self) -> None:
        """Open connections / load weights. Called once before first use."""

    def teardown(self) -> None:
        """Release resources."""

    @abstractmethod
    def generate(
        self,
        prompts: list[str],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        role: str = "prover",
    ) -> list[str]:
        """One completion per prompt, in input order.

        `role` is a hint identifying the caller (prover / verifier / corrector).
        Most backends ignore it; the mock uses it to pick a canned response.
        """
        ...

    def count_tokens(self, text: str) -> int:
        """Token count of `text`, for the loop's total-token accounting.

        The default is a whitespace split -- good enough to keep the budget
        axis honest for the mock. A real client overrides this with its own
        tokenizer so the count matches what it actually generated.
        """
        return len(text.split())

    def describe(self) -> dict:
        """Recorded in the agent's spec / the A2 sidecar."""
        return {"client": self.spec.client, "model": self.spec.model}


class MockModelClient(ModelClient):
    """A scriptable, CPU-only client for tests and demos.

    Pass a `responder(role, prompt) -> str` to drive the loop deterministically
    (e.g. a wrong proof first, a fixed one after). `role` is the free-form label
    the caller passes to :meth:`generate` so one responder can answer for prover,
    verifier and corrector.

    With no responder it falls back to a canned solution ending in a boxed
    answer, so a config-only `prove` run produces something gradeable without a
    script.
    """

    def __init__(
        self,
        spec: ModelSpec,
        responder: Callable[[str, str], str] | None = None,
    ):
        super().__init__(spec)
        self._responder = responder

    def generate(
        self,
        prompts: list[str],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        role: str = "prover",
    ) -> list[str]:
        return [self._one(role, p) for p in prompts]

    def _one(self, role: str, prompt: str) -> str:
        if self._responder is not None:
            return self._responder(role, prompt)
        if role == "verifier":
            # Default mock verifier accepts, so the loop terminates in one round.
            return "VERDICT: ACCEPT\nFLAWS:\n- none"
        return "Mock solution. The answer is \\boxed{42}."


class OpenAIClient(ModelClient):
    """OpenAI-compatible endpoint (e.g. a vLLM `--served-model-name`).

    NOT IMPLEMENTED. A conforming body would, lazily inside `setup()`, build an
    HTTP client for `spec.base_url` with the key read from
    `os.environ[spec.api_key_env]` (never the key inline), then POST each prompt
    to `/v1/chat/completions` with `model=spec.model`, `max_tokens`,
    `temperature`, `top_p`, and return `choices[0].message.content`.
    `count_tokens` should use the served model's tokenizer.
    """

    def setup(self) -> None:
        raise NotImplementedError(_OPENAI_SPEC)

    def generate(self, prompts, *, max_tokens, temperature, top_p, role="prover"):
        raise NotImplementedError(_OPENAI_SPEC)


class HFClient(ModelClient):
    """Local ``transformers`` generation with no server and no inference-time
    network.

    Runs any model with an ``AutoModelForCausalLM`` head named by ``spec.model``.
    Weights are loaded lazily in :meth:`setup` -- mirroring
    ``oracle/states/hf_extractor.py`` for the ``USE_TF=0`` opt-out, dtype
    selection, and CUDA fallback -- so importing ``frugalprover`` stays free of a
    torch dependency.

    :meth:`generate` batches all prompts, wraps each in a single user turn via the
    tokenizer's chat template (the role prompts are already fully-formed
    instructions), samples once, and decodes **only** the newly generated tokens.
    Generation is chunked into sub-batches of at most ``spec.max_batch_size`` so a
    large active set can't exhaust GPU memory in one call.

    Loaded weights are shared across all instances via the class-level
    :attr:`_cache`: two clients naming the same ``spec.model`` (e.g. a prover and
    a verifier on one 7B) reuse a single copy instead of loading it twice. The
    copy is reference-counted and freed when the last client using it tears down.
    """

    #: Class-level cache of loaded models, keyed by repo id / path, shared by
    #: every HFClient in the process. The verify-repair loop builds a separate
    #: client per role and a trio commonly reuses the same checkpoint, so without
    #: sharing each role would load its own copy and multiply GPU memory by the
    #: role count. dtype/device are derived deterministically from the hardware,
    #: so the model id alone identifies the artifact. `_lock` guards it so
    #: concurrent setups don't double-load.
    _cache: dict[str, _LoadedModel] = {}
    _lock = threading.Lock()

    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        self.model = None
        self.tokenizer = None
        self.device = None
        self._cached = False  # this client holds a reference in the class cache

    def setup(self) -> None:
        if self.model is not None:
            return
        key = self.spec.model
        with HFClient._lock:
            entry = HFClient._cache.get(key)
            if entry is None:
                entry = self._load(key)
                HFClient._cache[key] = entry
                log.info("loaded %s for agent client 'hf': device=%s", key, entry.device)
            else:
                log.info("reusing cached %s for agent client 'hf'", key)
            entry.refcount += 1
        self.model = entry.model
        self.tokenizer = entry.tokenizer
        self.device = entry.device
        self._cached = True

    def _load(self, model_id: str) -> _LoadedModel:
        # transformers probes TF/Flax at import and crashes the stage if either
        # is installed but broken; this path only uses torch (see hf_extractor).
        import os

        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("USE_FLAX", "0")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "agent client 'hf' needs torch and transformers: "
                "pip install 'frugalprover[gpu]'"
            ) from e

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # fp16 on CPU is slower than fp32 and unsupported for some ops; on GPU
        # prefer bf16 where available, else fp16, to halve the memory footprint.
        if device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Decoder generation must pad on the left, or right-padding tokens land
        # between the prompt and the continuation and corrupt every sample.
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype
        ).to(device).eval()
        return _LoadedModel(model=model, tokenizer=tokenizer, device=device)

    def generate(
        self,
        prompts: list[str],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        role: str = "prover",
    ) -> list[str]:
        if self.model is None:
            raise RuntimeError("call setup() before generate()")
        if not prompts:
            return []
        batch_size = max(1, self.spec.max_batch_size)
        out: list[str] = []
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start:start + batch_size]
            out.extend(self._generate_chunk(chunk, max_tokens, temperature, top_p))
        return out

    def _generate_chunk(
        self, prompts: list[str], max_tokens: int, temperature: float, top_p: float
    ) -> list[str]:
        import torch

        texts = [self._render(p) for p in prompts]
        enc = self.tokenizer(
            texts, return_tensors="pt", padding=True
        ).to(self.device)

        # temperature <= 0 means greedy; sampling params are ignored by
        # transformers when do_sample=False, so gate on it to avoid warnings.
        do_sample = temperature is not None and temperature > 0.0
        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if do_sample:
            gen_kwargs.update(temperature=temperature, top_p=top_p)

        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)

        # Left padding makes the prompt length uniform, so the continuation for
        # every row starts at the input width -- slice it off to keep only new
        # tokens (never echo the prompt back into the loop).
        new_tokens = out[:, enc["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    def _render(self, prompt: str) -> str:
        """Wrap a role prompt as a single user turn if the model is chat-tuned.

        The prover/verifier/corrector prompts are already complete instructions,
        so one user message is the whole conversation. A base model with no chat
        template is fed the prompt verbatim.
        """
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def count_tokens(self, text: str) -> int:
        if self.tokenizer is None:
            return super().count_tokens(text)
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def teardown(self) -> None:
        # Drop this client's references first, then release its hold on the
        # shared copy. The weights are freed only once the last client using
        # them tears down (refcount hits zero).
        self.model = None
        self.tokenizer = None
        if not self._cached:
            return
        self._cached = False
        freed = False
        with HFClient._lock:
            entry = HFClient._cache.get(self.spec.model)
            if entry is not None:
                entry.refcount -= 1
                if entry.refcount <= 0:
                    del HFClient._cache[self.spec.model]
                    freed = True
        if freed:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass


#: client name -> class. Mirrors oracle/budget's ESTIMATORS registry idiom.
MODEL_CLIENTS: dict[str, type[ModelClient]] = {
    "mock": MockModelClient,
    "openai": OpenAIClient,
    "hf": HFClient,
}


def build_model_client(spec: ModelSpec) -> ModelClient:
    """Construct the client named by `spec.client`."""
    try:
        cls = MODEL_CLIENTS[spec.client]
    except KeyError:
        raise ValueError(
            f"unknown model client {spec.client!r}. "
            f"Available: {sorted(MODEL_CLIENTS)}"
        ) from None
    return cls(spec)


_OPENAI_SPEC = (
    "agent model client 'openai' is not implemented yet.\n"
    "  - Use client: mock to run the loop on CPU.\n"
    "  - To implement: see the docstring of frugalprover.agent.model.OpenAIClient."
)
