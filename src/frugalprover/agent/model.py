"""Model backends for the agent's roles.

A role (prover, verifier, corrector) talks to a model through a `ModelClient`.
The interface is deliberately thin -- text in, text out -- so the loop in
`verify_repair.py` never knows whether it's hitting a mock, a vLLM endpoint, or
a local transformer.

Only `MockModelClient` is implemented; it needs no torch and no network, so the
whole loop runs and is testable on a laptop. `openai` and `hf` are registered
but raise `NotImplementedError` with a spec -- the same "seam visible, body
later" pattern as oracle/budget/sweep.py.

Nothing here imports from `oracle/`; the dependency runs one way (see
agent/README.md).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from frugalprover.common.config import ModelSpec


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
    """Local `transformers` generation.

    NOT IMPLEMENTED. A conforming body would, lazily inside `setup()`, load
    `AutoModelForCausalLM`/`AutoTokenizer` for `spec.model` (mirroring
    oracle/states/hf_extractor.py: `USE_TF=0`, dtype handling, cuda fallback),
    then in `generate` batch the prompts, call `model.generate(..., do_sample=True,
    max_new_tokens=max_tokens, temperature=temperature, top_p=top_p)`, and decode
    only the newly generated tokens (watch the tokenizer's padding side).
    """

    def setup(self) -> None:
        raise NotImplementedError(_HF_SPEC)

    def generate(self, prompts, *, max_tokens, temperature, top_p, role="prover"):
        raise NotImplementedError(_HF_SPEC)


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
_HF_SPEC = (
    "agent model client 'hf' is not implemented yet.\n"
    "  - Use client: mock to run the loop on CPU.\n"
    "  - To implement: see the docstring of frugalprover.agent.model.HFClient."
)
