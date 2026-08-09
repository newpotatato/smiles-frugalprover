"""Model backends for the agent's roles.

A role (prover, verifier, corrector) talks to a model through a `ModelClient`.
The interface is deliberately thin -- text in, text out -- so the loop in
`verify_repair.py` never knows whether it's hitting a mock, a vLLM endpoint, or
a local transformer.

`MockModelClient` needs no torch and no network, so the whole loop runs and is
testable on a laptop. `HFClient` runs an open model locally via `transformers`.
`OpenAIClient` drives an OpenAI-compatible server (vLLM) over HTTP, which is the
throughput path: the server batches continuously across every in-flight request
instead of the fixed-size waves `HFClient` is limited to, and this process needs
no torch at all.

Nothing here imports from `oracle/`; the dependency runs one way (see
agent/README.md).
"""
from __future__ import annotations

import os
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

from frugalprover.common.config import ModelSpec
from frugalprover.common.logging import get_logger

log = get_logger(__name__)


def per_prompt(value, n: int, what: str) -> list:
    """Broadcast a scalar to `n` items, or check a list is already that long.

    `max_tokens` and `seeds` are both "one per prompt, or one for all"; doing the
    widening once here keeps every client from re-deriving it, and turns a
    length mismatch into an error at the call rather than a silently truncated
    `zip` that would attach one problem's cap to another's prompt.
    """
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(f"{what}: expected {n} value(s) for {n} prompt(s), got {len(value)}")
        return list(value)
    return [value] * n


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
        max_tokens: int | list[int],
        temperature: float,
        top_p: float,
        role: str = "prover",
        seeds: list[int] | None = None,
    ) -> list[str]:
        """One completion per prompt, in input order.

        `max_tokens` is either one cap for the batch or one per prompt (see
        :func:`per_prompt`) -- the per-prompt form is what lets a single batched
        call serve attempts on different budgets.

        `seeds`, when given, is one decoding seed per prompt. A backend that
        honours it makes sampling reproducible, so the same problem run twice at
        different caps follows the same trajectory. Backends that can't are free
        to ignore it.

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
        max_tokens: int | list[int],
        temperature: float,
        top_p: float,
        role: str = "prover",
        seeds: list[int] | None = None,
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
    """OpenAI-compatible endpoint, e.g. ``vllm serve --served-model-name ...``.

    One HTTP request per prompt, dispatched concurrently through a thread pool
    and reassembled **in input order**. The server does its own continuous
    batching, so throughput comes from keeping many requests in flight
    (``spec.max_concurrency``) rather than from packing a tensor.

    Each prompt is sent as a single user message, so the server applies the
    model's own chat template -- the same conversation :meth:`HFClient._render`
    builds locally. Divergence there would mean hf-labelled and server-labelled
    runs are not the same experiment.

    Nothing in this class imports torch: with the model on a server, a CPU-only
    box can drive a full Stage 2 labeling run.
    """

    #: Completion-token counts are cached by returned text so `count_tokens` can
    #: answer from what the server already reported. Bounded because a long run
    #: generates far more text than needs to stay resident -- every call site
    #: counts a completion in the same round it was generated.
    _TOKEN_CACHE_MAX = 8192

    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        self._client = None
        self._http = None
        self._tokens: "OrderedDict[str, int]" = OrderedDict()
        self._tok_lock = threading.Lock()
        self._tokenize_url: str | None = None
        self._tokenize_dead = False
        self._warned_estimate = False

    # -- lifecycle ---------------------------------------------------------

    def setup(self) -> None:
        if self._client is not None:
            return
        try:
            import httpx
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "agent client 'openai' needs the openai SDK: "
                "pip install 'frugalprover[openai]'"
            ) from e

        base_url = self.spec.base_url or "http://127.0.0.1:8000/v1"
        # api_key_env names the variable, never holds the key. A server started
        # without --api-key accepts any bearer token, but the SDK refuses to send
        # an empty one, so "EMPTY" is the right default rather than "".
        key = os.environ.get(self.spec.api_key_env or "OPENAI_API_KEY") or "EMPTY"
        self._client = OpenAI(
            base_url=base_url,
            api_key=key,
            max_retries=6,
            # The SDK default is 600s. An 8k-token generation queued behind a few
            # hundred others exceeds that routinely, and the failure surfaces as a
            # random subset of requests dying rather than as a timeout -- which
            # would look like the model refusing to answer.
            timeout=httpx.Timeout(1800.0, connect=15.0),
        )
        # vLLM serves /tokenize at the server root, a level above the /v1 prefix.
        # Own httpx client rather than reaching into the SDK's private one.
        root = base_url.rstrip("/")
        root = root[: -len("/v1")] if root.endswith("/v1") else root
        self._tokenize_url = root + "/tokenize"
        self._http = httpx.Client(timeout=httpx.Timeout(30.0, connect=15.0),
                                  headers={"Authorization": f"Bearer {key}"})
        log.info("agent client 'openai': %s (model=%s, max_concurrency=%d)",
                 base_url, self.spec.model, self.spec.max_concurrency)

    def teardown(self) -> None:
        for attr in ("_client", "_http"):
            obj = getattr(self, attr)
            setattr(self, attr, None)
            if obj is not None:
                try:
                    obj.close()
                except Exception:  # noqa: BLE001 - teardown must not mask a real error
                    pass
        with self._tok_lock:
            self._tokens.clear()

    # -- generation --------------------------------------------------------

    def generate(
        self,
        prompts: list[str],
        *,
        max_tokens: int | list[int],
        temperature: float,
        top_p: float,
        role: str = "prover",
        seeds: list[int] | None = None,
    ) -> list[str]:
        if self._client is None:
            raise RuntimeError("call setup() before generate()")
        if not prompts:
            return []
        import time

        caps = per_prompt(max_tokens, len(prompts), "max_tokens")
        seed_list = per_prompt(seeds, len(prompts), "seeds") if seeds is not None else None
        n_workers = max(1, min(self.spec.max_concurrency, len(prompts)))
        distinct = sorted(set(caps))
        log.info("[%s] generating %d prompt(s) via %s, %d worker(s), max_tokens=%s, decoding=%s%s",
                 role, len(prompts), self.spec.base_url, n_workers,
                 distinct[0] if len(distinct) == 1 else f"{distinct[0]}-{distinct[-1]}",
                 "sampling" if (temperature and temperature > 0) else "greedy",
                 " (seeded)" if seed_list is not None else "")

        t0 = time.perf_counter()
        progress = {"done": 0, "last": t0}
        lock = threading.Lock()

        def one(prompt: str, cap: int, seed: int | None) -> str:
            kwargs = dict(
                model=self.spec.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=cap,
            )
            # `seed` is opt-in and comes from the caller, never invented here.
            # Left unset, n_samples sends identical prompts and a shared seed
            # would return identical completions, collapsing the
            # self-consistency baseline to a point mass -- so the loop derives a
            # per-*sample* seed (agent/base.py:stable_seed) or passes none.
            if seed is not None:
                kwargs["seed"] = seed
            if temperature is not None and temperature > 0.0:
                kwargs.update(temperature=temperature, top_p=top_p)
            else:
                kwargs["temperature"] = 0.0
            resp = self._client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            if usage is not None and usage.completion_tokens is not None:
                self._remember(text, usage.completion_tokens)
            # Heartbeat, mirroring HFClient's: a long batch is otherwise silent.
            with lock:
                progress["done"] += 1
                now = time.perf_counter()
                if now - progress["last"] >= 10.0:
                    log.info("[%s]   ...%d/%d completion(s) (%.0fs elapsed)",
                             role, progress["done"], len(prompts), now - t0)
                    progress["last"] = now
            return text

        with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix=f"gen-{role}") as ex:
            futures = [
                ex.submit(one, p, caps[i], seed_list[i] if seed_list else None)
                for i, p in enumerate(prompts)
            ]
            # Indexing the futures list -- NOT as_completed -- is what preserves
            # input order. The loop zips these results positionally against its
            # attempts, so an out-of-order return would silently attach one
            # problem's solution to another.
            out = [f.result() for f in futures]

        dt = time.perf_counter() - t0
        log.info("[%s] done: %d completion(s) in %.1fs (%.1f req/s)",
                 role, len(out), dt, len(out) / dt if dt > 0 else 0.0)
        return out

    # -- token accounting --------------------------------------------------

    def _remember(self, text: str, n: int) -> None:
        with self._tok_lock:
            self._tokens[text] = n
            self._tokens.move_to_end(text)
            while len(self._tokens) > self._TOKEN_CACHE_MAX:
                self._tokens.popitem(last=False)

    def count_tokens(self, text: str) -> int:
        """Tokens the server generated for `text`.

        This has to be right, not merely close. The count is the only input to
        the verify-repair loop's budget cap, so a systematic undercount means the
        cap never fires, every budget produces the same outcome, and `b_star`
        degenerates into a binary solved flag with nothing in the output looking
        wrong. The inherited whitespace default is a 3-5x undercount on
        LaTeX-heavy math, so it is never used here.
        """
        with self._tok_lock:
            n = self._tokens.get(text)
        if n is not None:
            return n                                # exact: the server told us
        n = self._count_via_server(text)
        if n is not None:
            self._remember(text, n)
            return n                                # exact: the server's tokenizer
        if not self._warned_estimate:
            log.warning(
                "[openai] no usage or /tokenize count available for %s -- falling back to a "
                "character heuristic. The budget axis will be approximate; check that the "
                "endpoint returns `usage` or supports POST /tokenize.", self.spec.model,
            )
            self._warned_estimate = True
        return max(1, len(text) // 4)

    def _count_via_server(self, text: str) -> int | None:
        """vLLM's POST /tokenize. Returns None if the backend doesn't have it."""
        if self._http is None or self._tokenize_dead or not self._tokenize_url:
            return None
        try:
            resp = self._http.post(
                self._tokenize_url,
                json={"model": self.spec.model, "prompt": text, "add_special_tokens": False},
            )
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code}")
            return int(resp.json()["count"])
        except Exception as e:  # noqa: BLE001 - any failure means "not available"
            self._tokenize_dead = True  # try once, never hammer a 404 per call
            log.debug("[openai] /tokenize unavailable (%s); using cached usage only", e)
            return None

    def describe(self) -> dict:
        # How tokens were counted belongs in the A2 sidecar next to the model id:
        # the server's completion_tokens and HFClient's decode-then-re-encode are
        # not the same measurement, so two runs counted differently are not one
        # labeling run.
        return {
            **super().describe(),
            "base_url": self.spec.base_url,
            "max_concurrency": self.spec.max_concurrency,
            "token_accounting": "server_usage",
        }


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

    ``spec.quantization`` (``8bit`` / ``4bit``, CUDA only) loads the weights
    through bitsandbytes instead of at full dtype -- the lever that fits a 7B on a
    16GB card.

    Loaded weights are shared across all instances via the class-level
    :attr:`_cache`: two clients naming the same ``spec.model`` (e.g. a prover and
    a verifier on one 7B) reuse a single copy instead of loading it twice. The
    copy is reference-counted and freed when the last client using it tears down.
    """

    #: Class-level cache of loaded models, keyed by (repo id / path, quantization),
    #: shared by every HFClient in the process. The verify-repair loop builds a
    #: separate client per role and a trio commonly reuses the same checkpoint, so
    #: without sharing each role would load its own copy and multiply GPU memory by
    #: the role count. dtype/device are derived deterministically from the hardware,
    #: so those two fields identify the loaded artifact. `_lock` guards it so
    #: concurrent setups don't double-load.
    _cache: dict[str, _LoadedModel] = {}
    _lock = threading.Lock()

    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        self.model = None
        self.tokenizer = None
        self.device = None
        self._cached = False  # this client holds a reference in the class cache
        self._warned_seed = False

    @property
    def _cache_key(self) -> str:
        return f"{self.spec.model}|{self._quantization}"

    @property
    def _quantization(self) -> str:
        return (self.spec.quantization or "none").lower()

    def setup(self) -> None:
        if self.model is not None:
            return
        key = self._cache_key
        with HFClient._lock:
            entry = HFClient._cache.get(key)
            if entry is None:
                entry = self._load(self.spec.model)
                HFClient._cache[key] = entry
                log.info(
                    "loaded %s for agent client 'hf': device=%s quantization=%s",
                    self.spec.model, entry.device, self._quantization,
                )
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

        quant = self._quantization
        # Checked before anything is fetched: a misconfigured run should fail in
        # seconds, not after pulling a checkpoint.
        if quant != "none" and device != "cuda":
            raise RuntimeError(
                f"quantization={quant!r} needs a CUDA GPU (bitsandbytes has no "
                "CPU kernels); set quantization: none to run on CPU."
            )
        # LLM.int8()'s matmul kernel only runs in fp16: handed bf16 activations it
        # casts them itself and warns once per call, which is thousands of lines
        # across a verify-repair sweep. Asking for fp16 up front is the same
        # arithmetic without the cast. 4bit has no such constraint -- it takes
        # bf16 as its compute dtype directly (see _quant_config).
        if quant == "8bit":
            dtype = torch.float16

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Decoder generation must pad on the left, or right-padding tokens land
        # between the prompt and the continuation and corrupt every sample.
        tokenizer.padding_side = "left"

        # low_cpu_mem_usage streams the checkpoint shard by shard into the target
        # dtype instead of materializing a full copy in RAM first -- the load-time
        # spike is what kills a small box before a single token is generated.
        kwargs = dict(torch_dtype=dtype, low_cpu_mem_usage=True)
        if quant == "none":
            model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
            model = model.to(device)
        else:
            kwargs["quantization_config"] = self._quant_config(quant, dtype)
            # bitsandbytes places the shards itself and a quantized model rejects a
            # later .to(), so pin every layer to GPU 0 at load and skip the move.
            kwargs["device_map"] = {"": 0}
            model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        return _LoadedModel(model=model.eval(), tokenizer=tokenizer, device=device)

    @staticmethod
    def _quant_config(quant: str, compute_dtype: Any) -> Any:
        """bitsandbytes config for `8bit` / `4bit`.

        4bit uses NF4 with double quantization -- the configuration the QLoRA
        paper reports as matching bf16 accuracy most closely, and the one worth
        defaulting to since the whole point here is to lose as little as possible
        for the memory saved.
        """
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as e:
            raise ImportError(
                f"agent model quantization={quant!r} needs bitsandbytes: "
                "pip install 'frugalprover[quant]'"
            ) from e
        from transformers import BitsAndBytesConfig

        if quant == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    def generate(
        self,
        prompts: list[str],
        *,
        max_tokens: int | list[int],
        temperature: float,
        top_p: float,
        role: str = "prover",
        seeds: list[int] | None = None,
    ) -> list[str]:
        if self.model is None:
            raise RuntimeError("call setup() before generate()")
        if not prompts:
            return []
        import time

        if seeds is not None and not self._warned_seed:
            # `model.generate` seeds from the global torch RNG for the whole
            # batch, so there is no per-sequence seed to honour here. Saying so
            # once beats letting a caller believe hf and vllm runs are paired.
            log.warning("[hf] per-prompt seeds are not supported by transformers "
                        "generation; sampling stays unseeded for this client.")
            self._warned_seed = True

        caps = per_prompt(max_tokens, len(prompts), "max_tokens")
        # One `generate` call emits the same number of new tokens for every row,
        # so prompts on different budgets cannot share a call without the
        # smallest cap truncating the rest. Group by cap, then chunk each group
        # for memory -- allocation policies draw from a short grid, so this is a
        # handful of groups, not one call per prompt.
        groups: dict[int, list[int]] = {}
        for i, cap in enumerate(caps):
            groups.setdefault(cap, []).append(i)

        batch_size = max(1, self.spec.max_batch_size)
        chunks = [
            (cap, idx[start:start + batch_size])
            for cap, idx in sorted(groups.items())
            for start in range(0, len(idx), batch_size)
        ]
        log.info("[%s] generating %d prompt(s) in %d chunk(s) over %d cap group(s), "
                 "max_new_tokens=%s, decoding=%s",
                 role, len(prompts), len(chunks), len(groups), sorted(groups),
                 "sampling" if (temperature and temperature > 0) else "greedy")

        out: list[str] = [""] * len(prompts)
        t0 = time.perf_counter()
        for ci, (cap, idx) in enumerate(chunks, 1):
            tc = time.perf_counter()
            texts = self._generate_chunk(
                [prompts[i] for i in idx], cap, temperature, top_p,
                role=role, chunk=(ci, len(chunks)))
            for i, text in zip(idx, texts):
                out[i] = text
            log.info("[%s] chunk %d/%d finished in %.1fs", role, ci, len(chunks),
                     time.perf_counter() - tc)
        log.info("[%s] done: %d completion(s) in %.1fs", role, len(out),
                 time.perf_counter() - t0)
        return out

    def _generate_chunk(
        self, prompts: list[str], max_tokens: int, temperature: float, top_p: float,
        *, role: str = "prover", chunk: tuple[int, int] = (1, 1),
    ) -> list[str]:
        import time

        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        texts = [self._render(p) for p in prompts]
        enc = self.tokenizer(
            texts, return_tensors="pt", padding=True
        ).to(self.device)
        prompt_len = enc["input_ids"].shape[1]
        ci, n_chunks = chunk
        log.info("[%s] chunk %d/%d: %d prompt(s), %d prompt tokens -> generating up to "
                 "%d new tokens", role, ci, n_chunks, len(prompts), prompt_len, max_tokens)

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

        # Heartbeat: `model.generate` is one blocking call that can run for
        # minutes at a large token cap. A StoppingCriteria is invoked after every
        # decoding step, so it's a zero-cost hook to log token progress -- it never
        # stops (always returns False), it just reports, turning an apparent freeze
        # into a visible "generating N/max tokens" trail.
        class _Heartbeat(StoppingCriteria):
            def __init__(self, every_seconds: float = 10.0):
                self.every = every_seconds
                self.start = time.perf_counter()
                self.last = self.start

            def __call__(self, input_ids, scores, **kwargs):
                now = time.perf_counter()
                if now - self.last >= self.every:
                    generated = input_ids.shape[1] - prompt_len
                    log.info("[%s]   ...generating %d/%d new tokens (%.0fs elapsed)",
                             role, generated, max_tokens, now - self.start)
                    self.last = now
                return False

        gen_kwargs["stopping_criteria"] = StoppingCriteriaList([_Heartbeat()])

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

    def describe(self) -> dict:
        # Quantization changes what the weights answer, so it belongs in the A2
        # sidecar next to the model id -- two runs of "the same model" at 4bit and
        # bf16 are not the same labeling run.
        return {**super().describe(), "quantization": self._quantization}

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
            entry = HFClient._cache.get(self._cache_key)
            if entry is not None:
                entry.refcount -= 1
                if entry.refcount <= 0:
                    del HFClient._cache[self._cache_key]
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
