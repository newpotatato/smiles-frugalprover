"""Hidden-state extraction with a HuggingFace transformer.

Forward pass only -- nothing is generated here. That's what makes this stage
cheap enough to run on the full problem pool while budget labeling runs on a
much smaller subset: a forward pass costs one pass over the prompt, while
labeling costs hundreds of sampled tokens times several budgets times several
attempts.

Because `output_hidden_states=True` computes every layer anyway, extracting all
of them costs essentially nothing extra over extracting one. `all_layers: true`
is usually the right default -- which layer carries the signal is the open
question, and re-running on a GPU to try a different one is the expensive
mistake.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from frugalprover.common.config import ExtractConfig
from frugalprover.common.records import ProblemRecord
from frugalprover.oracle.states.pooling import geometry, pool, resolve_layer


class TransformerHiddenStateExtractor:
    """Pools hidden states from a HuggingFace causal/base model."""

    def __init__(self, cfg: ExtractConfig):
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self.device = None
        self.n_hidden = 0
        self.hidden_size = 0
        self._pooled: list[tuple[int, str]] = []   # (resolved_layer, pooling)
        self._geometry: list[tuple[int, str]] = []

    # ------------------------------------------------------------------ setup

    def setup(self) -> None:
        # transformers probes for TensorFlow and Flax at import time and will
        # crash the whole stage if either is installed but broken (a common
        # state, since old TF builds don't work with numpy 2.x). This pipeline
        # only ever uses the torch backend, so opt out of the others.
        import os

        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("USE_FLAX", "0")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "hidden-state extraction needs torch and transformers: "
                "pip install 'frugalprover[gpu]'"
            ) from e

        cfg = self.cfg
        device = cfg.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            print(f"warning: device={device!r} requested but CUDA is unavailable; using CPU")
            device = "cpu"
        # fp16 on CPU is slower than fp32 and unsupported for some ops
        dtype = getattr(torch, cfg.dtype)
        if device == "cpu" and dtype in (torch.float16, torch.bfloat16):
            print(f"warning: {cfg.dtype} is not useful on CPU; using float32")
            dtype = torch.float32

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModel.from_pretrained(
            cfg.model_name, torch_dtype=dtype, output_hidden_states=True
        ).to(device).eval()

        config = self.model.config
        n_layers = getattr(config, "num_hidden_layers", None) or getattr(config, "n_layer")
        self.n_hidden = n_layers + 1  # index 0 is the embedding output
        self.hidden_size = getattr(config, "hidden_size", None) or getattr(config, "n_embd")

        self._resolve_columns()
        print(f"loaded {cfg.model_name}: {n_layers} layers, hidden_size={self.hidden_size}, "
              f"device={device}, dtype={str(dtype).replace('torch.', '')}")
        print(f"  extracting {len(self._pooled)} pooled + {len(self._geometry)} geometry columns")

    def _resolve_columns(self) -> None:
        """Turn config layer indices (possibly negative) into absolute ones,
        deduplicating so `all_layers` plus an explicit `-1` doesn't produce the
        same column twice."""
        cfg = self.cfg
        pooled = [(resolve_layer(f.layer, self.n_hidden), f.pooling) for f in cfg.features]
        if cfg.all_layers:
            pooled += [(i, cfg.all_layers_pooling) for i in range(self.n_hidden)]
        self._pooled = sorted(set(pooled))
        self._geometry = sorted({
            (resolve_layer(g.layer, self.n_hidden), g.metric) for g in cfg.geometry
        })
        if not self._pooled:
            raise ValueError(
                "no pooled features configured - set extract.features or "
                "extract.all_layers: true, otherwise the oracle has nothing to read"
            )

    # ---------------------------------------------------------------- extract

    def extract_batch(self, problems: list[ProblemRecord]) -> list[dict[str, Any]]:
        import torch

        cfg = self.cfg
        prompts = [cfg.prompt_template.format(problem=p.problem) for p in problems]
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_input_tokens,
        ).to(self.device)

        with torch.no_grad():
            out = self.model(**enc)

        hs, mask = out.hidden_states, enc["attention_mask"]
        columns: dict[str, np.ndarray] = {}
        for layer, kind in self._pooled:
            columns[f"L{layer}_{kind}"] = pool(hs[layer], mask, kind).float().cpu().numpy()
        for layer, metric in self._geometry:
            columns[f"L{layer}_{metric}"] = geometry(hs[layer], mask, metric).float().cpu().numpy()

        rows = []
        for i, p in enumerate(problems):
            row: dict[str, Any] = {"id": p.id, "type": p.type}
            for name, values in columns.items():
                v = values[i]
                # float32 halves the parquet size and is well past the
                # precision any of this analysis can resolve
                row[name] = v.astype(np.float32) if v.ndim else float(v)
            rows.append(row)
        return rows

    def teardown(self) -> None:
        self.model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @property
    def spec(self) -> dict:
        return {
            "model_name": self.cfg.model_name,
            "hidden_size": self.hidden_size,
            "n_hidden_states": self.n_hidden,
            "num_hidden_layers": self.n_hidden - 1,
            "prompt_template": self.cfg.prompt_template,
            "max_input_tokens": self.cfg.max_input_tokens,
            "dtype": self.cfg.dtype,
            "device": self.device,
            "pooled_columns": {f"L{l}_{k}": {"layer": l, "pooling": k} for l, k in self._pooled},
            "geometry_columns": {f"L{l}_{m}": {"layer": l, "metric": m} for l, m in self._geometry},
        }
