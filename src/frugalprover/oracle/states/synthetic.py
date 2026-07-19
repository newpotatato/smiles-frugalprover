"""Synthetic hidden states -- no model, no GPU.

Two real uses:

1. **A null baseline.** Random features should score at chance. If the oracle
   scores well on these, the score is coming from the surface features or from
   a leak, not from activations. Worth running once alongside any positive
   result.

2. **Exercising Stages 4-5** without a GPU or a working torch install, which is
   what `configs/smoke.yaml` does.

With `signal_strength > 0` one layer carries a planted, difficulty-correlated
direction. That makes the layer sweep have something to find, so you can tell a
broken pipeline (picks a noise layer) from a working one (picks the planted
layer). Set it to 0 for a true null.

Nothing produced here is a research result, and the sidecar says `synthetic:
true` so a run built on it can't be mistaken for one.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from frugalprover.common.config import ExtractConfig
from frugalprover.common.records import ProblemRecord


def _seed(problem_id: str, salt: int) -> int:
    """Stable 32-bit seed from a problem id, identical across processes."""
    import hashlib

    digest = hashlib.sha256(f"{salt}:{problem_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


class SyntheticExtractor:
    """Random pooled vectors, optionally with a planted signal in one layer."""

    def __init__(self, cfg: ExtractConfig):
        self.cfg = cfg
        self.n_layers = max(2, cfg.synthetic_layers)
        self.hidden_size = cfg.synthetic_hidden_size
        self.signal_layer = cfg.synthetic_signal_layer % self.n_layers
        self.strength = cfg.synthetic_signal_strength
        self._rng = None
        self._direction = None

    def setup(self) -> None:
        self._rng = np.random.default_rng(self.cfg.synthetic_seed)
        direction = self._rng.normal(size=self.hidden_size)
        self._direction = direction / np.linalg.norm(direction)
        print(f"synthetic extractor: {self.n_layers} layers x {self.hidden_size} dims, "
              f"signal in L{self.signal_layer}_mean (strength {self.strength})")
        if self.strength == 0:
            print("  strength=0: pure noise, a true null baseline")

    def extract_batch(self, problems: list[ProblemRecord]) -> list[dict[str, Any]]:
        rows = []
        for p in problems:
            # seed per problem id so a problem's vector is the same regardless
            # of batch size or how the run was restarted.
            #
            # NOT builtin hash(): Python randomizes string hashing per process
            # (PYTHONHASHSEED), so that would give different vectors on every
            # run and make the smoke pipeline's score wander for no reason.
            rng = np.random.default_rng(_seed(p.id, self.cfg.synthetic_seed))
            level = p.level if p.level is not None else 3
            difficulty = (level - 3) / 2.0  # roughly centered, in [-1, 1]

            row: dict[str, Any] = {"id": p.id, "type": p.type}
            for layer in range(self.n_layers):
                vec = rng.normal(size=self.hidden_size)
                if layer == self.signal_layer:
                    vec += self.strength * difficulty * self._direction
                row[f"L{layer}_mean"] = vec.astype(np.float32)
                row[f"L{layer}_l2_norm"] = float(np.linalg.norm(vec))
            rows.append(row)
        return rows

    def teardown(self) -> None:
        pass

    @property
    def spec(self) -> dict:
        return {
            "model_name": "synthetic",
            "synthetic": True,
            "hidden_size": self.hidden_size,
            "n_hidden_states": self.n_layers,
            "num_hidden_layers": self.n_layers - 1,
            "signal_layer": f"L{self.signal_layer}_mean",
            "signal_strength": self.strength,
            "seed": self.cfg.synthetic_seed,
            "pooled_columns": {
                f"L{i}_mean": {"layer": i, "pooling": "mean"} for i in range(self.n_layers)
            },
            "geometry_columns": {
                f"L{i}_l2_norm": {"layer": i, "metric": "l2_norm"} for i in range(self.n_layers)
            },
        }
