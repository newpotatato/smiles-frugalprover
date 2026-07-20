"""Collapsing a variable-length token cloud into fixed-size features.

A forward pass gives ``(batch, tokens, hidden)`` per layer, but the oracle needs
one vector per problem. Two families:

- **Pooling** -> a ``(hidden,)`` vector. Where in representation space the
  problem sits.
- **Geometry** -> a scalar. What *shape* the token cloud has. This is the
  direction the intrinsic-dimension literature points at: difficulty may show
  up in how spread out the representation is rather than where it is.

Everything here is mask-aware. Padding tokens are not real content, and a mean
that includes them measures how long the batch's longest problem was.

Lifted from extraction/generate_hidden_states.ipynb, with one correctness fix
noted in :func:`pool` under ``last``.
"""
from __future__ import annotations

POOLINGS = ["mean", "sum", "std", "max", "last"]
METRICS = ["l2_norm", "mean_token_norm", "token_norm_std", "anisotropy", "effective_rank"]


def pool(hidden, mask, kind: str):
    """``hidden`` (B, T, H), ``mask`` (B, T) in {0,1} -> (B, H)."""
    import torch

    m = mask.unsqueeze(-1).to(hidden.dtype)  # (B, T, 1)

    if kind == "mean":
        return (hidden * m).sum(1) / m.sum(1).clamp(min=1)

    if kind == "sum":
        return (hidden * m).sum(1)

    if kind == "std":
        # per-dimension spread across valid tokens
        counts = m.sum(1).clamp(min=1)
        mean = (hidden * m).sum(1) / counts
        var = ((hidden - mean.unsqueeze(1)) ** 2 * m).sum(1) / counts
        return var.clamp(min=0).sqrt()

    if kind == "max":
        neg = torch.finfo(hidden.dtype).min
        return hidden.masked_fill(m == 0, neg).max(1).values

    if kind == "last":
        # Index of the LAST position where mask == 1.
        #
        # The notebook used `mask.sum(1) - 1`, which counts real tokens and is
        # only correct under right-padding. Decoder-only models are commonly
        # left-padded, where the real tokens sit at the END and that formula
        # silently returns a padding position instead. Taking the argmax over
        # masked positions is correct for either padding side.
        positions = torch.arange(mask.size(1), device=mask.device)
        idx = (mask * positions).argmax(dim=1)
        return hidden[torch.arange(hidden.size(0), device=hidden.device), idx]

    raise ValueError(f"unknown pooling {kind!r}. Available: {POOLINGS}")


def geometry(hidden, mask, metric: str):
    """Scalar shape statistic of the token cloud. ``hidden`` (B, T, H) -> (B,)."""
    import torch

    counts = mask.sum(1).clamp(min=1)  # valid tokens per row
    h = hidden.float()                 # geometry in fp32: svdvals and pairwise
    fmask = mask.float()               # cosines are numerically touchy in fp16

    if metric == "l2_norm":
        # how far the cloud's center sits from the origin
        mean_vec = (h * fmask.unsqueeze(-1)).sum(1) / counts.unsqueeze(-1)
        return mean_vec.norm(dim=-1)

    token_norms = h.norm(dim=-1)  # (B, T)

    if metric == "mean_token_norm":
        return (token_norms * fmask).sum(1) / counts

    if metric == "token_norm_std":
        mean = ((token_norms * fmask).sum(1) / counts).unsqueeze(1)
        var = (((token_norms - mean) ** 2) * fmask).sum(1) / counts
        return var.clamp(min=0).sqrt()

    if metric == "anisotropy":
        # mean pairwise cosine similarity between distinct token vectors,
        # in closed form: (||sum u_i||^2 - T) / (T(T-1)) for unit vectors u_i.
        # High anisotropy = all tokens point the same way = a degenerate,
        # low-information representation.
        u = torch.nn.functional.normalize(h, dim=-1) * fmask.unsqueeze(-1)
        sq = (u.sum(1) ** 2).sum(-1)
        denom = (counts * (counts - 1)).clamp(min=1)
        return (sq - counts) / denom

    if metric == "effective_rank":
        # participation ratio of the centered cloud's singular spectrum:
        # (sum lambda)^2 / sum lambda^2. A linear proxy for intrinsic
        # dimension -- how many directions the cloud actually uses.
        out = []
        for b in range(h.size(0)):
            X = h[b][mask[b].bool()]
            X = X - X.mean(0, keepdim=True)
            lam = torch.linalg.svdvals(X) ** 2
            out.append((lam.sum() ** 2) / lam.pow(2).sum().clamp(min=1e-12))
        return torch.stack(out)

    raise ValueError(f"unknown geometry metric {metric!r}. Available: {METRICS}")


def resolve_layer(layer: int, n_hidden: int) -> int:
    """Negative index -> absolute position in the hidden_states tuple.

    Column names always use the absolute index, so ``-1`` on a 28-layer model
    becomes ``L28_...``, never ``L-1_...``. Two runs on the same model produce
    the same column names whether the config said -1 or 28.
    """
    resolved = layer if layer >= 0 else n_hidden + layer
    if not 0 <= resolved < n_hidden:
        raise ValueError(
            f"layer {layer} is out of range for a model with {n_hidden} hidden "
            f"states (0 = embeddings, {n_hidden - 1} = final layer)"
        )
    return resolved
