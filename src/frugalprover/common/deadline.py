"""Wall-clock stopping for long batched runs.

Both expensive stages -- Stage 2 labeling and an allocation run -- are given a
fixed window (a Colab session, a rented pod) and have to stop *cleanly* inside
it, having flushed whatever they finished. The rule is the same in both places,
so it lives in one.
"""
from __future__ import annotations

import time

from frugalprover.common.logging import get_logger

log = get_logger(__name__)


def should_stop(limit: float | None, t0: float, batch_s: float | None) -> bool:
    """Whether to stop before starting another batch.

    The predictive arm is the one that buys coverage: a batch started three
    minutes before the deadline finishes after it, wasting both those minutes and
    the batch. `batch_s` is an EMA of recent batch durations, so it
    self-calibrates to the hardware.
    """
    if limit is None:
        return False
    elapsed = time.perf_counter() - t0
    if elapsed >= limit:
        log.warning("time budget %.0fs reached after %.0fs -- stopping between batches",
                    limit, elapsed)
        return True
    if batch_s is not None and elapsed + batch_s > limit:
        log.warning("time budget %.0fs: %.0fs elapsed and the next batch needs ~%.0fs -- "
                    "stopping now rather than starting one that would be cut off",
                    limit, elapsed, batch_s)
        return True
    return False


def ema(previous: float | None, sample: float, weight: float = 0.3) -> float:
    """Exponential moving average of batch wall time."""
    return sample if previous is None else (1 - weight) * previous + weight * sample
