"""Answer extraction, normalization, grading, and surface features.

All of this is lifted from the pilot notebook and pipeline/smoke_test.py, where
it was duplicated in three places. It is deliberately dependency-free (stdlib
only) so both the CPU analysis path and the GPU generation path can use it.

Grading is intentionally crude: normalize both sides and compare strings. MATH
answers are short LaTeX expressions, so this catches the common formatting
mismatches (\\dfrac vs \\frac, stray spaces, \\left/\\right) without pulling in
a CAS. It will under-count on algebraically-equal-but-textually-different
answers; that biases p(B) down uniformly, which is acceptable for ranking
problems by effort but worth remembering before quoting an absolute accuracy.
"""
from __future__ import annotations

import math
import re

BOXED_RE = re.compile(r"\\boxed\{")
FINAL_ANSWER_RE = re.compile(r"final answer is[:\s]*\$?([^\n$]+)\$?", re.IGNORECASE)
LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]+")

#: Keys returned by :func:`surface_features`, in a stable order. This is the
#: "length baseline" the research plan calls confound #1: if activations don't
#: beat these seven numbers, there is no result.
SURFACE_KEYS = [
    "char_len",
    "word_len",
    "latex_cmd_count",
    "dollar_count",
    "brace_depth",
    "digit_count",
    "eq_count",
]


def extract_boxed(text: str) -> str | None:
    """Content of the *last* ``\\boxed{...}``, respecting nested braces.

    Last, not first, because MATH solutions often box an intermediate result
    before boxing the final one. Brace-aware because ``\\boxed{\\frac{1}{2}}``
    would otherwise truncate at the first ``}``.
    """
    matches = list(BOXED_RE.finditer(text))
    if not matches:
        return None
    start = matches[-1].end()
    depth, i = 1, start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start : i - 1].strip()


def normalize(ans: str | None) -> str:
    """Collapse the formatting differences that don't change an answer's value."""
    if ans is None:
        return ""
    a = ans.strip()
    a = a.replace("\\left", "").replace("\\right", "")
    a = a.replace(" ", "").replace("\\!", "")
    a = a.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    return a.rstrip(".")


def extract_answer(text: str) -> str | None:
    """Model's answer: prefer a boxed expression, fall back to "final answer is X"."""
    pred = extract_boxed(text)
    if pred is None:
        m = FINAL_ANSWER_RE.search(text)
        pred = m.group(1) if m else None
    return pred


def grade(generated_text: str, gold_answer: str) -> bool:
    """True if the generation's answer matches the gold answer.

    An empty extraction never counts as correct, so a model that produced no
    parseable answer can't accidentally match an empty gold.
    """
    pred = normalize(extract_answer(generated_text))
    return pred == normalize(gold_answer) and pred != ""


def max_brace_depth(s: str) -> int:
    """Deepest nesting of ``{}`` — a cheap proxy for expression complexity."""
    depth = maxd = 0
    for ch in s:
        if ch == "{":
            depth += 1
            maxd = max(maxd, depth)
        elif ch == "}":
            depth = max(0, depth - 1)
    return maxd


def surface_features(problem: str) -> dict[str, float]:
    """Cheap text statistics computable without any model. Keys are SURFACE_KEYS."""
    return {
        "char_len": float(len(problem)),
        "word_len": float(len(problem.split())),
        "latex_cmd_count": float(len(LATEX_CMD_RE.findall(problem))),
        "dollar_count": float(problem.count("$")),
        "brace_depth": float(max_brace_depth(problem)),
        "digit_count": float(sum(ch.isdigit() for ch in problem)),
        "eq_count": float(problem.count("=")),
    }


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Used instead of the normal approximation because n is tiny (3 attempts per
    budget is typical). At n=3, "2 successes" gives p=0.67 with a CI of roughly
    [0.21, 0.94] — the interval is what stops anyone from over-reading that 0.67.
    """
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))
