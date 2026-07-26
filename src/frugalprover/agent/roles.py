"""The three roles of the verify-repair loop.

Each role wraps one :class:`ModelClient` and its decoding params, renders its
prompt, and parses the raw completion into whatever the loop needs. The prover
and corrector return text; the verifier returns a structured :class:`Critique`.

The API is **batch-first**: every method takes a list of items and issues a
single `generate` call across all of them, so a batching backend (vLLM, HF)
processes the whole active set at once instead of one prompt at a time. Token
accounting is left to the caller, because one role instance serves many attempts
and the loop tracks tokens per attempt.

The `Critique.flaws` list is the feedback contract: the verifier hands the
corrector *specific diagnosed flaws*, not a bare pass/fail, because a repair
step can only fix what it's told is wrong.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from frugalprover.agent.model import ModelClient
from frugalprover.common.config import ModelSpec


@dataclass
class Critique:
    """One verifier's audit of a candidate solution."""

    accept: bool
    flaws: list[str] = field(default_factory=list)
    raw: str = ""


class _Role:
    """Common wiring: a client plus the decoding params it should use."""

    def __init__(self, client: ModelClient, spec: ModelSpec, prompt: str, role: str):
        self.client = client
        self.spec = spec
        self.prompt = prompt
        self.role = role

    def _generate(self, texts: list[str]) -> list[str]:
        if not texts:
            return []
        return self.client.generate(
            texts,
            max_tokens=self.spec.max_tokens,
            temperature=self.spec.temperature,
            top_p=self.spec.top_p,
            role=self.role,
        )

    def count_tokens(self, text: str) -> int:
        return self.client.count_tokens(text)


class Prover(_Role):
    """Proposes candidate solutions, one per problem."""

    def propose(self, problems: list[str]) -> list[str]:
        return self._generate([self.prompt.format(problem=p) for p in problems])


class Corrector(_Role):
    """Repairs candidates against their diagnosed flaws.

    `items` is a list of ``(problem, candidate, flaws)`` triples; returns one
    corrected candidate per item, in order.
    """

    def repair(self, items: list[tuple[str, str, list[str]]]) -> list[str]:
        texts = []
        for problem, candidate, flaws in items:
            flaw_text = "\n".join(f"- {f}" for f in flaws) or "- (unspecified)"
            texts.append(
                self.prompt.format(problem=problem, candidate=candidate, flaws=flaw_text)
            )
        return self._generate(texts)


class Verifier(_Role):
    """Audits candidates, returning one :class:`Critique` per item.

    `items` is a list of ``(problem, candidate)`` pairs. In debate mode, `peers`
    is a per-item list of the critiques earlier verifiers already gave for that
    same item, so this verifier can see them.
    """

    def audit(
        self,
        items: list[tuple[str, str]],
        peers: list[list[Critique]] | None = None,
    ) -> list[Critique]:
        texts = []
        for i, (problem, candidate) in enumerate(items):
            text = self.prompt.format(problem=problem, candidate=candidate)
            if peers and peers[i]:
                joined = "\n".join(f"- {f}" for c in peers[i] for f in c.flaws)
                text += f"\n\nOther reviewers raised:\n{joined}\n"
            texts.append(text)
        return [parse_critique(o) for o in self._generate(texts)]


def parse_critique(raw: str) -> Critique:
    """Parse a verifier completion into a :class:`Critique`.

    Expected shape (see config.VERIFY_PROMPT)::

        VERDICT: ACCEPT | REJECT
        FLAWS:
        - flaw one
        - flaw two

    Lenient by design: an unparseable or verdict-less response is treated as a
    REJECT with the raw text as the flaw, so a malformed audit fails safe (never
    a silent accept) rather than crashing the loop.
    """
    accept = False
    flaws: list[str] = []
    in_flaws = False
    found_verdict = False
    for line in raw.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("VERDICT:"):
            found_verdict = True
            accept = "ACCEPT" in upper and "REJECT" not in upper
            in_flaws = False
        elif upper.startswith("FLAWS"):
            in_flaws = True
        elif in_flaws and stripped.startswith("-"):
            flaw = stripped.lstrip("-").strip()
            if flaw and flaw.lower() != "none":
                flaws.append(flaw)
    if not found_verdict:
        return Critique(accept=False, flaws=[raw.strip() or "unparseable audit"], raw=raw)
    if not accept and not flaws:
        flaws = ["rejected without a stated flaw"]
    return Critique(accept=accept, flaws=flaws, raw=raw)
