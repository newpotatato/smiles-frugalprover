# FrugalProver: A Self-Improving Math Agents System that Learns Its
Boundaries

![figure 1](images\frugalproverBarannikov_img_001.png)

Serguei Barannikov1

1 Skolkovo Institute of Science and Technology (Skoltech)

## 1.1 Hypotheses

# Abstract

Today’s language models can already resolve some previously
open mathematical problems, but at large, unpredictable cost
and with errors that are hard to detect. We propose a com-
pact, auditable multi-agent system that proves theorems un-
der an explicit compute budget.
A budget oracle estimates,
for each problem, the effort (in tokens) a successful attempt
warrants—avoiding both overspending on easy problems and
premature termination on hard ones. A prover produces a can-
didate proof; skeptical verifiers audit it; a corrector repairs the
gaps; only candidates passing every check reach a human re-
viewer. The system also improves over time by adopting new,
sandbox-validated tactics. Its guiding premise—of interest be-
yond this system—is that the effort reasonable to spend on a
proof, or on its verification, for a successful attempt can be es-
timated in advance from a small model’s internal activations.
The pipeline runs entirely on open-weight models, making it
reproducible and inexpensive.

## 1.2 Goals

Index Terms: AI for mathematics; multi-agent reasoning; effi-
cient inference-time compute; automated theorem proving; geom-
etry and topology of neural representations.

# 1 Introduction

Judging a problem’s difficulty before attempting it is routine for
mathematicians and essential to allocating effort efficiently. We
turn this intuition into a single control variable—the per-problem
token budget—and ask whether a multi-agent system can predict
it, allocate it well, and improve at both. The system builds on
established propose–check–repair pipelines [4, 5, 8–12] and tool
use [6], keeping the solver offline until an independent verifier val-
idates each attempt.

## 1.3 Validation Protocol

Datasets.
Competition mathematics with known answers—the
MATH benchmark [3] and harder olympiad problems—allowing
automatic grading of final answers, with a human auditor checking
reasoning on a sample and on a few open-ended problems. The
predictor is lightweight (a probe plus an allocation rule) and needs
no backbone retraining.

Architecture.
Every call to the backbone first passes the bud-
get oracle O, which predicts a budget �B(x) and caps the response
length. A prover produces a candidate proof; an ensemble of k≥3
skeptical verifiers audits it; an offline corrector repairs the gaps; the
loop iterates until the verifiers concur. Accepted candidates pass to
a human gate—the single point of human sign-off. Independently,
an orchestrator proposes and sandbox-tests improved tactics before
deployment.

Metrics.
Problems-solved-per-token (area under the success-
versus-compute curve); tokens to first valid solution; budget-
prediction accuracy; wrongly accepted proofs at the gate; improve-
ment across rounds; and geometric/topological measures of the dif-
ficulty representation.

Baselines. Uniform budget; length-based budget; repeated sam-
pling / self-consistency [8] at matched compute; reduced verifier
count; and the system with the oracle disabled.

Formulation.
Let pi(B) be the probability of solving xi within B
tokens. Given a fixed total budget, the oracle allocates each ad-
ditional token where it yields the greatest marginal gain in suc-
cess [7]: max{Bi} ∑i pi(Bi) subject to ∑i Bi ≤Btot. It predicts ef-
fort from two inexpensive signals—surface features of the state-
ment φ(x) and a small model’s internal activations gℓ(x)—via
�B(x) = fθ(φ(x),gℓ(x)), updated online as outcomes accrue.

Timeline. Online weeks build the harness and a fixed-budget base-
line; week 1 trains the predictor and runs the allocation experi-
ments; week 2 adds self-improvement and the geometry/topology
analysis, with first results on the last day.

# References