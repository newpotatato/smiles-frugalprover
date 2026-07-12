"""FrugalProver pilot — shared library.

Common code used across the research phases (see docs/RESEARCH_PLAN.md):
data/results path anchoring (`paths`) and intrinsic-dimension estimators
(`id_estimators`). Install once with `pip install -e .` from the repo root so
every script and notebook can `from frugalprover import ...` regardless of the
directory it runs from.
"""
