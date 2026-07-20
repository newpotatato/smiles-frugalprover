"""Stage 5 - collect results. Produces `results/<run_name>/` (A5)."""
from __future__ import annotations

from frugalprover.oracle.reporting.report import run_report
from frugalprover.oracle.reporting.runs import collect_runs, print_runs_table

__all__ = ["run_report", "collect_runs", "print_runs_table"]
