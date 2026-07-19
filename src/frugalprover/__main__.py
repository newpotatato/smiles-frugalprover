"""Allows `python -m frugalprover ...` when the console script isn't on PATH."""
from frugalprover.cli import main

raise SystemExit(main())
