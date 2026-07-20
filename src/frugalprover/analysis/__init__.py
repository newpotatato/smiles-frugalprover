"""Research analyses. Consumers of the pipeline's artifacts, not stages of it.

These answer questions about the data rather than producing anything the
pipeline depends on: which layer carries signal, whether activations beat a
text baseline, how the oracle's accuracy grows as labels accrue, and what the
representation geometry looks like against difficulty.

Each is runnable as a module::

    python -m frugalprover.analysis.layer_probe --problems data/pilot/problems.jsonl \\
        --hidden-states data/pilot/hidden_states.parquet \\
        --budgets data/pilot/budgets.jsonl --out-dir results/analysis

They read the artifacts through `_records.load_records`, which flattens the
three files into the single-record shape they were originally written against.
New analyses should use `OracleDataset` directly instead.
"""
from frugalprover.analysis.id_estimators import mle_dimension, twonn_dimension

__all__ = ["twonn_dimension", "mle_dimension"]
